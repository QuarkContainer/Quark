// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use kvm_ioctls::{Kvm, VmFd};
use kvm_bindings::{kvm_userspace_memory_region, KVM_CAP_X86_DISABLE_EXITS, kvm_enable_cap, KVM_X86_DISABLE_EXITS_HLT, KVM_X86_DISABLE_EXITS_MWAIT};
use spin::Mutex;
use alloc::sync::Arc;
use std::{thread};
use core::sync::atomic::AtomicI32;
use core::sync::atomic::Ordering;
use lazy_static::lazy_static;
use std::os::unix::io::FromRawFd;

use super::super::super::qlib::common::*;
use super::super::super::qlib::pagetable::{PageTables};
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::ShareSpace;
use super::super::super::qlib::addr::AccessType;
use super::super::super::qlib::addr;
use super::super::super::qlib::config::*;
use super::super::super::qlib::perf_tunning::*;
use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::qmsg::*;
use super::super::super::syncmgr;
use super::super::super::runc::runtime::loader::*;
use super::super::super::qcall;
use super::super::super::kvm_vcpu::*;
use super::super::super::elf_loader::*;
use super::super::super::vmspace::*;
use super::super::super::{FD_NOTIFIER, VMS, PMA_KEEPER};
use super::super::super::ucall::ucall_server::*;

lazy_static! {
    static ref EXIT_STATUS : AtomicI32 = AtomicI32::new(-1);
}

#[inline]
pub fn IsRunning() -> bool {
    return EXIT_STATUS.load(Ordering::Relaxed) == -1
}

pub fn SetExitStatus(status: i32) {
    EXIT_STATUS.store(status, Ordering::Release);
}

pub fn GetExitStatus() -> i32 {
    return EXIT_STATUS.load(Ordering::Acquire)
}

pub struct BootStrapMem {
    pub startAddr: u64,
    pub vcpuCount: usize,
}

pub const KERNEL_HEAP_ORD : usize = 32; // 4GB
pub const PAGE_POOL_ORD: usize = KERNEL_HEAP_ORD - 8;

impl BootStrapMem {
    pub const PAGE_POOL_SIZE : usize = 1 << PAGE_POOL_ORD;

    pub fn New(startAddr: u64, vcpuCount: usize) -> Self {
        return Self {
            startAddr: startAddr,
            vcpuCount: vcpuCount,
        }
    }

    pub fn Size(&self) -> usize {
        let size = self.vcpuCount * VcpuBootstrapMem::AlignedSize() + Self::PAGE_POOL_SIZE;
        return size;
    }

    pub fn VcpuBootstrapMem(&self, idx: usize) -> &'static VcpuBootstrapMem {
        let addr = self.startAddr + (idx * VcpuBootstrapMem::AlignedSize()) as u64;
        return VcpuBootstrapMem::FromAddr(addr);
    }

    pub fn SimplePageAllocator(&self) -> SimplePageAllocator {
        let addr = self.startAddr + (self.vcpuCount * VcpuBootstrapMem::AlignedSize()) as u64;
        return SimplePageAllocator::New(addr, Self::PAGE_POOL_SIZE)
    }
}

pub struct VirtualMachine {
    pub kvm: Kvm,
    pub vmfd: VmFd,
    pub vcpus: Vec<Arc<Mutex<KVMVcpu>>>,
    pub elf: KernelELF,
}

impl VirtualMachine {
    pub fn SetMemRegion(slotId: u32, vm_fd: &VmFd, phyAddr: u64, hostAddr: u64, pageMmapsize: u64) -> Result<()> {
        info!("SetMemRegion phyAddr = {:x}, hostAddr={:x}; pageMmapsize = {:x} MB", phyAddr, hostAddr, (pageMmapsize >> 20));

        let mem_region = kvm_userspace_memory_region {
            slot: slotId,
            guest_phys_addr: phyAddr,
            memory_size: pageMmapsize,
            userspace_addr: hostAddr,
            flags: 0, //kvm_bindings::KVM_MEM_LOG_DIRTY_PAGES,
        };

        unsafe {
            vm_fd.set_user_memory_region(mem_region).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        }

        return Ok(())
    }

    pub fn Umask() -> u32 {
        let umask = unsafe{
            libc::umask(0)
        };

        return umask
    }

    #[cfg(debug_assertions)]
    pub const KERNEL_IMAGE : &'static str = "/usr/local/bin/qkernel_d.bin";

    #[cfg(not(debug_assertions))]
    pub const KERNEL_IMAGE : &'static str = "/usr/local/bin/qkernel.bin";

    pub fn Init(args: Args /*args: &Args, kvmfd: i32*/) -> Result<Self> {
        PerfGoto(PerfType::Other);

        /*{
            use super::super::super::qlib::cpuid::*;
            let featureSet = HostFeatureSet();
            error!("Host::hasXSAVEOPT is {}", featureSet.UseXsaveopt());
            error!("Host::hasXSAVE is {}", featureSet.UseXsave());
            error!("Host::hasFSGSBASE is {}", featureSet.HasFeature(Feature(X86Feature::X86FeatureFSGSBase as i32)));
            error!("Host::X86FeatureXSAVEOPT is {}", featureSet.HasFeature(Feature(X86Feature::X86FeatureXSAVEOPT as i32)));
            error!("Host::X86FeatureXSAVE is {}", featureSet.HasFeature(Feature(X86Feature::X86FeatureXSAVE as i32)));
            error!("Host::X86FeatureOSXSAVE is {}", featureSet.HasFeature(Feature(X86Feature::X86FeatureOSXSAVE as i32)));
        }*/

        let mut config = Config::default();
        config.Load();

        let kvmfd = args.KvmFd;
        let cpuCount = VMSpace::VCPUCount() - 1;
        let kernelMemRegionSize = config.KernelMemSize;
        let controlSock = args.ControlSock;

        let umask = Self::Umask();
        info!("reset umask from {:o} to {}, kernelMemRegionSize is {:x}", umask, 0, kernelMemRegionSize);

        let eventfd = FD_NOTIFIER.Eventfd();
        let kvm = unsafe { Kvm::from_raw_fd(kvmfd) };

        let KVM_MAX_CPUID_ENTRIES = 0x100;
        let kvm_cpuid = kvm.get_supported_cpuid(KVM_MAX_CPUID_ENTRIES).unwrap();

        let vm_fd = kvm.create_vm().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        let mut cap: kvm_enable_cap = Default::default();
        cap.cap = KVM_CAP_X86_DISABLE_EXITS;
        cap.args[0] = (KVM_X86_DISABLE_EXITS_HLT | KVM_X86_DISABLE_EXITS_MWAIT) as u64;
        vm_fd.enable_cap(&cap).unwrap();

        let mut elf = KernelELF::Init(&String::from(Self::KERNEL_IMAGE))?;
        info!("the end address is {:x}", elf.EndAddr().0);
        Self::SetMemRegion(1, &vm_fd, MemoryDef::PHY_LOWER_ADDR, MemoryDef::PHY_LOWER_ADDR, kernelMemRegionSize * MemoryDef::ONE_GB)?;

        info!("set map region start={:x}, end={:x}", MemoryDef::PHY_LOWER_ADDR, MemoryDef::PHY_LOWER_ADDR + 16 * MemoryDef::ONE_GB);

        let pageAllocatorBaseAddr;
        let pageAllocatorOrd;
        let autoStart;
        let bootstrapMem;

        {
            let memOrd = KERNEL_HEAP_ORD; // 8GB
            let kernelMemSize = 1 << memOrd;
            //pageMmap = KVMMachine::initKernelMem(&vm_fd, MemoryDef::PHY_LOWER_ADDR  + 64 * MemoryDef::ONE_MB, kernelMemSize)?;
            //pageAllocatorBaseAddr = pageMmap.as_ptr() as u64;

            info!("kernelMemSize is {:x}", kernelMemSize);
            let vms = &mut VMS.lock();
            //pageAllocatorBaseAddr = vms.pmaMgr.MapAnon(kernelMemSize, AccessType::ReadWrite().Val() as i32, true, false)?;
            pageAllocatorBaseAddr = PMA_KEEPER.lock().MapAnon(kernelMemSize, AccessType::ReadWrite().Val() as i32)?;
            info!("*******************alloc address is {:x}, expect{:x}", pageAllocatorBaseAddr, MemoryDef::PHY_LOWER_ADDR + MemoryDef::ONE_GB);

            //pageAlloc = PageAllocator::Init(pageMmap.as_ptr() as u64, memOrd - 12 /*1GB*/);
            pageAllocatorOrd = memOrd - 12 /*1GB*/;
            bootstrapMem = BootStrapMem::New(pageAllocatorBaseAddr, cpuCount);
            vms.allocator = Some(bootstrapMem.SimplePageAllocator());

            vms.hostAddrTop = MemoryDef::PHY_LOWER_ADDR + 64 * MemoryDef::ONE_MB + 2 * MemoryDef::ONE_GB;
            vms.pageTables = PageTables::New(vms.allocator.as_ref().unwrap())?;

            info!("the pageAllocatorBaseAddr is {:x}, the end of pageAllocator is {:x}", pageAllocatorBaseAddr, pageAllocatorBaseAddr + kernelMemSize);
            vms.KernelMapHugeTable(addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                                   addr::Addr(MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB),
                                   addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                                   addr::PageOpts::Zero().SetPresent().SetWrite().SetGlobal().Val())?;
            autoStart = args.AutoStart;
            vms.pivot = args.Pivot;
            vms.args = Some(args);
        }

        info!("before loadKernel");

        let entry = elf.LoadKernel()?;
        //let vdsoMap = VDSOMemMap::Init(&"/home/brad/rust/quark/vdso/vdso.so".to_string()).unwrap();
        elf.LoadVDSO(&"/usr/local/bin/vdso.so".to_string())?;
        VMS.lock().vdsoAddr = elf.vdsoStart;

        let p = entry as *const u8;
        info!("entry is 0x{:x}, data at entry is {:x}", entry, unsafe { *p } );

        //let usocket = USocket::InitServer(&ControlSocketAddr(&containerId))?;
        //let usocket = USocket::CreateServer(&ControlSocketAddr(&containerId), usockfd)?;
        InitUCallController(controlSock)?;

        {
            super::super::super::URING_MGR.lock();
        }

        let mut vcpus = Vec::with_capacity(cpuCount);
        for i in 0..cpuCount/*args.NumCPU*/ {
            let vcpu = Arc::new(Mutex::new(KVMVcpu::Init(i as usize,
                                                         cpuCount,
                                                         &vm_fd,
                                                         &bootstrapMem,
                                                         entry, pageAllocatorBaseAddr,
                                                         pageAllocatorOrd as u64,
                                                         eventfd,
                                                         autoStart)?));

            // enable cpuid in host
            vcpu.lock().vcpu.set_cpuid2(&kvm_cpuid).unwrap();
            vcpus.push(vcpu);
        }

        let vm = Self {
            kvm: kvm,
            vmfd: vm_fd,
            vcpus: vcpus,
            elf: elf,
        };

        PerfGofrom(PerfType::Other);
        Ok(vm)
    }

    pub fn run(&mut self) -> Result<i32> {
        let cpu = self.vcpus[0].clone();

        let mut threads = Vec::new();

        threads.push(thread::spawn(move || {
            cpu.lock().run().expect("vcpu run fail");
            info!("cpu#{} finish", 0);
        }));

        syncmgr::SyncMgr::WaitShareSpaceReady();
        info!("shareSpace ready...");

        for i in 1..self.vcpus.len() {
            let cpu = self.vcpus[i].clone();
            cpu.lock().shareSpace = VMS.lock().GetShareSpace();

            threads.push(thread::spawn(move || {
                info!("cpu#{} start", i);
                cpu.lock().run().expect("vcpu run fail");
                info!("cpu#{} finish", i);
            }));
        }

        threads.push(thread::spawn(move || {
            UcallSrvProcess().unwrap();
            info!("UcallSrvProcess finish...");
        }));


        threads.push(thread::spawn(move || {
            Self::Process();
            info!("IOThread  finish...");
        }));

        for t in threads {
            t.join().expect("the working threads has panicked");
        }

        Ok(GetExitStatus())
    }

    pub fn WakeAll(shareSpace: &ShareSpace) {
        shareSpace.scheduler.WakeAll();
    }

    pub fn Schedule(shareSpace: &ShareSpace, taskId: TaskIdQ) {
        shareSpace.scheduler.ScheduleQ(taskId.TaskId(), taskId.Queue());
    }

    pub const EVENT_COUNT: usize = 128;

    pub fn Process() {
        let shareSpace = VMS.lock().GetShareSpace();

        'main: loop {
            //PerfGoto(PerfType::QCall);
            while shareSpace.ReadyOutputMsgCnt() > 0 {
                unsafe {
                    let msg = shareSpace.AQHostOutputPop();

                    match msg {
                        None => {
                            llvm_asm!("pause" :::: "volatile");
                            //error!("get none output msg ...");
                        },
                        Some(HostOutputMsg::QCall(addr)) => {
                            let eventAddr = addr as *mut Event; // as &mut qlib::Event;
                            let event = &mut (*eventAddr);
                            let currTaskId = event.taskId;

                            //error!("qcall event is {:x?}", &event);

                            match qcall::qCall(addr, event) {
                                qcall::QcallRet::Normal => {
                                    if currTaskId.Addr() != 0 {
                                        Self::Schedule(shareSpace, currTaskId);
                                    }
                                }
                                qcall::QcallRet::Block => {
                                    //info!("start blocked wait ...........");
                                }
                            }
                        }
                        Some(msg) => {
                            //error!("qcall msg is {:x?}", &msg);
                            qcall::AQHostCall(msg, shareSpace);
                        }
                    }
                }
            }

            if !IsRunning() {
                VMS.lock().CloseVMSpace();
                return;
            }

            //PerfGofrom(PerfType::QCall);
            FD_NOTIFIER.WaitAndNotify(shareSpace, 0).unwrap();

            for _ in 0..10 {
                for _ in 0..20 {
                    if shareSpace.ReadyOutputMsgCnt() > 0 {
                        continue 'main
                    }

                    unsafe { llvm_asm!("pause" :::: "volatile"); }
                    unsafe { llvm_asm!("pause" :::: "volatile"); }
                    /*unsafe { llvm_asm!("pause" :::: "volatile"); }
                    unsafe { llvm_asm!("pause" :::: "volatile"); }
                    unsafe { llvm_asm!("pause" :::: "volatile"); }*/
                }
            }

            loop {
                //PerfGoto(PerfType::IdleWait);
                shareSpace.WaitInHost();
                if shareSpace.ReadyOutputMsgCnt() > 0 {
                    shareSpace.WakeInHost();
                    break;
                }

                //error!("io thread sleep... shareSpace.ReadyOutputMsgCnt() = {}", shareSpace.ReadyOutputMsgCnt());
                let _cnt = FD_NOTIFIER.WaitAndNotify(shareSpace, -1).unwrap();
                //error!("io thread wake...");

                if !IsRunning() {
                    VMS.lock().CloseVMSpace();
                    return;
                }
                shareSpace.WakeInHost();
                //PerfGofrom(PerfType::IdleWait);

                if shareSpace.ReadyOutputMsgCnt() > 0 {
                    break;
                }
            }
        }
    }
}
