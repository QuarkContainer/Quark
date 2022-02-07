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
//use kvm_bindings::{kvm_userspace_memory_region, KVM_CAP_X86_DISABLE_EXITS, kvm_enable_cap, KVM_X86_DISABLE_EXITS_HLT, KVM_X86_DISABLE_EXITS_MWAIT};
use kvm_bindings::*;
use alloc::sync::Arc;
use std::{thread};
use core::sync::atomic::AtomicI32;
use core::sync::atomic::Ordering;
use lazy_static::lazy_static;
use std::os::unix::io::FromRawFd;
use std::os::unix::io::AsRawFd;

use super::super::super::qlib::common::*;
use super::super::super::qlib::pagetable::{PageTables};
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::ShareSpace;
use super::super::super::SHARE_SPACE_STRUCT;
use super::super::super::SHARE_SPACE;
use super::super::super::qlib::addr;
use super::super::super::qlib::perf_tunning::*;
use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::kernel::SHARESPACE;
use super::super::super::qlib::kernel::KERNEL_STACK_ALLOCATOR;
use super::super::super::qlib::kernel::kernel::timer;
use super::super::super::qlib::kernel::kernel::futex;
use super::super::super::qlib::kernel::task;
use super::super::super::qlib::kernel::IOURING;
use super::super::super::qlib::kernel::KERNEL_PAGETABLE;
use super::super::super::qlib::kernel::PAGE_MGR;
use super::super::super::qlib::kernel::guestfdnotifier::GUEST_NOTIFIER;
use super::super::super::qlib::kernel::vcpu;
use super::super::super::qlib::pagetable::AlignedAllocator;
use super::super::super::print::LOG;
use super::super::super::syncmgr;
use super::super::super::runc::runtime::loader::*;
use super::super::super::kvm_vcpu::*;
use super::super::super::elf_loader::*;
use super::super::super::vmspace::*;
use super::super::super::{VMS, ROOT_CONTAINER_ID, PMA_KEEPER, QUARK_CONFIG, URING_MGR, KERNEL_IO_THREAD, THREAD_ID, ThreadId};

lazy_static! {
    static ref EXIT_STATUS : AtomicI32 = AtomicI32::new(-1);
}

const HEAP_OFFSET: u64 = 1 * MemoryDef::ONE_GB;

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


pub const KERNEL_HEAP_ORD : usize = 33; // 16GB


pub struct VirtualMachine {
    pub kvm: Kvm,
    pub vmfd: VmFd,
    pub vcpus: Vec<Arc<KVMVcpu>>,
    pub elf: KernelELF,
}

impl VirtualMachine {
    pub fn SetMemRegion(slotId: u32, vm_fd: &VmFd, phyAddr: u64, hostAddr: u64, pageMmapsize: u64) -> Result<()> {
        info!("SetMemRegion phyAddr = {:x}, hostAddr={:x}; pageMmapsize = {:x} MB", phyAddr, hostAddr, (pageMmapsize >> 20));

        // guest_phys_addr must be <512G
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

    pub const KVM_IOEVENTFD_FLAG_DATAMATCH : u32 = (1 << kvm_ioeventfd_flag_nr_datamatch);
    pub const KVM_IOEVENTFD_FLAG_PIO       : u32 =(1 << kvm_ioeventfd_flag_nr_pio);
    pub const KVM_IOEVENTFD_FLAG_DEASSIGN  : u32 =(1 << kvm_ioeventfd_flag_nr_deassign);
    pub const KVM_IOEVENTFD_FLAG_VIRTIO_CCW_NOTIFY : u32 = (1 << kvm_ioeventfd_flag_nr_virtio_ccw_notify);

    pub const KVM_IOEVENTFD : u64 = 0x4040ae79;

    pub fn IoEventfdAddEvent(vmfd: i32, addr: u64, eventfd: i32) {
        let kvmIoEvent = kvm_ioeventfd {
            addr: addr,
            len: 8,
            datamatch: 1,
            fd: eventfd,
            flags: Self::KVM_IOEVENTFD_FLAG_DATAMATCH,
            ..Default::default()
        };

        let ret = unsafe {
            libc::ioctl(vmfd, Self::KVM_IOEVENTFD, &kvmIoEvent as * const _ as u64)
        };

        assert!(ret ==0, "IoEventfdAddEvent ret is {}/{}/{}", ret, errno::errno().0, vmfd.as_raw_fd());
    }

    #[cfg(debug_assertions)]
    pub const KERNEL_IMAGE : &'static str = "/usr/local/bin/qkernel_d.bin";

    #[cfg(not(debug_assertions))]
    pub const KERNEL_IMAGE : &'static str = "/usr/local/bin/qkernel.bin";

    pub fn InitShareSpace(vmfd: &VmFd, cpuCount: usize, controlSock: i32) {
        SHARE_SPACE_STRUCT.lock().Init(cpuCount, controlSock);
        let spAddr = &(*SHARE_SPACE_STRUCT.lock()) as * const _ as u64;
        SHARE_SPACE.SetValue(spAddr);
        SHARESPACE.SetValue(spAddr);

        unsafe {
            vcpu::CPU_LOCAL.Init(&SHARESPACE.scheduler.VcpuArr);
        }

        let sharespace = SHARE_SPACE.Ptr();
        let logfd = super::super::super::print::LOG.lock().Logfd();
        URING_MGR.lock().Init(sharespace.config.read().DedicateUring);
        URING_MGR.lock().Addfd(logfd).unwrap();

        for i in 0..cpuCount {
            let addr = MemoryDef::KVM_IOEVENTFD_BASEADDR + (i as u64) * 8;
            Self::IoEventfdAddEvent(vmfd.as_raw_fd(), addr, sharespace.scheduler.VcpuArr[i].eventfd);
        }

        KERNEL_IO_THREAD.Init(sharespace.scheduler.VcpuArr[0].eventfd);
        URING_MGR.lock().SetupEventfd(sharespace.scheduler.VcpuArr[0].eventfd);
        URING_MGR.lock().Addfd(sharespace.HostHostEpollfd()).unwrap();
        URING_MGR.lock().Addfd(controlSock).unwrap();
        sharespace.SetIOUringsAddr(URING_MGR.lock().IOUringsAddr());
        IOURING.SetValue(sharespace.GetIOUringAddr());
        GUEST_NOTIFIER.SetValue(sharespace.GuestNotifierAddr());

        unsafe {
            KERNEL_PAGETABLE.SetRoot(VMS.lock().pageTables.GetRoot());
            PAGE_MGR.SetValue(sharespace.GetPageMgrAddr());

            KERNEL_STACK_ALLOCATOR.Init(AlignedAllocator::New(
                MemoryDef::DEFAULT_STACK_SIZE as usize,
                MemoryDef::DEFAULT_STACK_SIZE as usize,
            ));

            task::InitSingleton();
            futex::InitSingleton();
            timer::InitSingleton();
        }

        let syncPrint = sharespace.config.read().SyncPrint();
        super::super::super::print::SetSharespace(sharespace);
        super::super::super::print::SetSyncPrint(syncPrint);
    }

    pub fn Init(args: Args /*args: &Args, kvmfd: i32*/) -> Result<Self> {
        PerfGoto(PerfType::Other);

        *ROOT_CONTAINER_ID.lock() = args.ID.clone();
        if QUARK_CONFIG.lock().PerSandboxLog {
            LOG.lock().Reset(&args.ID[0..12]);
        }

        let kvmfd = args.KvmFd;

        let cnt = QUARK_CONFIG.lock().DedicateUring;

        if QUARK_CONFIG.lock().EnableRDMA {
            // use default rdma device
            let rdmaDeviceName = "";
            let lbPort = QUARK_CONFIG.lock().RDMAPort;
            super::super::super::vmspace::HostFileMap::rdma::RDMA.Init(rdmaDeviceName, lbPort);
        }

        let reserveCpuCount = QUARK_CONFIG.lock().ReserveCpuCount;
        let cpuCount = VMSpace::VCPUCount() - cnt - reserveCpuCount;
        VMS.lock().vcpuCount = cpuCount; //VMSpace::VCPUCount();
        VMS.lock().RandomVcpuMapping();
        let kernelMemRegionSize = QUARK_CONFIG.lock().KernelMemSize;
        let controlSock = args.ControlSock;

        let umask = Self::Umask();
        info!("reset umask from {:o} to {}, kernelMemRegionSize is {:x}", umask, 0, kernelMemRegionSize);

        let kvm = unsafe { Kvm::from_raw_fd(kvmfd) };

        let kvm_cpuid = kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES).unwrap();

        let vm_fd = kvm.create_vm().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        let mut cap: kvm_enable_cap = Default::default();
        cap.cap = KVM_CAP_X86_DISABLE_EXITS;
        cap.args[0] = (KVM_X86_DISABLE_EXITS_HLT | KVM_X86_DISABLE_EXITS_MWAIT) as u64;
        vm_fd.enable_cap(&cap).unwrap();

        let mut elf = KernelELF::New()?;
        Self::SetMemRegion(1, &vm_fd, MemoryDef::PHY_LOWER_ADDR, MemoryDef::PHY_LOWER_ADDR, kernelMemRegionSize * MemoryDef::ONE_GB)?;
        let memOrd = KERNEL_HEAP_ORD;
        let kernelMemSize = 1 << memOrd;
        let heapStartAddr = MemoryDef::PHY_LOWER_ADDR + HEAP_OFFSET;
        PMA_KEEPER.Init(heapStartAddr + kernelMemSize, kernelMemRegionSize * MemoryDef::ONE_GB - HEAP_OFFSET - kernelMemSize);

        info!("set map region start={:x}, end={:x}", MemoryDef::PHY_LOWER_ADDR, MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB);

        let autoStart;

        {
            info!("kernelMemSize is {:x}", kernelMemSize);
            let vms = &mut VMS.lock();
            vms.controlSock = controlSock;
            PMA_KEEPER.InitHugePages();

            vms.hostAddrTop = MemoryDef::PHY_LOWER_ADDR + 64 * MemoryDef::ONE_MB + 2 * MemoryDef::ONE_GB;
            vms.pageTables = PageTables::New(&vms.allocator)?;

            vms.KernelMap(addr::Addr(MemoryDef::KVM_IOEVENTFD_BASEADDR),
                          addr::Addr(MemoryDef::KVM_IOEVENTFD_BASEADDR + 0x1000),
                          addr::Addr(MemoryDef::KVM_IOEVENTFD_BASEADDR),
                          addr::PageOpts::Zero().SetPresent().SetWrite().SetGlobal().Val())?;

            //info!("the pageAllocatorBaseAddr is {:x}, the end of pageAllocator is {:x}", pageAllocatorBaseAddr, pageAllocatorBaseAddr + kernelMemSize);
            vms.KernelMapHugeTable(addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                                   addr::Addr(MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB),
                                   addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                                   addr::PageOpts::Zero().SetPresent().SetWrite().SetGlobal().Val())?;
            autoStart = args.AutoStart;
            vms.pivot = args.Pivot;
            vms.args = Some(args);
        }

        Self::InitShareSpace(&vm_fd, cpuCount, controlSock);

        info!("before loadKernel");

        let entry = elf.LoadKernel(Self::KERNEL_IMAGE)?;
        //let vdsoMap = VDSOMemMap::Init(&"/home/brad/rust/quark/vdso/vdso.so".to_string()).unwrap();
        elf.LoadVDSO(&"/usr/local/bin/vdso.so".to_string())?;
        VMS.lock().vdsoAddr = elf.vdsoStart;

        let p = entry as *const u8;
        info!("entry is 0x{:x}, data at entry is {:x}, heapStartAddr is {:x}", entry, unsafe { *p } , heapStartAddr);

        {
            super::super::super::URING_MGR.lock();
        }

        let mut vcpus = Vec::with_capacity(cpuCount);
        for i in 0..cpuCount/*args.NumCPU*/ {
            let vcpu = Arc::new(KVMVcpu::Init(i as usize,
                                                cpuCount,
                                                &vm_fd,
                                                entry,
                                                heapStartAddr,
                                                SHARE_SPACE.Value(),
                                                autoStart)?);

            // enable cpuid in host
            vcpu.vcpu.set_cpuid2(&kvm_cpuid).unwrap();
            VMS.lock().vcpus.push(vcpu.clone());
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
        let tgid = unsafe {
            libc::gettid()
        };

        threads.push(thread::Builder::new().name("0".to_string()).spawn(move || {
            THREAD_ID.with ( |f| {
                *f.borrow_mut() = 0;
            });
            cpu.run(tgid).expect("vcpu run fail");
            info!("cpu#{} finish", 0);
        }).unwrap());

        syncmgr::SyncMgr::WaitShareSpaceReady();
        info!("shareSpace ready...");

        for i in 1..self.vcpus.len() {
            let cpu = self.vcpus[i].clone();

            threads.push(thread::Builder::new().name(format!("{}", i)).spawn(move || {
                THREAD_ID.with ( |f| {
                    *f.borrow_mut() = i as i32;
                });
                info!("cpu#{} start", ThreadId());
                cpu.run(tgid).expect("vcpu run fail");
                info!("cpu#{} finish", ThreadId());
            }).unwrap());
        }

        for t in threads {
            t.join().expect("the working threads has panicked");
        }
        Ok(GetExitStatus())
    }

    pub fn WakeAll(shareSpace: &ShareSpace) {
        shareSpace.scheduler.WakeAll();
    }

    pub fn Schedule(shareSpace: &ShareSpace, taskId: TaskId) {
        shareSpace.scheduler.ScheduleQ(taskId, taskId.Queue());
    }

    pub fn PrintQ(shareSpace: &ShareSpace, vcpuId: u64) -> String {
        return shareSpace.scheduler.PrintQ(vcpuId)
    }
}

