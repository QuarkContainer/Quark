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

use alloc::sync::Arc;
use core::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::os::unix::io::AsRawFd;
use std::os::unix::io::FromRawFd;
use std::thread;

use kvm_bindings::*;
use kvm_ioctls::{Cap, Kvm, VmFd};
use lazy_static::lazy_static;
use nix::sys::signal;

use crate::qlib::MAX_VCPU_COUNT;
use crate::tsot_agent::TSOT_AGENT;
use crate::FD_NOTIFIER;
//use crate::vmspace::hibernate::HiberMgr;

use super::super::super::elf_loader::*;
use super::super::super::kvm_vcpu::*;
use super::super::super::print::LOG;
use super::super::super::qlib::addr;
use super::super::super::qlib::common::*;
use super::super::super::qlib::kernel::kernel::futex;
use super::super::super::qlib::kernel::kernel::timer;
use super::super::super::qlib::kernel::task;
use super::super::super::qlib::kernel::vcpu;
use super::super::super::qlib::kernel::IOURING;
use super::super::super::qlib::kernel::KERNEL_PAGETABLE;
use super::super::super::qlib::kernel::KERNEL_STACK_ALLOCATOR;
use super::super::super::qlib::kernel::PAGE_MGR;
use super::super::super::qlib::kernel::SHARESPACE;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::pagetable::AlignedAllocator;
use super::super::super::qlib::pagetable::PageTables;
use super::super::super::qlib::perf_tunning::*;
use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::ShareSpace;
use super::super::super::runc::runtime::loader::*;
use super::super::super::syncmgr;
use super::super::super::vmspace::*;
use super::super::super::SHARE_SPACE;
use super::super::super::SHARE_SPACE_STRUCT;
use super::super::super::{
    ThreadId, KERNEL_IO_THREAD, PMA_KEEPER, QUARK_CONFIG, ROOT_CONTAINER_ID, THREAD_ID, URING_MGR,
    VCPU, VMS,
};

pub const SANDBOX_UID_NAME: &str = "io.kubernetes.cri.sandbox-uid";

lazy_static! {
    static ref EXIT_STATUS: AtomicI32 = AtomicI32::new(-1);
    static ref DUMP: AtomicU64 = AtomicU64::new(0);
}

#[inline]
pub fn IsRunning() -> bool {
    return EXIT_STATUS.load(Ordering::Relaxed) == -1;
}

pub fn SetExitStatus(status: i32) {
    EXIT_STATUS.store(status, Ordering::Release);
}

pub fn GetExitStatus() -> i32 {
    return EXIT_STATUS.load(Ordering::Acquire);
}

pub fn Dump(id: usize) -> bool {
    assert!(id < MAX_VCPU_COUNT);
    return DUMP.load(Ordering::Acquire) & (0x1 << id) > 0;
}

pub fn SetDumpAll() {
    DUMP.store(u64::MAX, Ordering::Release);
}

pub fn ClearDump(id: usize) {
    assert!(id < MAX_VCPU_COUNT);
    DUMP.fetch_and(!(0x1 << id), Ordering::Release);
}

pub struct VirtualMachine {
    pub kvm: Kvm,
    pub vmfd: VmFd,
    pub vcpus: Vec<Arc<KVMVcpu>>,
    pub elf: KernelELF,
}

impl VirtualMachine {
    pub fn SetMemRegion(
        slotId: u32,
        vm_fd: &VmFd,
        phyAddr: u64,
        hostAddr: u64,
        pageMmapsize: u64,
    ) -> Result<()> {
        info!(
            "SetMemRegion phyAddr = {:x}, hostAddr={:x}; pageMmapsize = {:x} MB",
            phyAddr,
            hostAddr,
            (pageMmapsize >> 20)
        );

        // guest_phys_addr must be <512G
        let mem_region = kvm_userspace_memory_region {
            slot: slotId,
            guest_phys_addr: phyAddr,
            memory_size: pageMmapsize,
            userspace_addr: hostAddr,
            flags: 0, //kvm_bindings::KVM_MEM_LOG_DIRTY_PAGES,
        };

        unsafe {
            vm_fd
                .set_user_memory_region(mem_region)
                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        }

        return Ok(());
    }

    pub fn Umask() -> u32 {
        let umask = unsafe { libc::umask(0) };

        return umask;
    }

    pub const KVM_IOEVENTFD_FLAG_DATAMATCH: u32 = (1 << kvm_ioeventfd_flag_nr_datamatch);
    pub const KVM_IOEVENTFD_FLAG_PIO: u32 = (1 << kvm_ioeventfd_flag_nr_pio);
    pub const KVM_IOEVENTFD_FLAG_DEASSIGN: u32 = (1 << kvm_ioeventfd_flag_nr_deassign);
    pub const KVM_IOEVENTFD_FLAG_VIRTIO_CCW_NOTIFY: u32 =
        (1 << kvm_ioeventfd_flag_nr_virtio_ccw_notify);

    pub const KVM_IOEVENTFD: u64 = 0x4040ae79;

    pub fn IoEventfdAddEvent(vmfd: i32, addr: u64, eventfd: i32) {
        let kvmIoEvent = kvm_ioeventfd {
            addr: addr,
            len: 8,
            datamatch: 1,
            fd: eventfd,
            flags: Self::KVM_IOEVENTFD_FLAG_DATAMATCH,
            ..Default::default()
        };

        let ret = unsafe { libc::ioctl(vmfd, Self::KVM_IOEVENTFD, &kvmIoEvent as *const _ as u64) };

        assert!(
            ret == 0,
            "IoEventfdAddEvent ret is {}/{}/{}",
            ret,
            errno::errno().0,
            vmfd.as_raw_fd()
        );
    }

    #[cfg(debug_assertions)]
    pub const KERNEL_IMAGE: &'static str = "/usr/local/bin/qkernel_d.bin";

    #[cfg(not(debug_assertions))]
    pub const KERNEL_IMAGE: &'static str = "/usr/local/bin/qkernel.bin";

    pub const VDSO_PATH: &'static str = "/usr/local/bin/vdso.so";

    pub fn InitShareSpace(cpuCount: usize, controlSock: i32, rdmaSvcCliSock: i32, podId: [u8; 64]) {
        SHARE_SPACE_STRUCT
            .lock()
            .Init(cpuCount, controlSock, rdmaSvcCliSock, podId);

        let spAddr = &(*SHARE_SPACE_STRUCT.lock()) as *const _ as u64;
        SHARE_SPACE.SetValue(spAddr);
        SHARESPACE.SetValue(spAddr);

        unsafe {
            vcpu::CPU_LOCAL.Init(&SHARESPACE.scheduler.VcpuArr);
        }

        let sharespace = SHARE_SPACE.Ptr();

        KERNEL_IO_THREAD.Init(sharespace.scheduler.VcpuArr[0].eventfd);
        FD_NOTIFIER.EpollCtlAdd(controlSock, EVENT_READ).unwrap();
        IOURING.SetValue(sharespace.GetIOUringAddr());

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

        if SHARESPACE.config.read().EnableTsot {
            // initialize the tost_agent
            TSOT_AGENT.NextReqId();
            SHARESPACE.dnsSvc.Init().unwrap();
        };

        *SHARESPACE.bootId.lock() = uuid::Uuid::new_v4().to_string();

        let syncPrint = sharespace.config.read().SyncPrint();
        super::super::super::print::SetSyncPrint(syncPrint);
    }

    pub fn Init(args: Args) -> Result<Self> {
        PerfGoto(PerfType::Other);

        *ROOT_CONTAINER_ID.lock() = args.ID.clone();
        if QUARK_CONFIG.lock().PerSandboxLog {
            let sandboxName = match args.Spec.annotations.get("io.kubernetes.cri.sandbox-name") {
                None => args.ID[0..12].to_owned(),
                Some(name) => name.clone(),
            };
            LOG.Reset(&sandboxName);
        }

        let cpuCount = args.GetCpuCount();

        let kvmfd = args.KvmFd;

        let reserveCpuCount = QUARK_CONFIG.lock().ReserveCpuCount;
        let cpuCount = if cpuCount == 0 {
            VMSpace::VCPUCount() - reserveCpuCount
        } else {
            cpuCount.min(VMSpace::VCPUCount() - reserveCpuCount)
        };

        if cpuCount < 2 {
            // only do cpu affinit when there more than 2 cores
            VMS.lock().cpuAffinit = false;
        } else {
            VMS.lock().cpuAffinit = true;
        }

        match args.Spec.annotations.get(SANDBOX_UID_NAME) {
            None => (),
            Some(podUid) => {
                VMS.lock().podUid = podUid.clone();
            }
        }

        let cpuCount = cpuCount.max(2); // minimal 2 cpus

        VMS.lock().vcpuCount = cpuCount;
        VMS.lock().RandomVcpuMapping();
        let kernelMemRegionSize = QUARK_CONFIG.lock().KernelMemSize;
        let controlSock = args.ControlSock;

        let rdmaSvcCliSock = args.RDMASvcCliSock;

        let umask = Self::Umask();
        info!(
            "reset umask from {:o} to {}, kernelMemRegionSize is {:x}",
            umask, 0, kernelMemRegionSize
        );

        let kvm = unsafe { Kvm::from_raw_fd(kvmfd) };

        #[cfg(target_arch = "x86_64")]
        let kvm_cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap();

        let vm_fd = kvm
            .create_vm()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        {
            let mut cap: kvm_enable_cap = Default::default();
            cap.cap = KVM_CAP_X86_DISABLE_EXITS;
            cap.args[0] = (KVM_X86_DISABLE_EXITS_HLT | KVM_X86_DISABLE_EXITS_MWAIT) as u64;
            vm_fd.enable_cap(&cap).unwrap();
        }

        if !kvm.check_extension(Cap::ImmediateExit) {
            panic!("KVM_CAP_IMMEDIATE_EXIT not supported");
        }

        let mut elf = KernelELF::New()?;
        Self::SetMemRegion(
            1,
            &vm_fd,
            MemoryDef::PHY_LOWER_ADDR,
            MemoryDef::PHY_LOWER_ADDR,
            MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB,
        )?;

        let heapStartAddr = MemoryDef::HEAP_OFFSET;

        PMA_KEEPER.Init(MemoryDef::FILE_MAP_OFFSET, MemoryDef::FILE_MAP_SIZE);

        info!(
            "set map region start={:x}, end={:x}",
            MemoryDef::PHY_LOWER_ADDR,
            MemoryDef::PHY_LOWER_ADDR + MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB
        );

        let autoStart;
        let podIdStr = args.ID.clone();
        let mut podId = [0u8; 64];
        podId.clone_from_slice(podIdStr.as_bytes());

        {
            let vms = &mut VMS.lock();
            vms.controlSock = controlSock;
            PMA_KEEPER.InitHugePages();

            vms.hostAddrTop =
                MemoryDef::PHY_LOWER_ADDR + 64 * MemoryDef::ONE_MB + 2 * MemoryDef::ONE_GB;
            vms.pageTables = PageTables::New(&vms.allocator)?;

            let mut opts = addr::PageOpts::Zero();
            opts.SetPresent().SetWrite().SetGlobal();
            #[cfg(target_arch = "aarch64")]
            opts.SetMtNormal().SetDirty().SetAccessed();

            vms.KernelMapHugeTable(
                addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                addr::Addr(MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB),
                addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                opts.Val(),
            )?;

            #[cfg(target_arch = "aarch64")]
            {
                let mut opts = addr::PageOpts::Zero();

                opts.SetWrite().SetGlobal().SetPresent().SetAccessed();
                opts.SetMMIOPage();
                vms.KernelMap(
                    addr::Addr(MemoryDef::HYPERCALL_MMIO_BASE),
                    addr::Addr(MemoryDef::HYPERCALL_MMIO_BASE + MemoryDef::HYPERCALL_MMIO_SIZE),
                    addr::Addr(MemoryDef::HYPERCALL_MMIO_BASE),
                    opts.Val(),
                )?;
            }

            autoStart = args.AutoStart;
            vms.pivot = args.Pivot;
            vms.args = Some(args);
        }

        Self::InitShareSpace(cpuCount, controlSock, rdmaSvcCliSock, podId);

        let entry = elf.LoadKernel(Self::KERNEL_IMAGE)?;
        elf.LoadVDSO(Self::VDSO_PATH)?;
        VMS.lock().vdsoAddr = elf.vdsoStart;

        // let p = entry as *const u8;
        // info!(
        //     "entry is 0x{:x}, data at entry is {:x}, heapStartAddr is {:x}",
        //     entry,
        //     unsafe { *p },
        //     heapStartAddr
        // );

        {
            super::super::super::URING_MGR.lock();
        }

        #[cfg(target_arch = "aarch64")]
        let kvi = get_kvm_vcpu_init(&vm_fd)?;

        let mut vcpus = Vec::with_capacity(cpuCount);
        for i in 0..cpuCount
        /*args.NumCPU*/
        {
            let vcpu = Arc::new(KVMVcpu::Init(
                i as usize,
                cpuCount,
                &vm_fd,
                entry,
                heapStartAddr,
                SHARE_SPACE.Value(),
                autoStart,
            )?);
            // enable cpuid in host
            #[cfg(target_arch = "x86_64")]
            vcpu.vcpu.set_cpuid2(&kvm_cpuid).unwrap();
            VMS.lock().vcpus.push(vcpu.clone());
            vcpus.push(vcpu);
        }

        #[cfg(target_arch = "aarch64")]
        for vcpu in vcpus.iter() {
            vcpu.vcpu
                .vcpu_init(&kvi)
                .map_err(|e| Error::SysError(e.errno()))?;
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
        // start the io thread
        let cpu = self.vcpus[0].clone();
        SetSigusr1Handler();
        let mut threads = Vec::new();
        let tgid = unsafe { libc::gettid() };
        threads.push(
            thread::Builder::new()
                .name("0".to_string())
                .spawn(move || {
                    THREAD_ID.with(|f| {
                        *f.borrow_mut() = 0;
                    });
                    VCPU.with(|f| {
                        *f.borrow_mut() = Some(cpu.clone());
                    });
                    cpu.run(tgid).expect("vcpu run fail");
                    info!("cpu0 finish");
                })
                .unwrap(),
        );

        syncmgr::SyncMgr::WaitShareSpaceReady();
        info!("shareSpace ready...");
        // start the vcpu threads
        for i in 1..self.vcpus.len() {
            let cpu = self.vcpus[i].clone();

            threads.push(
                thread::Builder::new()
                    .name(format!("{}", i))
                    .spawn(move || {
                        THREAD_ID.with(|f| {
                            *f.borrow_mut() = i as i32;
                        });
                        VCPU.with(|f| {
                            *f.borrow_mut() = Some(cpu.clone());
                        });
                        info!("cpu#{} start", ThreadId());
                        cpu.run(tgid).expect("vcpu run fail");
                        info!("cpu#{} finish", ThreadId());
                    })
                    .unwrap(),
            );
        }

        for t in threads {
            t.join().expect("the working threads has panicked");
        }

        URING_MGR.lock().Close();
        Ok(GetExitStatus())
    }

    pub fn WakeAll(shareSpace: &ShareSpace) {
        shareSpace.scheduler.WakeAll();
    }

    pub fn Schedule(shareSpace: &ShareSpace, taskId: TaskId, cpuAff: bool) {
        shareSpace
            .scheduler
            .ScheduleQ(taskId, taskId.Queue(), cpuAff);
    }

    pub fn PrintQ(shareSpace: &ShareSpace, vcpuId: u64) -> String {
        return shareSpace.scheduler.PrintQ(vcpuId);
    }
}

#[cfg(target_arch = "aarch64")]
fn get_kvm_vcpu_init(vmfd: &VmFd) -> Result<kvm_vcpu_init> {
    let mut kvi = kvm_vcpu_init::default();
    vmfd.get_preferred_target(&mut kvi)
        .map_err(|e| Error::SysError(e.errno()))?;
    kvi.features[0] |= 1 << kvm_bindings::KVM_ARM_VCPU_PSCI_0_2;
    Ok(kvi)
}

fn SetSigusr1Handler() {
    let sig_action = signal::SigAction::new(
        signal::SigHandler::Handler(handleSigusr1),
        signal::SaFlags::empty(),
        signal::SigSet::empty(),
    );
    unsafe {
        signal::sigaction(signal::Signal::SIGUSR1, &sig_action)
            .expect("sigusr1 sigaction set fail");
    }
}

extern "C" fn handleSigusr1(_signal: i32) {
    SetDumpAll();
    let vms = VMS.lock();
    for vcpu in &vms.vcpus {
        if vcpu.state.load(Ordering::Acquire) == KVMVcpuState::HOST as u64 {
            vcpu.dump().unwrap_or_default();
        }
        vcpu.Signal(Signal::SIGCHLD);
    }
    drop(vms);
}
