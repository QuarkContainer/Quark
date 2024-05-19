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

//use kvm_bindings::{kvm_userspace_memory_region, KVM_CAP_X86_DISABLE_EXITS, kvm_enable_cap, KVM_X86_DISABLE_EXITS_HLT, KVM_X86_DISABLE_EXITS_MWAIT};
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
//use crate::vmspace::hibernate::HiberMgr;

use super::super::super::elf_loader::*;
use super::super::super::kvm_vcpu::*;
#[cfg(target_arch = "aarch64")]
use super::super::super::kvm_vcpu_aarch64::KVMVcpuInit;
use super::super::super::print::LOG;
use super::super::super::qlib::addr;
use super::super::super::qlib::common::*;
use super::super::super::qlib::kernel::kernel::futex;
use super::super::super::qlib::kernel::kernel::timer;
use super::super::super::qlib::kernel::vcpu;
use super::super::super::qlib::kernel::SHARESPACE;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::pagetable::PageTables;
use super::super::super::qlib::perf_tunning::*;
use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::ShareSpace;
use super::super::super::runc::runtime::loader::*;
use super::super::super::syncmgr;
use super::super::super::vmspace::*;
use super::super::super::SHARE_SPACE;
use super::super::super::{
    ThreadId, KERNEL_IO_THREAD, PMA_KEEPER, QUARK_CONFIG, ROOT_CONTAINER_ID, THREAD_ID, URING_MGR,
    VCPU, VMS,
};
#[cfg(feature = "cc")]
use crate::qlib::cc::sev_snp::{check_amd, check_snp_support, set_cbit_mask};
#[cfg(not(feature = "cc"))]
use crate::qlib::kernel::IOURING;
#[cfg(feature = "cc")]
use crate::qlib::kernel::Kernel::{ENABLE_CC, IS_SEV_SNP};

pub const SANDBOX_UID_NAME : &str = "io.kubernetes.cri.sandbox-uid";

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


    #[cfg(not(feature = "cc"))]
    pub fn InitShareSpace(
        cpuCount: usize,
        controlSock: i32,
        rdmaSvcCliSock: i32,
        podId: [u8; 64],
    ) {
        let supportMemoryBarrier = VMS.read().haveMembarrierGlobal;
        crate::SHARE_SPACE_STRUCT
            .lock()
            .Init(cpuCount, controlSock, rdmaSvcCliSock, podId, supportMemoryBarrier);
        let spAddr = &(*crate::SHARE_SPACE_STRUCT.lock()) as *const _ as u64;
        SHARE_SPACE.SetValue(spAddr);
        SHARESPACE.SetValue(spAddr);

        unsafe {
            vcpu::CPU_LOCAL.Init(&SHARESPACE.scheduler.VcpuArr);
        }
  
        let sharespace = SHARE_SPACE.Ptr();
        let logfd = super::super::super::print::LOG.Logfd();
        
        URING_MGR.lock().Addfd(logfd).unwrap();

        KERNEL_IO_THREAD.Init(sharespace.scheduler.VcpuArr[0].eventfd);
        URING_MGR
            .lock()
            .Addfd(sharespace.HostHostEpollfd())
            .unwrap();
        URING_MGR.lock().Addfd(controlSock).unwrap();
        IOURING.SetValue(sharespace.GetIOUringAddr());

        unsafe {

            crate::qlib::kernel::KERNEL_PAGETABLE.SetRoot(VMS.read().pageTables.GetRoot());
            crate::qlib::kernel::PAGE_MGR.SetValue(sharespace.GetPageMgrAddr());
            crate::qlib::kernel::KERNEL_STACK_ALLOCATOR.Init(crate::qlib::pagetable::AlignedAllocator::New(
                MemoryDef::DEFAULT_STACK_SIZE as usize,
                MemoryDef::DEFAULT_STACK_SIZE as usize,
            ));

            crate::qlib::kernel::task::InitSingleton();
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


    #[cfg(feature = "cc")]
    pub fn InitShareSpace(
        sharedSpace: &mut ShareSpace,
        cpuCount: usize,
        controlSock: i32,
        rdmaSvcCliSock: i32,
        podId: [u8; 64],
        haveMembarrierGlobal: bool,
    ) {
        let sp = ShareSpace::New();
        let sp_size = core::mem::size_of_val(&sp);
        let sharedSpace_size = core::mem::size_of_val(sharedSpace);
        assert!(sp_size == sharedSpace_size, "sp_size != sharedSpace_size, sp_size {}, 
                                            sharedSpace_size {}", sp_size, sharedSpace_size);
        unsafe {
			core::ptr::write(sharedSpace as *mut ShareSpace, sp);
		}

        sharedSpace.Init(cpuCount, controlSock, rdmaSvcCliSock, podId, haveMembarrierGlobal);

        let spAddr =  sharedSpace as *const _ as u64;
        SHARE_SPACE.SetValue(spAddr);
        SHARESPACE.SetValue(spAddr);

        unsafe {
            vcpu::CPU_LOCAL.Init(&SHARESPACE.scheduler.VcpuArr);
        }


        let sharespace = SHARE_SPACE.Ptr();
        let logfd = super::super::super::print::LOG.Logfd();
        
        URING_MGR.lock().Addfd(logfd).unwrap();
        KERNEL_IO_THREAD.Init(sharespace.scheduler.VcpuArr[0].eventfd);
        URING_MGR
            .lock()
            .Addfd(sharespace.HostHostEpollfd())
            .unwrap();

        URING_MGR.lock().Addfd(controlSock).unwrap();

        unsafe {
            futex::InitSingleton();
            timer::InitSingleton();
        }

        if SHARESPACE.config.read().EnableTsot {
            // initialize the tost_agent
            TSOT_AGENT.NextReqId();
            SHARESPACE.dnsSvc.Init().unwrap();
        };

        let syncPrint = sharespace.config.read().SyncPrint();
        super::super::super::print::SetSyncPrint(syncPrint);

    }

    pub fn Init_vm(args: Args, enable_cc: bool) -> Result<Self> {
        if enable_cc {
            #[cfg(feature = "cc")]
            if check_amd() && check_snp_support() {
                ENABLE_CC.store(true, Ordering::Release);
                IS_SEV_SNP.store(true, Ordering::Release);
                set_cbit_mask();
                return Self::InitSevSnp(args);
            }
            return Err(Error::InvalidInput);
        } else {
            return Self::Init(args);
        }
    }

    #[cfg(not(feature = "cc"))]
    pub fn Init(args: Args /*args: &Args, kvmfd: i32*/) -> Result<Self> {
        PerfGoto(PerfType::Other);

        *ROOT_CONTAINER_ID.lock() = args.ID.clone();
        if QUARK_CONFIG.lock().PerSandboxLog {
            let sandboxName = match args.Spec.annotations.get("io.kubernetes.cri.sandbox-name") {
                None => {
                    args.ID[0..12].to_owned()
                }
                Some(name) => name.clone()
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
            VMS.write().cpuAffinit = false;
        } else {
            VMS.write().cpuAffinit = true;
        }


        match args.Spec.annotations.get(SANDBOX_UID_NAME) {
            None => (),
            Some(podUid) => {
                VMS.write().podUid = podUid.clone();
            }
        }

        let cpuCount = cpuCount.max(2); // minimal 2 cpus

        VMS.write().vcpuCount = cpuCount; //VMSpace::VCPUCount();
        VMS.write().RandomVcpuMapping();
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

        let mut cap: kvm_enable_cap = Default::default();
        cap.cap = KVM_CAP_X86_DISABLE_EXITS;
        cap.args[0] = (KVM_X86_DISABLE_EXITS_HLT | KVM_X86_DISABLE_EXITS_MWAIT) as u64;
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        vm_fd.enable_cap(&cap).unwrap();
        if !kvm.check_extension(Cap::ImmediateExit) {
            panic!("KVM_CAP_IMMEDIATE_EXIT not supported");
        }

        let mut elf = KernelELF::New()?;
        Self::SetMemRegion(
            1,
            &vm_fd,
            MemoryDef::phy_lower_gpa(),
            MemoryDef::phy_lower_gpa(),
            MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB,
        )?;

        Self::SetMemRegion(
            2,
            &vm_fd,
            MemoryDef::NVIDIA_START_ADDR,
            MemoryDef::NVIDIA_START_ADDR,
            MemoryDef::NVIDIA_ADDR_SIZE,
        )?;

        let heapStartAddr = MemoryDef::guest_private_init_heap_offset_gpa();

        PMA_KEEPER.Init(MemoryDef::file_map_offset_gpa(), MemoryDef::FILE_MAP_SIZE);

        info!(
            "set map region start={:x}, end={:x}",
            MemoryDef::phy_lower_gpa(),
            MemoryDef::phy_lower_gpa() + MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB
        );

        let autoStart;
        let podIdStr = args.ID.clone();
        let mut podId = [0u8; 64];
        podId.clone_from_slice(podIdStr.as_bytes());
        // let mut podId: [u8; 64] = [0; 64];
        // debug!("VM::Initxxxxx, podIdStr: {}", podIdStr);
        // if podIdStr.len() != podId.len() {
        //     panic!("podId len: {} is not equal to podIdStr len: {}", podId.len(), podIdStr.len());
        // }

        // podIdStr.bytes()
        //     .zip(podId.iter_mut())
        //     .for_each(|(b, ptr)| *ptr = b);
        {
            let vms = &mut VMS.write();
            vms.controlSock = controlSock;
            PMA_KEEPER.InitHugePages();

            vms.pageTables = PageTables::New(&vms.allocator)?;

            vms.KernelMapHugeTable(
                addr::Addr(MemoryDef::phy_lower_gpa()),
                addr::Addr(MemoryDef::phy_lower_gpa() + kernelMemRegionSize * MemoryDef::ONE_GB),
                addr::Addr(MemoryDef::phy_lower_gpa()),
                addr::PageOpts::Zero()
                    .SetPresent()
                    .SetWrite()
                    .SetGlobal()
                    .Val(),
            )?;


            vms.KernelMapHugeTable(
                addr::Addr(MemoryDef::NVIDIA_START_ADDR),
                addr::Addr(MemoryDef::NVIDIA_START_ADDR + MemoryDef::NVIDIA_ADDR_SIZE),
                addr::Addr(MemoryDef::NVIDIA_START_ADDR),
                addr::PageOpts::Zero()
                    .SetPresent()
                    .SetWrite()
                    .SetGlobal()
                    .Val(),
            )?;
            autoStart = args.AutoStart;
            vms.pivot = args.Pivot;
            vms.args = Some(args);
        }

        Self::InitShareSpace(cpuCount, controlSock, rdmaSvcCliSock, podId);

        let entry_gpa = elf.LoadKernel(Self::KERNEL_IMAGE)?;
        //let vdsoMap = VDSOMemMap::Init(&"/home/brad/rust/quark/vdso/vdso.so".to_string()).unwrap();
        elf.LoadVDSO(&"/usr/local/bin/vdso.so".to_string())?;
        VMS.write().vdsoAddrGpa = MemoryDef::hva_to_gpa(elf.vdsoStartHva);

        {
            super::super::super::URING_MGR.lock();
        }

        #[cfg(target_arch = "aarch64")]
        set_kvm_vcpu_init(&vm_fd)?;

        let mut vcpus = Vec::with_capacity(cpuCount);
        for i in 0..cpuCount
        /*args.NumCPU*/
        {
            let vcpu = Arc::new(KVMVcpu::Init(
                i as usize,
                cpuCount,
                &vm_fd,
                entry_gpa,
                heapStartAddr,
                SHARE_SPACE.Value(),
                autoStart,
            )?);

            // enable cpuid in host
            #[cfg(target_arch = "x86_64")]
            vcpu.vcpu.set_cpuid2(&kvm_cpuid).unwrap();
            VMS.write().vcpus.push(vcpu.clone());
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

    #[cfg(feature = "cc")]
    pub fn Init(args: Args /*args: &Args, kvmfd: i32*/) -> Result<Self> {
        PerfGoto(PerfType::Other);

        *ROOT_CONTAINER_ID.lock() = args.ID.clone();
        if QUARK_CONFIG.lock().PerSandboxLog {
            let sandboxName = match args.Spec.annotations.get("io.kubernetes.cri.sandbox-name") {
                None => {
                    args.ID[0..12].to_owned()
                }
                Some(name) => name.clone()
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
            VMS.write().cpuAffinit = false;
        } else {
            VMS.write().cpuAffinit = true;
        }


        match args.Spec.annotations.get(SANDBOX_UID_NAME) {
            None => (),
            Some(podUid) => {
                VMS.write().podUid = podUid.clone();
            }
        }

        let cpuCount = cpuCount.max(4); // minimal 2 cpus
        //let cpuCount = 3;
        VMS.write().vcpuCount = cpuCount; //VMSpace::VCPUCount();
        VMS.write().RandomVcpuMapping();
        let kernelMemRegionSize = QUARK_CONFIG.lock().KernelMemSize;
        let controlSock = args.ControlSock;

        VMS.write().rdmaSvcCliSock = args.RDMASvcCliSock;
        let podIdStr = args.ID.clone();
        VMS.write().podId.clone_from_slice(podIdStr.as_bytes());

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

        let mut cap: kvm_enable_cap = Default::default();
        cap.cap = KVM_CAP_X86_DISABLE_EXITS;
        cap.args[0] = (KVM_X86_DISABLE_EXITS_MWAIT) as u64;
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        vm_fd.enable_cap(&cap).unwrap();
        if !kvm.check_extension(Cap::ImmediateExit) {
            panic!("KVM_CAP_IMMEDIATE_EXIT not supported");
        }

        let mut elf = KernelELF::New()?;

        let guest_private_size = MemoryDef::guest_private_running_heap_end_gpa()  -  MemoryDef::phy_lower_gpa();
        // Private Region
        Self::SetMemRegion(
            1,
            &vm_fd,
            MemoryDef::phy_lower_gpa(),
            MemoryDef::phy_lower_hva(),
            guest_private_size,
        )?;

        // Shared Region
        let guest_host_shared = MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB - guest_private_size; 
        Self::SetMemRegion(
            2,
            &vm_fd,
            MemoryDef::guest_host_shared_heap_offest_gpa(),
            MemoryDef::guest_host_shared_heap_offset_hva(),
            guest_host_shared,
        )?;
        
        if !QUARK_CONFIG.lock().EnableCC {
            Self::SetMemRegion(
                3,
                &vm_fd,
                MemoryDef::NVIDIA_START_ADDR,
                MemoryDef::NVIDIA_START_ADDR,
                MemoryDef::NVIDIA_ADDR_SIZE,
            )?;
        }


        PMA_KEEPER.Init(MemoryDef::file_map_offset_hva(), MemoryDef::FILE_MAP_SIZE);


        let autoStart;
        {
            let vms = &mut VMS.write();
            vms.controlSock = controlSock;
            PMA_KEEPER.InitHugePages(); 

            vms.pageTables = PageTables::New(&vms.allocator)?;

            vms.KernelMapHugeTable(
                addr::Addr(MemoryDef::phy_lower_gpa()),
                addr::Addr(MemoryDef::phy_lower_gpa() + kernelMemRegionSize * MemoryDef::ONE_GB),
                addr::Addr(MemoryDef::phy_lower_gpa()),
                addr::PageOpts::Zero()
                    .SetPresent()
                    .SetWrite()
                    .SetGlobal()
                    .Val(),
            )?;

            if !QUARK_CONFIG.lock().EnableCC { 
                vms.KernelMapHugeTable(
                    addr::Addr(MemoryDef::NVIDIA_START_ADDR),
                    addr::Addr(MemoryDef::NVIDIA_START_ADDR + MemoryDef::NVIDIA_ADDR_SIZE),
                    addr::Addr(MemoryDef::NVIDIA_START_ADDR),
                    addr::PageOpts::Zero()
                        .SetPresent()
                        .SetWrite()
                        .SetGlobal()
                        .Val(),
                )?;
            }

            autoStart = args.AutoStart;
            vms.pivot = args.Pivot;
            vms.args = Some(args);
            vms.kvmfd = kvmfd;
            vms.vmfd = vm_fd.as_raw_fd();
        }

        let entry_gpa = elf.LoadKernel(Self::KERNEL_IMAGE)?;
        elf.LoadVDSO(&"/usr/local/bin/vdso.so".to_string())?;
        VMS.write().vdsoAddrGpa = MemoryDef::hva_to_gpa(elf.vdsoStartHva);


        {
            super::super::super::URING_MGR.lock();
        }

        #[cfg(target_arch = "aarch64")]
        set_kvm_vcpu_init(&vm_fd)?;

        let mut vcpus = Vec::with_capacity(cpuCount);
        for i in 0..cpuCount
        /*args.NumCPU*/
        {
            let vcpu = Arc::new(KVMVcpu::Init(
                i as usize,
                cpuCount,
                &vm_fd,
                entry_gpa,
                autoStart,
            )?);

            // enable cpuid in host
            #[cfg(target_arch = "x86_64")]
            vcpu.vcpu.set_cpuid2(&kvm_cpuid).unwrap();
            #[cfg(target_arch = "x86_64")]
            vcpu.x86_init()?;
            VMS.write().vcpus.push(vcpu.clone());
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
        // start the io thread
        let cpu = self.vcpus[0].clone();
        SetSigusr1Handler();
        let mut threads = Vec::new();
        let tgid = unsafe { libc::gettid() };
        let kvmfd = VMS.read().kvmfd;
        let vmfd = VMS.read().vmfd;
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
                    cpu.run(tgid, kvmfd, vmfd).expect("vcpu run fail");
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
                        cpu.run(tgid, kvmfd, vmfd).expect("vcpu run fail");
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
const _KVM_ARM_PREFERRED_TARGET:u64  = 0x8020aeaf;

#[cfg(target_arch = "aarch64")]
fn set_kvm_vcpu_init(vmfd: &VmFd) -> Result<()> {
    use crate::kvm_vcpu_aarch64::KVM_VCPU_INIT;

    let mut kvm_vcpu_init = KVMVcpuInit::default();
    let raw_fd = vmfd.as_raw_fd();
    let ret = unsafe { libc::ioctl(raw_fd, _KVM_ARM_PREFERRED_TARGET, &kvm_vcpu_init as *const _ as u64) };
    if ret != 0 {
        return Err(Error::SysError(ret));
    }
    kvm_vcpu_init.set_psci_0_2();
    unsafe { KVM_VCPU_INIT.Init(kvm_vcpu_init); }
    Ok(())
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
    let vms = VMS.read();
    for vcpu in &vms.vcpus {
        if vcpu.state.load(Ordering::Acquire) == KVMVcpuState::HOST as u64 {
            vcpu.dump().unwrap_or_default();
        }
        vcpu.Signal(Signal::SIGCHLD);
    }
    drop(vms);
}
