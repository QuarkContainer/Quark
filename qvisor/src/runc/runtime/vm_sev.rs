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
use std::os::unix::io::FromRawFd;
use kvm_bindings::*;
use kvm_ioctls::{Cap, Kvm, VmFd};
use crate::qlib::cpuid::HostID;

use crate::tsot_agent::TSOT_AGENT;
//use crate::vmspace::hibernate::HiberMgr;

use super::super::super::elf_loader::*;
use super::super::super::kvm_vcpu::*;
use super::vm::*;
#[cfg(target_arch = "aarch64")]
use super::super::super::kvm_vcpu_aarch64::KVMVcpuInit;
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
use super::super::super::qlib::cc::cpuid_page::*;
use super::super::super::runc::runtime::loader::*;
use super::super::super::vmspace::*;
use super::super::super::SHARE_SPACE;
use super::super::super::SHARE_SPACE_STRUCT;
use super::super::super::{
     KERNEL_IO_THREAD, PMA_KEEPER, QUARK_CONFIG, ROOT_CONTAINER_ID,  URING_MGR, VMS,
};

impl CpuidPage {
    pub fn FillCpuidPage(&mut self, kvm_cpuid_entries:&CpuId ) -> Result<()> {
        let mut has_entries = false;
    
        for kvm_entry in kvm_cpuid_entries.as_slice(){
            if kvm_entry.function == 0 && kvm_entry.index == 0 && has_entries {
                break;
            }
    
            if kvm_entry.function == 0xFFFFFFFF {
                break;
            }
    
            // range check, see:
            // SEV Secure Nested Paging Firmware ABI Specification
            // 8.14.2.6 PAGE_TYPE_CPUID
            if !((0..0xFFFF).contains(&kvm_entry.function)
                || (0x8000_0000..0x8000_FFFF).contains(&kvm_entry.function))
            {
                continue;
            }
            has_entries = true;
    
            let mut snp_cpuid_entry = SnpCpuidFunc{
                eax_in: kvm_entry.function,
                ecx_in: {
                    if (kvm_entry.flags & KVM_CPUID_FLAG_SIGNIFCANT_INDEX) != 0 {
                        kvm_entry.index
                    } else {
                        0
                    }
                },
                xcr0_in: 0,
                xss_in: 0,
                eax: kvm_entry.eax,
                ebx: kvm_entry.ebx,
                ecx: kvm_entry.ecx,
                edx: kvm_entry.edx,
                ..Default::default()
            };
            if snp_cpuid_entry.eax_in == 0xD
                && (snp_cpuid_entry.ecx_in == 0x0 || snp_cpuid_entry.ecx_in == 0x1)
            {
                /*
                * Guest kernels will calculate EBX themselves using the 0xD
                * subfunctions corresponding to the individual XSAVE areas, so only
                * encode the base XSAVE size in the initial leaves, corresponding
                * to the initial XCR0=1 state.
                */
                snp_cpuid_entry.ebx = 0x240;
                snp_cpuid_entry.xcr0_in = 1;
                snp_cpuid_entry.xss_in = 0;
            }
            self.AddEntry(&snp_cpuid_entry)
                .expect("Failed to add CPUID entry to the CPUID page");
        }
        Ok(())
    }
}
//for assumption
pub fn get_page_from_allocator(_len: usize) -> *mut u64 {
    todo!()
}

//get c_bit position
pub fn get_c_bit() -> u64 {
    let (_, ebx, _, _) = HostID(0x8000001f, 0);
    return (ebx&0x3f) as u64;
}

impl VirtualMachine {
    pub fn SetMemRegionCC(
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

    pub fn InitShareSpaceSevSnp(
        cpuCount: usize,
        controlSock: i32,
        rdmaSvcCliSock: i32,
        podId: [u8; 64],
    ) {
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
            KERNEL_PAGETABLE.SetRoot(VMS.lock().pageTables.GetRoot());
            PAGE_MGR.SetValue(sharespace.GetPageMgrAddr());

            // used for created new task from host
            // see Create(runFnAddr: u64, para: *const u8, kernel: bool) -> &'static mut Self {
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

        let syncPrint = sharespace.config.read().SyncPrint();
        super::super::super::print::SetSyncPrint(syncPrint);
    }


    pub fn InitSevSnp(args: Args /*args: &Args, kvmfd: i32*/) -> Result<Self> {
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

        VMS.lock().vcpuCount = cpuCount; //VMSpace::VCPUCount();
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

        //get one page from private allocator
        
        let cpuid_page_ptr = get_page_from_allocator(0x1000);
        let mut cpuid_page = unsafe{ *(cpuid_page_ptr as *const CpuidPage)};
        cpuid_page.FillCpuidPage(&kvm_cpuid).expect("Fail to fill Cpuid Page!");
        let _secret_page_ptr = get_page_from_allocator(0x1000);
        
        //kvm.create_vm_with_type(vm_type) in sev-snp
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
        //private region, offset 4kb,
        Self::SetMemRegion(
            0,
            &vm_fd,
            MemoryDef::PHY_LOWER_ADDR,
            MemoryDef::KERNEL_MEM_INIT_PRIVATE_REGION_OFFSET,
            MemoryDef::KERNEL_MEM_INIT_PRIVATE_REGION_SIZE * MemoryDef::ONE_GB,
        )?;


        Self::SetMemRegion(
            1,
            &vm_fd,
            MemoryDef::PHY_LOWER_ADDR+MemoryDef::KERNEL_MEM_INIT_PRIVATE_REGION_SIZE * MemoryDef::ONE_GB,
            MemoryDef::KERNEL_MEM_INIT_SHARE_REGION_OFFSET,
            MemoryDef::KERNEL_MEM_INIT_SHARE_REGION_SIZE * MemoryDef::ONE_GB,
        )?;

        Self::SetMemRegion(
            2,
            &vm_fd,
            MemoryDef::NVIDIA_START_ADDR,
            MemoryDef::NVIDIA_START_ADDR,
            MemoryDef::NVIDIA_ADDR_SIZE,
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
        // let mut podId: [u8; 64] = [0; 64];
        // debug!("VM::Initxxxxx, podIdStr: {}", podIdStr);
        // if podIdStr.len() != podId.len() {
        //     panic!("podId len: {} is not equal to podIdStr len: {}", podId.len(), podIdStr.len());
        // }

        // podIdStr.bytes()
        //     .zip(podId.iter_mut())
        //     .for_each(|(b, ptr)| *ptr = b);
        {
            let vms = &mut VMS.lock();
            vms.controlSock = controlSock;
            PMA_KEEPER.InitHugePages(); 
            vms.hostAddrTop =
                MemoryDef::PHY_LOWER_ADDR + 64 * MemoryDef::ONE_MB + 2 * MemoryDef::ONE_GB;
            vms.pageTables = PageTables::New(&vms.allocator)?;

            /* 
            let c_bit = get_c_bit();
            vms.KernelMapHugeTablePrivate(
                addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                addr::Addr(MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB),
                addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                addr::PageOpts::Zero()
                    .SetPresent()
                    .SetWrite()
                    .SetGlobal()
                    .Val(),
                    c_bit,
            )?;
            
            vms.KernelMapHugeTablePrivate(
                addr::Addr(MemoryDef::NVIDIA_START_ADDR),
                addr::Addr(MemoryDef::NVIDIA_START_ADDR + MemoryDef::NVIDIA_ADDR_SIZE),
                addr::Addr(MemoryDef::NVIDIA_START_ADDR),
                addr::PageOpts::Zero()
                    .SetPresent()
                    .SetWrite()
                    .SetGlobal()
                    .Val(),
                    c_bit,
            )?;
            */
            
            vms.KernelMapHugeTable(
                addr::Addr(MemoryDef::PHY_LOWER_ADDR),
                addr::Addr(MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB),
                addr::Addr(MemoryDef::PHY_LOWER_ADDR),
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

            vms.rdmaSvcCliSock = rdmaSvcCliSock;
            vms.podId = podId;
            

        }

        //Self::InitShareSpaceSevSnp(cpuCount, controlSock, rdmaSvcCliSock, podId);

        let entry = elf.LoadKernelwithOffset(Self::KERNEL_IMAGE, MemoryDef::PAGE_SIZE)?;
        //let vdsoMap = VDSOMemMap::Init(&"/home/brad/rust/quark/vdso/vdso.so".to_string()).unwrap();
        elf.LoadVDSO(&"/usr/local/bin/vdso.so".to_string())?;
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
        set_kvm_vcpu_init(&vm_fd)?;

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

        let vm = Self {
            kvm: kvm,
            vmfd: vm_fd,
            vcpus: vcpus,
            elf: elf,
        };

        PerfGofrom(PerfType::Other);
        Ok(vm)
    }

}