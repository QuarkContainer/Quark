use alloc::sync::Arc;
use kvm_bindings::*;
use kvm_ioctls::{Cap, Kvm, VmFd};
use std::os::fd::AsRawFd;
use std::os::unix::io::FromRawFd;

//use crate::vmspace::hibernate::HiberMgr;

use super::super::super::elf_loader::*;
use super::super::super::kvm_vcpu::*;
use super::super::super::print::LOG;
use super::super::super::qlib::addr;
use super::super::super::qlib::cc::sev_snp::cpuid_page::*;
use super::super::super::qlib::cc::sev_snp::get_cbit;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::mem::list_allocator::MAXIMUM_PAGE_START;
use super::super::super::qlib::pagetable::PageTables;
use super::super::super::qlib::perf_tunning::*;
use super::super::super::runc::runtime::loader::*;
use super::super::super::vmspace::*;
use super::super::super::{PMA_KEEPER, QUARK_CONFIG, ROOT_CONTAINER_ID, VMS};
use super::vm::*;
use core::slice::from_raw_parts_mut;
use core::sync::atomic::Ordering;
use sev::firmware::host::Firmware;
use sev::launch::snp::*;
pub const KVM_MEM_GUEST_MEMFD: u32 = 1 << 2;
pub const KVM_MEMORY_ATTRIBUTE_PRIVATE: u64 = 1 << 3;

impl CpuidPage {
    pub fn FillCpuidPage(&mut self, kvm_cpuid_entries: &CpuId) -> Result<()> {
        let mut has_entries = false;

        for kvm_entry in kvm_cpuid_entries.as_slice() {
            if kvm_entry.function == 0 && kvm_entry.index == 0 && has_entries {
                break;
            }

            if kvm_entry.function == 0xFFFFFFFF {
                break;
            }

            // range check, see:
            // SEV Secure Nested Paging Firmware ABI Specification
            // 8.17.2.6 PAGE_TYPE_CPUID
            if !((0x0000_0000..=0x0000_FFFF).contains(&kvm_entry.function)
                || (0x8000_0000..=0x8000_FFFF).contains(&kvm_entry.function))
            {
                continue;
            }
            has_entries = true;

            let mut snp_cpuid_entry = SnpCpuidFunc {
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
        let gmem = kvm_create_guest_memfd {
            size: pageMmapsize,
            ..Default::default()
        };
        let gmem_fd = vm_fd
            .create_guest_memfd(&gmem)
            .expect("Fail to create guest memory") as u32;
        // guest_phys_addr must be <512G
        let mem_region = kvm_userspace_memory_region2 {
            slot: slotId,
            flags: KVM_MEM_GUEST_MEMFD,
            guest_phys_addr: phyAddr,
            memory_size: pageMmapsize,
            userspace_addr: hostAddr,
            gmem_offset: 0,
            gmem_fd: gmem_fd,
            ..Default::default()
        };

        unsafe {
            vm_fd
                .set_user_memory_region2(mem_region)
                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        }

        return Ok(());
    }

    pub fn InitSevSnp(args: Args /*args: &Args, kvmfd: i32*/) -> Result<Self> {
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

        let cpuCount = cpuCount.max(3); // minimal 3 cpus
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
            .create_vm_with_type(3)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        let sev = Firmware::open().expect("Unable to open /dev/sev");
        let launcher = Launcher::new(vm_fd, sev).unwrap();
        let start = Start::new(
            Policy {
                flags: PolicyFlags::SMT,
                ..Default::default()
            },
            [0; 16],
        );

        let mut launcher = launcher.start(start).unwrap();
        let mut cap: kvm_enable_cap = Default::default();
        cap.cap = KVM_CAP_X86_DISABLE_EXITS;
        //cap.args[0] = (KVM_X86_DISABLE_EXITS_HLT | KVM_X86_DISABLE_EXITS_MWAIT) as u64;
        cap.args[0] = KVM_X86_DISABLE_EXITS_MWAIT as u64;
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        launcher.as_ref().enable_cap(&cap).unwrap();
        if !kvm.check_extension(Cap::ImmediateExit) {
            panic!("KVM_CAP_IMMEDIATE_EXIT not supported");
        }

        let mut elf = KernelELF::New()?;
        Self::SetMemRegionCC(
            1,
            launcher.as_ref(),
            MemoryDef::phy_lower_gpa(),
            MemoryDef::phy_lower_gpa(),
            MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB,
        )?;

        PMA_KEEPER.Init(MemoryDef::file_map_offset_gpa(), MemoryDef::FILE_MAP_SIZE);

        let autoStart;
        {
            let vms = &mut VMS.write();
            vms.controlSock = controlSock;
            PMA_KEEPER.InitHugePages();
            vms.pageTables = PageTables::New(&vms.allocator)?;

            vms.KernelMapHugeTableSevSnp(
                addr::Addr(MemoryDef::phy_lower_gpa()),
                addr::Addr(MemoryDef::phy_lower_gpa() + kernelMemRegionSize * MemoryDef::ONE_GB),
                addr::Addr(MemoryDef::phy_lower_gpa()),
                addr::PageOpts::Zero()
                    .SetPresent()
                    .SetWrite()
                    .SetGlobal()
                    .Val(),
                get_cbit(),
            )?;
            autoStart = args.AutoStart;
            vms.pivot = args.Pivot;
            vms.args = Some(args);
            vms.kvmfd = kvmfd;
            vms.vmfd = launcher.as_ref().as_raw_fd();
        }

        let cpuid_page_addr = MemoryDef::CPUID_PAGE;
        let secret_page_addr = MemoryDef::SECRET_PAGE;
        let cpuid_page = CpuidPage::get_ref(cpuid_page_addr);
        cpuid_page
            .FillCpuidPage(&kvm_cpuid)
            .expect("Fail to fill cpuid page");
        //cpuid_page.dump_cpuid();
        let entry = elf.LoadKernel(Self::KERNEL_IMAGE)?;
        //let vdsoMap = VDSOMemMap::Init(&"/home/brad/rust/quark/vdso/vdso.so".to_string()).unwrap();
        elf.LoadVDSO(&"/usr/local/bin/vdso.so".to_string())?;
        VMS.write().vdsoAddrGpa = MemoryDef::hva_to_gpa(elf.vdsoStartHva);

        {
            super::super::super::URING_MGR.lock();
        }

        let mut vcpus = Vec::with_capacity(cpuCount);
        for i in 0..cpuCount
        /*args.NumCPU*/
        {
            let vcpu = Arc::new(KVMVcpu::Init(
                i as usize,
                cpuCount,
                launcher.as_ref(),
                entry,
                autoStart,
            )?);
            // enable cpuid in host
            #[cfg(target_arch = "x86_64")]
            vcpu.vcpu.set_cpuid2(&kvm_cpuid).unwrap();
            vcpu.x86_init()?;
            VMS.write().vcpus.push(vcpu.clone());
            vcpus.push(vcpu);
        }

        let memory_attributes = kvm_memory_attributes {
            address: MemoryDef::phy_lower_gpa(),
            size: kernelMemRegionSize * MemoryDef::ONE_GB,
            attributes: KVM_MEMORY_ATTRIBUTE_PRIVATE,
            flags: 0,
        };
        launcher
            .as_ref()
            .set_memory_attributes(&memory_attributes)
            .expect("Unable to convert memory to private");

        //update initial private heap including private allocator, page table, gdt etc.
        let maximum_pagetable_page = MAXIMUM_PAGE_START.load(Ordering::Acquire);
        info!("MAXIMUM_PAGE_START is 0x{:x}", maximum_pagetable_page);

        let pt_space: &mut [u8] = unsafe {
            from_raw_parts_mut(
                MemoryDef::guest_private_init_heap_offset_gpa () as *mut u8,
                (maximum_pagetable_page + MemoryDef::PAGE_SIZE
                    - MemoryDef::guest_private_init_heap_offset_gpa ()) as usize,
            )
        };
        let update_pt = Update::new(
            MemoryDef::guest_private_init_heap_offset_gpa() >> 12,
            pt_space,
            PageType::Normal,
        );
        launcher.update_data(update_pt).unwrap();
        // let p = entry as *const u8;
        // info!(
        //     "entry is 0x{:x}, data at entry is {:x}, heapStartAddr is {:x}",
        //     entry,
        //     unsafe { *p },
        //     heapStartAddr
        // );

        //update kernel
        let kernel_space: &mut [u8] = unsafe {
            from_raw_parts_mut(
                elf.startAddrHva.0 as *mut u8,
                (elf.endAddrHva.0 + elf.vdsoLen - elf.startAddrHva.0) as usize,
            )
        };
        let update_kernel = Update::new(elf.startAddrHva.0 >> 12, kernel_space, PageType::Normal);
        launcher.update_data(update_kernel).unwrap();

        //update cpuid_page
        let cpuid_space: &mut [u8] = unsafe {
            from_raw_parts_mut(cpuid_page_addr as *mut u8, MemoryDef::PAGE_SIZE as usize)
        };
        let update_cpuid = Update::new(cpuid_page_addr >> 12, cpuid_space, PageType::Cpuid);
        //Retry again if udpate failed
        match launcher.update_data(update_cpuid) {
            Ok(_) => (),
            Err(_) => {
                //cpuid_page.dump_cpuid();
                launcher.update_data(update_cpuid).unwrap();
            }
        };
        //update secret page
        let secret_space: &mut [u8] = unsafe {
            from_raw_parts_mut(secret_page_addr as *mut u8, MemoryDef::PAGE_SIZE as usize)
        };
        let update_secret = Update::new(secret_page_addr >> 12, secret_space, PageType::Secrets);
        launcher.update_data(update_secret).unwrap();
        info!("update finished");
        let finish = Finish::new(None, None, [0u8; 32]);
        let (vm_fd, _sev) = launcher.finish(finish).unwrap();
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
