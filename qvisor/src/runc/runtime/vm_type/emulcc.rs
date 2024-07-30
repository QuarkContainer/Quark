// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,x
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{os::fd::FromRawFd, sync::{atomic::Ordering, Arc}};

use kvm_bindings::kvm_enable_cap;
use kvm_ioctls::{Cap, Kvm, VmFd};

use crate::FD_NOTIFIER;
use crate::{arch::{tee::util::{adjust_addr_to_guest, adjust_addr_to_host},
            vm::vcpu::ArchVirtCpu}, elf_loader::KernelELF, print::LOG,
            qlib::{addr::{Addr, PageOpts}, common::Error, kernel::{kernel::{futex, timer},
            vcpu::CPU_LOCAL, SHARESPACE, IOURING}, linux_def::{MemoryDef, EVENT_READ}, pagetable::PageTables,
            pagetable::HugePageType, ShareSpace}, runc::runtime::{loader::Args,
            vm::{self, VirtualMachine}}, tsot_agent::TSOT_AGENT, CCMode, VMSpace,
            KERNEL_IO_THREAD, PMA_KEEPER, QUARK_CONFIG, ROOT_CONTAINER_ID,
            SHARE_SPACE, URING_MGR, VMS};
use crate::arch::VirtCpu;
use super::{resources::{GuestPrivateMemLayout, MemArea, MemLayoutConfig, VmResources}, VmType};
#[cfg(feature = "cc")]
use crate::qlib::kernel::Kernel::IDENTICAL_MAPPING;
#[cfg(feature = "cc")]
use crate::qlib::kernel::Kernel::ENABLE_CC;


#[derive(Debug)]
pub struct VmCcEmul {
    vm_resources: VmResources,
    entry_address: u64,
    vdso_address: u64,
    emul_cc_mode: CCMode,
}

#[cfg(feature = "cc")]
impl VmType for VmCcEmul {
    fn init(args: Option<&Args>) -> Result<(Box<dyn VmType>, KernelELF), Error> {
        ENABLE_CC.store(true, Ordering::Release);
        let _pod_id = args.expect("VM creation expects arguments").ID.clone();
        let default_min_vcpus = 3;
        let _emul_type: CCMode = QUARK_CONFIG.lock().CCMode;
        if _emul_type == CCMode::Normal {
            IDENTICAL_MAPPING.store(true, Ordering::Release);
        } else {
            IDENTICAL_MAPPING.store(false, Ordering::Release);
        };

        let guest_priv_mem_layout = GuestPrivateMemLayout {
            private_heap_mem_base_host:
                adjust_addr_to_host(MemoryDef::GUEST_PRIVATE_HEAP_OFFSET, _emul_type),
            private_heap_mem_base_guest: MemoryDef::GUEST_PRIVATE_HEAP_OFFSET,
            private_heap_init_mem_size: MemoryDef::GUEST_PRIVATE_INIT_HEAP_SIZE,
            private_heap_total_mem_size: MemoryDef::GUEST_PRIVATE_HEAP_SIZE,
        };

        let mem_layout_config = MemLayoutConfig {
            guest_private_mem_layout: Some(guest_priv_mem_layout),
            shared_heap_mem_base_guest: MemoryDef::GUEST_HOST_SHARED_HEAP_OFFSET,
            shared_heap_mem_base_host: MemoryDef::GUEST_HOST_SHARED_HEAP_OFFSET,
            shared_heap_mem_size: MemoryDef::GUEST_HOST_SHARED_HEAP_SIZE,
            kernel_base_guest: MemoryDef::PHY_LOWER_ADDR,
            kernel_base_host: adjust_addr_to_host(MemoryDef::PHY_LOWER_ADDR, _emul_type),
            kernel_init_region_size: MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB,
            file_map_area_base_host: MemoryDef::FILE_MAP_OFFSET,
            file_map_area_base_guest: MemoryDef::FILE_MAP_OFFSET,
            file_map_area_size: MemoryDef::FILE_MAP_SIZE,
            //NOTE: Not backed by the host
            #[cfg(target_arch = "aarch64")]
            hypercall_mmio_base: MemoryDef::HYPERCALL_MMIO_BASE,
            #[cfg(target_arch = "aarch64")]
            hypercall_mmio_size: MemoryDef::HYPERCALL_MMIO_SIZE,
            stack_size: MemoryDef::DEFAULT_STACK_SIZE as usize,
        };
        let default_mem_layout = mem_layout_config;
        let _kernel_bin_path = VirtualMachine::KERNEL_IMAGE.to_string();
        let _vdso_bin_path = VirtualMachine::VDSO_PATH.to_string();
        let _sbox_uid_name = vm::SANDBOX_UID_NAME.to_string();

        let mut elf = KernelELF::New().expect("Failed to create elf object.");
        let _kernel_entry = elf
            .LoadKernel(_kernel_bin_path.as_str())
            .expect("Failed to load kernel from given path.");
        elf.LoadVDSO(_vdso_bin_path.as_str())
            .expect("Failed to load vdso from given path.");
        let _vdso_address = adjust_addr_to_guest(elf.vdsoStart, _emul_type);

        let vm_cc_emul = Self {
            vm_resources: VmResources {
                min_vcpu_amount: default_min_vcpus,
                kernel_bin_path: _kernel_bin_path,
                vdso_bin_path: _vdso_bin_path,
                sandbox_uid_name: _sbox_uid_name,
                pod_id: _pod_id,
                mem_layout: default_mem_layout,
            },
            entry_address: _kernel_entry,
            vdso_address: _vdso_address,
            emul_cc_mode: _emul_type,
        };
        let box_type: Box<dyn VmType> = Box::new(vm_cc_emul);

        Ok((box_type, elf))
    }

    fn create_vm(
        self: Box<VmCcEmul>,
        kernel_elf: KernelELF,
        args: Args,
    ) -> Result<VirtualMachine, Error> {
        crate::GLOBAL_ALLOCATOR.InitPrivateAllocator();
        *ROOT_CONTAINER_ID.lock() = args.ID.clone();
        if QUARK_CONFIG.lock().PerSandboxLog {
            let sandbox_name = match args
                .Spec
                .annotations
                .get(self.vm_resources.sandbox_uid_name.as_str())
            {
                None => args.ID[0..12].to_owned(),
                Some(name) => name.clone(),
            };
            LOG.Reset(&sandbox_name);
        }

        let cpu_count = args.GetCpuCount();
        let reserve_cpu_count = QUARK_CONFIG.lock().ReserveCpuCount;
        let cpu_count = if cpu_count == 0 {
            VMSpace::VCPUCount() - reserve_cpu_count
        } else {
            cpu_count.min(VMSpace::VCPUCount() - reserve_cpu_count)
        };

        if let Err(e) = self.vm_space_initialize(cpu_count, args) {
            error!("VM creation failed on VM-Space initialization.");
            return Err(e);
        } else {
            info!("VM creation - VM-Space initialization finished.");
        }

        {
            URING_MGR.lock();
        }

        let _kvm: Kvm;
        let vm_fd: VmFd;
        let _kvm_fd = VMS.lock().args.as_ref().unwrap().KvmFd;
        match self.create_kvm_vm(_kvm_fd) {
            Ok((__kvm, __vm_fd)) => {
                _kvm = __kvm;
                vm_fd = __vm_fd;
                info!("VM cration - kvm-vm_fd initialized.");
            }
            Err(e) => {
                error!("VM creation failed on kvm-vm creation.");
                return Err(e);
            }
        };

        self.vm_memory_initialize(&vm_fd)
            .expect("VM creation failed on memory initialization.");
        let (_, pheap, _) = self.vm_resources.mem_area_info(MemArea::PrivateHeapArea).unwrap();
        let _vcpu_total = VMS.lock().vcpuCount;
        let _auto_start = VMS.lock().args.as_ref().unwrap().AutoStart;
        let _vcpus = self
            .vm_vcpu_initialize(
                &_kvm,
                &vm_fd,
                _vcpu_total,
                self.entry_address,
                _auto_start,
                Some(pheap),
                None)
            .expect("VM creation failed on vcpu creation.");

        let _vm_type: Box<dyn VmType> = self;
        let vm = VirtualMachine {
            kvm: _kvm,
            vmfd: vm_fd,
            vm_type: _vm_type,
            vcpus: _vcpus,
            elf: kernel_elf,
        };
        Ok(vm)
    }

    fn vm_space_initialize(&self, vcpu_count: usize, args: Args) -> Result<(), Error> {
        let vms = &mut VMS.lock();
        vms.vcpuCount = vcpu_count.max(self.vm_resources.min_vcpu_amount);
        vms.cpuAffinit = true;
        vms.RandomVcpuMapping();
        vms.controlSock = args.ControlSock;
        vms.vdsoAddr = self.vdso_address;
        vms.pivot = args.Pivot;
        if let Some(id) = args
            .Spec
            .annotations
            .get(self.vm_resources.sandbox_uid_name.as_str())
        {
            vms.podUid = id.clone();
        } else {
            info!("No sandbox id found in specification.");
        }

        let (fmap_base_host, _, fmap_size) = self.vm_resources
            .mem_area_info(MemArea::FileMapArea).unwrap();
        PMA_KEEPER.Init(fmap_base_host, fmap_size);
        PMA_KEEPER.InitHugePages();
        vms.pageTables = PageTables::New(&vms.allocator)?;

        let page_opt = PageOpts::Kernel();
        let (_, kmem_base_guest, kmem_init_region) = self.vm_resources
            .mem_area_info(MemArea::KernelArea).unwrap();
        vms.KernelMapHugeTable(Addr(kmem_base_guest), Addr(kmem_base_guest + kmem_init_region),
            Addr(kmem_base_guest), page_opt.Val(),)?;

        #[cfg(target_arch = "aarch64")]
        {
            let mut page_opt = PageOpts::Zero();
            page_opt.SetWrite().SetGlobal().SetPresent().SetAccessed().SetMMIOPage();
            let (_, hcall_base, hcall_size) =
                self.vm_resources.mem_area_info(MemArea::HypercallMmioArea).unwrap();
            vms.KernelMap(Addr(hcall_base), Addr(hcall_base + hcall_size),
                Addr(hcall_base), page_opt.Val())?;
        }
        vms.args = Some(args);

        Ok(())
    }

    fn vm_memory_initialize(&self, vm_fd: &VmFd) -> Result<(), Error> {
        let (fmap_base_host, fmap_base_guest, fmap_region) = self.vm_resources
            .mem_area_info(MemArea::FileMapArea).unwrap();
        let (kmem_base_host, kmem_base_guest, _) = self.vm_resources
            .mem_area_info(MemArea::KernelArea).unwrap();
        let kmem_private_region = fmap_base_guest - kmem_base_guest;
        let kvm_kmem_region = kvm_bindings::kvm_userspace_memory_region {
            slot: 1,
            guest_phys_addr: kmem_base_guest,
            memory_size: kmem_private_region,
            userspace_addr: kmem_base_host,
            flags: 0,
        };

        let (pheap_base_host, pheap_base_guest, pheap_region) = self.vm_resources
            .mem_area_info(MemArea::PrivateHeapArea).unwrap();
        let kvm_private_heap_region = kvm_bindings::kvm_userspace_memory_region {
            slot: 2,
            guest_phys_addr: pheap_base_guest,
            memory_size: pheap_region,
            userspace_addr: pheap_base_host,
            flags: 0,
        };

        let kvm_file_map_region = kvm_bindings::kvm_userspace_memory_region {
            slot: 3,
            guest_phys_addr: fmap_base_guest,
            memory_size: fmap_region,
            userspace_addr: fmap_base_host,
            flags: 0,
        };

        let (shared_heap_base_host, shared_heap_base_guest, shared_heap_region) = self.vm_resources
            .mem_area_info(MemArea::SharedHeapArea).unwrap();
        let kvm_shared_heap_region = kvm_bindings::kvm_userspace_memory_region {
            slot: 4,
            guest_phys_addr: shared_heap_base_guest,
            memory_size: shared_heap_region,
            userspace_addr: shared_heap_base_host,
            flags: 0,
        };

        unsafe {
            vm_fd.set_user_memory_region(kvm_kmem_region).map_err(|e| {
                Error::IOError(format!("Failed to set kvm kernel memory region - error:{:?}",
                    e))})?;
            vm_fd.set_user_memory_region(kvm_private_heap_region).map_err(|e| {
                Error::IOError(format!("Failed to set kvm private heap memory region - error:{:?}",
                    e))})?;
            vm_fd.set_user_memory_region(kvm_shared_heap_region).map_err(|e| {
                Error::IOError(format!("Failed to set kvm shared heap memory region - error:{:?}",
                    e))})?;
            vm_fd.set_user_memory_region(kvm_file_map_region).map_err(|e| {
                Error::IOError(format!("Failed to set kvm file map memory region - error:{:?}",
                    e))})?;
        }

        info!("KernelMemRegion - Guest-phyAddr:{:#x}, host-VA:{:#x}, page mmap-size:{} MB",
            kmem_base_guest, kmem_base_host, kmem_private_region >> 20);
        info!("PrivateHeapMemRegion - Guest-phyAddr:{:#x}, host-VA:{:#x}, page mmap-size:{} MB",
            pheap_base_guest, pheap_base_host, pheap_region >> 20);
        info!("SharedMemRegion - Guest-phyAddr:{:#x}, host-VA:{:#x}, page mmap-size:{} MB",
            shared_heap_base_guest, shared_heap_base_host, shared_heap_region >> 20);
        info!("FileMapMemRegion - Guest-phyAddr:{:#x}, host-VA:{:#x}, page mmap-size:{} MB",
            fmap_base_guest, fmap_base_host, fmap_region >> 20);

        Ok(())
    }

    fn create_kvm_vm(&self, kvm_fd: i32) -> Result<(Kvm, VmFd), Error> {
        let kvm = unsafe { Kvm::from_raw_fd(kvm_fd) };

        if !kvm.check_extension(Cap::ImmediateExit) {
            panic!("Can not create VM - KVM_CAP_IMMEDIATE_EXIT is not supported.");
        }

        let vm_fd = kvm
            .create_vm()
            .map_err(|e| Error::IOError(format!("Failed to create a kvm-vm with error:{:?}", e)))?;

        #[cfg(target_arch = "x86_64")]
        {
            let mut cap: kvm_bindings::kvm_enable_cap = Default::default();
            cap.cap = kvm_bindings::KVM_CAP_X86_DISABLE_EXITS;
            cap.args[0] = (kvm_bindings::KVM_X86_DISABLE_EXITS_HLT
                | kvm_bindings::KVM_X86_DISABLE_EXITS_MWAIT) as u64;
            vm_fd.enable_cap(&cap).unwrap();
        }

        Ok((kvm, vm_fd))
    }

    fn init_share_space(vcpu_count: usize, control_sock: i32, rdma_svc_cli_sock: i32,
                     pod_id: [u8; 64], share_space_addr: Option<u64>,
                     _has_global_mem_barrier: Option<bool>) -> Result<(), Error> {
        use core::sync::atomic;
        crate::GLOBAL_ALLOCATOR.vmLaunched.store(true, atomic::Ordering::SeqCst);
        let shared_space_obj = unsafe {
            &mut *(share_space_addr.expect("Failed to initialize shared space in host\
                            - shared-space-table address is missing") as *mut ShareSpace)
        };
        let default_share_space_table = ShareSpace::New();
        let def_sh_space_tab_size = core::mem::size_of_val(&default_share_space_table);
        let sh_space_obj_size = core::mem::size_of_val(shared_space_obj);
        assert!(sh_space_obj_size == def_sh_space_tab_size,
            "Guest passed shared-space address does not match to a shared-space object.\
                Expected obj size:{:#x} - found:{:#x}", def_sh_space_tab_size, sh_space_obj_size);
        unsafe {
            core::ptr::write(shared_space_obj as *mut ShareSpace, default_share_space_table);
        }

        {
            let mut vms = VMS.lock();
            let shared_copy = vms.args.as_ref().unwrap().Spec.Copy();
            vms.args.as_mut().unwrap().Spec = shared_copy;
        }

        shared_space_obj.Init(vcpu_count, control_sock, rdma_svc_cli_sock, pod_id);
        SHARE_SPACE.SetValue(share_space_addr.unwrap());
        SHARESPACE.SetValue(share_space_addr.unwrap());
        let share_space_ptr = SHARE_SPACE.Ptr();
        KERNEL_IO_THREAD.Init(share_space_ptr.scheduler.VcpuArr[0].eventfd);
        FD_NOTIFIER.EpollCtlAdd(control_sock, EVENT_READ).unwrap();
        
        unsafe {
            CPU_LOCAL.Init(&SHARESPACE.scheduler.VcpuArr);
            futex::InitSingleton();
            timer::InitSingleton();
        }

        if SHARESPACE.config.read().EnableTsot {
            TSOT_AGENT.NextReqId();
            SHARESPACE.dnsSvc.Init().unwrap();
        }
        *SHARESPACE.bootId.lock() = uuid::Uuid::new_v4().to_string();
        crate::print::SetSyncPrint(share_space_ptr.config.read().SyncPrint());

        Ok(())
    }

    fn post_memory_initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn vm_vcpu_initialize(&self, kvm: &Kvm, vm_fd: &VmFd, total_vcpus: usize, entry_addr: u64,
                        auto_start: bool, page_allocator_addr: Option<u64>,
                        share_space_addr: Option<u64>) -> Result<Vec<Arc<ArchVirtCpu>>, Error> {
        let mut vcpus: Vec<Arc<ArchVirtCpu>> = Vec::with_capacity(total_vcpus);

        for vcpu_id in 0..total_vcpus {
            let vcpu = Arc::new(ArchVirtCpu::new_vcpu(
                vcpu_id as usize,
                total_vcpus,
                &vm_fd,
                entry_addr,
                page_allocator_addr,
                share_space_addr,
                auto_start,
                self.vm_resources.mem_layout.stack_size,
                Some(&kvm),
                self.emul_cc_mode)?);
            vcpus.push(vcpu);
        }

        for vcpu_id in 0..total_vcpus {
            let _ = vcpus[vcpu_id].vcpu_init();
            vcpus[vcpu_id].initialize_sys_registers()
                .expect("VM: Failed to initialize systetem registers");
            vcpus[vcpu_id].initialize_cpu_registers()
                .expect("VM: Failed to initialize GPR-registers");
        }
        VMS.lock().vcpus = vcpus.clone();

        Ok(vcpus)
    }

    fn post_vm_initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn post_init_upadate(&mut self) -> Result<(), Error> {
        Ok(())
    }
}
