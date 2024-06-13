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

use super::super::vm::VirtualMachine;
use super::{resources::{MemArea, MemLayoutConfig, VmResources}, VmType};
use crate::arch::{VirtCpu, ConfCompType::NoConf, vm::vcpu::ArchVirtCpu};
use crate::qlib::kernel::{vcpu, IOURING, KERNEL_PAGETABLE, PAGE_MGR};
use crate::{elf_loader::KernelELF, print::LOG, qlib, runc::runtime, tsot_agent::TSOT_AGENT,
            vmspace::VMSpace, KERNEL_IO_THREAD, PMA_KEEPER, QUARK_CONFIG, ROOT_CONTAINER_ID,
            SHARE_SPACE, SHARE_SPACE_STRUCT, URING_MGR, VMS};
use addr::{Addr, PageOpts};
use kernel::{kernel::futex, kernel::timer, task, KERNEL_STACK_ALLOCATOR, SHARESPACE};
use kvm_bindings::kvm_enable_cap;
use kvm_ioctls::{Cap, Kvm, VmFd};
use pagetable::{AlignedAllocator, PageTables};
use qlib::{addr, common::Error, kernel, linux_def::MemoryDef, pagetable};
use runtime::{loader::Args, vm};
use std::fmt;
use std::ops::Deref;
use std::os::fd::FromRawFd;
use std::sync::Arc;

#[cfg(target_arch = "aarch64")]
use MemArea::HypercallMmioArea;
use MemArea::{FileMapArea, KernelArea, SharedHeapArea};

pub struct VmNormal {
    vm_resources: VmResources,
    entry_address: u64,
    vdso_address: u64,
}

impl fmt::Debug for VmNormal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VmNormal[\nVM Resources:{:?},\nEntry Address:{:#x},\nVDSO Address:{:#x}]",
            self.vm_resources, self.entry_address, self.vdso_address)
    }
}

impl VmType for VmNormal {
    //NOTE: Use defaults for now, but we can improve it, e.g. configuration file.
    fn init(args: Option<&Args>) -> Result<(Box<dyn VmType>, KernelELF), Error> {
        let _pod_id = args.expect("VM creation expects arguments").ID.clone();
        let default_min_vcpus = 2;
        let mem_layout_config = MemLayoutConfig {
            shared_heap_mem_base: MemoryDef::HEAP_OFFSET,
            shared_heap_mem_size: MemoryDef::HEAP_SIZE,
            kernel_base: MemoryDef::PHY_LOWER_ADDR,
            kernel_size: MemoryDef::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB,
            file_map_area_base: MemoryDef::FILE_MAP_OFFSET,
            file_map_area_size: MemoryDef::FILE_MAP_SIZE,
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
        let _vdso_address = elf.vdsoStart;

        let normal_vm = Self {
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
        };
        let box_type: Box<dyn VmType> = Box::new(normal_vm);

        Ok((box_type, elf))
    }

    fn create_vm(
        self: Box<VmNormal>,
        kernel_elf: KernelELF,
        args: Args,
    ) -> Result<VirtualMachine, Error> {
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

        let mut pod_id = [0u8; 64];
        pod_id.copy_from_slice(VMS.lock().args.as_ref().unwrap().ID.clone().as_bytes());
        let _vcpu_total = VMS.lock().vcpuCount;
        let _ctrl_sock = VMS.lock().controlSock;
        let _rdma_sock = VMS.lock().args.as_ref().unwrap().RDMASvcCliSock;
        if let Err(e) =
            self.init_share_space(_vcpu_total, _ctrl_sock, _rdma_sock, pod_id, None, None)
        {
            error!("VM creation failed on VM-Space initialization.");
            return Err(e);
        } else {
            info!("VM creation - shared-space initialization finished.");
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
        let (heap_base, _) = self.vm_resources.mem_area_info(SharedHeapArea).unwrap();
        let _auto_start = VMS.lock().args.as_ref().unwrap().AutoStart;
        let _vcpus = self
            .vm_vcpu_initialize(
                &_kvm,
                &vm_fd,
                _vcpu_total,
                self.entry_address,
                _auto_start,
                Some(heap_base),
                Some(SHARE_SPACE.Value()),
            )
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

        let (fmap_base, fmap_size) = self.vm_resources.mem_area_info(FileMapArea).unwrap();
        PMA_KEEPER.Init(fmap_base, fmap_size);
        PMA_KEEPER.InitHugePages();
        vms.pageTables = PageTables::New(&vms.allocator)?;

        let mut page_opt = PageOpts::Zero();
        page_opt = PageOpts::Kernel();
        let (kmem_base, kmem_size) = self.vm_resources.mem_area_info(KernelArea).unwrap();
        vms.KernelMapHugeTable(Addr(kmem_base), Addr(kmem_base + kmem_size),
            Addr(kmem_base), page_opt.Val(),)?;

        #[cfg(target_arch = "aarch64")]
        {
            page_opt = PageOpts::Zero();
            page_opt.SetWrite().SetGlobal().SetPresent().SetAccessed().SetMMIOPage();
            let (hcall_base, hcall_size) =
                self.vm_resources.mem_area_info(HypercallMmioArea).unwrap();
            vms.KernelMap(Addr(hcall_base), Addr(hcall_base + hcall_size),
                Addr(hcall_base), page_opt.Val())?;
        }
        vms.args = Some(args);

        Ok(())
    }

    fn init_share_space(
        &self,
        cpu_count: usize,
        control_sock: i32,
        rdma_svc_cli_sock: i32,
        pod_id: [u8; 64],
        _share_space_addr: Option<u64>,
        _has_global_mem_barrierr: Option<bool>,
    ) -> Result<(), Error> {
        SHARE_SPACE_STRUCT
            .lock()
            .Init(cpu_count, control_sock, rdma_svc_cli_sock, pod_id);
        let shared_space_table = SHARE_SPACE_STRUCT.lock().deref().Addr();
        SHARE_SPACE.SetValue(shared_space_table);
        SHARESPACE.SetValue(shared_space_table);
        let share_space_ptr = SHARE_SPACE.Ptr();
        KERNEL_IO_THREAD.Init(share_space_ptr.scheduler.VcpuArr[0].eventfd);
        IOURING.SetValue(share_space_ptr.GetIOUringAddr());

        unsafe {
            vcpu::CPU_LOCAL.Init(SHARESPACE.scheduler.VcpuArr.as_ref());
            KERNEL_PAGETABLE.SetRoot(VMS.lock().pageTables.GetRoot());
            PAGE_MGR.SetValue(share_space_ptr.GetPageMgrAddr());
            KERNEL_STACK_ALLOCATOR.Init(AlignedAllocator::New(
                MemoryDef::DEFAULT_STACK_SIZE as usize,
                MemoryDef::DEFAULT_STACK_SIZE as usize,
            ));
            task::InitSingleton();
            futex::InitSingleton();
            timer::InitSingleton();
        }

        if SHARESPACE.config.read().EnableTsot {
            //Initialize tsot_agent
            TSOT_AGENT.NextReqId();
            SHARESPACE.dnsSvc.Init().unwrap();
        }

        *SHARESPACE.bootId.lock() = uuid::Uuid::new_v4().to_string();
        crate::print::SetSyncPrint(share_space_ptr.config.read().SyncPrint());
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
            let mut cap: kvm_enable_cap = Default::default();
            cap.cap = kvm_bindings::KVM_CAP_X86_DISABLE_EXITS;
            cap.args[0] = (kvm_bindings::KVM_X86_DISABLE_EXITS_HLT
                | kvm_bindings::KVM_X86_DISABLE_EXITS_MWAIT) as u64;
            vm_fd.enable_cap(&cap).unwrap();
        }

        Ok((kvm, vm_fd))
    }

    fn vm_memory_initialize(&self, vm_fd: &VmFd) -> Result<(), Error> {
        let (kmem_base, kmem_size) = self.vm_resources.mem_area_info(KernelArea).unwrap();
        let kvm_mem_region = kvm_bindings::kvm_userspace_memory_region {
            slot: 1,
            guest_phys_addr: kmem_base,
            memory_size: kmem_size,
            userspace_addr: kmem_base,
            flags: 0,
        };

        unsafe {
            vm_fd.set_user_memory_region(kvm_mem_region).map_err(|e| {
                Error::IOError(format!("Failed to set kvm memory region - error:{:?}", e))})?;
        }

        info!("SetMemRegion - phyAddr:{:#x}, hostAddr:{:#x}, page mmap-size:{:#x} MB",
            kmem_base, kmem_base,kmem_size >> 20);

        Ok(())
    }

    fn vm_vcpu_initialize(
        &self,
        kvm: &Kvm,
        vm_fd: &VmFd,
        total_vcpus: usize,
        entry_addr: u64,
        auto_start: bool,
        page_allocator_addr: Option<u64>,
        share_space_addr: Option<u64>,
    ) -> Result<Vec<Arc<ArchVirtCpu>>, Error> {
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
                NoConf)?);
            vcpus.push(vcpu);
        }
        VMS.lock().vcpus = vcpus.clone();

        Ok(vcpus)
    }

    fn post_memory_initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn post_vm_initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn post_init_upadate(&mut self) -> Result<(), Error> {
        Ok(())
    }
}
