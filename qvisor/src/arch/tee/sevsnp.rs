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

use crate::arch::Register;
use crate::qlib::linux_def::MemoryDef;
use crate::runc::runtime::vm_type::emulcc::VmCcEmul;
use crate::runc::runtime::vm_type::VmType;
use crate::sharepara::ShareParaPage;
use crate::VMS;
use crate::{arch::ConfCompExtension, qlib};
use kvm_bindings::{kvm_memory_attributes, KVM_MEMORY_ATTRIBUTE_PRIVATE, KVM_HC_MAP_GPA_RANGE};
use kvm_ioctls::{HypercallExit, VcpuExit, VmFd};

use qlib::common::Error;
use qlib::config::CCMode;

static mut DUMMY_U64: u64 = 0u64;

pub struct SevSnp<'a> {
    kvm_exits_list: [VcpuExit<'a>; 1],
    hypercalls_list: [u16; 1],
    pub cc_mode: CCMode,
    pub share_space_table_addr: Option<u64>,
    pub page_allocator_addr: u64,
}



impl ConfCompExtension for SevSnp<'_> {
    fn initialize_conf_extension(
        _share_space_table_addr: Option<u64>,
        _page_allocator_base_addr: Option<u64>,
    ) -> Result<Box<dyn ConfCompExtension>, crate::qlib::common::Error>
    where
        Self: Sized,
    {
        let ret_val = Box::new(0u64);
        let _self: Box<dyn ConfCompExtension> = Box::new(SevSnp {
            kvm_exits_list: [VcpuExit::Hypercall(HypercallExit {
                nr: KVM_HC_MAP_GPA_RANGE,
                args: [0; 6],
                ret: Box::leak(ret_val),  // mutable reference points to variable
                longmode: 1,                  // assuming 1 (true) for long mode
            })],
            hypercalls_list: [qlib::HYPERCALL_SHARESPACE_INIT],
            share_space_table_addr: None,
            page_allocator_addr: _page_allocator_base_addr
                .expect("Exptected address of the page allocator - found None"),
            cc_mode: CCMode::SevSnp,
        });
        Ok(_self)
    }

    fn set_cpu_registers(
        &self,
        vcpu_fd: &kvm_ioctls::VcpuFd,
        _regs: Option<Vec<Register>>,
    ) -> Result<(), crate::qlib::common::Error> {
        self._set_cpu_registers(&vcpu_fd)
    }

    fn get_hypercall_arguments(
        &self,
        vcpu_fd: &kvm_ioctls::VcpuFd,
        vcpu_id: usize,
    ) -> Result<(u64, u64, u64, u64), crate::qlib::common::Error> {
        self._get_hypercall_arguments(vcpu_fd, vcpu_id)
    }

    fn should_handle_kvm_exit(&self, kvm_exit: &kvm_ioctls::VcpuExit) -> bool {
        self.kvm_exits_list.iter().any(|exit| match (exit, kvm_exit) {
            (VcpuExit::Hypercall(a), VcpuExit::Hypercall(b)) => a.nr == b.nr,
            // Add other arms if needed
            _ => false,
        })
    }

    fn should_handle_hypercall(&self, hypercall: u16) -> bool {
        self.hypercalls_list.contains(&hypercall)
    }

    fn handle_kvm_exit(
        &self,
        kvm_exit: &mut kvm_ioctls::VcpuExit,
        _vcpu_id: usize,
        vm_fd: Option<&VmFd>,
    ) -> Result<bool, crate::qlib::common::Error> {
        let mut _exit = false;
        _exit = match kvm_exit {
            VcpuExit::Hypercall(HypercallExit {
                nr: KVM_HC_MAP_GPA_RANGE,
                args: [address, num_pages, attrs, ..],
                ret,
                ..
            }) => {
                let private = (*attrs & (1 << 3)) != 0;
                let res = self._handle_kvm_hypercall(*address, *num_pages, private, vm_fd.unwrap());
                match res {
                    Ok(_) => **ret = 0,
                    Err(_) => **ret = 1,
                }
                return res
            }
            _ => false,
        };

        Ok(_exit)
    }

    fn handle_hypercall(
        &self,
        hypercall: u16,
        arg0: u64,
        arg1: u64,
        arg2: u64,
        arg3: u64,
        vcpu_id: usize,
    ) -> Result<bool, crate::qlib::common::Error> {
        let mut _exit = false;
        _exit = match hypercall {
            qlib::HYPERCALL_SHARESPACE_INIT => {
                self._handle_hcall_shared_space_init(arg0, arg1, arg2, arg3, vcpu_id)?
            }
            _ => false,
        };

        Ok(_exit)
    }
}

impl SevSnp<'_> {
    fn _confidentiality_type(&self) -> CCMode {
        self.cc_mode
    }

    fn _get_hypercall_arguments(
        &self,
        _vcpu_fd: &kvm_ioctls::VcpuFd,
        vcpu_id: usize,
    ) -> Result<(u64, u64, u64, u64), Error> {
        let shared_param_buffer =
            unsafe { *(MemoryDef::HYPERCALL_PARA_PAGE_OFFSET as *const ShareParaPage) };
        let passed_params = shared_param_buffer.SharePara[vcpu_id];
        let _arg0 = passed_params.para1;
        let _arg1 = passed_params.para2;
        let _arg2 = passed_params.para3;
        let _arg3 = passed_params.para4;

        Ok((_arg0, _arg1, _arg2, _arg3))
    }

    pub(self) fn _handle_hcall_shared_space_init(
        &self,
        arg0: u64,
        _arg1: u64,
        _arg2: u64,
        _arg3: u64,
        _vcpu_id: usize,
    ) -> Result<bool, Error> {
        let ctrl_sock: i32;
        let vcpu_count: usize;
        let rdma_svc_cli_sock: i32;
        let mut pod_id = [0u8; 64]; //TODO: Hardcoded length of ID set it as cost to check on
        {
            let vms = VMS.lock();
            ctrl_sock = vms.controlSock;
            vcpu_count = vms.vcpuCount;
            rdma_svc_cli_sock = vms.args.as_ref().unwrap().RDMASvcCliSock;
            let id_bytes = vms.args.as_ref().unwrap().ID.as_bytes();
            let n = id_bytes.len().min(64);
            pod_id[..n].copy_from_slice(&id_bytes[..n]);
        }
        if let Err(e) = VmCcEmul::init_share_space(
            vcpu_count,
            ctrl_sock,
            rdma_svc_cli_sock,
            pod_id,
            Some(arg0),
            None,
        ) {
            error!("Vcpu: hypercall failed on shared-space initialization.");
            return Err(e);
        } else {
            info!("Vcpu: finished shared-space initialization.");
        }

        Ok(false)
    }

    pub(self) fn _handle_kvm_hypercall(
        &self,
        gpa: u64,
        num_pages: u64,
        private: bool,
        vm_fd: &VmFd,
    ) -> Result<bool, Error> {
        debug!("HypercallExit, gpa {:x}, num_pages: {:?}, private {:?}", gpa, num_pages,  private);
        let attr = if private {
            KVM_MEMORY_ATTRIBUTE_PRIVATE
        } else {
            0
        };
        let memory_attributes = kvm_memory_attributes {
            address: gpa,
            size: num_pages * MemoryDef::PAGE_SIZE,
            attributes: attr as u64,
            flags: 0,
        };
        vm_fd
            .set_memory_attributes(memory_attributes)
            .expect("Unable to convert memory to private");
        Ok(false)
    }
}
