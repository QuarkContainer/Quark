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

use crate::arch::vm::vcpu::kvm_vcpu::Register;
use crate::qlib::linux_def::MemoryDef;
use crate::runc::runtime::vm_type::VmType;
use crate::runc::runtime::vm_type::emulcc::VmCcEmul;
use crate::VMS;
use crate::{arch::ConfCompExtension, qlib, QUARK_CONFIG};
use kvm_ioctls::{VcpuExit, VcpuFd};
use qlib::config::CCMode;
use qlib::common::Error;

pub struct EmulCc<'a> {
    kvm_exits_list: Option<[VcpuExit<'a>; 0]>,
    hypercalls_list: [u16; 1],
    pub cc_mode: CCMode,
    pub share_space_table_addr: Option<u64>,
    pub page_allocator_addr: u64,
}

impl ConfCompExtension for EmulCc<'_> {
    fn initialize_conf_extension(_share_space_table_addr: Option<u64>,
        _page_allocator_base_addr: Option<u64>)
        -> Result<Box<dyn ConfCompExtension>, crate::qlib::common::Error>
        where Self: Sized {
        let _cc_mode = QUARK_CONFIG.lock().CCMode;
        let _self: Box<dyn ConfCompExtension> = Box::new(EmulCc{
            kvm_exits_list: None,
            hypercalls_list:[qlib::HYPERCALL_SHARESPACE_INIT],
            share_space_table_addr: None,
            page_allocator_addr: _page_allocator_base_addr
                .expect("Exptected address of the page allocator - found None"),
            cc_mode: _cc_mode,
        });
        Ok(_self)
    }

    fn set_cpu_registers(&self, vcpu_fd: &VcpuFd, _regs: Option<Vec<Register>>)
        -> Result<(), Error> {
        self._set_cpu_registers(&vcpu_fd)
    }

    fn get_hypercall_arguments(&self, vcpu_fd: &kvm_ioctls::VcpuFd, vcpu_id: usize)
        -> Result<(u64, u64, u64, u64), crate::qlib::common::Error> {
        self._get_hypercall_arguments(vcpu_fd, vcpu_id)
    }

    fn should_handle_kvm_exit(&self, _kvm_exit: &kvm_ioctls::VcpuExit) -> bool {
        self.kvm_exits_list.is_some()
    }

    fn should_handle_hypercall(&self, hypercall: u16) -> bool {
        if hypercall == self.hypercalls_list[0] {
            true
        } else {
            false
        }
    }

    fn handle_hypercall(&self, hypercall: u16, arg0: u64, arg1: u64, arg2: u64,
        arg3: u64, vcpu_id: usize) -> Result<bool , crate::qlib::common::Error> {
        let mut _exit = false;
        _exit = match hypercall {
            qlib::HYPERCALL_SHARESPACE_INIT =>
                self._handle_hcall_shared_space_init(arg0, arg1, arg2, arg3, vcpu_id)?,
            _ => false,
        };

        Ok(_exit)
    }
}

impl EmulCc<'_> {
    fn _get_hypercall_arguments(&self, _vcpu_fd: &kvm_ioctls::VcpuFd, vcpu_id: usize)
        -> Result<(u64, u64, u64, u64), Error> {
        use crate::sharepara::ShareParaPage;
        let shared_param_buffer = unsafe {
            *(MemoryDef::HYPERCALL_PARA_PAGE_OFFSET as *const ShareParaPage)
        };
        let passed_params = shared_param_buffer.SharePara[vcpu_id];
        let _arg0 = passed_params.para1;
        let _arg1 = passed_params.para2;
        let _arg2 = passed_params.para3;
        let _arg3 = passed_params.para4;

        Ok((_arg0, _arg1, _arg2, _arg3))
    }

    pub(in self) fn _handle_hcall_shared_space_init(&self, arg0: u64, _arg1: u64, _arg2: u64,
        _arg3: u64, _vcpu_id: usize) -> Result<bool, Error> {
        let ctrl_sock: i32;
        let vcpu_count: usize;
        let rdma_svc_cli_sock: i32;
        let mut pod_id = [0u8; 64]; //TODO: Hardcoded length of ID set it as cost to check on
        {
            let vms = VMS.lock();
            ctrl_sock = vms.controlSock;
            vcpu_count = vms.vcpuCount;
            rdma_svc_cli_sock = vms.args.as_ref().unwrap().RDMASvcCliSock;
            pod_id.copy_from_slice(vms.args.as_ref().unwrap().ID.clone().as_bytes());
        }
        if let Err(e) = VmCcEmul::init_share_space(vcpu_count, ctrl_sock, rdma_svc_cli_sock,
            pod_id, Some(arg0), None) {
            error!("Vcpu: hypercall failed on shared-space initialization.");
            return Err(e);
        } else {
            info!("Vcpu: finished shared-space initialization.");
        }

        Ok(false)
    }
}
