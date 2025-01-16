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

pub mod emulcc;

use kvm_ioctls::{VcpuExit, VcpuFd};

use super::{ConfCompExtension, vm::vcpu::kvm_vcpu::Register};
use crate::{qlib::common::Error, CCMode};


pub struct NonConf<'a> {
    kvm_exits_list: Option<[VcpuExit<'a>; 0]>,
    hypercalls_list: Option<[u16; 0]>,
    pub cc_mode: CCMode,
    pub share_space_table_addr: u64,
    pub page_allocator_addr: u64,
}

impl ConfCompExtension for NonConf<'_> {
    fn initialize_conf_extension(_share_space_addr: Option<u64>, _page_allocator: Option<u64>)
        -> Result<Box<dyn ConfCompExtension>, Error>
        where Self: Sized {
        let _self: Box<dyn ConfCompExtension> = Box::new(NonConf{
            kvm_exits_list: None,
            hypercalls_list: None,
            share_space_table_addr: _share_space_addr
                .expect("Exptected base address of the share space - found None"),
            page_allocator_addr: _page_allocator
                .expect("Exptected address of the page allocator - found None"),
            cc_mode: CCMode::None,
        });
        Ok(_self)
    }

    fn should_handle_hypercall(&self, _hypercall: u16) -> bool {
        self.hypercalls_list.is_some()
    }

    fn should_handle_kvm_exit(&self, _kvm_exit: &VcpuExit) -> bool {
        self.kvm_exits_list.is_some()
    }

    fn set_cpu_registers(&self, vcpu_fd: &VcpuFd, _regs: Option<Vec<Register>>)
        -> Result<(), Error> {
        self._set_cpu_registers(&vcpu_fd)
    }

    fn get_hypercall_arguments(&self, vcpu_fd: &VcpuFd, _vcpu_id: usize)
        -> Result<(u64, u64, u64, u64), Error> {
        self._get_hypercall_arguments(&vcpu_fd, _vcpu_id)
    }

    fn handle_hypercall(&self, _hypercall: u16, _arg0: u64, _arg1: u64, _arg2: u64,
        _arg3: u64, _vcpu_id: usize) -> Result<bool, Error> { Ok(false) }
}

pub mod util {
    use crate::{qlib::linux_def::MemoryDef, CCMode};

    #[inline]
    pub fn get_offset(confidentiality_type: CCMode) -> u64 {
        let offset = match confidentiality_type {
            CCMode::None | CCMode::Normal =>
                0,
            CCMode::NormalEmu =>
                MemoryDef::UNIDENTICAL_MAPPING_OFFSET,
            _ => panic!(""),
        };
        offset
    }

    #[inline]
    pub fn adjust_addr_to_guest(host_addr: u64, confidentiality_type: CCMode) -> u64 {
        host_addr - get_offset(confidentiality_type)
    }

    #[inline]
    pub fn adjust_addr_to_host(addr: u64, confidentiality_type: CCMode) -> u64 {
        addr + get_offset(confidentiality_type)
    }
}
