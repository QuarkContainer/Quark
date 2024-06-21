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

#[cfg(target_arch = "aarch64")]
#[path = "./aarch64/mod.rs"]
pub mod vm;

#[cfg(target_arch = "x86_64")]
#[path = "./x86_64/mod.rs"]
pub mod vm;

pub mod tee;

use std::os::fd::RawFd;
use kvm_ioctls::{Kvm, VcpuExit, VcpuFd, VmFd};
use vm::vcpu::ArchVirtCpu;
use crate::{qlib::common::Error, CCMode};

pub trait VirtCpu {
    fn new_vcpu(vcpu_id: usize, total_vcpus: usize, vm_fd: &VmFd, entry_addr: u64,
        page_allocator_base_addr: Option<u64>, share_space_table_addr: Option<u64>,
        auto_start: bool, stack_size: usize, kvm: Option<&Kvm>, conf_extension: CCMode)
        -> Result<ArchVirtCpu, Error>;
    fn initialize_sys_registers(&self) -> Result<(), Error>;
    fn initialize_cpu_registers(&self) -> Result<(), Error>;
    fn default_hypercall_handler(&self, hypercall: u16, data: &[u8],
        arg0: u64, arg1: u64, arg2: u64, arg3: u64) -> Result<bool, Error>;
    fn default_kvm_exit_handler(&self, kvm_exit: VcpuExit) -> Result<bool, Error>;
    fn vcpu_run(&self, tgid: i32, _kvm_raw_fd: Option<RawFd>, _vm_raw_fd: Option<RawFd>)
        -> Result<(), Error>;
}

pub trait ConfCompExtension: Send + Sync {
    fn initialize_conf_extension(share_space_table_addr: Option<u64>,
        page_allocator_base_addr: Option<u64>) -> Result<Box<dyn ConfCompExtension>, Error>
        where Self: Sized;
    fn set_sys_registers(&self, vcpu_fd: &VcpuFd) -> Result<(), Error>;
    fn set_cpu_registers(&self, vcpu_fd: &VcpuFd) -> Result<(), Error>;
    fn get_hypercall_arguments(&self, vcpu_fd: &VcpuFd, vcpu_id: usize)
        -> Result<(u64, u64, u64, u64), Error>;
    fn should_handle_kvm_exit(&self, kvm_exit: &VcpuExit) -> bool;
    fn should_handle_hypercall(&self, hypercall: u16) -> bool;
    fn handle_kvm_exit(&self, kvm_exit: &mut VcpuExit, vcpu_id: usize, _vm_fd: Option<&VmFd>)
        -> Result<bool, Error>;
    fn handle_hypercall(&self, hypercall: u16, data: &[u8], arg0: u64, arg1: u64, arg2: u64,
        arg3: u64, vcpu_id: usize) -> Result<bool , Error>;
    fn confidentiality_type(&self) -> CCMode;
}
