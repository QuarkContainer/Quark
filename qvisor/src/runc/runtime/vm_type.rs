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


pub mod noncc;
pub mod emulcc;
pub mod resources;

use std::sync::Arc;
use kvm_ioctls::{Kvm, VmFd};
use crate::{arch::vm::vcpu::ArchVirtCpu, elf_loader::KernelELF, qlib::common::Error,
            runc::runtime};
use runtime::{vm::VirtualMachine, loader::Args};

pub trait VmType: std::fmt::Debug {
    fn init(args: Option<&Args>) -> Result<(Box<dyn VmType>, KernelELF), Error>
        where Self: Sized;
    /// Entry point for creating a VM
    #[allow(patterns_in_fns_without_body)]
    fn create_vm(mut self: Box<Self>, kernel_elf: KernelELF, args: Args)
        -> Result<VirtualMachine, Error>;
    fn vm_space_initialize(&self, vcpu_count: usize, args: Args) -> Result<(), Error>;
    fn init_share_space(vcpu_count: usize, control_sock: i32, rdma_svc_cli_sock: i32,
                     pod_id: [u8; 64], share_space_addr: Option<u64>,
                     has_global_mem_barrier: Option<bool>) -> Result<(), Error>
        where Self: Sized;
    fn create_kvm_vm(&mut self, kvm_fd: i32) -> Result<(Kvm, VmFd), Error>;
    fn vm_memory_initialize(&mut self, vm_fd: &VmFd) -> Result<(), Error>;
    fn post_memory_initialize(&mut self, _vm_fd: &mut VmFd) -> Result<(), Error> { Ok(()) }
    fn vm_vcpu_initialize(&self, kvm: &Kvm, vm_fd: &VmFd, total_vcpus: usize, entry_addr: u64,
                        auto_start: bool, page_allocator_addr: Option<u64>,
                        share_space_addr: Option<u64>) -> Result<Vec<Arc<ArchVirtCpu>>, Error>;
    fn post_vm_initialize(&mut self, _vm_fd: &mut VmFd) -> Result<(), Error> { Ok(()) }
    fn post_init_update(&mut self, _vm_fd: &mut VmFd) -> Result<(), Error> { Ok(()) }
}
