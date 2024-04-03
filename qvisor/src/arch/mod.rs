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

use std::sync::mpsc::Sender;
use kvm_bindings::CpuId;
use spin::MutexGuard;
use crate::qlib::common::Error;

#[cfg(target_arch = "x86_64")]
#[path = "./x86_64/mod.rs"]
pub mod __cpu_arch;

pub trait vCPU {
    fn new (kvm_vm_fd: &kvm_ioctls::VmFd, vCPU_id: usize) -> Self;
    fn init(&mut self, vcpu_id: usize, kvm_vcpu_id: CpuId) -> Result<(), Error>;
    fn run(&self, entry_addr: u64, stack_start_addr: u64, heap_start_addr: u64,
           share_space_addr: u64, vcpu_id: u64, vdso_addr: u64, cpus_total: u64,
           auto_start: bool) -> Result<(), Error>;
    fn interrupt_guest(&self);
    fn get_interrupt_lock(&self) -> MutexGuard<'_, (bool, Vec<Sender<()>>)>;
    fn vcpu_fd(&self) -> &kvm_ioctls::VcpuFd;
    fn dump(&self, vcpu_id: u64) -> Result<(), Error>;
}
