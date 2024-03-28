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

use kvm_bindings::kvm_regs;
use std::sync::mpsc::Sender;
use std::os::unix::io::AsRawFd;
use libc::ioctl;
use spin::{Mutex, MutexGuard};

use crate::arch::vCPU;
use crate::qlib::common::{KERNEL_FLAGS_SET, Error};
use crate::qlib::linux_def::MemoryDef;

pub mod context;
pub mod vm;

#[derive(Default, Debug)]
pub struct x86_64vCPU {
    gdtAddr: u64,
    idtAddr: u64,
    tssIntStackStart: u64,
    tssAddr: u64,
    vcpu_fd: Option<kvm_ioctls::VcpuFd>,
    interrupting: Mutex<(bool, Vec<Sender<()>>)>
}

pub type ArchvCPU = x86_64vCPU;

impl vCPU for x86_64vCPU {
    fn interrupt_guest(&self) {
        let kvm_interrupt: u64 = 0x4004ae86;
        let bounce: u32 = 20; //VirtualizationException
        let ret = unsafe {
            ioctl(
                self.vcpu_fd().unwrap().as_raw_fd(),
                kvm_interrupt,
                &bounce as *const _ as u64,
            )
        };

        assert!(
            ret == 0,
            "InterruptGuest ret is {}/{}/{}",
            ret,
            errno::errno().0,
            self.vcpu_fd().unwrap().as_raw_fd()
        );
    }

    fn get_interrupt_lock(&self) -> MutexGuard<'_, (bool, Vec<Sender<()>>)> {
        self.interrupting.lock()
    }

    fn vcpu_fd(&self) -> Option<&kvm_ioctls::VcpuFd> {
        if self.vcpu_fd.is_none() {
            None
        } else {
            self.vcpu_fd.as_ref()
        }
    }

    fn new(kvm_vm_fd: &kvm_ioctls::VmFd, vCPU_id: usize) -> Self {
        let kvm_vcpu_fd = kvm_vm_fd
            .create_vcpu(vCPU_id as u64)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))
            .expect("create vcpu fail");

        Self {
            vcpu_fd: Some(kvm_vcpu_fd),
            ..Default::default()
        }
    }

    fn init(&mut self, id: usize) -> Result<(), Error> {
        self.vcpu_reg_init(id)?;
        info!("The tssIntStackStart is {:x}, tssAddr address is {:x}, idt addr is {:x}, gdt addr is {:x}",
             self.tssIntStackStart, self.tssAddr, self.idtAddr, self.gdtAddr);
        info!(
            "[{}] - The tssSegment stack is {:x}",
            id,
            self.tssIntStackStart + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE
        );
        Ok(())
    }

    fn run(&self, entry_addr: u64, stack_start_addr: u64, heap_start_addr: u64,
           share_space_addr: u64, id: u64, vdso_addr: u64, cpus_total: u64,
           auto_start: bool) -> Result<(), Error> {
        let regs: kvm_regs = kvm_regs {
            rflags: KERNEL_FLAGS_SET,
            rip: entry_addr,
            rsp: stack_start_addr,
            rax: 0x11,
            rbx: 0xdd,
            //arg0
            rdi: heap_start_addr,
            //arg1
            rsi: share_space_addr,
            //arg2
            rdx: id,
            //arg3
            rcx: vdso_addr,
            //arg4
            r8: cpus_total,
            //arg5
            r9: auto_start as u64,
            ..Default::default()
        };

        self.vcpu_fd
            .unwrap()
            .set_regs(&regs)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        self.vcpu_run(id)
    }
}
