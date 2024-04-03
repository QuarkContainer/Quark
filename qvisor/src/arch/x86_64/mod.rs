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

use kvm_bindings::{kvm_regs, CpuId};
use kvm_ioctls::Kvm;
use std::sync::mpsc::Sender;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::convert::TryInto;
use libc::ioctl;
use spin::{Mutex, MutexGuard};

use crate::QUARK_CONFIG;
use crate::arch::vCPU;
use crate::runc::runtime::vm::{Dump, ClearDump};
use crate::qlib::common::{KERNEL_FLAGS_SET, Error};
use crate::qlib::linux_def::MemoryDef;

pub mod context;
pub mod vm;

#[derive(Debug)]
pub struct x86_64vCPU {
    gdtAddr: u64,
    idtAddr: u64,
    tssIntStackStart: u64,
    tssAddr: u64,
    //
    //Not Option-> Error if not present in initalisation
    //
    vcpu_fd: kvm_ioctls::VcpuFd,
    interrupting: Mutex<(bool, Vec<Sender<()>>)>
}

pub type ArchvCPU = x86_64vCPU;

impl vCPU for x86_64vCPU {
    fn interrupt_guest(&self) {
        let kvm_interrupt: u64 = 0x4004ae86;
        let bounce: u32 = 20; //VirtualizationException
        let ret = unsafe {
            ioctl(
                self.vcpu_fd()
                    .as_raw_fd(),
                kvm_interrupt,
                &bounce as *const _ as u64,
            )
        };

        assert!(
            ret == 0,
            "InterruptGuest ret is {}/{}/{}",
            ret,
            errno::errno().0,
            self.vcpu_fd().as_raw_fd()
        );
    }

    fn dump(&self, vcpu_id: u64) -> Result<(), Error> {
        if !Dump(vcpu_id
                 .try_into()
                 .unwrap()) {
            return Ok(());
        }
        defer!(ClearDump(vcpu_id
                         .try_into()
                         .unwrap()));
        let regs = self
            .vcpu_fd()
            .get_regs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let sregs = self
            .vcpu_fd()
            .get_sregs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        error!("vcpu regs: {:#x?}", regs);
        let ss = sregs.ss.selector as u64;
        let isUser = (ss & 0x3) != 0;
        if isUser {
            error!("vcpu {} is in user mode, skip", vcpu_id);
            return Ok(());
        }
        let kernelMemRegionSize = QUARK_CONFIG.lock().KernelMemSize;
        let mut frames = String::new();
        crate::qlib::backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
            frames.push_str(&format!("{:#x?}\n", frame));
            if frame.rbp < MemoryDef::PHY_LOWER_ADDR
                || frame.rbp >= MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB
            {
                false
            } else {
                true
            }
        });
        error!("vcpu {} stack: {}", vcpu_id, frames);
        Ok(())
    }

    fn get_interrupt_lock(&self) -> MutexGuard<'_, (bool, Vec<Sender<()>>)> {
        self.interrupting.lock()
    }

    fn vcpu_fd(&self) -> &kvm_ioctls::VcpuFd {
            &self.vcpu_fd
    }

    fn new(kvm_vm_fd: &kvm_ioctls::VmFd, vCPU_id: usize) -> Self {
        let kvm_vcpu_fd = kvm_vm_fd
            .create_vcpu(vCPU_id as u64)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))
            .expect("create vcpu fail");

        Self {
            gdtAddr: 0,
            idtAddr: 0,
            tssIntStackStart: 0,
            tssAddr: 0,
            vcpu_fd: kvm_vcpu_fd,
            interrupting: Mutex::new((false, vec![])),
        }
    }

    fn init(&mut self, vcpu_id: usize, kvm_cpu_id: CpuId) -> Result<(), Error> {

        self.vcpu_fd
            .set_cpuid2(&kvm_cpu_id).unwrap();
        self.vcpu_reg_init(vcpu_id)?;
        info!("The tssIntStackStart is {:x}, tssAddr address is {:x}, idt addr is {:x}, gdt addr is {:x}",
             self.tssIntStackStart, self.tssAddr, self.idtAddr, self.gdtAddr);
        info!(
            "[{}] - The tssSegment stack is {:x}",
            vcpu_id,
            self.tssIntStackStart + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE
        );
        Ok(())
    }

    fn run(&self, entry_addr: u64, stack_start_addr: u64, heap_start_addr: u64,
           share_space_addr: u64, vcpu_id: u64, vdso_addr: u64, cpus_total: u64,
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
            rdx: vcpu_id,
            //arg3
            rcx: vdso_addr,
            //arg4
            r8: cpus_total,
            //arg5
            r9: auto_start as u64,
            ..Default::default()
        };

        self.vcpu_fd()
            .set_regs(&regs)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        self.vcpu_run(vcpu_id)
    }
}
