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

use crate::{qlib::{common::Error, linux_def::MemoryDef}, runc::runtime::vm::{ClearDump, Dump},
    KVMVcpu, QUARK_CONFIG};

pub type Register = u64;

impl KVMVcpu {
    pub fn get_frequency(&self) -> Result<u64, Error> {
        Ok((self.vcpu_fd.get_tsc_khz().map_err(|e| Error::SysError(e.errno()))? as u64) * 1000)
    }

    pub fn interrupt_guest(&self) {
        use::std::os::fd::AsRawFd;
        const KVM_INTERRUPT: u64 = 0x4004ae86;
        let bounce: u32 = 20; //VirtualizationException
        let ret = unsafe {
            libc::ioctl(
                self.vcpu_fd.as_raw_fd(),
                KVM_INTERRUPT,
                &bounce as *const _ as u64,
            )
        };

        assert!(
            ret == 0,
            "InterruptGuest ret is {}/{}/{}",
            ret,
            errno::errno().0,
            self.vcpu_fd.as_raw_fd()
        );
    }

    pub fn dump(&self) -> Result<(), Error> {
        if !Dump(self.id) {
            return Ok(());
        }
        defer!(ClearDump(self.id));
        let regs = self            .vcpu_fd
            .get_regs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let sregs = self            .vcpu_fd
            .get_sregs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        error!("vcpu regs: {:#x?}", regs);
        let ss = sregs.ss.selector as u64;
        let isUser = (ss & 0x3) != 0;
        if isUser {
            error!("vcpu {} is in user mode, skip", self.id);
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
        error!("vcpu {} stack: {}", self.id, frames);
        Ok(())
    }
}
