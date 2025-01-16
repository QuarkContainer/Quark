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

use kvm_ioctls:: VcpuFd;
use crate::arch::tee::emulcc::EmulCc;
use crate::arch::tee::NonConf;
use crate::qlib::common::Error;

impl NonConf<'_> {
    pub(in crate::arch) fn _set_cpu_registers(&self, vcpu_fd: &VcpuFd) -> Result<(), Error> {
        let mut cpu_regs = vcpu_fd.get_regs().unwrap();
        //arg0
        cpu_regs.rdi = self.page_allocator_addr;
        //arg1
        cpu_regs.rsi = self.share_space_table_addr;
        vcpu_fd.set_regs(&cpu_regs)
            .expect("vCPU - failed to set up cpu registers.");
        Ok(())
    }

    #[inline]
    pub(in crate::arch) fn _get_hypercall_arguments(&self, vcpu_fd: &VcpuFd, _vcpu_id: usize)
        -> Result<(u64, u64, u64, u64), Error> {
        let vcpu_regs = vcpu_fd.get_regs().map_err(|e|
            Error::IOError(format!("Failed to get vcpu regs - error:{}", e)))?;
        let _arg0 = vcpu_regs.rsi;
        let _arg1 = vcpu_regs.rcx;
        let _arg2 = vcpu_regs.rdi;
        let _arg3 = vcpu_regs.r10;
        Ok((_arg0, _arg1, _arg2, _arg3))
    }
}

impl EmulCc<'_> {
    pub(in crate::arch) fn _set_cpu_registers(&self, vcpu_fd: &kvm_ioctls::VcpuFd) -> Result<(), Error> {
        let mut cpu_regs = vcpu_fd.get_regs().unwrap();
        //arg0
        cpu_regs.rdi = self.page_allocator_addr;
        //arg1
        cpu_regs.rsi = self.cc_mode as u64;
        vcpu_fd.set_regs(&cpu_regs)
            .expect("vCPU - failed to set up cpu registers.");
        Ok(())
    }
}
