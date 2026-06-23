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

use kvm_ioctls::VcpuFd;
use super::vcpu::kvm_vcpu::{KvmAarch64Reg::{X0, X1, X2, X3, X4, X5}, get_one_reg_u64, set_one_reg_u64};

use crate::{arch::tee::{NonConf, emulcc::EmulCc}, qlib::common::Error};

impl NonConf<'_> {
    pub(in crate::arch) fn _set_cpu_registers(&self, vcpu_fd: &VcpuFd) -> Result<(), Error> {
        set_one_reg_u64(vcpu_fd, X0 as u64, self.page_allocator_addr)?;
        set_one_reg_u64(vcpu_fd, X1 as u64, self.share_space_table_addr)?;

        Ok(())
    }

    #[inline]
    pub(in crate::arch) fn _get_hypercall_arguments(&self, vcpu_fd: &VcpuFd, _vcpu_id: usize)
        -> Result<(u64, u64, u64, u64), Error> {
        // reading hypercall parameters from vcpu register file
        // x0 and x1 (w1) are used by the str instruction
        // the 64-bit parameters 1,2,3,4 are passed via
        // x2,x3,x4,x5
        let para1 = get_one_reg_u64(vcpu_fd, X2 as u64)?;
        let para2 = get_one_reg_u64(vcpu_fd, X3 as u64)?;
        let para3 = get_one_reg_u64(vcpu_fd, X4 as u64)?;
        let para4 = get_one_reg_u64(vcpu_fd, X5 as u64)?;
        Ok((para1, para2, para3, para4))
    }
}

impl EmulCc<'_> {
    pub(in crate::arch) fn _set_cpu_registers(&self, vcpu_fd: &kvm_ioctls::VcpuFd) -> Result<(), Error> {
        //arg0
        set_one_reg_u64(vcpu_fd, X0 as u64, self.page_allocator_addr)?;
        set_one_reg_u64(vcpu_fd, X1 as u64, self.cc_mode as u64)?;
        Ok(())
    }
}
