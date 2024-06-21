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
use super::vcpu::kvm_vcpu::KvmAarch64Reg::{R0, R1, R2, R3, R4, R5};

use crate::{arch::tee::NonConf, qlib::common::Error};

impl NonConf<'_> {
    pub(in crate::arch) fn _set_cpu_registers(&self, vcpu_fd: &VcpuFd) -> Result<(), Error> {
        vcpu_fd.set_one_reg(R0 as u64, self.page_allocator_addr)
            .map_err(|e| Error::SysError(e.errno()))?;
        vcpu_fd.set_one_reg(R1 as u64, self.share_space_table_addr)
            .map_err(|e| Error::SysError(e.errno()))?;

        Ok(())
    }

    pub(in crate::arch) fn _get_hypercall_arguments(&self, vcpu_fd: &VcpuFd, _vcpu_id: usize)
        -> Result<(u64, u64, u64, u64), Error> {
        // reading hypercall parameters from vcpu register file
        // x0 and x1 (w1) are used by the str instruction
        // the 64-bit parameters 1,2,3,4 are passed via
        // x2,x3,x4,x5
        let para1 = vcpu_fd.get_one_reg(R2 as u64)
            .map_err(|e| Error::SysError(e.errno()))?;
        let para2 = vcpu_fd.get_one_reg(R3 as u64)
            .map_err(|e| Error::SysError(e.errno()))?;
        let para3 = vcpu_fd.get_one_reg(R4 as u64)
            .map_err(|e| Error::SysError(e.errno()))?;
        let para4 = vcpu_fd.get_one_reg(R5 as u64)
            .map_err(|e| Error::SysError(e.errno()))?;
        Ok((para1, para2, para3, para4))
    }
}
