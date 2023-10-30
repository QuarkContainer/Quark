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

use core::mem;

use super::super::arch::__arch::context::MAX_ADDR64;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::Task;
use super::super::vcpu::*;

pub fn IsValidSegmentBase(addr: u64) -> bool {
    return addr < MAX_ADDR64;
}

pub fn SysArchPrctl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let code = args.arg0;
    let addr = args.arg1 as u64;

    if code != PrCtlEnum::ARCH_SET_GS as u64
        && code != PrCtlEnum::ARCH_SET_FS as u64
        && code != PrCtlEnum::ARCH_GET_FS as u64
        && code != PrCtlEnum::ARCH_GET_GS as u64
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let cmdCode: PrCtlEnum = unsafe { mem::transmute(code as u16) };

    match cmdCode {
        PrCtlEnum::ARCH_SET_GS => {
            SetGs(addr);
        }
        PrCtlEnum::ARCH_SET_FS => {
            if !IsValidSegmentBase(addr) {
                return Err(Error::SysError(SysErr::EPERM));
            }
            SetTLS(addr);
            task.context.set_tls(addr);
            //info!("ARCH_SET_FS: the val is {:x}", unsafe {*(addr as * const u64)});
            //info!("after ARCH_SET_FS, the input value is {:x}, the get fs result is {:x}", addr, ReadMsr(MSR::MSR_FS_BASE as u32));
        }
        PrCtlEnum::ARCH_GET_FS => {
            task.CopyOutObj(&GetFs(), addr)?;
        }
        PrCtlEnum::ARCH_GET_GS => {
            task.CopyOutObj(&GetGs(), addr)?;
        }
    }
    return Ok(0);
}
