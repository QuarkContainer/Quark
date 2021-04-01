// Copyright (c) 2021 Quark Container Authors
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

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::vcpu::*;
use super::super::task::Task;
use super::super::arch::x86_64::context::*;

pub fn IsValidSegmentBase(addr: u64) -> bool {
    return addr < MAX_ADDR64
}

pub fn SysArchPrctl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let code = args.arg0;
    let addr = args.arg1 as u64;

    if code != 0x1001 && code != 0x1002 && code != 0x1003 && code != 0x1004 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let cmdCode: PrCtlEnum = unsafe { mem::transmute(code as u16) };

    match cmdCode {
        PrCtlEnum::ARCH_SET_GS => {
            //WriteMsr(MSR::MSR_KERNEL_GS_BASE as u32, addr);
            SetGs(addr);
        }
        PrCtlEnum::ARCH_SET_FS => {
            if !IsValidSegmentBase(addr) {
                return Err(Error::SysError(SysErr::EPERM));
            }
            SetFs(addr);
            task.context.fs = addr;
            //info!("ARCH_SET_FS: the val is {:x}", unsafe {*(addr as * const u64)});
            //info!("after ARCH_SET_FS, the input value is {:x}, the get fs result is {:x}", addr, ReadMsr(MSR::MSR_FS_BASE as u32));
        }
        PrCtlEnum::ARCH_GET_FS => {
            *task.GetTypeMut::<u64>(addr)? = GetFs();
        }
        PrCtlEnum::ARCH_GET_GS => {
            *task.GetTypeMut::<u64>(addr)? = GetGs();
            //unsafe {*(addr as *mut u64) = ReadMsr(MSR::MSR_KERNEL_GS_BASE as u32)}
        }
    }

    return Ok(0);
}