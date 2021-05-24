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

use alloc::vec::Vec;

use super::super::qlib::common::*;
use super::super::Kernel::HostSpace;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::Task;

pub fn SysGetRandom(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0;
    let mut length = args.arg1 as u32;
    let flags = args.arg2 as i32;

    // Flags are checked for validity but otherwise ignored. See above.
    if flags & !(_GRND_NONBLOCK | _GRND_RANDOM) != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if length > core::i32::MAX as u32 {
        length = core::i32::MAX as u32;
    }

    let mut iovs = Vec::new();
    task.V2P(addr, length as u64, &mut iovs, true)?;

    for iov in iovs {
        let ret = HostSpace::GetRandom(iov.start, iov.len as u64, flags as u32);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32))
        }
    }

    return Ok(length as i64)
}
