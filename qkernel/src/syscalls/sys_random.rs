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

    let buf = DataBuff::New(length as usize);

    let ret = HostSpace::GetRandom(buf.Ptr(), buf.Len() as u64, flags as u32);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    task.CopyOutSlice(&buf.buf[0..ret as usize], addr, length as usize)?;

    return Ok(ret as i64)
}
