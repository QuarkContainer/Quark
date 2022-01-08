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


use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::linux::membarrier::*;
use super::super::syscalls::syscalls::*;

// Membarrier implements syscall membarrier(2).
pub fn SysMembarrier(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let cmd = args.arg0 as i32;
    let flags = args.arg1 as u32;

    match cmd {
        MEMBARRIER_CMD_QUERY => {
            if flags != 0 {
                return Err(Error::SysError(SysErr::EINVAL))
            }

            let supportedCommands = 0;
            return Ok(supportedCommands)
        }
        // todo: enable membarrier
        _ => {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    }
}