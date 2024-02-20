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
use super::super::qlib::kernel::kernel::kernel::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

pub const SYSLOG_ACTION_READ_ALL: i32 = 3;
pub const SYSLOG_ACTION_SIZE_BUFFER: i32 = 10;

// logBufLen is the default syslog buffer size on Linux.
const LOG_BUF_LEN: usize = 1 << 17; //128 KB

// Syslog implements part of Linux syscall syslog.
//
// Only the unpriviledged commands are implemented, allowing applications to
// read a dummy dmesg.
pub fn SysSysLog(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let cmd = args.arg0 as i32;
    let buf = args.arg1 as u64;
    let size = args.arg2 as i32;

    match cmd {
        SYSLOG_ACTION_READ_ALL => {
            if size < 0 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut size = size as usize;
            if size > LOG_BUF_LEN {
                size = LOG_BUF_LEN;
            }

            let log = GetKernel().Syslog().Log();
            if size > log.len() {
                size = log.len();
            }

            task.CopyOutSlice(&log[0..size], buf, size)?;
            return Ok(size as _);
        }
        SYSLOG_ACTION_SIZE_BUFFER => return Ok(LOG_BUF_LEN as _),
        _ => {
            return Err(Error::SysError(SysErr::ENOSYS));
        }
    }
}
