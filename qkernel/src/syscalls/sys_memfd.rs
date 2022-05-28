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


use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::kernel::fs::file::*;
use crate::syscalls::syscalls::*;
use crate::qlib::kernel::kernel::fd_table::FDFlags;
use crate::task::*;

pub const MEMFD_PREFIX : &str = "memfd:";
pub const MEMFD_MAX_NAME_LEN : usize = NAME_MAX - MEMFD_PREFIX.len();
pub const MEMFD_ALL_FLAGS : u32 = MfdType::MFD_CLOEXEC | MfdType::MFD_ALLOW_SEALING;

// MemfdCreate implements Linux syscall memfd_create(2).
pub fn SysMemfdCreate(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let flags = args.arg1 as u32;

    if flags & !MEMFD_ALL_FLAGS != 0 {
        // Unknown bits in flags.
        return Err(Error::SysError(SysErr::EINTR));
    }

    let (fileName, err) = task.CopyInString(addr, MEMFD_MAX_NAME_LEN);
    match err {
        Err(e) => return Err(e),
        _ => (),
    }

    let file = File::NewMemfdFile(task, &fileName, &task.FileOwner(), flags)?;
    let fd = task.NewFDFrom(
        0,
        &file,
        &FDFlags {
            CloseOnExec: flags & Flags::O_CLOEXEC as u32 != 0,
        },
    )?;
    return Ok(fd as i64)
}