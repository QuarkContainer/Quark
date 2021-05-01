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


use super::super::task::*;
use super::super::qlib::common::*;
use super::super::fs::file::*;
use super::super::fs::host::hostinodeop::*;
use super::super::Kernel::HostSpace;
use super::super::syscalls::syscalls::*;

// Sync implements linux system call sync(2).
pub fn SysSync(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    HostSpace::SysSync();
    return Ok(0)
}

// Syncfs implements linux system call syncfs(2).
pub fn SyncFs(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;

    let file = task.GetFile(fd)?;
    let inode = file.Dirent.Inode();
    let iops = inode.lock().InodeOp.clone();
    match iops.as_any().downcast_ref::<HostInodeOp>() {
        None => {
            return Ok(0)
        },
        Some(h) => {
            h.SyncFs()?;
            return Ok(0)
        }
    }
}

// Fsync implements linux syscall fsync(2).
pub fn SysFsync(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;

    let file = task.GetFile(fd)?;

    file.Fsync(task, 0, FILE_MAX_OFFSET, SyncType::SyncAll)?;
    return Ok(0)
}

// Fdatasync implements linux syscall fdatasync(2).
//
// At the moment, it just calls Fsync, which is a big hammer, but correct.
pub fn SysDatasync(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;

    let file = task.GetFile(fd)?;

    file.Fsync(task, 0, FILE_MAX_OFFSET, SyncType::SyncData)?;
    return Ok(0)
}