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


use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;
use super::super::fs::flags::*;
use super::super::kernel::eventfd::*;
use super::super::kernel::fd_table::*;
use super::super::syscalls::syscalls::*;

pub fn Eventfd2(task: &mut Task, initVal: i32, flags: i32) -> Result<i64> {
    let allOps = EFD_SEMAPHORE | EFD_NONBLOCK | EFD_CLOEXEC;

    if flags & !allOps != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let event = NewEventfd(task, initVal as u64, flags & EFD_SEMAPHORE != 0);
    event.SetFlags(task, SettableFileFlags{
        NonBlocking: flags & EFD_NONBLOCK != 0,
        ..Default::default()
    });
    event.flags.lock().0.NonSeekable = true;

    let fd = task.NewFDFrom(0, &event, &FDFlags{
        CloseOnExec: flags & EFD_CLOEXEC != 0,
    })?;

    return Ok(fd as i64)
}

pub fn SysEventfd2(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let initVal = args.arg0 as i32;
    let flags = args.arg1 as i32;

    return Eventfd2(task, initVal, flags)
}

pub fn SysEventfd(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let initVal = args.arg0 as i32;

    return Eventfd2(task, initVal, 0)
}