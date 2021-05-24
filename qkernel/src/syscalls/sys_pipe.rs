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


use super::super::kernel::pipe::pipe::*;
use super::super::fs::flags::*;
use super::super::kernel::fd_table::*;
use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;

pub fn Pipe2(task: &mut Task, addr: u64, flags: i32) -> Result<i64> {
    if flags & !(Flags::O_NONBLOCK | Flags::O_CLOEXEC) != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let (r, w) = NewConnectedPipe(task, DEFAULT_PIPE_SIZE, MemoryDef::PAGE_SIZE as usize);

    r.SetFlags(task, FileFlags::FromFlags(flags as u32).SettableFileFlags());
    r.flags.lock().0.NonSeekable = true;
    w.SetFlags(task, FileFlags::FromFlags(flags as u32).SettableFileFlags());
    w.flags.lock().0.NonSeekable = true;

    let fds : &mut [i32; 2] = task.GetTypeMut(addr)?;

    let rfd = task.NewFDFrom(0, &r, &FDFlags {
        CloseOnExec: flags & Flags::O_CLOEXEC != 0,
    })?;

    let wfd = task.NewFDFrom(0, &w, &FDFlags {
        CloseOnExec: flags & Flags::O_CLOEXEC != 0,
    })?;

    fds[0] = rfd;
    fds[1] = wfd;

    info!("Pipe2 the fds is {:?}", &fds);

    return Ok(0)
}

pub fn SysPipe(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    return Pipe2(task, addr, 0);
}

pub fn SysPipe2(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let flags = args.arg1 as i32;

    return Pipe2(task, addr, flags);
}
