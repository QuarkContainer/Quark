// Copyright (c) 2021 QuarkSoft LLC
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


use super::super::kernel::timer::timer::*;
use super::super::kernel::fd_table::*;
use super::super::qlib::linux::time::*;
use super::super::fs::timerfd::*;
use super::super::fs::flags::*;
use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;

// TimerfdCreate implements Linux syscall timerfd_create(2).
pub fn SysTimerfdCreateSysRead(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let clockID = args.arg0 as i32;
    let flags = args.arg1 as i32;

    if flags & !(TFD_CLOEXEC | TFD_NONBLOCK) != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let f = NewTimerfd(task, clockID)?;

    f.SetFlags(task, SettableFileFlags{
        NonBlocking: flags & EFD_NONBLOCK != 0,
        ..Default::default()
    });
    f.flags.lock().0.NonSeekable = true;

    let fd = task.NewFDFrom(0, &f, &FDFlags{
        CloseOnExec: flags & EFD_CLOEXEC != 0,
    })?;

    return Ok(fd as i64)
}

// TimerfdSettime implements Linux syscall timerfd_settime(2).
pub fn SysTimerfdSettime(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let flags = args.arg1 as i32;
    let newValAddr = args.arg2 as u64;
    let oldValAddr = args.arg3 as u64;

    if flags & !TFD_TIMER_ABSTIME != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let file = task.GetFile(fd)?;

    let tf = match file.FileOp.as_any().downcast_ref::<TimerOperations>() {
        Some(tf) => tf,
        None => return Err(Error::SysError(SysErr::EINVAL)),
    };

    let newVal : Itimerspec = *task.GetType(newValAddr)?;
    let clock = tf.Clock();
    let newS = Setting::FromItimerspec(&newVal, flags & TFD_TIMER_ABSTIME != 0, &clock)?;

    let (tm, oldS) = tf.SetTime(&newS);
    if oldValAddr != 0 {
        let oldVal = ItimerspecFromSetting(tm, oldS);
        let v : &mut Itimerspec = task.GetTypeMut(oldValAddr)?;
        *v = oldVal;
    }

    return Ok(0)
}

// TimerfdGettime implements Linux syscall timerfd_gettime(2).
pub fn SysTimerfdGettime(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let curValAddr = args.arg1 as u64;

    let file = task.GetFile(fd)?;

    let tf = match file.FileOp.as_any().downcast_ref::<TimerOperations>() {
        Some(tf) => tf,
        None => return Err(Error::SysError(SysErr::EINVAL)),
    };

    let (tm, s) = tf.GetTime();
    let curVal = ItimerspecFromSetting(tm, s);
    let v :&mut Itimerspec = task.GetTypeMut(curValAddr)?;
    *v = curVal;

    return Ok(0)
}