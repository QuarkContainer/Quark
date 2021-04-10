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
use super::super::syscalls::syscalls::*;
use super::super::qlib::linux::time::*;

// copyItimerValIn copies an ItimerVal from the untrusted app range to the
// kernel.  The ItimerVal may be either 32 or 64 bits.
// A NULL address is allowed because because Linux allows
// setitimer(which, NULL, &old_value) which disables the timer.
// There is a KERN_WARN message saying this misfeature will be removed.
// However, that hasn't happened as of 3.19, so we continue to support it.
pub fn CopyItimerValIn(task: &Task, addr: u64) -> Result<ItimerVal> {
    if addr == 0 {
        return Ok(ItimerVal::default())
    }

    let itv : ItimerVal = *task.GetType(addr)?;
    return Ok(itv)
}

// copyItimerValOut copies an ItimerVal to the untrusted app range.
// The ItimerVal may be either 32 or 64 bits.
// A NULL address is allowed, in which case no copy takes place
pub fn CopyItimerValOut(task: &mut Task, addr: u64, itv: &ItimerVal) -> Result<()> {
    if addr == 0 {
        return Ok(())
    }

    *task.GetTypeMut(addr)? = *itv;

    return Ok(())
}

// Getitimer implements linux syscall getitimer(2).
pub fn SysGetitimer(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let timerID = args.arg0 as i32;
    let val = args.arg1 as u64;

    let thread = task.Thread();
    let olditv = thread.Getitimer(timerID)?;

    CopyItimerValOut(task, val, &olditv)?;
    return Ok(0)
}

// Setitimer implements linux syscall setitimer(2).
pub fn SysSetitimer(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let timerID = args.arg0 as i32;
    let newVal = args.arg1 as u64;
    let oldVal = args.arg2 as u64;

    let newitv = CopyItimerValIn(task, newVal)?;

    let thread = task.Thread();
    let olditv = thread.Setitimer(timerID, &newitv)?;

    CopyItimerValOut(task, oldVal, &olditv)?;
    return Ok(0)
}

// Alarm implements linux syscall alarm(2).
pub fn SysAlarm(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let duration = args.arg0 as u32 as i64 * SECOND;

    let olditv = task.Thread().Setitimer(ITIMER_REAL, &ItimerVal{
        Interval: Timeval::default(),
        Value: Timeval::FromNs(duration),
    })?;

    let olddur = olditv.Value.ToDuration();
    let mut secs = olddur / SECOND;
    if secs == 0 && olddur != 0 {
        // We can't return 0 if an alarm was previously scheduled.
        secs = 1;
    }

    return Ok(secs)
}