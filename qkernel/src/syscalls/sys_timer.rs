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
use super::super::qlib::linux::time::*;
use super::super::syscalls::syscalls::*;
use super::super::SignalDef::*;
use super::sys_time::*;

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

    let itv : ItimerVal = task.CopyInObj(addr)?;
    return Ok(itv)
}

// copyItimerValOut copies an ItimerVal to the untrusted app range.
// The ItimerVal may be either 32 or 64 bits.
// A NULL address is allowed, in which case no copy takes place
pub fn CopyItimerValOut(task: &mut Task, addr: u64, itv: &ItimerVal) -> Result<()> {
    if addr == 0 {
        return Ok(())
    }

    //*task.GetTypeMut(addr)? = *itv;
    task.CopyOutObj(itv, addr)?;
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

// TimerCreate implements linux syscall timer_create(2).
pub fn SysTimerCreate(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let clockID = args.arg0 as i32;
    let sevp = args.arg1 as u64;
    let timerIDp = args.arg2 as u64;

    let c = GetClock(task, clockID)?;

    let mut sev = Sigevent::default();
    if sevp != 0 {
        sev = task.CopyInObj(sevp)?;
    }

    let id = task.Thread().IntervalTimerCreate(&c, &mut sev)?;

    //let timerID = task.GetTypeMut(timerIDp)?;
    //*timerID = id;

    task.CopyOutObj(&id, timerIDp)?;
    return Ok(0)
}

// TimerSettime implements linux syscall timer_settime(2).
pub fn SysTimerSettime(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let timerID = args.arg0 as i32;
    let flags = args.arg1 as i32;
    let newValAddr = args.arg2 as u64;
    let oldValAddr = args.arg3 as u64;

    let newVal : Itimerspec = task.CopyInObj(newValAddr)?;

    let oldVal = task.Thread().IntervalTimerSettime(timerID, &newVal, flags & TIMER_ABSTIME != 0)?;
    if oldValAddr != 0 {
        //*task.GetTypeMut(oldValAddr)? = oldVal;
        task.CopyOutObj(&oldVal, oldValAddr)?;
    }

    return Ok(0)
}

// TimerGettime implements linux syscall timer_gettime(2).
pub fn SysTimerGettime(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let timerID = args.arg0 as i32;
    let curValAddr = args.arg1 as u64;

    let curVal = task.Thread().IntervalTimerGettime(timerID)?;
    //*task.GetTypeMut(curValAddr)? = curVal;
    task.CopyOutObj(&curVal, curValAddr)?;
    return Ok(0)
}

// TimerGetoverrun implements linux syscall timer_getoverrun(2).
pub fn SysTimerGetoverrun(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let timerID = args.arg0 as i32;

    let o = task.Thread().IntervalTimerGetoverrun(timerID)?;

    return Ok(o as i64)
}

// TimerDelete implements linux syscall timer_delete(2).
pub fn SysTimerDelete(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let timerID = args.arg0 as i32;

    task.Thread().IntervalTimerDelete(timerID)?;
    return Ok(0)
}