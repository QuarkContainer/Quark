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

use alloc::boxed::Box;

use crate::GUEST_HOST_SHARED_ALLOCATOR;
use super::super::kernel::timer::timer::*;
use super::super::kernel::timer::*;
use super::super::qlib::common::*;
use super::super::qlib::linux::time::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::super::taskMgr::*;
use super::super::threadmgr::task_syscall::*;
use super::super::threadmgr::thread::*;
use super::super::Kernel::HostSpace;
use super::sys_poll::TIMEOUT_PROCESS_TIME;

// The most significant 29 bits hold either a pid or a file descriptor.
pub fn PidOfClockID(c: i32) -> i32 {
    return !(c >> 3);
}

// whichCPUClock returns one of CPUCLOCK_PERF, CPUCLOCK_VIRT, CPUCLOCK_SCHED or
// CLOCK_FD.
pub fn WhichCPUClock(c: i32) -> i32 {
    return c & CPUCLOCK_CLOCK_MASK;
}

// isCPUClockPerThread returns true if the CPUCLOCK_PERTHREAD bit is set in the
// clock id.
pub fn IsCPUClockPerThread(c: i32) -> bool {
    return c & CPUCLOCK_PERTHREAD_MASK != 0;
}

// isValidCPUClock returns checks that the cpu clock id is valid.
pub fn IsValidCPUClock(c: i32) -> bool {
    // Bits 0, 1, and 2 cannot all be set.
    if c & 7 == 7 {
        return false;
    }

    if WhichCPUClock(c) >= CPUCLOCK_MAX {
        return false;
    }

    return true;
}

pub fn TargetThread(task: &Task, c: i32) -> Option<Thread> {
    let pid = PidOfClockID(c);
    if c == 0 {
        return Some(task.Thread());
    }

    return task.Thread().PIDNamespace().TaskWithID(pid);
}

pub fn GetClock(task: &Task, clockId: i32) -> Result<Clock> {
    if clockId < 0 {
        if !IsValidCPUClock(clockId) {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let targetThread = match TargetThread(task, clockId) {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(t) => t,
        };

        if IsCPUClockPerThread(clockId) {
            let target = targetThread;
            match WhichCPUClock(clockId) {
                CPUCLOCK_VIRT => return Ok(target.UserCPUClock()),
                CPUCLOCK_PROF | CPUCLOCK_SCHED => {
                    // CPUCLOCK_SCHED is approximated by CPUCLOCK_PROF.
                    return Ok(target.CPUClock());
                }
                _ => return Err(Error::SysError(SysErr::EINVAL)),
            }
        } else {
            let target = targetThread.ThreadGroup();
            match WhichCPUClock(clockId) {
                CPUCLOCK_VIRT => return Ok(target.UserCPUClock()),
                CPUCLOCK_PROF | CPUCLOCK_SCHED => {
                    // CPUCLOCK_SCHED is approximated by CPUCLOCK_PROF.
                    return Ok(target.CPUClock());
                }
                _ => return Err(Error::SysError(SysErr::EINVAL)),
            }
        }
    }

    match clockId {
        CLOCK_REALTIME | CLOCK_REALTIME_COARSE => return Ok(REALTIME_CLOCK.clone()),

        CLOCK_MONOTONIC | CLOCK_MONOTONIC_COARSE | CLOCK_MONOTONIC_RAW | CLOCK_BOOTTIME => {
            return Ok(MONOTONIC_CLOCK.clone())
        }

        CLOCK_PROCESS_CPUTIME_ID => return Ok(task.Thread().ThreadGroup().CPUClock()),
        CLOCK_THREAD_CPUTIME_ID => return Ok(task.Thread().CPUClock()),
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }
}

pub fn SysClockGetRes(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let clockID = args.arg0 as i32;
    let addr = args.arg1 as u64;

    GetClock(task, clockID)?;

    if addr == 0 {
        return Ok(0);
    }

    /*let ts : &mut Timespec = task.GetTypeMut(addr)?;
    *ts = Timespec {
        tv_sec: 0,
        tv_nsec: 1,
    };*/

    let ts = Timespec {
        tv_sec: 0,
        tv_nsec: 1,
    };
    task.CopyOutObj(&ts, addr)?;

    return Ok(0);
}

pub fn SysClockGetTime(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let clockID = args.arg0 as i32;
    let addr = args.arg1 as u64;

    //let clockID = 1;

    let clock = GetClock(task, clockID)?;
    //let ts : &mut Timespec = task.GetTypeMut(addr)?;
    //*ts = clock.Now().Timespec();

    let ts = clock.Now().Timespec();
    task.CopyOutObj(&ts, addr)?;
    //info!("SysClockGetTime: output is {:?}", ts);

    return Ok(0);
}

pub fn SysClockSettime(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::EPERM));
}

pub fn SysTime(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    let now = REALTIME_CLOCK.Now().0 / 1_000_000_000;
    if addr == 0 {
        return Ok(now);
    }

    //let ptr : &mut i64 = task.GetTypeMut(addr)?;
    //*ptr = now;

    task.CopyOutObj(&now, addr)?;

    return Ok(0);
}

pub fn SysNanoSleep(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let req = args.arg0 as u64;
    let rem = args.arg1 as u64;

    let ts: Timespec = task.CopyInObj(req)?;

    if !ts.IsValid() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let dur = ts.ToNs()?;

    Yield();
    if dur < TIMEOUT_PROCESS_TIME {
        return Ok(0);
    }

    let clock = GetClock(task, CLOCK_MONOTONIC)?;

    let now = clock.Now();
    let end = now.0 + dur;
    return NansleepUntil(task, clock, end, rem, true);
}

// SysClockNanosleep implements linux syscall clock_nanosleep(2).
pub fn SysClockNanosleep(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let clockID = args.arg0 as i32;
    let flags = args.arg1 as i32;
    let addr = args.arg2 as u64;
    let rem = args.arg3 as u64;

    let ts: Timespec = task.CopyInObj(addr)?;

    if !ts.IsValid() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let input = ts.ToNs()?;

    // Only allow clock constants also allowed by Linux.
    if clockID >= 0 {
        if clockID != CLOCK_REALTIME
            && clockID != CLOCK_MONOTONIC
            && clockID != CLOCK_PROCESS_CPUTIME_ID
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }

    let clock = GetClock(task, clockID)?;

    let end;
    if flags & TIMER_ABSTIME != 0 {
        end = input;
    } else {
        let now = clock.Now();
        end = now.0 + input;
    }

    return NansleepUntil(task, clock, end, rem, flags & TIMER_ABSTIME == 0);
}

pub fn NansleepUntil(
    task: &mut Task,
    clock: Clock,
    end: i64,
    rem: u64,
    needRestartBlock: bool,
) -> Result<i64> {
    let timer = task.blocker.GetTimerWithClock(&clock);

    let now = clock.Now();
    let dur = end - now.0;

    if dur < TIMEOUT_PROCESS_TIME {
        Yield();
        return Ok(0);
    }

    let (remaining, res) = task.blocker.BlockWithTimeout(timer, false, Some(dur));

    if rem != 0 && remaining != 0 {
        let timeleft = Timespec::FromNs(remaining);
        //*task.GetTypeMut(rem)? = timeleft;
        task.CopyOutObj(&timeleft, rem)?;
    }

    match res {
        Err(Error::ErrInterrupted) => {
            if !needRestartBlock {
                return Err(Error::SysError(SysErr::ERESTART));
            }

            let b = Box::new(NanosleepRestartBlock {
                clock: clock,
                end: end,
                rem: rem,
            });
            task.SetSyscallRestartBlock(b);
            return Err(Error::SysError(SysErr::ERESTART_RESTARTBLOCK));
        }
        Err(Error::SysError(SysErr::ETIMEDOUT)) => return Ok(0),
        Err(e) => return Err(e),
        Ok(()) => {
            panic!("NansleepFor:: impossible to get Ok result")
        }
    }
}

pub struct NanosleepRestartBlock {
    pub clock: Clock,
    pub end: i64,
    pub rem: u64,
}

impl SyscallRestartBlock for NanosleepRestartBlock {
    fn Restart(&self, task: &mut Task) -> Result<i64> {
        return NansleepUntil(task, self.clock.clone(), self.end, self.rem, true);
    }
}

pub fn SysGettimeofday(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tvAddr = args.arg0 as u64;
    let tzAddr = args.arg1 as u64;

    let mut timeV = Box::new_in(Timeval::default(), GUEST_HOST_SHARED_ALLOCATOR);
    let mut timezone = Box::new_in([0u32; 2], GUEST_HOST_SHARED_ALLOCATOR);

    let ret = HostSpace::GetTimeOfDay(
        &mut *timeV as *mut _ as u64,
        &mut timezone[0] as *mut _ as u64,
    );
    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }

    if tvAddr != 0 {
        //let tv : &mut Timeval = task.GetTypeMut(tvAddr)?;
        //*tv = timeV;

        task.CopyOutObj(&*timeV, tvAddr)?;
    }

    if tzAddr != 0 {
        //let tz : &mut [u32; 2] = task.GetTypeMut(tzAddr)?;
        //*tz = timezone;

        task.CopyOutObj(&*timezone, tzAddr)?;
    }

    return Ok(0);
}
