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

use alloc::boxed::Box;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;
use super::super::qlib::linux::time::*;
use super::super::qlib::linux::futex::*;
use super::super::kernel::time::*;
use super::super::threadmgr::task_syscall::*;
use super::super::syscalls::syscalls::*;

// futexWaitRestartBlock encapsulates the state required to restart futex(2)
// via restart_syscall(2).
pub struct FutexWaitRestartBlock {
    pub dur: Duration,
    pub addr: u64,
    pub private: bool,
    pub val: u32,
    pub mask: u32,
}

impl SyscallRestartBlock for FutexWaitRestartBlock {
    fn Restart(&self, task: &mut Task) -> Result<i64> {
        FutexWaitDuration(task, Some(self.dur), self.addr, self.private, self.val, self.mask)
    }
}

// futexWaitAbsolute performs a FUTEX_WAIT_BITSET, blocking until the wait is
// complete.
//
// The wait blocks forever if forever is true, otherwise it blocks until ts.
//
// If blocking is interrupted, the syscall is restarted with the original
// arguments.
fn FutexWaitAbsolute(task: &mut Task, realtime: bool, ts: Option<Timespec>, addr: u64, private: bool, val: u32, mask: u32) -> Result<i64> {
    let waitEntry = task.blocker.generalEntry.clone();
    task.futexMgr.WaitPrepare(&waitEntry, task, addr, private, val, mask)?;

    let res = match ts {
        None => task.blocker.BlockWithRealTimer(true, None),
        Some(ts) => {
            let ns = ts.ToDuration()?;
            if realtime {
                task.blocker.BlockWithRealTimer(true, Some(Time(ns)))
            } else {
                task.blocker.BlockWithMonoTimer(true, Some(Time(ns)))
            }
        }
    };

    task.futexMgr.WaitComplete(&waitEntry);
    match res {
        Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
        Err(e) => return Err(e),
        Ok(()) => return Ok(0),
    }
}

// futexWaitDuration performs a FUTEX_WAIT, blocking until the wait is
// complete.
//
// The wait blocks forever if forever is true, otherwise is blocks for
// duration.
//
// If blocking is interrupted, forever determines how to restart the
// syscall. If forever is true, the syscall is restarted with the original
// arguments. If forever is false, duration is a relative timeout and the
// syscall is restarted with the remaining timeout.
fn FutexWaitDuration(task: &mut Task, dur: Option<Duration>, addr: u64, private: bool, val: u32, mask: u32) -> Result<i64> {
    let waitEntry = task.blocker.generalEntry.clone();
    task.futexMgr.WaitPrepare(&waitEntry, task, addr, private, val, mask)?;

    let (remain, res) = task.blocker.BlockWithMonoTimeout(true, dur);
    task.futexMgr.WaitComplete(&waitEntry);
    match res {
        Ok(_) => return Ok(0),
        Err(Error::ErrInterrupted) => (),
        // The wait was unsuccessful for some reason other than interruption. Simply
        // forward the error.
        Err(e) => return Err(e),
    };

    // The wait was interrupted and we need to restart. Decide how.

    // The wait duration was absolute, restart with the original arguments.
    if dur.is_none() {
        //wait forever
        return Err(Error::SysError(SysErr::ERESTARTSYS))
    }

    let b = Box::new(FutexWaitRestartBlock {
        dur: remain,
        addr: addr,
        private: private,
        val: val,
        mask: mask
    });

    task.SetSyscallRestartBlock(b);

    return Err(Error::SysError(SysErr::ERESTART_RESTARTBLOCK));
}

fn FutexLockPI(task: &mut Task, ts: Option<Timespec>, addr: u64, private: bool) -> Result<()> {
    let waitEntry = task.blocker.generalEntry.clone();
    let tid = task.Thread().ThreadID();
    let locked = task.futexMgr.LockPI(&waitEntry, task, addr, tid as u32, private, false)?;

    if locked {
        return Ok(())
    }

    let res = match ts {
        None => task.blocker.BlockWithRealTimer(true, None),
        Some(ts) => {
            let ns = ts.ToDuration()?;
            task.blocker.BlockWithRealTimer(true, Some(Time(ns)))
        }
    };

    task.futexMgr.WaitComplete(&waitEntry);
    match res {
        Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
        Err(e) => return Err(e),
        Ok(()) => return Ok(()),
    }
}

fn TryLockPid(task: &mut Task, addr: u64, private: bool) -> Result<()> {
    let waitEntry = task.blocker.generalEntry.clone();
    let tid = task.Thread().ThreadID();
    let locked = task.futexMgr.LockPI(&waitEntry, task, addr, tid as u32, private, false)?;
    if !locked {
        task.futexMgr.WaitComplete(&waitEntry);
        return Err(Error::SysError(SysErr::EWOULDBLOCK));
    }

    return Ok(())
}

// Futex implements linux syscall futex(2).
// It provides a method for a program to wait for a value at a given address to
// change, and a method to wake up anyone waiting on a particular address.
pub fn SysFutex(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0;
    let futexOp = args.arg1 as i32;
    let val = args.arg2 as i32;
    let nreq = args.arg3 as i32;
    let timeout = args.arg3;
    let naddr = args.arg4;
    let val3 = args.arg5 as i32;

    let cmd = futexOp & !(FUTEX_PRIVATE_FLAG | FUTEX_CLOCK_REALTIME);
    let private = (futexOp & FUTEX_PRIVATE_FLAG) != 0;
    let realtime = (futexOp & FUTEX_CLOCK_REALTIME) != 0;
    let mut mask = val3 as u32;

    match cmd {
        FUTEX_WAIT | FUTEX_WAIT_BITSET => {
            // WAIT{_BITSET} wait forever if the timeout isn't passed.
            let forever = timeout == 0;
            let timespec = if !forever {
                Some(*task.GetType::<Timespec>(timeout)?)
            } else {
                None
            };

            match cmd {
                FUTEX_WAIT => {
                    //info!("FUTEX_WAIT...");
                    // WAIT uses a relative timeout.
                    let mask = !0;
                    let timeoutDur = if forever {
                        None
                    } else {
                        Some(timespec.unwrap().ToDuration()?)
                    };
                    return FutexWaitDuration(task, timeoutDur, addr, private, val as u32, mask);
                }
                FUTEX_WAIT_BITSET => {
                    //info!("FUTEX_WAIT_BITSET...");
                    // WAIT_BITSET uses an absolute timeout which is either
                    // CLOCK_MONOTONIC or CLOCK_REALTIME.
                    if mask == 0 {
                        return Err(Error::SysError(SysErr::EINVAL))
                    }

                    return FutexWaitAbsolute(task, realtime, timespec, addr, private, val as u32, mask);
                }
                _ => panic!("not reachable")
            }
        }
        FUTEX_WAKE | FUTEX_WAKE_BITSET => {
            if cmd == FUTEX_WAKE {
                //info!("FUTEX_WAKE...");
                mask = !0;
            } else {
                //info!("FUTEX_WAKE_BITSET...");
                if mask == 0 {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }

            // linux awake 1 waiter when val is 0
            let n = if val == 0 {
                1
            } else {
                val
            };

            let res = task.futexMgr.Wake(task, addr, private, mask, n)?;
            return Ok(res as i64);
        }
        FUTEX_REQUEUE => {
            //info!("FUTEX_REQUEUE...");
            let n = task.futexMgr.Requeue(task, addr, naddr, private, val, nreq)?;
            return Ok(n as i64);
        }
        FUTEX_CMP_REQUEUE => {
            //info!("FUTEX_CMP_REQUEUE...");
            // 'val3' contains the value to be checked at 'addr' and
            // 'val' is the number of waiters that should be woken up.
            let nval = val3 as u32;
            let n = task.futexMgr.RequeueCmp(task, addr, naddr, private, nval, val, nreq)?;
            return Ok(n as i64);
        }
        FUTEX_WAKE_OP => {
            //info!("FUTEX_WAKE_OP...");
            let op = val3 as u32;
            let n = task.futexMgr.WakeOp(task, addr, naddr, private, val, nreq, op)?;
            return Ok(n as i64);
        }
        FUTEX_LOCK_PI => {
            //info!("FUTEX_LOCK_PI...");
            let forever = timeout == 0;
            let timespec = if forever {
                None
            } else {
                Some(*task.GetType::<Timespec>(timeout)?)
            };

            FutexLockPI(task, timespec, addr, private)?;
            return Ok(0)
        }
        FUTEX_TRYLOCK_PI => {
            //info!("FUTEX_TRYLOCK_PI...");
            TryLockPid(task, addr, private)?;
            return Ok(0)
        }
        FUTEX_UNLOCK_PI => {
            //info!("FUTEX_UNLOCK_PI...");
            let tid = task.Thread().ThreadID();
            task.futexMgr.UnlockPI(task, addr, tid as u32, private)?;
            return Ok(0)
        }
        //FUTEX_WAIT_REQUEUE_PI | FUTEX_CMP_REQUEUE_PI
        _ => {
            return Err(Error::SysError(SysErr::ENOSYS))
        }
    }
}