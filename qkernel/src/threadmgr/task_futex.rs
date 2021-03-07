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



use super::super::qlib::linux::futex::*;
use super::super::kernel::futex::*;
use super::super::task::*;
use super::super::threadmgr::thread::*;

impl Thread {
    pub fn ExitRobustList(&self, task: &Task) {
        let addr = {
            let mut t = self.lock();
            let ret = t.robust_list_head;
            t.robust_list_head = 0;
            ret
        };

        if addr == 0 {
            return;
        }

        let rl : RobustListHead = match task.GetType(addr) {
            Err(_) => return,
            Ok(p) => *p,
        };

        let mut next = rl.List;
        let mut done = 0;
        let mut pendingLockAddr = 0;
        if rl.ListOpPending != 0 {
            pendingLockAddr = rl.ListOpPending + rl.FutexOffset;
        }

        // Wake up normal elements.
        while next != addr {
            // We traverse to the next element of the list before we
            // actually wake anything. This prevents the race where waking
            // this futex causes a modification of the list.
            let thisLockAddr = (next as i64 + rl.FutexOffset as i64) as u64;

            // Try to decode the next element in the list before waking the
            // current futex. But don't check the error until after we've
            // woken the current futex. Linux does it in this order too
            next = match task.GetType(next) {
                Err(_) => {
                    if thisLockAddr != pendingLockAddr {
                        self.WakeRobustListOne(task, thisLockAddr)
                    };

                    // ignore error
                    return;
                }
                Ok(next) => {
                    if thisLockAddr != pendingLockAddr {
                        self.WakeRobustListOne(task, thisLockAddr)
                    };
                    *next
                }
            };

            // This is a user structure, so it could be a massive list, or
            // even contain a loop if they are trying to mess with us. We
            // cap traversal to prevent that.
            done += 1;
            if done >= ROBUST_LIST_LIMIT {
                break;
            }
        }

        // Is there a pending entry to wake?
        if pendingLockAddr != 0 {
            self.WakeRobustListOne(task, pendingLockAddr)
        }
    }

    // wakeRobustListOne wakes a single futex from the robust list.
    pub fn WakeRobustListOne(&self, task: &Task, addr: u64)  {
        // Bit 0 in address signals PI futex.
        let pi = addr & 1 == 1;
        let addr = addr & !1;

        // Load the futex.
        let mut f = match task.LoadU32(addr) {
            // if Can't read this single value? Ignore the problem.
            // We can wake the other futexes in the list.
            Err(_) => return,
            Ok(f) => f,
        };

        let tid = self.ThreadID() as u32;

        loop {
            // Is this held by someone else?
            if f & FUTEX_TID_MASK != tid {
                return
            }

            // This thread is dying and it's holding this futex. We need to
            // set the owner died bit and wake up any waiters.
            let newF = (f & FUTEX_WAITERS) | FUTEX_OWNER_DIED;
            let curF = match task.CompareAndSwapU32(addr, f, newF) {
                // if Can't read this single value? Ignore the problem.
                // We can wake the other futexes in the list.
                Err(_) => return,
                Ok(v) => v,
            };

            if curF != f {
                f = curF;
                continue;
            }

            // Wake waiters if there are any.
            if f & FUTEX_WAITERS != 0 {
                let private = f & FUTEX_WAITERS != 0;
                if pi {
                    task.futexMgr.UnlockPI(task, addr, tid, private).ok();
                    return
                }

                task.futexMgr.Wake(task, addr, private, FUTEX_BITSET_MATCH_ANY, 1).ok();
            }

            return;
        }
    }
}