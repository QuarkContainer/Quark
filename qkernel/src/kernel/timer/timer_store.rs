// Copyright (c) 2021 Quark Container Authors.
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

use alloc::collections::btree_map::BTreeMap;
use alloc::collections::btree_set::BTreeSet;
use core::cmp::Ordering;
use core::ops::Deref;
use spin::Mutex;

use super::super::super::IOURING;
use super::*;

#[derive(Debug, Copy, Clone)]
pub struct TimerUnit {
    pub expire: i64,
    pub timerId: u64,
    pub seqNo: u64,
}

impl TimerUnit {
    pub fn New(taskId: u64, seqNo: u64, expire: i64) -> Self {
        return Self {
            timerId: taskId,
            seqNo: seqNo,
            expire: expire,
        }
    }

    pub fn Fire(&self) {
        super::FireTimer(self.timerId, self.seqNo);
    }
}

impl Ord for TimerUnit {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.expire != other.expire {
            return self.expire.cmp(&other.expire)
        } else {
            return self.timerId.cmp(&other.timerId)
        }
    }
}

impl PartialOrd for TimerUnit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for TimerUnit {}

impl PartialEq for TimerUnit {
    fn eq(&self, other: &Self) -> bool {
        self.timerId == other.timerId && self.seqNo == other.seqNo
    }
}

#[derive(Default)]
pub struct TimerStore(Mutex<TimerStoreIntern>);

impl Deref for TimerStore {
    type Target = Mutex<TimerStoreIntern>;

    fn deref(&self) -> &Mutex<TimerStoreIntern> {
        &self.0
    }
}

impl TimerStore {
    // the timeout need to process a timer, <PROCESS_TIME means the timer will be triggered immediatelyfa
    pub const PROCESS_TIME : i64 = 2000;

    pub fn Trigger(&self, expire: i64) {
        let mut now;
        loop {
            now = MONOTONIC_CLOCK.Now().0 + Self::PROCESS_TIME;
            let tu = self.lock().GetFirst(now);
            match tu {
                Some(timer) => {
                    timer.Fire();
                }
                None => break,
            }
        }

        let mut tm = self.lock();

        if expire != tm.nextExpire // not triggered by the the timer's timeout
            && now > tm.nextExpire { // the nextExpire has passed
            tm.RemoveUringTimer();
        }

        let firstExpire = match tm.timerSeq.first() {
            None => return,
            Some(t) => t.expire,
        };

         if firstExpire < tm.nextExpire || tm.nextExpire == 0 {
            tm.RemoveUringTimer();
            tm.SetUringTimer(firstExpire);
        }
    }

    pub fn ResetTimer(&mut self, timerId: u64, seqNo: u64, timeout: i64) {
        self.lock().ResetTimer(timerId, seqNo, timeout);
        self.Trigger(0);
    }


    pub fn CancelTimer(&self, timerId: u64) {
        self.lock().RemoveTimer(timerId);
        self.Trigger(0);
    }
}

#[derive(Default)]
pub struct TimerStoreIntern {
    pub timerSeq: BTreeSet<TimerUnit>, // order by expire time
    pub timers: BTreeMap<u64, TimerUnit>, // timerid -> TimerUnit
    pub nextExpire: i64,
    pub uringId: u64,
}

impl TimerStoreIntern {
    // return: existing or not
    pub fn RemoveTimer(&mut self, timerId: u64) -> bool {
        let tu = match self.timers.remove(&timerId) {
            None => {
                return false
            },
            Some(tu) => tu,
        };

        self.timerSeq.remove(&tu);
        return true;
    }


    pub fn ResetTimer(&mut self, timerId: u64, seqNo: u64, timeout: i64) {
        self.RemoveTimer(timerId);

        let current = MONOTONIC_CLOCK.Now().0;
        let expire = current + timeout;

        let tu = TimerUnit {
            expire: expire,
            timerId: timerId,
            seqNo: seqNo,
        };

        self.timerSeq.insert(tu.clone());
        self.timers.insert(timerId, tu);
    }

    pub fn RemoveUringTimer(&mut self) {
        if self.nextExpire != 0 {
            IOURING.AsyncTimerRemove(self.uringId);
            self.nextExpire = 0;
        }
    }

    pub fn SetUringTimer(&mut self, expire: i64) {
        let now = MONOTONIC_CLOCK.Now().0;
        let expire = if expire < now {
            now
        } else {
            expire
        };
        assert!(self.nextExpire == 0);
        self.nextExpire = expire;
        self.uringId = IOURING.Timeout(expire, expire - now) as u64;
    }

    pub fn GetFirst(&mut self, now: i64) -> Option<TimerUnit> {
        let tu = match self.timerSeq.first() {
            None => return None,
            Some(t) => *t,
        };

        if tu.expire > now {
            return None;
        }

        let timerId = tu.timerId;
        self.RemoveTimer(timerId);

        return Some(tu)
    }
}