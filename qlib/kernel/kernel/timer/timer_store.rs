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

use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Bound::Included;
use core::ops::Deref;

use super::super::super::IOURING;
use super::timer::*;
use super::*;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::GuestHostSharedAllocator;

#[derive(Debug, Copy, Clone)]
pub struct TimerUnit {
    pub timerId: u64,
    pub expire: i64,
}

impl TimerUnit {
    pub const MAX: Self = Self::Max();
    pub const MIN: Self = Self::Min();

    pub const fn Max() -> Self {
        return Self {
            timerId: u64::MAX,
            expire: i64::MAX,
        };
    }

    pub const fn Min() -> Self {
        return Self {
            timerId: u64::MIN,
            expire: i64::MIN,
        };
    }
}

impl Ord for TimerUnit {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.expire != other.expire {
            return self.expire.cmp(&other.expire);
        } else {
            return self.timerId.cmp(&other.timerId);
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
        self.timerId == other.timerId && self.expire == other.expire
    }
}

#[derive(Default)]
pub struct TimerStore(QMutex<TimerStoreIntern>);

impl Deref for TimerStore {
    type Target = QMutex<TimerStoreIntern>;

    fn deref(&self) -> &QMutex<TimerStoreIntern> {
        &self.0
    }
}

impl TimerStore {
    pub fn Trigger(&self) -> i64 {
        return self.lock().Trigger();
    }

    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn ResetTimer(&self, timer: &Timer, timeout: i64) {
        let mut ts = self.lock();
        ts.ResetTimer(timer, timeout);
        ts.Trigger();
    }

    pub fn CancelTimer(&self, timer: &Timer) {
        let mut ts = self.lock();

        ts.RemoveTimer(timer);
        ts.Trigger();
    }
}

pub struct TimerStoreIntern {
    // expire time -> Timer
    pub timerSeq: BTreeMap<TimerUnit, Timer, GuestHostSharedAllocator>, // order by expire time
    pub nextExpire: i64,
    pub uringExpire: i64,
    pub uringId: u64,
}

impl Default for TimerStoreIntern {
    fn default() -> Self {
        let res = Self {
            timerSeq: BTreeMap::new_in(GUEST_HOST_SHARED_ALLOCATOR),
            nextExpire: 0,
            uringExpire: 0,
            uringId: 0,
        };

        return res;
    }
}

impl TimerStoreIntern {
    // the timeout need to process a timer, <PROCESS_TIME means the timer will be triggered immediatelyfa
    pub const PROCESS_TIME: i64 = 30_000;

    pub fn Print(&self) -> String {
        let keys: Vec<TimerUnit> = self.timerSeq.keys().cloned().collect();
        return format!("TimerStoreIntern seq is {:#?}", keys);
    }

    pub fn Trigger(&mut self) -> i64 {
        let mut now = MONOTONIC_CLOCK.Now().0;
        while now + Self::PROCESS_TIME >= self.nextExpire {
            let timer = self.GetFirst(now + Self::PROCESS_TIME);
            match timer {
                Some(timer) => {
                    timer.Fire(self);
                }
                None => break,
            }

            now = MONOTONIC_CLOCK.Now().0;
        }

        if self.nextExpire == 0 {
            return -1;
        }

        assert!(
            self.nextExpire > now,
            "next expire is {}, now is {}",
            self.nextExpire,
            now
        );
        return self.nextExpire - now;
        /*if self.nextExpire != self.uringExpire {
            self.RemoveUringTimer();

            if self.nextExpire != 0 {
                self.SetUringTimer(self.nextExpire);
            }
        }*/
    }

    // return: existing or not
    pub fn RemoveTimer(&mut self, timer: &Timer) -> bool {
        let timer = timer.lock();

        if timer.Expire > 0 {
            self.timerSeq.remove(&timer.TimerUnit());
            return true;
        }

        return false;
    }

    pub fn ResetTimer(&mut self, timer: &Timer, timeout: i64) {
        let mut tl = timer.lock();
        if tl.Expire > 0 {
            self.timerSeq.remove(&tl.TimerUnit());
        }

        if timeout == 0 {
            return;
        }

        let current = MONOTONIC_CLOCK.Now().0;
        tl.Expire = current + timeout;

        if self.nextExpire == 0 || self.nextExpire > tl.Expire {
            self.nextExpire = tl.Expire;
        }

        self.timerSeq.insert(tl.TimerUnit(), timer.clone());
    }

    pub fn RemoveUringTimer(&mut self) {
        if self.uringExpire != 0 {
            IOURING.AsyncTimerRemove(self.uringId);
            self.uringExpire = 0;
        }
    }

    pub fn SetUringTimer(&mut self, expire: i64) {
        let now = MONOTONIC_CLOCK.Now().0;
        let expire = if expire < now + Self::PROCESS_TIME {
            now + Self::PROCESS_TIME
        } else {
            expire
        };
        assert!(self.uringExpire == 0);
        assert!(
            expire > now,
            "Expire {}, now {}, expire - now {}",
            expire,
            now,
            expire - now
        );
        self.uringExpire = expire;
        self.uringId = IOURING.Timeout(expire, expire - now) as u64;
    }

    // return (Expire, Timer)
    pub fn GetFirst(&mut self, now: i64) -> Option<Timer> {
        if self.nextExpire == 0 || self.nextExpire > now {
            return None;
        }

        let mut firstKey = None;

        for (&key, _) in self
            .timerSeq
            .range((Included(&TimerUnit::MIN), Included(&TimerUnit::MAX)))
        {
            firstKey = Some(key);
            break;
        }

        let firstKey = match firstKey {
            None => return None,
            Some(key) => key,
        };

        let timer = match self.timerSeq.remove(&firstKey) {
            None => return None,
            Some(timer) => timer,
        };

        if self.timerSeq.len() == 0 {
            self.nextExpire = 0;
        }

        for (&key, _) in self
            .timerSeq
            .range((Included(&TimerUnit::MIN), Included(&TimerUnit::MAX)))
        {
            self.nextExpire = key.expire;
            break;
        }

        return Some(timer);
    }
}
