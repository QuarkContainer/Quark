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

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use libc::*;

use super::*;
use super::super::qlib::*;
use super::super::qlib::perf_tunning::*;
use super::super::qlib::qmsg::*;

#[derive(Debug, Copy, Clone)]
pub struct Timer {
    pub expire: i64,
    pub timerId: u64,
    pub seqNo: u64,
}

impl Timer {
    pub fn New(taskId: u64, seqNo: u64, expire: i64) -> Self {
        return Self {
            timerId: taskId,
            seqNo: seqNo,
            expire: expire,
        }
    }
}

impl Ord for Timer {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.expire != other.expire {
            return self.expire.cmp(&other.expire)
        } else {
            return self.timerId.cmp(&other.timerId)
        }
    }
}

impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Timer {}

impl PartialEq for Timer {
    fn eq(&self, other: &Self) -> bool {
        self.timerId == other.timerId
    }
}

impl Hash for Timer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.timerId.hash(state);
    }
}

#[derive(Default)]
pub struct TimerKeeper {
    pub TimerList: BTreeSet<Timer>,
    //sort by <expire, timerId>
    pub TimerSet: HashSet<Timer>,
    //id by <timerId>
    pub timerfd: i32,
}

impl TimerKeeper {
    pub fn New(timerfd: i32) -> Self {
        return Self {
            timerfd: timerfd,
            ..Default::default()
        }
    }

    pub fn ResetTimer(&mut self, clockId: i32, timerId: u64, seqNo: u64, expire: i64, ss: &'static ShareSpace) {
        let current = if clockId == CLOCK_MONOTONIC {
            HostTime::Monotime().unwrap()
        } else {
            HostTime::Realtime().unwrap()
        };

        if expire < 10_000 {
            ss.AQHostInputCall(&HostInputMsg::FireTimer(FireTimer {
                TimerId: timerId,
                SeqNo: seqNo,
            }));

            return;
        }

        let expire = expire + current;
        let needSet = match self.TimerList.first() {
            None => true,
            Some(t) => {
                t.expire > expire
            },
        };

        self.Add(Timer {
            expire: expire,
            timerId: timerId,
            seqNo: seqNo
        });

        if needSet {
            self.SetTimer(expire);
        }
    }

    pub fn SetTimer(&mut self, expire: i64) {
        let E9 = 1_000_000_000;
        let interval = timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };

        let val = timespec {
            tv_sec: expire / E9,
            tv_nsec: expire % E9,
        };

        let newVal = itimerspec {
            it_interval: interval,
            it_value: val,
        };

        let ret = unsafe {
            timerfd_settime(self.timerfd, TFD_TIMER_ABSTIME, &newVal, 0 as *mut itimerspec)
        };

        if ret == -1 {
            error!("panic: TimerKeeper::SetTimer fail..., timerfd is {}, error is {}", self.timerfd, errno::errno().0);
            panic!("TimerKeeper::SetTimer fail..., timerfd is {}, error is {}", self.timerfd, errno::errno().0);
        }
    }

    pub fn NextExpire(&mut self, now: i64, ss: &'static ShareSpace) -> bool {
        let mut hasMsg = false;

        self.Expire(now, |timer| {
            ss.AQHostInputCall(&HostInputMsg::FireTimer(FireTimer {
                TimerId: timer.timerId,
                SeqNo: timer.seqNo,
            }));

            if timer.timerId == 1 {
                PerfPrint();
            }

            hasMsg = true;
        });

        let next = match self.TimerList.first() {
            None => 0, //0 means cancel timer
            Some(t) => t.expire,
        };

        self.SetTimer(next);

        return hasMsg;
    }

    pub fn Add(&mut self, t: Timer) {
        self.TimerList.insert(t);
        self.TimerSet.insert(t);
    }

    pub fn Expire(&mut self, now: i64, mut f: impl FnMut(&Timer)) {
        loop {
            let timer = match self.TimerList.first() {
                None => break,
                Some(t) => *t,
            };

            if timer.expire > now {
                break;
            }

            f(&timer);

            self.TimerList.remove(&timer);
            self.TimerSet.remove(&timer);
        }
    }

    pub fn StopTimer(&mut self, timerId: u64) {
        let t = self.TimerSet.get(&Timer {
            timerId: timerId,
            seqNo: 0,
            expire: 0,
        });

        let timer = match t {
            None => {
                return
            }
            Some(t) => {
                *t
            }
        };

        self.TimerList.remove(&timer);
        self.TimerSet.remove(&timer);
    }
}