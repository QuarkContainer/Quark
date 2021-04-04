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

use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;
use alloc::collections::btree_map::BTreeMap;

use super::raw_timer::*;
use super::timer::*;

pub struct TimerMgrInternal {
    pub nextId: u64,
    //next timer id
    pub timers: BTreeMap<u64, RawTimer>,
}

impl Default for TimerMgrInternal {
    fn default() -> Self {
        return Self {
            nextId: 1,
            timers: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Default)]
pub struct TimerMgr(Arc<Mutex<TimerMgrInternal>>);

impl Deref for TimerMgr {
    type Target = Arc<Mutex<TimerMgrInternal>>;

    fn deref(&self) -> &Arc<Mutex<TimerMgrInternal>> {
        &self.0
    }
}

impl TimerMgr {
    pub fn NewTimer(&self, timer: &Timer) -> RawTimer {
        let mut tm = self.lock();
        let id = tm.nextId;
        tm.nextId += 1;

        let timer = RawTimer::New(id, self, timer);

        tm.timers.insert(id, timer.clone());
        return timer;
    }

    pub fn RemoveTimer(&self, timer: &RawTimer) {
        let id = timer.lock().Id;
        self.lock().timers.remove(&id);
    }

    pub fn Fire(&self, timerId: u64, seqNo: u64) {
        let timer = {
            let tm = self.lock();
            match tm.timers.get(&timerId) {
                None => return,
                Some(timer) => {
                    timer.clone()
                }
            }
        };

        timer.Fire(seqNo);
    }
}