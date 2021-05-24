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

use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::Ordering;

//use super::super::super::threadmgr::thread::*;
use super::super::super::threadmgr::task_block::*;
use super::queue::*;
use super::entry::*;
use super::*;

#[derive(Default)]
pub struct WaitGroupInternal {
    pub cnt: AtomicI32,
    pub queue: Queue,
    pub mutex: Mutex<()>,
}

#[derive(Default, Clone)]
pub struct WaitGroup(Arc<WaitGroupInternal>);

impl Deref for WaitGroup {
    type Target = Arc<WaitGroupInternal>;

    fn deref(&self) -> &Arc<WaitGroupInternal> {
        &self.0
    }
}

impl WaitGroup {
    pub fn New(cnt: i32) -> Self {
        let internal = WaitGroupInternal {
            cnt: AtomicI32::new(cnt),
            queue: Queue::default(),
            mutex: Mutex::new(()),
        };

        return Self(Arc::new(internal))
    }

    // Add adds delta, which may be negative, to the WaitGroup counter.
    // If the counter becomes zero, all threads blocked on Wait are released.
    // If the counter goes negative, Add panics.
    //
    // Note that calls with a positive delta that occur when the counter is zero
    // must happen before a Wait. Calls with a negative delta, or calls with a
    // positive delta that start when the counter is greater than zero, may happen
    // at any time.
    // Typically this means the calls to Add should execute before the statement
    // creating the thread or other event to be waited for.
    // If a WaitGroup is reused to wait for several independent sets of events,
    // new Add calls must happen after all previous Wait calls have returned.
    // See the WaitGroup example.
    pub fn Add(&self, delta: i32) {
        let w = self;

        w.mutex.lock();
        let prev = w.cnt.fetch_add(delta, Ordering::SeqCst);
        let val = prev + delta;

        if val < 0 {
            panic!("sync: negative WaitGroup counter prev is {} delta is {}", prev, delta)
        }

        if val == 0 {
            w.queue.Notify(!0);
            w.queue.write().RemoveAll();
        }
    }

    // Done decrements the WaitGroup counter by one.
    pub fn Done(&self) {
        self.Add(-1);
    }

    #[inline]
    pub fn Count(&self) -> i32 {
        return self.cnt.load(Ordering::SeqCst);
    }

    //if return == true, it is blocked, otherwise it can return
    pub fn EventRegister(&self, task: &Task,e: &WaitEntry, mask: EventMask) -> bool {
        let w = self;
        w.mutex.lock();
        if w.cnt.load(Ordering::SeqCst) == 0 {
            //e.Notify(!0);
            return false;
        }

        e.Clear();
        w.queue.EventRegister(task, e, mask);
        return true;
    }
}

impl Blocker {
    pub fn WaitGroupWait(&self, task: &Task, wg: &WaitGroup) {
        let block = wg.EventRegister(task, &self.generalEntry, 1);
        if !block {
            //fast path
            return
        }

        return self.BlockGeneralOnly();
    }
}