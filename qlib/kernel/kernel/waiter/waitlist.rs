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

use super::super::futex::*;
use super::entry::*;
use super::*;

#[derive(Default)]
pub struct WaitList {
    head: Option<WaitEntry>,
    tail: Option<WaitEntry>,
}

impl WaitList {
    //remove all of the wait entries
    pub fn Reset(&mut self) {
        let mut cur = self.head.clone();

        self.head = None;
        self.tail = None;

        while cur.is_some() {
            let tmp = cur.clone().unwrap();
            let next = tmp.lock().next.clone();
            tmp.Reset();

            cur = next;
        }
    }

    pub fn Empty(&self) -> bool {
        return self.head.is_none()
    }

    pub fn Front(&self) -> Option<WaitEntry> {
        return self.head.clone();
    }

    pub fn Back(&self) -> Option<WaitEntry> {
        return self.tail.clone();
    }

    pub fn PushFront(&mut self, e: &WaitEntry) {
        assert!(e.InitState(), "waitlist PushFront WaitEntry is not in init statue");
        //e.Reset();

        if self.head.is_none() {
            //empty
            self.head = Some(e.clone());
            self.tail = Some(e.clone())
        } else {
            let head = self.head.clone().unwrap();
            e.lock().next = Some(head.clone());
            head.lock().prev = Some(e.Downgrade());
            self.head = Some(e.clone());
        }
    }

    pub fn PushBack(&mut self, e: &WaitEntry) {
        assert!(e.InitState(), "waitlist PushBack WaitEntry is not in init statue");
        //e.Reset();

        if self.head.is_none() {
            //empty
            self.head = Some(e.clone());
            self.tail = Some(e.clone())
        } else {
            let tail = self.tail.clone().unwrap();
            e.lock().prev = Some(tail.Downgrade());
            tail.lock().next = Some(e.clone());
            self.tail = Some(e.clone());
        }
    }

    pub fn RemoveAll(&mut self) {
        self.Reset();
    }

    pub fn Remove(&mut self, e: &WaitEntry) {
        if e.lock().prev.is_none() {
            //head
            self.head = e.lock().next.clone();
        } else {
            let lock = e.lock();
            lock.prev.clone().unwrap().Upgrade().unwrap().lock().next = lock.next.clone();
        }

        if e.lock().next.is_none() {
            //tail
            self.tail = match &e.lock().prev {
                None => None,
                Some(ref p) => p.Upgrade()
            };
        } else {
            let lock = e.lock();
            lock.next.clone().unwrap().lock().prev = lock.prev.clone();
        }

        e.Reset();
    }

    // for futex, wakeLocked wakes up to n waiters matching the bitmask at the addr for this
    // queue and returns the number of waiters woken.
    // unlike Notify, Wake will remove the trigged waitentry
    pub fn WakeLocked(&mut self, mask: EventMask, n: i32) -> i32 {
        let q = self;

        let mut done = 0;
        let mut entry = q.Front();
        while entry.is_some() && done < n {
            let tmp = entry.clone().unwrap();
            entry = tmp.lock().next.clone();

            /*let triggered = tmp.Notify(mask);
            if triggered {
                done += 1;
                q.Remove(&tmp)
            }

            tmp.lock().key = Key::default();*/
            if q.WakeWaiterLocked(&tmp, mask) {
                done += 1;
            }
        }

        return done;
    }

    pub fn WakeWaiterLocked(&mut self, w: &WaitEntry, mask: EventMask) -> bool {
        let triggered = w.Notify(mask);
        if triggered {
            self.Remove(&w)
        }

        w.lock().context.ThreadContext().key = Key::default();
        return triggered;
    }

    // requeueLocked takes n waiters from the waitlist and moves them to naddr on the
    // waitlist "to".
    pub fn RequeueLocked(&mut self, to: &mut Self, nKey: &Key, n: i32) -> i32 {
        let mut done = 0;
        let mut w = self.Front();

        while w.is_some() && done < n {
            let tmp = w.clone().unwrap();
            let requeued = tmp.clone();
            w = tmp.lock().next.clone();

            self.Remove(&requeued);
            to.PushBack(&requeued);
            requeued.lock().context.ThreadContext().key = *nKey;

            done += 1;
        }

        return done;
    }
}
