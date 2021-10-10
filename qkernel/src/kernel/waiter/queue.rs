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
use ::qlib::mutex::*;
use core::ops::Deref;

use super::waitlist::*;
use super::entry::*;
use super::*;

#[derive(Default, Clone)]
pub struct Queue(Arc<QRwLock<WaitList>>);

impl Deref for Queue {
    type Target = Arc<QRwLock<WaitList>>;

    fn deref(&self) -> &Arc<QRwLock<WaitList>> {
        &self.0
    }
}

impl Waitable for Queue {
    fn Readiness(&self, _task: &Task,_mask: EventMask) -> EventMask {
        return 0;
    }

    fn EventRegister(&self, _task: &Task,e: &WaitEntry, mask: EventMask) {
        let mut q = self.write();
        e.lock().mask = mask;
        q.PushBack(e)
    }

    fn EventUnregister(&self, _task: &Task,e: &WaitEntry) {
        let mut q = self.write();
        q.Remove(e)
    }
}

impl Queue {
    //notify won't remove the trigged waitentry
    pub fn Notify(&self, mask: EventMask) {
        let q = self.read();
        let mut entry = q.Front();
        while entry.is_some() {
            let tmp = entry.clone().unwrap();
            tmp.Notify(mask);
            entry = tmp.lock().next.clone();
        }
    }

    pub fn Clear(&self) {
        let q = self.read();
        let mut entry = q.Front();
        while entry.is_some() {
            let tmp = entry.clone().unwrap();
            tmp.Clear();
            entry = tmp.lock().next.clone();
        }
    }

    pub fn Events(&self) -> EventMask {
        let mut ret = 0;

        let q = self.read();
        let mut entry = q.Front();
        while entry.is_some() {
            let tmp = entry.clone().unwrap();
            ret |= tmp.Mask();
            entry = tmp.lock().next.clone();
        }

        return ret;
    }
}