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

use crate::qlib::mutex::*;
use alloc::sync::Arc;
use core::fmt;
use core::ops::Deref;

use crate::qlib::mem::list_allocator::GuestHostSharedAllocator;
use super::entry::*;
use super::waitlist::*;
use super::*;

#[derive(Clone)]
pub struct Queue(Arc<QRwLock<WaitList>,GuestHostSharedAllocator>);
impl Default for Queue{
    fn default() -> Self{
        return Queue(Arc::new_in(QRwLock::new(WaitList::default()), crate::GUEST_HOST_SHARED_ALLOCATOR));
    }
}
impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Queue",)
    }
}

impl Deref for Queue {
    type Target = Arc<QRwLock<WaitList>,GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QRwLock<WaitList>,GuestHostSharedAllocator> {
        &self.0
    }
}

impl Waitable for Queue {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        return 0;
    }

    fn EventRegister(&self, _task: &Task, e: &WaitEntry, mask: EventMask) {
        let mut q = self.write();
        e.lock().mask = mask;
        q.PushBack(e)
    }

    fn EventUnregister(&self, _task: &Task, e: &WaitEntry) {
        let mut q = self.write();
        e.Clear();
        q.Remove(e)
    }
}

impl Queue {
    //notify won't remove the trigged waitentry
    pub fn Notify(&self, mask: EventMask) {
        let q = self.read();
        let mut entry = q.Front();
        loop {
            let tmp = if let Some(tmp) = entry {
                tmp
            } else {
                break;
            };
            tmp.Notify(mask);
            entry = tmp.lock().next.clone();
        }
    }

    pub fn Clear(&self) {
        let q = self.read();
        let mut entry = q.Front();
        loop {
            let tmp = if let Some(tmp) = entry {
                tmp
            } else {
                break;
            };
            tmp.Clear();
            entry = tmp.lock().next.clone();
        }
    }

    pub fn Events(&self) -> EventMask {
        let mut ret = 0;

        let q = self.read();
        let mut entry = q.Front();
        loop {
            let tmp = if let Some(tmp) = entry {
                tmp
            } else {
                break;
            };
            ret |= tmp.Mask();
            entry = tmp.lock().next.clone();
        }

        return ret;
    }
}
