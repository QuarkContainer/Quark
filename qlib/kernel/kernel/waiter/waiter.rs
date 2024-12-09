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
use core::ops::Deref;

use crate::qlib::mutex::*;

use super::super::super::super::task_mgr::*;
use super::super::super::taskMgr;
use super::super::super::SHARESPACE;

use super::entry::*;
use super::*;

pub type WaiterID = u32;

use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::GuestHostSharedAllocator;

#[derive(Copy, Clone, Debug)]
pub struct WaiterInternal {
    pub bitmap: u64,
    pub mask: u64,
    pub idCnt: u32,
    pub state: WaitState,
    pub taskId: TaskId,
}

impl Default for WaiterInternal {
    fn default() -> Self {
        return Self {
            bitmap: 0,
            mask: 0,
            idCnt: 0,
            state: WaitState::default(),
            taskId: TaskId::New(0),
        };
    }
}

#[derive(Clone)]
pub struct Waiter(Arc<QMutex<WaiterInternal>, GuestHostSharedAllocator>);

impl Default for Waiter {
    fn default() -> Self {
        return Waiter(Arc::new_in(
            QMutex::new(WaiterInternal::default()),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
    }
}

impl Deref for Waiter {
    type Target = Arc<QMutex<WaiterInternal>, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QMutex<WaiterInternal>, GuestHostSharedAllocator> {
        &self.0
    }
}

impl Waiter {
    pub fn New(taskId: u64) -> Self {
        let internal = WaiterInternal {
            taskId: TaskId::New(taskId),
            ..Default::default()
        };

        return Self(Arc::new_in(
            QMutex::new(internal),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
    }

    fn NextWaiterId(&self) -> WaiterID {
        let mut b = self.lock();
        if b.idCnt == 64 {
            panic!("Waiter overflow");
        }

        b.idCnt += 1;
        return b.idCnt - 1;
    }

    pub const GENERAL_WAITID: WaiterID = 0;
    pub const INTERRUPT_WAITID: WaiterID = 1;
    pub const TIMER_WAITID: WaiterID = 2;

    pub fn NewWaitEntry(&self, waitId: WaiterID, mask: EventMask) -> WaitEntry {
        //let waitId = self.NextWaiterId();
        return WaitEntry::NewThreadContext(self, waitId, mask);
    }

    pub fn Trigger(&self, id: WaiterID) {
        let mut b = self.lock();
        assert!(id <= Self::TIMER_WAITID, "Waiter out of range");

        b.bitmap |= 1 << id as usize;

        //info!("Waiter::trigger 1 taskid is {:x?}, stat is {:?}, id is {}, mask is {:x}",
        //    b.taskId, b.state, id, b.mask);
        if b.state == WaitState::Waiting && (b.bitmap & b.mask) != 0 {
            b.state = WaitState::Running;
            // error!("Waiter::trigger 2 taskid is {:x?}, stat is {:?}, id is {}, bitmap is {:x}",
            //    b.taskId, b.state, id, b.bitmap);
            if id == Self::TIMER_WAITID {
                SHARESPACE.scheduler.Schedule(b.taskId, false);
            } else {
                SHARESPACE.scheduler.Schedule(b.taskId, true);
            }
        }
    }

    pub fn Check(&self, mask: u64) -> Option<WaiterID> {
        let b = self.lock();
        let bitmap = b.bitmap & mask;
        if bitmap > 0 {
            return Some(bitmap.trailing_zeros());
        } else {
            return None;
        }
    }

    pub fn Wait(&self, mask: u64) -> WaiterID {
        loop {
            {
                loop {
                    let mut b = self.lock();
                    //error!("b.bitmap {:b} mask is {:b}", b.bitmap, mask);
                    let bitmap = b.bitmap & mask;

                    if bitmap != 0 {
                        let idx = bitmap.trailing_zeros() as usize;
                        b.state = WaitState::Running;
                        return idx as WaiterID;
                    }

                    b.mask = mask;
                    b.state = WaitState::Waiting;
                    break;
                }
            }

            taskMgr::Wait();
        }
    }

    pub fn Clear(&self, id: WaiterID) {
        self.lock().bitmap &= !(1 << id); //clear the bit
    }

    pub fn IsReady(&self, entry: &WaitEntry) -> bool {
        let id = entry.lock().context.ThreadContext().waiterID;
        let w = self.lock();
        let bitmap = 1 << id;
        if w.bitmap & bitmap != 0 {
            return true;
        }

        return false;
    }

    pub fn TryWait(&self, entry: &WaitEntry, clear: bool) -> bool {
        let id = entry.lock().context.ThreadContext().waiterID;
        let mut w = self.lock();
        let bitmap = 1 << id;
        if w.bitmap & bitmap != 0 {
            if clear {
                w.bitmap &= !bitmap; //clear
            }

            return true;
        }

        return false;
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum WaitState {
    Running,
    Waiting,
}

impl Default for WaitState {
    fn default() -> Self {
        return WaitState::Running;
    }
}
