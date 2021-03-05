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

use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;

use super::super::super::qlib::task_mgr::*;
use super::super::super::taskMgr;
use super::super::super::SHARESPACE;

use super::entry::*;
use super::*;

pub type WaiterID = u32;

#[derive(Copy, Clone)]
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
        }
    }
}

#[derive(Clone, Default)]
pub struct Waiter(Arc<Mutex<WaiterInternal>>);

impl Deref for Waiter {
    type Target = Arc<Mutex<WaiterInternal>>;

    fn deref(&self) -> &Arc<Mutex<WaiterInternal>> {
        &self.0
    }
}

impl Waiter {
    pub fn New(taskId: u64) -> Self {
        let internal = WaiterInternal {
            taskId: TaskId::New(taskId),
            ..Default::default()
        };

        return Self(Arc::new(Mutex::new(internal)));
    }

    fn NextWaiterId(&self) -> WaiterID {
        let mut b = self.lock();
        if b.idCnt == 64 {
            panic!("Waiter overflow");
        }

        b.idCnt += 1;
        return b.idCnt - 1
    }

    pub const TIMER_WAITID : WaiterID = 0;
    pub const INTERRUPT_WAITID : WaiterID = 1;
    pub const GENERAL_WAITID : WaiterID = 2;

    pub fn NewWaitEntry(&self, waitId: WaiterID, mask: EventMask) -> WaitEntry {
        //let waitId = self.NextWaiterId();
        return WaitEntry::NewThreadContext(self, waitId, mask)
    }

    pub fn Trigger(&self, id: WaiterID) {
        //info!("Waiter::trigger 1 id is {}", id);
        let mut b = self.lock();
        assert!(id <= Self::GENERAL_WAITID, "Waiter out of range");

        b.bitmap |= 1 << id as usize;

        if b.state == WaitState::Waiting && (b.bitmap & b.mask) != 0 {
            b.state = WaitState::Running;
            SHARESPACE.scheduler.Schedule(b.taskId);
        }
    }

    pub fn Wait(&self, entries: &[Option<WaitEntry>]) -> WaitEntry {
        let mut mask = 0;

        for e in entries {
            match e {
                None => continue,
                Some(e) => {
                    mask |= 1 << e.lock().context.ThreadContext().waiterID;
                }
            }
        }

        loop {
            {
                'r: loop {
                    let mut b = self.lock();

                    for e in entries {
                        match e {
                            None => continue,
                            Some(e) => {
                                if let Some(elock) = e.try_lock() {
                                    let waiterID = elock.context.ThreadContext().waiterID;
                                    //let mut b = self.lock();

                                    if b.bitmap & (1 << waiterID) != 0 {
                                        b.bitmap &= !(1 << waiterID); //clear the bit
                                        b.state = WaitState::Running;
                                        core::mem::drop(elock);
                                        return e.clone()
                                    }
                                } else {
                                    continue 'r;
                                }

                            }
                        }
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

    pub fn TryWait(&self, entry: &WaitEntry) -> bool {
        let id = entry.lock().context.ThreadContext().waiterID;
        let mut w = self.lock();
        let bitmap = 1 << id;
        if w.bitmap & bitmap != 0 {
            w.bitmap &= !bitmap; //clear
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

