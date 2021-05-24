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

use super::super::super::threadmgr::task_block::*;
use super::super::super::qlib::common::*;
use super::entry::*;
use super::queue::*;
use super::*;

#[derive(Default)]
pub struct QLockInternal {
    pub queue: Queue,
    pub locked: bool,
}

#[derive(Default, Clone)]
pub struct QLock(Arc<Mutex<QLockInternal>>);

impl Deref for QLock {
    type Target = Arc<Mutex<QLockInternal>>;

    fn deref(&self) -> &Arc<Mutex<QLockInternal>> {
        &self.0
    }
}

impl QLock {
    //if return == true, it is blocked, otherwise it can return
    pub fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) -> bool {
        let mut l = self.lock();
        if l.locked == false {
            //fast path, got lock
            l.locked = true;
            return false;
        }

        e.Clear();
        l.queue.EventRegister(task, e, mask);
        return true;
    }

    pub fn Unlock(&self) {
        let mut l = self.lock();
        assert!(l.locked == true, "QLock::Unlock misrun");
        l.locked = false;
        l.queue.Notify(!0);

        l.queue.write().RemoveAll();
    }

    pub fn Lock(&self, task: &Task) -> Result<QLockGuard> {
        let blocker = task.blocker.clone();

        loop {
            let block = self.EventRegister(task, &blocker.generalEntry, 1);
            info!("Qlock block is {}", block);
            if !block {
                //fast path
                return Ok(QLockGuard {
                    lock: Some(self.clone()),
                })
            }
            match blocker.BlockGeneral() {
                Err(e) => {
                    self.lock().queue.EventUnregister(task, &blocker.generalEntry);
                    return Err(e)
                }
                Ok(()) => ()
            }
            self.lock().queue.EventUnregister(task, &blocker.generalEntry);
        }

        //return blocker.Qlock(task, self);
    }
}

impl Blocker {
    pub fn Qlock(&self, task: &Task, l: &QLock) -> Result<QLockGuard> {
        loop {
            let block = l.EventRegister(task, &self.generalEntry, 1);
            if !block {
                //fast path
                return Ok(QLockGuard {
                    lock: Some(l.clone()),
                })
            }
            match self.BlockGeneral() {
                Err(e) => {
                    l.lock().queue.EventUnregister(task, &self.generalEntry);
                    return Err(e)
                }
                Ok(()) => ()
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct QLockGuard {
    pub lock: Option<QLock>
}

impl Drop for QLockGuard {
    fn drop(&mut self) {
        let lock = self.lock.take();

        if lock.is_some() {
            lock.unwrap().Unlock();
        }
    }
}