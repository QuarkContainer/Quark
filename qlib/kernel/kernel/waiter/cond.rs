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
use crate::qlib::mutex::*;
use core::ops::Deref;

use super::super::super::super::common::*;
use super::super::super::task::*;
use super::queue::*;
use super::*;

#[derive(Default)]
pub struct CondInternal {
    pub queue: Queue,
    pub signaled: bool,
}

#[derive(Default, Clone)]
pub struct Cond(Arc<QMutex<CondInternal>>);

impl Deref for Cond {
    type Target = Arc<QMutex<CondInternal>>;

    fn deref(&self) -> &Arc<QMutex<CondInternal>> {
        &self.0
    }
}

impl Cond {
    //return: Ok:get notification; Err: interrupted.
    pub fn Wait(&self, task: &Task) -> Result<()> {
        let e;
        {
            let c = self.lock();
            if c.signaled {
                return Ok(());
            }

            e = task.blocker.generalEntry.clone();
            e.Clear();
            c.queue.EventRegister(task, &e, !0);
        }

        let signal = self.lock().signaled;
        if signal {
            self.lock().queue.EventUnregister(task, &e);
            return Ok(());
        }

        let res = task.blocker.block(true, None);
        self.lock().queue.EventUnregister(task, &e);

        return res;
    }

    pub fn Broadcast(&self) {
        let mut c = self.lock();
        c.signaled = true;
        c.queue.Notify(!0);
    }

    pub fn Reset(&self) {
        let mut c = self.lock();
        c.signaled = false;
    }
}