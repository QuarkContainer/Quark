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

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::queue::*;
use super::*;
use super::super::super::task::*;

// Chan with >=1 chanel size
pub struct ChanInternel<T: Sized> {
    pub data : Option<T>,

    // is there thread waiting for reading this chan?
    pub waiting: bool,
    pub queue: Queue,
    pub closed: bool,
}

#[derive(Clone)]
pub struct Chan<T>(Arc<Mutex<ChanInternel<T>>>);

impl <T> Deref for Chan<T> {
    type Target = Arc<Mutex<ChanInternel<T>>>;

    fn deref(&self) -> &Arc<Mutex<ChanInternel<T>>> {
        &self.0
    }
}

impl <T> Chan <T> {
    pub fn New() -> Self {
        let internel = ChanInternel {
            data: None,
            waiting: false,
            queue: Queue::default(),
            closed: false,
        };

        return Self(Arc::new(Mutex::new(internel)))
    }

    pub fn Write(&self, task: &Task, data: T) -> Result<()> {
        loop {
            {
                let mut c = self.lock();
                if c.closed {
                    return Err(Error::ChanClose)
                }

                if c.waiting {
                    c.data = Some(data);
                    c.waiting = false;
                    c.queue.Notify(EVENT_IN);
                    return Ok(())
                }

                let block = task.blocker.clone();
                c.queue.EventRegister(task, &block.generalEntry, EVENT_OUT);
            }

            task.blocker.BlockGeneral()?;
            {
                let c = self.lock();
                let block = task.blocker.clone();
                c.queue.EventUnregister(task, &block.generalEntry);
            }
        }
    }

    //unblock write, return true if write successfully. otherwise false.
    pub fn TryWrite(&self, _task: &Task, data: T) -> Result<bool> {
        let mut c = self.lock();

        if c.closed {
            return Err(Error::ChanClose)
        }

        if c.waiting {
            c.data = Some(data);
            c.waiting = false;
            c.queue.Notify(EVENT_IN);
            return Ok(true)
        }

        return Ok(false);
    }

    pub fn Read(&self, task: &Task) -> Result<T> {
        loop {
            {
                let mut c = self.lock();

                if c.closed {
                    return Err(Error::ChanClose)
                }

                let data = c.data.take();
                if data.is_some() {
                    c.queue.Notify(EVENT_OUT);
                    return Ok(data.unwrap());
                }

                c.waiting = true;

                let block = task.blocker.clone();
                c.queue.EventRegister(task, &block.generalEntry, EVENT_IN);
            }

            task.blocker.BlockGeneral()?;
            {
                let c = self.lock();
                let block = task.blocker.clone();
                c.queue.EventUnregister(task, &block.generalEntry);
            }
        }
    }

    /*

    //we can't TryRead, because we can't set c.waiting
    pub fn TryRead(&self, _task: &Task) -> Result<Option<T>> {
        let mut c = self.lock();

        if c.closed {
            return Err(Error::ChanClose)
        }

        let data = c.data.take();
        if data.is_some() {
            c.queue.Notify(EVENT_OUT);
        }

        return Ok(None);
    }
    */

    pub fn Close(&self) {
        let mut c = self.lock();
        c.queue.Notify(!0);
        c.closed = true;
    }
}