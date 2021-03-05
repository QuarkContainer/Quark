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

use alloc::collections::vec_deque::*;
use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::queue::*;
use super::*;
use super::super::super::task::*;

// Chan with >=1 chanel size
pub struct BufChanInternel<T: Sized> {
    pub buf : VecDeque<T>,
    pub space: usize,
    pub queue: Queue,
    pub closed: bool,
}

#[derive(Clone)]
pub struct BufChan<T>(Arc<Mutex<BufChanInternel<T>>>);

impl <T> Deref for BufChan<T> {
    type Target = Arc<Mutex<BufChanInternel<T>>>;

    fn deref(&self) -> &Arc<Mutex<BufChanInternel<T>>> {
        &self.0
    }
}

impl <T> BufChan <T> {
    pub fn New(size: usize) -> Self {
        let internel = BufChanInternel {
            buf: VecDeque::with_capacity(size),
            space: size,
            queue: Queue::default(),
            closed: false,
        };

        return Self(Arc::new(Mutex::new(internel)))
    }

    // Get the items waiting in the buffer
    pub fn Len(&self) -> usize {
        return self.lock().buf.len();
    }

    pub fn Write(&self, task: &Task, data: T) -> Result<()> {
        loop {
            {
                let mut c = self.lock();
                if c.closed {
                    return Err(Error::ChanClose)
                }

                if c.space > 0 {
                    c.buf.push_back(data);
                    c.space -= 1;
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

        if c.space > 0 {
            c.buf.push_back(data);
            c.space -= 1;
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

                if c.buf.len() > 0 {
                    let ret = c.buf.pop_front().unwrap();
                    c.space += 1;
                    c.queue.Notify(EVENT_OUT);
                    return Ok(ret);
                }

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

    pub fn TryRead(&self) -> Result<Option<T>> {
        let mut c = self.lock();

        if c.closed {
            return Err(Error::ChanClose)
        }

        if c.buf.len() > 0 {
            let ret = c.buf.pop_front().unwrap();
            c.space += 1;
            c.queue.Notify(EVENT_OUT);
            return Ok(Some(ret));
        }

        return Ok(None);
    }

    pub fn Close(&self) {
        let mut c = self.lock();
        c.queue.Notify(!0);
        c.closed = true;
    }
}