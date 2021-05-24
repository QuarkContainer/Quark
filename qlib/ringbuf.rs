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

use ringbuf::*;
use spin::Mutex;
use core::cell::RefCell;
use core::marker::PhantomData;
use alloc::collections::vec_deque::VecDeque;
use core::ops::Deref;

use super::common::*;

// multple producer single consumer
pub struct MpScRing<T> {
    pub consumer: RefCell<Consumer<T>>,
    pub producer: Mutex<Producer<T>>,
    pub resource_type: PhantomData<T>,
}

unsafe impl <T> Sync for MpScRing<T> {}

impl <T> MpScRing <T> {
    pub fn New(size: usize) -> Self {
        let r = RingBuffer::new(size);
        let (p, c) = r.split();
        return Self {
            consumer: RefCell::new(c),
            producer: Mutex::new(p),
            resource_type: PhantomData,
        }
    }

    pub fn Push(&self, data: T) -> Result<()> {
        match self.producer.lock().push(data) {
            Ok(()) => return Ok(()),
            _ => return Err(Error::QueueFull)
        }
    }

    pub fn TryPush(&self, data: T) -> Option<T> {
        let mut p = match self.producer.try_lock() {
            None => return Some(data),
            Some(p) => p
        };

        if p.is_full() {
            return Some(data);
        }

        match p.push(data) {
            Ok(()) => (),
            _ => panic!("TryPush fail"),
        }
        return None;
    }

    pub fn Pop(&self) -> Option<T> {
        return self.consumer.borrow_mut().pop()
    }

    pub fn Count(&self) -> usize {
        return self.producer.lock().len();
    }

    pub fn IsFull(&self) -> bool {
        return self.producer.lock().is_full();
    }

    pub fn IsEmpty(&self) -> bool {
        return self.producer.lock().is_empty();
    }

    pub fn CountLockless(&self) -> usize {
        return self.consumer.borrow().len();
    }
}

//single producer multple consumer
pub struct SpMcRing  <T> {
    pub consumer: Mutex<Consumer<T>>,
    pub producer: RefCell<Producer<T>>,
    pub resource_type: PhantomData<T>,
}

unsafe impl <T> Sync for SpMcRing<T> {}

impl <T> SpMcRing <T> {
    pub fn New(size: usize) -> Self {
        let r = RingBuffer::new(size);
        let (p, c) = r.split();
        return Self {
            consumer: Mutex::new(c),
            producer: RefCell::new(p),
            resource_type: PhantomData,
        }
    }

    pub fn Push(&self, data: T) -> Result<()> {
        match self.producer.borrow_mut().push(data) {
            Ok(()) => return Ok(()),
            _ => return Err(Error::QueueFull)
        }
    }

    pub fn Pop(&self) -> Option<T> {
        return self.consumer.lock().pop()
    }

    pub fn TryPop(&self) -> Option<T> {
        let mut c = match self.consumer.try_lock() {
            None => return None,
            Some(d) => d,
        };

        return c.pop();
    }

    pub fn Count(&self) -> usize {
        return self.consumer.lock().len();
    }

    pub fn IsFull(&self) -> bool {
        return self.consumer.lock().is_full();
    }

    pub fn IsEmpty(&self) -> bool {
        return self.consumer.lock().is_empty();
    }

    pub fn CountLockless(&self) -> usize {
        return self.producer.borrow().len();
    }
}

pub struct QRingBuf<T>(Mutex<VecDeque<T>>);

impl <T> Deref for QRingBuf <T> {
    type Target = Mutex<VecDeque<T>>;

    fn deref(&self) -> &Mutex<VecDeque<T>> {
        &self.0
    }
}

impl <T> QRingBuf <T> {
    pub fn New(size: usize) -> Self {
        return Self(Mutex::new(VecDeque::with_capacity(size)))
    }

    pub fn Push(&self, data: T) -> Result<()> {
        let mut p = self.lock();

        if p.len() == p.capacity() {
            return Err(Error::QueueFull);
        }

        p.push_back(data);
        return Ok(());
    }

    pub fn TryPush(&self, data: T) -> Option<T> {
        let mut p = match self.try_lock() {
            None => return Some(data),
            Some(p) => p
        };

        if p.len() == p.capacity() {
            return Some(data);
        }

        p.push_back(data);
        return None;
    }

    pub fn Pop(&self) -> Option<T> {
        return self.lock().pop_front()
    }

    pub fn TryPop(&self) -> Option<T> {
        let mut p = match self.try_lock() {
            None => return None,
            Some(p) => p
        };

        return p.pop_front();
    }

    pub fn Count(&self) -> usize {
        return self.lock().len();
    }

    pub fn IsFull(&self) -> bool {
        let q = self.lock();
        return q.len() == q.capacity();
    }

    pub fn IsEmpty(&self) -> bool {
        return self.lock().len() == 0;
    }

    pub fn CountLockless(&self) -> usize {
        return self.lock().len();
    }
}
