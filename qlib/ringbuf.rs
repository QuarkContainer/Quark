// Copyright (c) 2021 Quark Container Authors
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

use super::mutex::*;
use alloc::collections::vec_deque::VecDeque;
use core::ops::Deref;

use super::common::*;

#[derive(Default)]
pub struct QRingQueue<T: Clone>(QMutex<VecDeque<T>>);

impl<T: Clone> Deref for QRingQueue<T> {
    type Target = QMutex<VecDeque<T>>;

    fn deref(&self) -> &QMutex<VecDeque<T>> {
        &self.0
    }
}

impl<T: Clone> QRingQueue<T> {
    pub fn New(size: usize) -> Self {
        return Self(QMutex::new(VecDeque::with_capacity(size)));
    }

    pub fn Push(&self, data: &T) -> Result<()> {
        let mut p = self.lock();

        if p.len() == p.capacity() {
            return Err(Error::QueueFull);
        }

        p.push_back(data.clone());
        return Ok(());
    }

    pub fn TryPush(&self, data: &T) -> Result<()> {
        let mut p = match self.try_lock() {
            None => return Err(Error::NoData),
            Some(p) => p,
        };

        if p.len() == p.capacity() {
            return Err(Error::QueueFull);
        }

        p.push_back(data.clone());
        return Ok(());
    }

    pub fn Pop(&self) -> Option<T> {
        return self.lock().pop_front();
    }

    pub fn TryPop(&self) -> Option<T> {
        let mut p = match self.try_lock() {
            None => return None,
            Some(p) => p,
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
