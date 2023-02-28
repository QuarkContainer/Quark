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

use core::ops::Deref;
use crossbeam_queue::ArrayQueue;

use super::common::*;

pub struct QRingQueue<T: Clone>(ArrayQueue<T>);

pub const MSG_QUEUE_SIZE: usize = 256;

impl<T: Clone> Default for QRingQueue<T> {
    fn default() -> Self {
        return Self::New(MSG_QUEUE_SIZE);
    }
}

impl<T: Clone> Deref for QRingQueue<T> {
    type Target = ArrayQueue<T>;

    fn deref(&self) -> &ArrayQueue<T> {
        &self.0
    }
}

impl<T: Clone> QRingQueue<T> {
    pub fn New(size: usize) -> Self {
        return Self(ArrayQueue::new(size));
    }

    pub fn Push(&self, data: &T) -> Result<()> {
        match self.0.push(data.clone()) {
            Err(_) => {
                error!("QRingQueue full...");
                return Err(Error::QueueFull);
            }
            _ => return Ok(()),
        }
    }

    pub fn Pop(&self) -> Option<T> {
        return self.0.pop();
    }

    pub fn Count(&self) -> usize {
        return self.0.len();
    }

    pub fn IsFull(&self) -> bool {
        return self.0.is_full();
    }

    pub fn IsEmpty(&self) -> bool {
        return self.0.is_empty();
    }
}
