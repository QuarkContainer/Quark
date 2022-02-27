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

use core::sync::atomic::AtomicU32;
use core::sync::atomic::Ordering;

pub const COUNT: usize = 128;
pub struct RingQueue <T: 'static + Default> {
    pub data: [T; COUNT],
    pub ringMask: AtomicU32,
    pub head: AtomicU32,
    pub tail: AtomicU32,
}

impl <T: 'static + Default + Copy> RingQueue <T> {
    pub fn Init(&self) {
        self.ringMask.store(COUNT as u32 -1, Ordering::Release);
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
    }

    #[inline]
    pub fn RingMask(&self) -> u32 {
        return self.ringMask.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn Count(&self) -> usize {
        return self.ringMask.load(Ordering::Relaxed) as usize + 1
    }

    // pop
    pub fn Pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        if available == 0 {
            return None
        }

        let idx = head & self.RingMask();
        let data = self.data[idx as usize];
        self.head.store(head.wrapping_add(1),  Ordering::Release);
        return Some(data);
    }

    pub fn DataCount(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        return available
    }

    //push
    pub fn SpaceCount(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        return self.Count() - available;
    }

    // precondition: there must be at least one free space
    pub fn Push(&mut self, data: T) {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        let available = tail.wrapping_sub(head) as usize;
        assert!(available < self.Count());

        let idx = tail & self.RingMask();
        self.data[idx as usize] = data;
        self.tail.store(tail.wrapping_add(1),  Ordering::Release);
    }
}