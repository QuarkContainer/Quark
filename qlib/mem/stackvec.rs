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

use alloc::vec::Vec;

//const ON_STACK_COUNT: usize = 128;

pub struct StackVec<T: Copy + Default, const ON_STACK_COUNT: usize> {
    pub buf: [T; ON_STACK_COUNT],
    pub len: usize,
    pub vec: Vec<T>,
    pub onStack: bool,
}

impl<T: Copy + Default, const ON_STACK_COUNT: usize> StackVec<T, ON_STACK_COUNT> {
    pub fn New() -> Self {
        return Self {
            buf: [T::default(); ON_STACK_COUNT],
            len: 0,
            vec: Vec::with_capacity(0),
            onStack: true,
        };
    }

    pub fn OnStack(&self) -> bool {
        return self.onStack;
    }

    pub fn Len(&self) -> usize {
        return self.len;
    }

    pub fn Push(&mut self, val: T) {
        if self.len == self.buf.len() {
            self.vec = self.buf.to_vec();
            self.onStack = false
        }

        if self.OnStack() {
            self.buf[self.len] = val;
        } else {
            self.vec.push(val);
        }

        self.len += 1;
    }

    pub fn Pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        let idx = self.len;
        self.len -= 1;
        if self.OnStack() {
            return Some(self.buf[idx]);
        } else {
            return self.vec.pop();
        }
    }

    pub fn Slice(&self) -> &[T] {
        if self.OnStack() {
            return &self.buf[0..self.len];
        } else {
            return &self.vec[0..self.len];
        }
    }

    pub fn SliceMut(&mut self) -> &mut [T] {
        if self.OnStack() {
            return &mut self.buf[0..self.len];
        } else {
            return &mut self.vec[0..self.len];
        }
    }
}
