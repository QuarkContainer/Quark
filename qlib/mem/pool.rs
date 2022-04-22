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

use super::super::mutex::*;
use alloc::collections::vec_deque::VecDeque;
use alloc::sync::Arc;
use core::ops::Deref;

#[derive(Default)]
pub struct PoolInternal<T: Default> {
    pub stack: VecDeque<T>,
    pub size: usize,
}

pub struct Pool<T: Default>(Arc<QMutex<PoolInternal<T>>>);

impl<T: Default> Deref for Pool<T> {
    type Target = Arc<QMutex<PoolInternal<T>>>;

    fn deref(&self) -> &Arc<QMutex<PoolInternal<T>>> {
        &self.0
    }
}

impl<T: Default> Pool<T> {
    pub fn New(size: usize) -> Self {
        let internal = PoolInternal {
            stack: VecDeque::with_capacity(size),
            size: size,
        };

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn Pop(&self) -> Option<T> {
        return self.lock().stack.pop_front();
    }

    pub fn Push(&self, v: T) {
        let mut p = self.lock();
        if p.stack.len() == p.size {
            return;
        }

        p.stack.push_front(v);
    }
}
