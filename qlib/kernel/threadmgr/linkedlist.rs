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
use alloc::sync::Weak;
use ::qlib::mutex::*;
use core::ops::Deref;
use core::ops::DerefMut;

pub struct LinkedList<T> {
    pub head: Arc<QMutex<LinkEntry<T>>>,
    pub tail: Arc<QMutex<LinkEntry<T>>>,
    pub count: usize,
}

impl<T> Default for LinkedList<T> {
    fn default() -> Self {
        let head = Arc::new(QMutex::new(LinkEntry::default()));
        let tail = Arc::new(QMutex::new(LinkEntry::default()));

        head.lock().next = Some(tail.clone());
        tail.lock().prev = Some(Arc::downgrade(&head));

        return Self {
            head: head,
            tail: tail,
            count: 0,
        }
    }
}

impl<T> LinkedList<T> {
    pub fn Reset(&mut self) {
        self.head.lock().next = Some(self.tail.clone());
        self.tail.lock().prev = Some(Arc::downgrade(&self.head));
    }

    pub fn Empty(&self) -> bool {
        return self.count == 0;
    }

    pub fn Front(&self) -> Option<Arc<QMutex<LinkEntry<T>>>> {
        return match &self.head.lock().next {
            None => None,
            Some(ref e) => Some(e.clone()),
        }
    }

    pub fn Back(&self) -> Option<Arc<QMutex<LinkEntry<T>>>> {
        return match &self.tail.lock().prev {
            None => None,
            Some(ref e) => Some(e.upgrade().unwrap().clone()),
        }
    }

    pub fn PushFront(&mut self, entry: Arc<QMutex<LinkEntry<T>>>) {
        let next = self.head.lock().next.take().unwrap();

        next.lock().prev = Some(Arc::downgrade(&entry));
        entry.lock().next = Some(next.clone());

        entry.lock().prev = Some(Arc::downgrade(&self.head));
        self.head.lock().next = Some(entry);

        self.count += 1;
    }

    pub fn PopFront(&mut self) -> Option<Arc<QMutex<LinkEntry<T>>>> {
        if self.count == 0 {
            return None;
        }

        self.count -= 1;

        let ret = self.head.lock().next.as_ref().unwrap().clone();
        ret.lock().Remove();

        return Some(ret);
    }

    pub fn PushBack(&mut self, entry: Arc<QMutex<LinkEntry<T>>>) {
        let prev = self.tail.lock().prev.take().unwrap().upgrade().unwrap();

        entry.lock().prev = Some(Arc::downgrade(&prev));
        entry.lock().next = Some(self.tail.clone());
        self.tail.lock().prev = Some(Arc::downgrade(&entry));
        prev.lock().next = Some(entry);

        self.count += 1;
    }

    pub fn PopBack(&mut self) -> Option<Arc<QMutex<LinkEntry<T>>>> {
        if self.count == 0 {
            return None;
        }

        self.count -= 1;

        let ret = self.tail.lock().prev.as_ref().unwrap().upgrade().unwrap();
        ret.lock().Remove();

        return Some(ret);
    }
}

pub struct LinkEntry<T> {
    pub prev: Option<Weak<QMutex<LinkEntry<T>>>>,
    pub next: Option<Arc<QMutex<LinkEntry<T>>>>,
    pub data: Option<T>,
}

impl<T> Deref for LinkEntry<T> {
    type Target = Option<T>;

    fn deref(&self) -> &Option<T> {
        &self.data
    }
}

impl<T> DerefMut for LinkEntry<T> {
    fn deref_mut(&mut self) -> &mut Option<T> {
        &mut self.data
    }
}

impl<T> Default for LinkEntry<T> {
    fn default() -> Self {
        return Self {
            prev: None,
            next: None,
            data: None,
        }
    }
}

impl<T> LinkEntry<T> {
    pub fn New(data: T) -> Self {
        return Self {
            prev: None,
            next: None,
            data: Some(data),
        }
    }

    pub fn Remove(&mut self) {
        let prev = self.prev.take().expect("prev is null").upgrade().unwrap();
        let next = self.next.take().expect("next is null");

        (*prev).lock().next = Some(next.clone());
        (*next).lock().prev = Some(Arc::downgrade(&prev));
    }
}
