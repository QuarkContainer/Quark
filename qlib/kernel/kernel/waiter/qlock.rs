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
use crate::qlib::mutex::*;
use crate::GuestHostSharedAllocator;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::ops::Deref;
use core::ops::DerefMut;

use super::super::super::super::common::*;
use super::super::super::taskMgr;
use super::queue::*;
use super::*;
/*
#[derive(Default)]
pub struct QLock<T: ?Sized> {
    pub locked: QMutex<bool>,
    pub queue: Queue,
    pub data: UnsafeCell<T>,
}

pub struct QLockGuard<'a, T: ?Sized + 'a> {
    pub lock: &'a QLock<T>,
}

// Same unsafe impls as `std::sync::QMutex`
unsafe impl<T: ?Sized + Send> Sync for QLock<T> {}
unsafe impl<T: ?Sized + Send> Send for QLock<T> {}

impl<T> QLock<T> {
    pub fn New(data: T) -> Self {
        return Self {
            locked: QMutex::new(false),
            queue: Queue::default(),
            data: UnsafeCell::new(data),
        };
    }
}

impl<T: ?Sized> QLock<T> {
    pub fn Unlock(&self) {
        let mut l = self.locked.lock();
        assert!(*l == true, "QLock::Unlock misrun");
        *l = false;
        self.queue.Notify(!0);

        //self.queue.write().RemoveAll();
    }

    pub fn lock(&self) -> QLockGuard<T> {
        let task = super::super::super::task::Task::Current();
        loop {
            match self.Lock(task) {
                Err(_) => continue,
                Ok(ret) => return ret,
            }
        }
    }

    pub fn Lock(&self, task: &Task) -> Result<QLockGuard<T>> {
        let blocker = task.blocker.clone();

        loop {
            let block = {
                let mut l = self.locked.lock();
                if *l == false {
                    //fast path, got lock
                    *l = true;
                    false
                } else {
                    blocker.generalEntry.Clear();
                    self.queue.EventRegister(task, &blocker.generalEntry, 1);
                    true
                }
            };

            if !block {
                //fast path
                return Ok(QLockGuard { lock: self });
            }

            match blocker.BlockGeneral() {
                Err(e) => {
                    self.queue.EventUnregister(task, &blocker.generalEntry);
                    return Err(e);
                }
                Ok(()) => (),
            }

            self.queue.EventUnregister(task, &blocker.generalEntry);
        }
    }
}

impl<'a, T: ?Sized + 'a> Deref for QLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        let data = unsafe { &mut *self.lock.data.get() };
        &*data
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for QLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        let data = unsafe { &mut *self.lock.data.get() };
        &mut *data
    }
}

impl<'a, T: ?Sized + 'a> Drop for QLockGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.Unlock();
    }
}

#[derive(Default, Clone)]
pub struct QAsyncLock {
    pub locked: Arc<QMutex<bool>>,
}

#[derive(Default)]
pub struct QAsyncLockGuard {
    pub locked: Arc<QMutex<bool>>,
}

impl QAsyncLock {
    pub fn Lock(&self, _task: &Task) -> QAsyncLockGuard {
        loop {
            let mut l = self.locked.lock();
            if *l == false {
                *l = true;
                break;
            }

            taskMgr::Yield();
        }

        return QAsyncLockGuard {
            locked: self.locked.clone(),
        }
    }
}

impl QAsyncLockGuard {
    pub fn Unlock(&self) {
        let mut l = self.locked.lock();
        assert!(*l == true, "QLock::Unlock misrun");
        *l = false;
    }
}

impl Drop for QAsyncLockGuard {
    fn drop(&mut self) {
        self.Unlock();
    }
}*/

#[derive(Clone)]
pub struct QAsyncLock {
    pub locked: Arc<QMutex<bool>, GuestHostSharedAllocator>,
    pub queue: Queue,
}

impl Default for QAsyncLock {
    fn default() -> Self {
        Self {
            locked: Arc::new_in(QMutex::default(), GUEST_HOST_SHARED_ALLOCATOR),
            queue: Queue::default(),
        }
    }
}

pub struct QAsyncLockGuard {
    pub locked: Arc<QMutex<bool>, GuestHostSharedAllocator>,
    pub queue: Queue,
}

impl Default for QAsyncLockGuard {
    fn default() -> Self {
        Self {
            locked: Arc::new_in(QMutex::default(), GUEST_HOST_SHARED_ALLOCATOR),
            queue: Queue::default(),
        }
    }
}

impl QAsyncLock {
    pub fn Lock(&self, task: &Task) -> QAsyncLockGuard {
        loop {
            match self.Block(task) {
                Err(_) => continue,
                Ok(ret) => return ret,
            }
        }
    }

    pub fn Block(&self, task: &Task) -> Result<QAsyncLockGuard> {
        let blocker = task.blocker.clone();

        loop {
            let block = {
                let mut l = self.locked.lock();
                if *l == false {
                    //fast path, got lock
                    *l = true;
                    false
                } else {
                    blocker.generalEntry.Clear();
                    self.queue.EventRegister(task, &blocker.generalEntry, 1);
                    true
                }
            };

            if !block {
                //fast path
                return Ok(QAsyncLockGuard {
                    locked: self.locked.clone(),
                    queue: self.queue.clone(),
                });
            }

            match blocker.BlockGeneral() {
                Err(e) => {
                    self.queue.EventUnregister(task, &blocker.generalEntry);
                    return Err(e);
                }
                Ok(()) => (),
            }

            self.queue.EventUnregister(task, &blocker.generalEntry);
        }
    }
}

impl QAsyncLockGuard {
    pub fn Unlock(&self) {
        let mut l = self.locked.lock();
        assert!(*l == true, "QLock::Unlock misrun");
        *l = false;
        self.queue.Notify(!0);
    }
}

impl Drop for QAsyncLockGuard {
    fn drop(&mut self) {
        self.Unlock();
    }
}

pub enum RWState {
    NoLock,
    Write,
    Read(u32),
}

impl Default for RWState {
    fn default() -> Self {
        return Self::NoLock;
    }
}

#[derive(Clone)]
pub struct QAsyncRwLock {
    pub locked: Arc<QMutex<RWState>, GuestHostSharedAllocator>,
    pub queue: Queue,
}

impl Default for QAsyncRwLock {
    fn default() -> Self {
        Self {
            locked: Arc::new_in(QMutex::<RWState>::default(), GUEST_HOST_SHARED_ALLOCATOR),
            queue: Queue::default(),
        }
    }
}

pub struct QAsyncReadLockGuard {
    pub locked: Arc<QMutex<RWState>, GuestHostSharedAllocator>,
    pub queue: Queue,
}

impl Default for QAsyncReadLockGuard {
    fn default() -> Self {
        Self {
            locked: Arc::new_in(QMutex::<RWState>::default(), GUEST_HOST_SHARED_ALLOCATOR),
            queue: Queue::default(),
        }
    }
}

pub struct QAsyncWriteLockGuard {
    pub locked: Arc<QMutex<RWState>, GuestHostSharedAllocator>,
    pub queue: Queue,
}

impl Default for QAsyncWriteLockGuard {
    fn default() -> Self {
        Self {
            locked: Arc::new_in(QMutex::<RWState>::default(), GUEST_HOST_SHARED_ALLOCATOR),
            queue: Queue::default(),
        }
    }
}

impl QAsyncRwLock {
    pub fn Read(&self, task: &Task) -> QAsyncReadLockGuard {
        loop {
            match self.BlockRead(task) {
                Err(_) => continue,
                Ok(ret) => return ret,
            }
        }
    }

    pub fn Write(&self, task: &Task) -> QAsyncWriteLockGuard {
        loop {
            match self.BlockWrite(task) {
                Err(_) => continue,
                Ok(ret) => return ret,
            }
        }
    }

    pub fn BlockRead(&self, task: &Task) -> Result<QAsyncReadLockGuard> {
        let blocker = task.blocker.clone();

        loop {
            let getLock = {
                let mut l = self.locked.lock();

                match *l {
                    RWState::NoLock => {
                        *l = RWState::Read(1);
                        true
                    }
                    RWState::Read(count) => {
                        *l = RWState::Read(count + 1);
                        true
                    }
                    RWState::Write => {
                        self.queue.EventRegister(task, &blocker.generalEntry, 1);
                        false
                    }
                }
            };

            if getLock {
                //fast path
                return Ok(QAsyncReadLockGuard {
                    locked: self.locked.clone(),
                    queue: self.queue.clone(),
                });
            }

            match blocker.BlockGeneral() {
                Err(e) => {
                    self.queue.EventUnregister(task, &blocker.generalEntry);
                    return Err(e);
                }
                Ok(()) => (),
            }

            self.queue.EventUnregister(task, &blocker.generalEntry);
        }
    }

    pub fn BlockWrite(&self, task: &Task) -> Result<QAsyncWriteLockGuard> {
        let blocker = task.blocker.clone();

        loop {
            let getLock = {
                let mut l = self.locked.lock();

                match *l {
                    RWState::NoLock => {
                        *l = RWState::Write;
                        true
                    }
                    RWState::Read(_) => {
                        self.queue.EventRegister(task, &blocker.generalEntry, 1);
                        false
                    }
                    RWState::Write => {
                        self.queue.EventRegister(task, &blocker.generalEntry, 1);
                        false
                    }
                }
            };

            if getLock {
                //fast path
                return Ok(QAsyncWriteLockGuard {
                    locked: self.locked.clone(),
                    queue: self.queue.clone(),
                });
            }

            match blocker.BlockGeneral() {
                Err(e) => {
                    self.queue.EventUnregister(task, &blocker.generalEntry);
                    return Err(e);
                }
                Ok(()) => (),
            }

            self.queue.EventUnregister(task, &blocker.generalEntry);
        }
    }
}

impl QAsyncReadLockGuard {
    pub fn Unlock(&self) {
        let mut l = self.locked.lock();
        match *l {
            RWState::NoLock => {
                panic!("QAsyncReadLockGuard unlock found RWState::NoLock");
            }
            RWState::Read(count) => {
                if count == 1 {
                    *l = RWState::NoLock;
                    self.queue.Notify(!0);
                } else {
                    *l = RWState::Read(count - 1)
                }
            }
            RWState::Write => {
                panic!("QAsyncReadLockGuard unlock found RWState::Write");
            }
        }
    }
}

impl Drop for QAsyncReadLockGuard {
    fn drop(&mut self) {
        self.Unlock();
    }
}

impl QAsyncWriteLockGuard {
    pub fn Unlock(&self) {
        let mut l = self.locked.lock();
        match *l {
            RWState::NoLock => {
                panic!("QAsyncWriteLockGuard unlock found RWState::NoLock");
            }
            RWState::Read(count) => {
                panic!(
                    "QAsyncWriteLockGuard unlock found RWState::Read count = {}",
                    count
                );
            }
            RWState::Write => {
                *l = RWState::NoLock;
                self.queue.Notify(!0);
            }
        }
    }
}

impl Drop for QAsyncWriteLockGuard {
    fn drop(&mut self) {
        self.Unlock();
    }
}

#[derive(Default)]
pub struct QLock<T: ?Sized> {
    pub locked: QMutex<bool>,
    pub data: UnsafeCell<T>,
}

pub struct QLockGuard<'a, T: ?Sized + 'a> {
    pub lock: &'a QLock<T>,
}

// Same unsafe impls as `std::sync::QMutex`
unsafe impl<T: ?Sized + Send> Sync for QLock<T> {}
unsafe impl<T: ?Sized + Send> Send for QLock<T> {}

impl<T> QLock<T> {
    pub fn New(data: T) -> Self {
        return Self {
            locked: QMutex::new(false),
            data: UnsafeCell::new(data),
        };
    }
}

impl<T: ?Sized> QLock<T> {
    pub fn Unlock(&self) {
        let mut l = self.locked.lock();
        assert!(*l == true, "QLock::Unlock misrun");
        *l = false;
    }

    pub fn lock(&self) -> QLockGuard<T> {
        loop {
            let mut l = self.locked.lock();
            if *l == false {
                *l = true;
                break;
            } else {
                taskMgr::Yield();
            }
        }

        return QLockGuard { lock: self };
    }

    pub fn Lock(&self, _task: &Task) -> Result<QLockGuard<T>> {
        return Ok(self.lock());
    }
}

impl<'a, T: ?Sized + 'a> Deref for QLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        let data = unsafe { &mut *self.lock.data.get() };
        &*data
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for QLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        let data = unsafe { &mut *self.lock.data.get() };
        &mut *data
    }
}

impl<'a, T: ?Sized + 'a> Drop for QLockGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.Unlock();
    }
}
