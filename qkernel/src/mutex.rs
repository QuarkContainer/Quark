// Copyright (c) 2021 Quark Container Authors / https://github.com/mvdnes/spin-rs
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

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};
use core::marker::PhantomData;
use core::hint::spin_loop;
use spin::*;

use super::task::*;

pub struct Spin;
pub struct QMutex<T: ?Sized, R = Spin> {
    phantom: PhantomData<R>,
    pub(crate) lock: AtomicU64,
    data: UnsafeCell<T>,
}

pub struct QMutexGuard<'a, T: ?Sized + 'a> {
    lock: &'a AtomicU64,
    data: &'a mut T,
}

unsafe impl<T: ?Sized + Send> Sync for QMutex<T> {}
unsafe impl<T: ?Sized + Send> Send for QMutex<T> {}

impl<T, R> QMutex<T, R> {
   #[inline(always)]
    pub const fn new(data: T) -> Self {
        QMutex {
            lock: AtomicU64::new(0),
            data: UnsafeCell::new(data),
            phantom: PhantomData,
        }
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.data.get()
    }
}

impl<T: ?Sized> QMutex<T> {
    #[inline(always)]
    pub fn lock(&self) -> QMutexGuard<T> {
        // Can fail to lock even if the spinlock is not locked. May be more efficient than `try_lock`
        // when called in a loop.
        let id = Task::Current().taskId;

        let val = self.lock.compare_and_swap(0, id, Ordering::Acquire);
        if val == 0{
            return QMutexGuard {
                lock: &self.lock,
                data: unsafe { &mut *self.data.get() },
            }
        }

        debug!("QMutex lock by {:x}", val);

        loop  {
            let val = self.lock.compare_and_swap(0, id, Ordering::Acquire);
            if val == 0 {
                break;
            }

            while self.is_locked() {
                spin_loop();
            }
        }

        return QMutexGuard {
            lock: &self.lock,
            data: unsafe { &mut *self.data.get() },
        }
    }

    #[inline(always)]
    pub fn is_locked(&self) -> bool {
        self.lock.load(Ordering::Relaxed) != 0
    }

    #[inline(always)]
    pub fn try_lock(&self) -> Option<QMutexGuard<T>> {
        let id = Task::Current().taskId;

        let val = self.lock.compare_and_swap(0, id, Ordering::Acquire);
        if val == 0 {
            Some(QMutexGuard {
                lock: &self.lock,
                data: unsafe { &mut *self.data.get() },
            })
        } else {
            None
        }
    }
}

impl<T: ?Sized + Default, R> Default for QMutex<T, R> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T, R> From<T> for QMutex<T, R> {
    fn from(data: T) -> Self {
        Self::new(data)
    }
}

impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for QMutexGuard<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<'a, T: ?Sized + fmt::Display> fmt::Display for QMutexGuard<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<'a, T: ?Sized> Deref for QMutexGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.data
    }
}

impl<'a, T: ?Sized> DerefMut for QMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.data
    }
}

impl<'a, T: ?Sized> Drop for QMutexGuard<'a, T> {
    /// The dropping of the MutexGuard will release the lock it was created from.
    fn drop(&mut self) {
        self.lock.store(0, Ordering::Release);
    }
}


pub struct QRwLock<T: ?Sized> {
    data: RwLock<T>,
}

pub struct QRwLockReadGuard<'a, T: 'a + ?Sized> {
    data: RwLockReadGuard<'a, T>,
}

pub struct QRwLockWriteGuard<'a, T: 'a + ?Sized> {
    data: RwLockWriteGuard<'a, T>,
}

pub struct QRwLockUpgradableGuard<'a, T: 'a + ?Sized> {
    data: RwLockUpgradableGuard<'a, T>,
}

unsafe impl<T: ?Sized + Send> Send for QRwLock<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for QRwLock<T> {}

impl<T> QRwLock<T> {
    #[inline]
    pub const fn new(data: T) -> Self {
        return Self {
            data: RwLock::new(data)
        }
    }
}
impl<T: ?Sized> QRwLock<T> {
    #[inline]
    pub fn read(&self) -> QRwLockReadGuard<T> {
        return QRwLockReadGuard {
            data: self.data.read()
        }
    }

    #[inline]
    pub fn write(&self) -> QRwLockWriteGuard<T> {
        return QRwLockWriteGuard {
            data: self.data.write()
        }
    }

    #[inline]
    pub fn upgradeable_read(&self) -> QRwLockUpgradableGuard<T> {
        return QRwLockUpgradableGuard {
            data: self.data.upgradeable_read()
        }
    }

    #[inline]
    pub fn try_read(&self) -> Option<QRwLockReadGuard<T>> {
        match self.data.try_read() {
            None => None,
            Some(g) => Some(QRwLockReadGuard{
                data: g
            })
        }
    }

    #[inline]
    pub fn try_write(&self) -> Option<QRwLockWriteGuard<T>> {
        match self.data.try_write() {
            None => None,
            Some(g) => Some(QRwLockWriteGuard{
                data: g
            })
        }
    }

    #[inline]
    pub fn try_upgradeable_read(&self) -> Option<QRwLockUpgradableGuard<T>> {
        match self.data.try_upgradeable_read() {
            None => None,
            Some(g) => Some(QRwLockUpgradableGuard {
                data: g
            })
        }
    }
}
