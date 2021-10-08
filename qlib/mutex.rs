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
use core::sync::atomic::{AtomicU64};
use core::marker::PhantomData;
use core::hint::spin_loop;
//use spin::*;

use super::linux_def::QOrdering;
//use super::super::asm::*;

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

#[inline(always)]
pub fn CmpExchg(addr: u64, old: u64, new: u64) -> u64 {
    let mut ret : u64;
    unsafe {
        llvm_asm!("
              lock cmpxchgq $2, ($3)
            "
            : "={rax}"(ret)
            : "{rax}"(old), "{rdx}"(new), "{rcx}"(addr)
            : "memory" : "volatile"  );
    };

    return ret;
}

#[inline(always)]
pub fn WriteOnce(addr: u64, val: u64) {
    unsafe {
        llvm_asm!("
               mfence
               mov $1, ($0)
            "
            :
            : "r"(addr), "r"(val)
            : "memory" : "volatile"  );
    };
}

#[inline(always)]
pub fn LoadOnce(addr: u64) ->  u64 {
    let ret: u64;
    unsafe {
        llvm_asm!("
               movq ($1), $0
               lfence
            "
            : "={rax}"(ret)
            : "{rdi}"(addr)
            : "memory" : "volatile"  );
    };

    return ret;
}

impl<T: ?Sized> QMutex<T> {
    #[inline(always)]
    pub fn CmpExchg(&self, old: u64, new: u64) -> u64 {
        /*match self.lock.compare_exchange(old, new, QOrdering::ACQUIRE, QOrdering::RELAXED) {
            Ok(v) => return v,
            Err(v) => return v,
        }*/

        return CmpExchg(&self.lock as * const _ as u64, old, new)
        //return self.lock.compare_and_swap(old, new, QOrdering::ACQUIRE);
    }

    pub fn Addr(&self) -> u64 {
        return &self.lock as * const _ as u64
    }

    pub fn MutexId(&self) -> u64 {
        return &self.lock as * const _ as u64;
    }

    #[inline(always)]
    pub fn lock(&self) -> QMutexGuard<T> {
        // Can fail to lock even if the spinlock is not locked. May be more efficient than `try_lock`
        // when called in a loop.
        let id = Self::GetID();
        /*if id < 0x4040000000 {
            raw!(0x122, id, &self.lock as * const _ as u64);
        }*/

        let mut val = 0;
        for _ in 0..10000 {
            super::super::asm::mfence();
            //val = self.lock.compare_and_swap(0, id, QOrdering::ACQUIRE);
            val = self.CmpExchg(0, id);
            if val == 0 {
                return QMutexGuard {
                    lock: &self.lock,
                    data: unsafe { &mut *self.data.get() },
                }
            }

            spin_loop();
        }

        raw!(0x123, val, &self.lock as * const _ as u64);
        defer!(raw!(0x122, val, &self.lock as * const _ as u64));

        loop  {
            super::super::asm::mfence();
            //let val = self.lock.compare_and_swap(0, id, QOrdering::ACQUIRE);
            let val = self.CmpExchg(0, id);
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
        //self.lock.load(QOrdering::RELAXED) != 0

        return LoadOnce(self.Addr()) != 0;
    }

    #[inline(always)]
    pub fn try_lock(&self) -> Option<QMutexGuard<T>> {
        let id = Self::GetID();

        super::super::asm::mfence();
        let val = self.CmpExchg(0, id);
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

impl<T: ?Sized + fmt::Debug> fmt::Debug for QMutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.try_lock() {
            Some(guard) => write!(f, "QMutex {{ data: ")
                .and_then(|()| (&*guard).fmt(f))
                .and_then(|()| write!(f, "}}")),
            None => write!(f, "QMutex {{ <locked> }}"),
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
    /// The dropping of the QMutexGuard will release the lock it was created from.
    fn drop(&mut self) {
        self.lock.store(0, QOrdering::RELEASE);

        //WriteOnce(self.lock as * const _ as u64, 0);
        //super::super::asm::mfence();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

pub struct QRwLock<T: ?Sized> {
    data: QMutex<T>,
}

pub struct QRwLockReadGuard<'a, T: 'a + ?Sized> {
    data: QMutexGuard<'a, T>,
}

pub struct QRwLockWriteGuard<'a, T: 'a + ?Sized> {
    data: QMutexGuard<'a, T>,
}


unsafe impl<T: ?Sized + Send> Send for QRwLock<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for QRwLock<T> {}

impl<T> QRwLock<T> {
    #[inline]
    pub const fn new(data: T) -> Self {
        return Self {
            data: QMutex::new(data)
        }
    }
}

impl<T: ?Sized> QRwLock<T> {
    #[inline]
    pub fn read(&self) -> QRwLockReadGuard<T> {
        return QRwLockReadGuard {
            data: self.data.lock()
        }
    }

    #[inline]
    pub fn write(&self) -> QRwLockWriteGuard<T> {
        super::super::asm::mfence();
        return QRwLockWriteGuard {
            data: self.data.lock()
        }
    }

    #[inline]
    pub fn try_read(&self) -> Option<QRwLockReadGuard<T>> {
        match self.data.try_lock() {
            None => None,
            Some(g) => Some(QRwLockReadGuard{
                data: g
            })
        }
    }

    #[inline]
    pub fn try_write(&self) -> Option<QRwLockWriteGuard<T>> {
        match self.data.try_lock() {
            None => None,
            Some(g) => Some(QRwLockWriteGuard{
                data: g
            })
        }
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockReadGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockWriteGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> DerefMut for QRwLockWriteGuard<'rwlock, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T: ?Sized + Default> Default for QRwLock<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

//////////////////////////////////////////////////////////////////

/*
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
        super::super::asm::mfence();
        return QRwLockReadGuard {
            data: self.data.read()
        }
    }

    #[inline]
    pub fn write(&self) -> QRwLockWriteGuard<T> {
        super::super::asm::mfence();
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
        super::super::asm::mfence();
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
        super::super::asm::mfence();
        match self.data.try_upgradeable_read() {
            None => None,
            Some(g) => Some(QRwLockUpgradableGuard {
                data: g
            })
        }
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockReadGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockUpgradableGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockWriteGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> DerefMut for QRwLockWriteGuard<'rwlock, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T: ?Sized + Default> Default for QRwLock<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}
*/