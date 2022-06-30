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

use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::fmt;
use core::hint::spin_loop;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use spin::*;

use super::kernel::uid::*;
use super::linux_def::QOrdering;

//use super::super::asm::*;

pub struct Spin;

pub type QMutex<T> = Mutex<T>;
pub type QMutexGuard<'a, T> = MutexGuard<'a, T>;

//pub type QMutex<T> = QMutexIntern<T>;
//pub type QMutexGuard<'a, T> = QMutexInternGuard<'a, T>;

pub type QRwLock<T> = RwLock<T>;
pub type QRwLockReadGuard<'a, T> = RwLockReadGuard<'a, T>;
pub type QRwLockWriteGuard<'a, T> = RwLockWriteGuard<'a, T>;

//pub type QRwLock<T> = QRwLockIntern<T>;
//pub type QRwLockReadGuard<'a, T> = QRwLockInternReadGuard<'a, T>;
//pub type QRwLockWriteGuard<'a, T> = QRwLockInternWriteGuard<'a, T>;

pub struct QMutexIntern<T: ?Sized, R = Spin> {
    phantom: PhantomData<R>,
    pub(crate) lock: AtomicU64,
    data: UnsafeCell<T>,
}

pub struct QMutexInternGuard<'a, T: ?Sized + 'a> {
    lock: &'a AtomicU64,
    data: &'a mut T,
}

unsafe impl<T: ?Sized + Send> Sync for QMutexIntern<T> {}
unsafe impl<T: ?Sized + Send> Send for QMutexIntern<T> {}

impl<T, R> QMutexIntern<T, R> {
    #[inline(always)]
    pub const fn new(data: T) -> Self {
        QMutexIntern {
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
    let mut ret: u64;
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
pub fn LoadOnce(addr: u64) -> u64 {
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

impl<T: ?Sized> QMutexIntern<T> {
    #[inline(always)]
    pub fn CmpExchg(&self, old: u64, new: u64) -> u64 {
        /*match self.lock.compare_exchange(old, new, QOrdering::ACQUIRE, QOrdering::RELAXED) {
            Ok(v) => return v,
            Err(v) => return v,
        }*/

        return CmpExchg(&self.lock as *const _ as u64, old, new);
        //return self.lock.compare_and_swap(old, new, QOrdering::ACQUIRE);
    }

    pub fn Addr(&self) -> u64 {
        return &self.lock as *const _ as u64;
    }

    pub fn MutexId(&self) -> u64 {
        return &self.lock as *const _ as u64;
    }

    #[inline(always)]
    pub fn lock(&self) -> QMutexInternGuard<T> {
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
                return QMutexInternGuard {
                    lock: &self.lock,
                    data: unsafe { &mut *self.data.get() },
                };
            }

            spin_loop();
        }

        raw!(0x123, val, &self.lock as *const _ as u64);
        defer!(raw!(0x122, val, &self.lock as *const _ as u64));

        loop {
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

        return QMutexInternGuard {
            lock: &self.lock,
            data: unsafe { &mut *self.data.get() },
        };
    }

    #[inline(always)]
    pub fn is_locked(&self) -> bool {
        //self.lock.load(QOrdering::RELAXED) != 0

        return LoadOnce(self.Addr()) != 0;
    }

    #[inline(always)]
    pub fn try_lock(&self) -> Option<QMutexInternGuard<T>> {
        let id = Self::GetID();

        super::super::asm::mfence();
        let val = self.CmpExchg(0, id);
        if val == 0 {
            Some(QMutexInternGuard {
                lock: &self.lock,
                data: unsafe { &mut *self.data.get() },
            })
        } else {
            None
        }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for QMutexIntern<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.try_lock() {
            Some(guard) => write!(f, "QMutexIntern {{ data: ")
                .and_then(|()| (&*guard).fmt(f))
                .and_then(|()| write!(f, "}}")),
            None => write!(f, "QMutexIntern {{ <locked> }}"),
        }
    }
}

impl<T: ?Sized + Default, R> Default for QMutexIntern<T, R> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T, R> From<T> for QMutexIntern<T, R> {
    fn from(data: T) -> Self {
        Self::new(data)
    }
}

impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for QMutexInternGuard<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<'a, T: ?Sized + fmt::Display> fmt::Display for QMutexInternGuard<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<'a, T: ?Sized> Deref for QMutexInternGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.data
    }
}

impl<'a, T: ?Sized> DerefMut for QMutexInternGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.data
    }
}

impl<'a, T: ?Sized> Drop for QMutexInternGuard<'a, T> {
    /// The dropping of the QMutexInternGuard will release the lock it was created from.
    fn drop(&mut self) {
        self.lock.store(0, QOrdering::RELEASE);

        //WriteOnce(self.lock as * const _ as u64, 0);
        //super::super::asm::mfence();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

pub struct QRwLockIntern<T: ?Sized> {
    data: QMutexIntern<T>,
}

pub struct QRwLockInternReadGuard<'a, T: 'a + ?Sized> {
    data: QMutexInternGuard<'a, T>,
}

pub struct QRwLockInternWriteGuard<'a, T: 'a + ?Sized> {
    data: QMutexInternGuard<'a, T>,
}

unsafe impl<T: ?Sized + Send> Send for QRwLockIntern<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for QRwLockIntern<T> {}

impl<T> QRwLockIntern<T> {
    #[inline]
    pub const fn new(data: T) -> Self {
        return Self {
            data: QMutexIntern::new(data),
        };
    }
}

impl<T: ?Sized> QRwLockIntern<T> {
    #[inline]
    pub fn read(&self) -> QRwLockInternReadGuard<T> {
        return QRwLockInternReadGuard {
            data: self.data.lock(),
        };
    }

    #[inline]
    pub fn write(&self) -> QRwLockInternWriteGuard<T> {
        super::super::asm::mfence();
        return QRwLockInternWriteGuard {
            data: self.data.lock(),
        };
    }

    #[inline]
    pub fn try_read(&self) -> Option<QRwLockInternReadGuard<T>> {
        match self.data.try_lock() {
            None => None,
            Some(g) => Some(QRwLockInternReadGuard { data: g }),
        }
    }

    #[inline]
    pub fn try_write(&self) -> Option<QRwLockInternWriteGuard<T>> {
        match self.data.try_lock() {
            None => None,
            Some(g) => Some(QRwLockInternWriteGuard { data: g }),
        }
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockInternReadGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockInternWriteGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> DerefMut for QRwLockInternWriteGuard<'rwlock, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T: ?Sized + Default> Default for QRwLockIntern<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

const READER: u64 = 1 << 2;
const UPGRADED: u64 = 1 << 1;
const WRITER: u64 = 1;

#[inline(always)]
fn compare_exchange(
    atomic: &AtomicU64,
    current: u64,
    new: u64,
    success: Ordering,
    failure: Ordering,
    strong: bool,
) -> Result<u64, u64> {
    if strong {
        atomic.compare_exchange(current, new, success, failure)
    } else {
        atomic.compare_exchange_weak(current, new, success, failure)
    }
}

#[derive(Clone)]
pub struct QUpgradableLock {
    lock: Arc<AtomicU64>,
    id: u64,
}

impl Default for QUpgradableLock {
    fn default() -> Self {
        return Self {
            lock: Default::default(),
            id: NewUID(),
        };
    }
}

pub struct QUpgradableLockGuard {
    lock: QUpgradableLock,
    write: AtomicBool,
}

impl QUpgradableLock {
    #[inline(always)]
    pub fn TryRead(&self) -> Option<QUpgradableLockGuard> {
        if self.lock.load(Ordering::Acquire) & (WRITER | UPGRADED) != 0 {
            return None;
        }
        let value = self.lock.fetch_add(READER, Ordering::Acquire);

        if value & (WRITER | UPGRADED) != 0 {
            // Lock is taken, undo.
            self.lock.fetch_sub(READER, Ordering::Release);
            None
        } else {
            //error!("RWLock {}: Read 2 cnt {:x}", self.id, value);
            Some(QUpgradableLockGuard {
                lock: self.clone(),
                write: AtomicBool::new(false),
            })
        }
    }

    #[inline(always)]
    pub fn ForceIncrRead(&self) -> u64 {
        return (self.lock.fetch_add(READER, Ordering::Acquire) >> 2) + 1;
    }

    #[inline(always)]
    pub fn ForceDecrRead(&self) -> u64 {
        return self.lock.fetch_sub(READER, Ordering::Release);
    }

    #[inline]
    pub fn Read(&self) -> QUpgradableLockGuard {
        //error!("RWLock {}: Read 1 {:x}", self.id, self.Value());
        loop {
            match self.TryRead() {
                None => spin_loop(),
                Some(g) => return g,
            }
        }
    }

    #[inline(always)]
    pub fn TryWriteIntern(&self, strong: bool) -> Option<QUpgradableLockGuard> {
        if compare_exchange(
            &self.lock,
            0,
            WRITER,
            Ordering::Acquire,
            Ordering::Relaxed,
            strong,
        )
        .is_ok()
        {
            Some(QUpgradableLockGuard {
                lock: self.clone(),
                write: AtomicBool::new(true),
            })
        } else {
            None
        }
    }

    pub fn Value(&self) -> u64 {
        return self.lock.load(Ordering::Acquire);
    }

    #[inline]
    pub fn TryWrite(&self) -> Option<QUpgradableLockGuard> {
        return self.TryWriteIntern(true);
    }

    pub fn Write(&self) -> QUpgradableLockGuard {
        //error!("RWLock {}: Write 1 {:x}", self.id, self.Value());
        //defer!(error!("RWLock {}: Write 2 {:x}", self.id, self.Value()));
        loop {
            match self.TryWrite() {
                None => spin_loop(),
                Some(g) => return g,
            }
        }
    }

    pub fn TryUpgrade(&self) -> bool {
        if self.lock.fetch_or(UPGRADED, Ordering::Acquire) & (WRITER | UPGRADED) == 0 {
            return true;
        } else {
            // We can't unflip the UPGRADED bit back just yet as there is another upgradeable or write lock.
            // When they unlock, they will clear the bit.
            return false;
        }
    }
}

impl Drop for QUpgradableLockGuard {
    fn drop(&mut self) {
        if self.Writable() {
            //error!("RWLock {}: write free {:x}", self.lock.id, self.lock.Value());
            self.lock
                .lock
                .fetch_and(!(WRITER | UPGRADED), Ordering::Release);
        } else {
            let _cnt = self.lock.ForceDecrRead();
            //error!("RWLock {}: read free {:x}", self.lock.id, cnt);
        }
    }
}

impl QUpgradableLockGuard {
    pub fn Writable(&self) -> bool {
        return self.write.load(Ordering::Acquire);
    }

    pub fn TryUpgradeToWriteIntern(&self) -> bool {
        return compare_exchange(
            &self.lock.lock,
            UPGRADED,
            WRITER,
            Ordering::Acquire,
            Ordering::Relaxed,
            true,
        )
        .is_ok();
    }

    pub fn Upgrade(&self) {
        //error!("RWLock {}: Upgrade1 {:x}", self.lock.id, self.lock.Value());
        assert!(!self.Writable());
        self.lock.ForceDecrRead();
        loop {
            if self.lock.TryUpgrade() {
                break;
            } else {
                spin_loop()
            }
        }

        //error!("RWLock {}: Upgrade2 {:x}", self.lock.id, self.lock.Value());
        loop {
            if self.TryUpgradeToWriteIntern() {
                break;
            } else {
                spin_loop()
            }
        }
        //error!("RWLock {}: Upgrade3 {:x}", self.lock.id, self.lock.Value());

        self.write.store(true, Ordering::Release)
    }

    pub fn Downgrade(&self) {
        assert!(self.Writable());
        //error!("RWLock {}: Downgrade1 {:x}", self.lock.id, self.lock.Value());
        self.lock.ForceIncrRead();
        self.lock
            .lock
            .fetch_and(!(UPGRADED | WRITER), Ordering::Release);
        self.write.store(false, Ordering::Release);
        //error!("RWLock {}: Downgrade2 {:x}", self.lock.id, self.lock.Value());
    }
}

//////////////////////////////////////////////////////////////////

/*
pub struct QRwLockIntern<T: ?Sized> {
    data: RwLock<T>,
}

pub struct QRwLockInternReadGuard<'a, T: 'a + ?Sized> {
    data: RwLockReadGuard<'a, T>,
}

pub struct QRwLockInternWriteGuard<'a, T: 'a + ?Sized> {
    data: RwLockWriteGuard<'a, T>,
}

pub struct QRwLockInternUpgradableGuard<'a, T: 'a + ?Sized> {
    data: RwLockUpgradableGuard<'a, T>,
}

unsafe impl<T: ?Sized + Send> Send for QRwLockIntern<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for QRwLockIntern<T> {}

impl<T> QRwLockIntern<T> {
    #[inline]
    pub const fn new(data: T) -> Self {
        return Self {
            data: RwLock::new(data)
        }
    }
}

impl<T: ?Sized> QRwLockIntern<T> {
    #[inline]
    pub fn read(&self) -> QRwLockInternReadGuard<T> {
        super::super::asm::mfence();
        return QRwLockInternReadGuard {
            data: self.data.read()
        }
    }

    #[inline]
    pub fn write(&self) -> QRwLockInternWriteGuard<T> {
        super::super::asm::mfence();
        return QRwLockInternWriteGuard {
            data: self.data.write()
        }
    }

    #[inline]
    pub fn upgradeable_read(&self) -> QRwLockInternUpgradableGuard<T> {
        return QRwLockInternUpgradableGuard {
            data: self.data.upgradeable_read()
        }
    }

    #[inline]
    pub fn try_read(&self) -> Option<QRwLockInternReadGuard<T>> {
        super::super::asm::mfence();
        match self.data.try_read() {
            None => None,
            Some(g) => Some(QRwLockInternReadGuard{
                data: g
            })
        }
    }

    #[inline]
    pub fn try_write(&self) -> Option<QRwLockInternWriteGuard<T>> {
        match self.data.try_write() {
            None => None,
            Some(g) => Some(QRwLockInternWriteGuard{
                data: g
            })
        }
    }

    #[inline]
    pub fn try_upgradeable_read(&self) -> Option<QRwLockInternUpgradableGuard<T>> {
        super::super::asm::mfence();
        match self.data.try_upgradeable_read() {
            None => None,
            Some(g) => Some(QRwLockInternUpgradableGuard {
                data: g
            })
        }
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockInternReadGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockInternUpgradableGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> Deref for QRwLockInternWriteGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<'rwlock, T: ?Sized> DerefMut for QRwLockInternWriteGuard<'rwlock, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T: ?Sized + Default> Default for QRwLockIntern<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}
*/
