
use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};
use core::marker::PhantomData;
use core::hint::spin_loop;

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

        error!("QMutex lock by {:x}", val);

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
