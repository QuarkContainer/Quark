use core::sync::atomic::AtomicU64;
use alloc::sync::Arc;
use core::hint::spin_loop;

use super::super::qlib::mutex::*;

use super::super::qlib::linux_def::QOrdering;

pub type MultiWait = Arc<MultiWaitIntern>;

#[derive(Default)]
pub struct MultiWaitIntern {
    pub count: AtomicU64
}

impl MultiWaitIntern {
    pub fn New(&self) -> Self {
        return Self::default();
    }

    pub fn AddWait(&self) -> u64 {
        return self.count.fetch_add(1, QOrdering::ACQUIRE) + 1
    }

    pub fn Done(&self) -> u64 {
        return self.count.fetch_sub(1, QOrdering::ACQUIRE) - 1
    }

    pub fn Wait(&self) {
        while self.count.load(QOrdering::ACQUIRE) > 0 {
            while self.count.load(QOrdering::RELAXED) > 0 {
                spin_loop();
            }
        }
    }

    pub fn TryWait(&self) -> u64 {
        return self.count.load(QOrdering::ACQUIRE)
    }
}

#[derive(Clone)]
pub struct Future<T: Clone + Copy> {
    data: Arc<QMutex<T>>,
}

impl <T: Clone + Copy> Future <T> {
    pub fn New(t: T) -> Self {
        return Self {
            data: Arc::new(QMutex::new(t))
        };
    }

    pub fn Set(&self, t: T) {
        *self.data.lock() = t;
    }

    pub fn Get(&self) -> T {
        return *self.data.lock()
    }
}