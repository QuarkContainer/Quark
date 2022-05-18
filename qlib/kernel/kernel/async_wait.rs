use alloc::sync::Arc;
use core::ops::Deref;
use core::sync::atomic::AtomicU64;

use super::super::super::common::*;
use super::super::super::mutex::*;
use super::super::super::task_mgr::*;
//use super::super::taskMgr::*;

use super::super::super::linux_def::QOrdering;

#[derive(Clone)]
pub struct MultiWait(Arc<MultiWaitIntern>);

impl MultiWait {
    pub fn New(taskId: TaskId) -> Self {
        return Self(Arc::new(MultiWaitIntern::New(taskId)));
    }
}

impl Deref for MultiWait {
    type Target = Arc<MultiWaitIntern>;

    fn deref(&self) -> &Arc<MultiWaitIntern> {
        &self.0
    }
}

pub struct MultiWaitIntern {
    pub count: AtomicU64,
    pub taskId: TaskId,
}

impl MultiWaitIntern {
    pub fn New(taskId: TaskId) -> Self {
        return Self {
            count: AtomicU64::new(1),
            taskId: taskId,
        };
    }

    pub fn AddWait(&self) -> u64 {
        return self.count.fetch_add(1, QOrdering::ACQUIRE) + 1;
    }

    pub fn Done(&self) -> u64 {
        let ret = self.count.fetch_sub(1, QOrdering::ACQUIRE) - 1;
        /*if ret == 0 {
            ScheduleQ(self.taskId);
        }*/

        return ret;
    }

    pub fn Wait(&self) {
        self.Done();
        //Wait();
        while self.count.load(QOrdering::ACQUIRE) != 0 {
            while self.count.load(QOrdering::RELAXED) != 0 {
                core::hint::spin_loop();
            }
        }
    }

    // return the waiting work item count
    pub fn TryWait(&self) -> u64 {
        return self.count.load(QOrdering::ACQUIRE);
    }
}

#[derive(Clone)]
pub struct Future<T: Clone + Copy> {
    data: Arc<QMutex<(Result<T>, bool)>>,
}

impl<T: Clone + Copy> Future<T> {
    pub fn New(t: T) -> Self {
        return Self {
            data: Arc::new(QMutex::new((Ok(t), false))),
        };
    }

    pub fn Wait(&self) -> Result<T> {
        loop {
            let lock = self.data.lock();
            if !lock.1 {
                continue;
            }

            return lock.0.clone();
        }
    }

    pub fn TryWait(&self) -> Option<Result<T>> {
        let lock = self.data.lock();
        if lock.1 {
            return Some(lock.0.clone());
        }

        return None;
    }

    pub fn Set(&self, t: Result<T>) {
        let mut lock = self.data.lock();
        lock.0 = t;
        lock.1 = true;
    }
}
