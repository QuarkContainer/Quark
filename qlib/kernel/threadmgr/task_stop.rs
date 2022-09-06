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
use alloc::vec::Vec;

use super::super::threadmgr::thread::*;
use super::super::threadmgr::threads::*;

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum TaskStopType {
    GROUPSTOP,
    PTRACESTOP,
    EXECSTOP,
    VFORKSTOP,
    OTHER,
    //DUMMY
}

pub trait TaskStop: Sync + Send {
    fn Type(&self) -> TaskStopType;
    fn Killable(&self) -> bool;
}

impl ThreadInternal {
    pub fn beginInternalStop<T: TaskStop + 'static>(&mut self, s: &Arc<T>) {
        let pidns = self.tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let lock = self.tg.lock().signalLock.clone();
        let _s = lock.lock();

        self.beginInternalStopLocked(s);
        return;
    }

    pub fn beginInternalStopLocked<T: TaskStop + 'static>(&mut self, s: &Arc<T>) {
        if self.stop.is_some() {
            panic!("Attempting to enter internal stop when already in internal stop");
        }

        self.stop = Some(s.clone());
        self.beginStopLocked();
    }

    // endInternalStopLocked indicates the end of an internal stop that applies to
    // t. endInternalStopLocked does not wait for the task to resume.
    //
    // The caller is responsible for ensuring that the internal stop they expect
    // actually applies to t; this requires holding the signal mutex which protects
    // t.stop, which is why there is no endInternalStop that locks the signal mutex
    // for you.
    //
    // Preconditions: The signal mutex must be locked. The task must be in an
    // internal stop (i.e. t.stop != nil).
    pub fn endInternalStopLocked(&mut self) {
        if self.stop.is_none() {
            panic!("Attempting to leave non-existent internal stop")
        }

        self.stop = None;
        self.endStopLocked();
    }

    // beginStopLocked increments t.stopCount to indicate that a new internal or
    // external stop applies to t.
    //
    // Preconditions: The signal mutex must be locked.
    pub fn beginStopLocked(&mut self) {
        self.stopCount.Add(1);
    }

    // endStopLocked decerements t.stopCount to indicate that an existing internal
    // or external stop no longer applies to t.
    //
    // Preconditions: The signal mutex must be locked.
    pub fn endStopLocked(&mut self) {
        self.stopCount.Done();
    }
}

impl Thread {
    // BeginExternalStop indicates the start of an external stop that applies to t.
    // BeginExternalStop does not wait for t's task goroutine to stop.
    pub fn BeginExternalStop(&self) {
        let mut t = self.lock();
        let pidns = t.tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let tg = t.tg.clone();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        t.beginStopLocked();
        t.interrupt();
    }

    // EndExternalStop indicates the end of an external stop started by a previous
    // call to Task.BeginExternalStop. EndExternalStop does not wait for t's task
    // goroutine to resume.
    pub fn EndExternalStop(&self) {
        let mut t = self.lock();
        let pidns = t.tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let tg = t.tg.clone();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        t.endStopLocked();
    }
}

impl TaskSet {
    // BeginExternalStop indicates the start of an external stop that applies to
    // all current and future tasks in ts. BeginExternalStop does not wait for
    // task thread to stop.
    pub fn BeginExternalStop(&self) {
        let _l = self.WriteLock();
        let mut ts = self.write();

        ts.stopCount += 1;
        if ts.stopCount <= 0 {
            panic!("BeginExternalStop: Invalid stopCount: {}", ts.stopCount)
        }

        if ts.root.is_none() {
            return;
        }

        let pidns = ts.root.clone().unwrap();
        let threads: Vec<_> = pidns.lock().tids.keys().cloned().collect();
        for t in &threads {
            {
                let tg = t.lock().tg.clone();
                let lock = tg.lock().signalLock.clone();
                let _s = lock.lock();

                t.lock().beginStopLocked();
            }

            t.lock().interrupt();
        }
    }

    // EndExternalStop indicates the end of an external stop started by a previous
    // call to TaskSet.BeginExternalStop. EndExternalStop does not wait for task
    // thread to resume.
    pub fn EndExternalStop(&self) {
        let _l = self.WriteLock();
        let mut ts = self.write();

        ts.stopCount -= 1;
        if ts.stopCount < 0 {
            return;
            //panic!("EndExternalStop: Invalid stopCount: {}", ts.stopCount)
        }

        if ts.root.is_none() {
            return;
        }

        let pidns = ts.root.clone().unwrap();
        let threads: Vec<_> = pidns.lock().tids.keys().cloned().collect();
        for t in &threads {
            let tg = t.lock().tg.clone();
            let lock = tg.lock().signalLock.clone();
            let _s = lock.lock();

            t.lock().endStopLocked();
        }
    }
}
