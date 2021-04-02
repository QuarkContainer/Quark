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

use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;

use super::super::super::qlib::linux::time::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::timer::*;
use super::super::super::IOURING;
use super::super::super::task::*;
use super::timermgr::*;

pub trait Notifier: Sync + Send {
    fn Timeout(&self);
    fn Reset(&self);
}

#[derive(Clone)]
pub struct DummyNotifier {}

impl Notifier for DummyNotifier {
    fn Timeout(&self) {}

    fn Reset(&self) {}
}

#[derive(Eq, PartialEq, Clone, Copy)]
pub enum TimerState {
    Expired,
    Running,
    Stopped,
}

impl Default for TimerState {
    fn default() -> Self {
        return Self::Stopped;
    }
}

pub struct RawTimerInternal {
    pub Id: u64,
    pub ClockId: i32,
    pub Expire: Time,
    pub Timer: Timer,
    pub State: TimerState,
    pub SeqNo: u64,
    pub TM: TimerMgr,
    pub userData: u64,
}

#[derive(Clone)]
pub struct RawTimer(Arc<Mutex<RawTimerInternal>>);

impl Drop for RawTimer {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            self.Drop();
        }
    }
}

impl Deref for RawTimer {
    type Target = Arc<Mutex<RawTimerInternal>>;

    fn deref(&self) -> &Arc<Mutex<RawTimerInternal>> {
        &self.0
    }
}

impl RawTimer {
    pub fn New(clockId: i32, id: u64, tm: &TimerMgr, timer: &Timer) -> Self {
        let internal = RawTimerInternal {
            Id: id,
            ClockId: clockId,
            Expire: Time(0),
            Timer: timer.clone(),
            State: TimerState::default(),
            SeqNo: 0,
            TM: tm.clone(),
            userData: 0,
        };

        return Self(Arc::new(Mutex::new(internal)))
    }

    // Stop prevents the Timer from firing.
    // It returns true if the call stops the timer, false if the timer has already
    // expired or been stopped.
    // Stop does not close the channel, to prevent a read from the channel succeeding
    // incorrectly.
    pub fn Stop(&self) -> bool {
        let (state, userData) = {
            let mut t = self.lock();
            let state = t.State;
            t.State = TimerState::Stopped;
            (state, t.userData)
        };

        // we need to call the TimerRemove out of lock to avoid deadlock
        if state == TimerState::Running {
            let task = Task::Current();
            IOURING.TimerRemove(task, userData);
        }

        return false;
    }

    // Reset changes the timer to expire after duration d.
    // It returns true if the timer had been active, false if the timer had
    // expired or been stopped.
    pub fn Reset(&self, expire: Time) -> bool {
        assert!(expire.0 >= 0, "Timer::Reset get negtive expire");
        let mut t = self.lock();

        let task = Task::Current();

        if expire.0 == 0 { // cancel the timer
            if t.State != TimerState::Running {
                return false; //one out of data fire.
            }

            t.SeqNo += 1;

            //HostSpace::StopTimer(t.ClockId, t.Id);

            IOURING.TimerRemove(task, t.userData);
            return true;
        }

        let now = if t.ClockId == CLOCK_MONOTONIC {
            MonotonicNow()
        } else {
            RealNow()
        };

        let mut delta = expire.0 - now ;
        if delta <= 0 {
            delta = 0;
        }

        t.Timer.Reset();
        t.Expire = Time(delta);
        t.State = TimerState::Running;
        t.SeqNo += 1;

        let userData = IOURING.Timeout(task, t.Id, t.SeqNo, delta) as u64;
        t.userData = userData;
        return false;
    }

    pub fn Fire(&self, SeqNo: u64) {
        let timer = {
            let mut t = self.lock();
            if SeqNo != t.SeqNo || t.State != TimerState::Running {
                return; //one out of data fire.
            }

            t.State = TimerState::Expired;
            t.Timer.clone()
        };

        //t.WaitEntry.Notify(1);
        timer.Timeout();
    }

    pub fn Drop(&mut self) {
        self.Stop();
        let tm = self.lock().TM.clone();
        tm.RemoveTimer(self);
    }
}

