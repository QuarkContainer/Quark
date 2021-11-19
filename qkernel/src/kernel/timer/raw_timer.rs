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
use ::qlib::mutex::*;
use core::ops::Deref;

use super::super::super::kernel::timer::*;
use super::timer_store::*;
use super::TIMER_STORE;
use super::super::super::uid::*;

pub trait Notifier: Sync + Send {
    fn Timeout(&self) -> i64;
    fn Reset(&self);
}

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
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
    pub Expire: i64,
    pub Timer: Timer,
    pub State: TimerState,
}

#[derive(Clone)]
pub struct RawTimer(Arc<QMutex<RawTimerInternal>>);

impl Drop for RawTimer {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            self.Drop();
        }
    }
}

impl Deref for RawTimer {
    type Target = Arc<QMutex<RawTimerInternal>>;

    fn deref(&self) -> &Arc<QMutex<RawTimerInternal>> {
        &self.0
    }
}

impl RawTimerInternal {
    pub fn TimerUnit(&self) -> TimerUnit {
        return TimerUnit {
            timerId: self.Id,
            expire: self.Expire,
        }
    }
}

impl RawTimer {
    pub fn New(timer: &Timer) -> Self {
        let id = NewUID();
        let internal = RawTimerInternal {
            Id: id,
            Timer: timer.clone(),
            State: TimerState::default(),
            Expire: 0,
        };

        return Self(Arc::new(QMutex::new(internal)))
    }

    // Stop prevents the Timer from firing.
    // It returns true if the call stops the timer, false if the timer has already
    // expired or been stopped.
    pub fn Stop(&self) -> bool {
        let state = self.lock().State;
        if state != TimerState::Running {
            return false
        }

        TIMER_STORE.CancelTimer(self);
        self.lock().State = TimerState::Stopped;
        return true;
    }

    // Reset changes the timer to expire after duration d.
    // It returns true if the timer had been active, false if the timer had
    // expired or been stopped.
    pub fn Reset(&self, timeout: i64) -> bool {
        if timeout == 0 {
            return self.Stop();
        }

        assert!(timeout > 0, "Timer::Reset get negtive delta");

        TIMER_STORE.ResetTimer(self, timeout);
        self.lock().State = TimerState::Running;
        return false;
    }

    pub fn Fire(&self, ts: &mut TimerStoreIntern) {
        let timer = {
            let mut t = self.lock();

            t.State = TimerState::Expired;
            t.Timer.clone()
        };

        let delta = timer.Timeout();
        if delta > 0 {
            ts.ResetTimer(self, delta);
            self.lock().State = TimerState::Running;
        }
    }

    pub fn Drop(&mut self) {
        self.Stop();
    }
}

