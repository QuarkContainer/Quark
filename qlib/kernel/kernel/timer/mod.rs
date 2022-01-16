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

pub mod sampler;
pub mod parameters;
pub mod calibratedClock;
pub mod timekeeper;
pub mod timer;
pub mod timer_store;

//pub use self::raw_timer::*;

use self::timekeeper::*;
use self::timer_store::*;
use self::timer::*;
use super::super::super::singleton::*;
use super::super::super::object_ref::*;

pub static TIME_KEEPER : Singleton<TimeKeeper> = Singleton::<TimeKeeper>::New();
pub static REALTIME_CLOCK : Singleton<Clock> = Singleton::<Clock>::New();
pub static MONOTONIC_CLOCK : Singleton<Clock> = Singleton::<Clock>::New();
pub static TIMER_STORE : Singleton<TimerStore> = Singleton::<TimerStore>::New();

pub unsafe fn InitSingleton() {
    TIME_KEEPER.Init(TimeKeeper::default());
    REALTIME_CLOCK.Init(TIME_KEEPER.NewClock(REALTIME));
    MONOTONIC_CLOCK.Init(TIME_KEEPER.NewClock(MONOTONIC));
    TIMER_STORE.Init(TimerStore::default());
}

pub struct TimerUpdater {}

impl TimerListener for TimerUpdater {
    fn Notify(&self, _exp: u64) {
        TIME_KEEPER.write().Update();
    }

    fn Destroy(&self) {}
}

pub fn InitTimeKeeper(vdsoParamPageAddr: u64) {
    TIME_KEEPER.Initialization(vdsoParamPageAddr)
}

pub fn GetVDSOParamPageAddr() -> u64 {
    return TIME_KEEPER.read().params.GetParamPageAddr();
}

pub fn RealNow() -> i64 {
    return TIME_KEEPER.GetTime(REALTIME).expect("RealNow fail");
}

pub fn MonotonicNow() -> i64 {
    return TIME_KEEPER.GetTime(MONOTONIC).expect("MonotonicNow fail");
}

pub fn Timeout() {
    TIMER_STORE.Trigger();
}

pub type ClockID = i32;

pub const REALTIME: ClockID = 0;
pub const MONOTONIC: ClockID = 1;

pub type TimerKeeperRef = ObjectRef<TimeKeeper>;