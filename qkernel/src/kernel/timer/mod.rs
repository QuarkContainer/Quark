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

pub mod raw_timer;
pub mod timermgr;
pub mod sampler;
pub mod parameters;
pub mod calibratedClock;
pub mod timekeeper;
pub mod timer;

pub use self::raw_timer::*;

use lazy_static::lazy_static;

use self::timermgr::*;
use self::timekeeper::*;
use self::timer::*;

lazy_static! {
    static ref TIMER_MGR: TimerMgr = TimerMgr::default();
    pub static ref TIME_KEEPER: TimeKeeper = TimeKeeper::default();
    pub static ref REALTIME_CLOCK: Arc<TimeKeeperClock> = Arc::new(TIME_KEEPER.NewClock(REALTIME));
    pub static ref MONOTONIC_CLOCK: Arc<TimeKeeperClock> = Arc::new(TIME_KEEPER.NewClock(MONOTONIC));
}

pub struct TimerUpdater {}

impl TimerListener for TimerUpdater {
    fn Notify(&self, _exp: u64) {
        TIME_KEEPER.write().Update();
    }

    fn Destroy(&self) {}
}

pub fn InitTimeKeeper(vdsoParamPageAddr: u64) {
    TIME_KEEPER.Init(vdsoParamPageAddr)
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

pub fn DummyTimer() -> RawTimer {
    return TIMER_MGR.DummyTimer();
}

pub fn NewRawTimer<T: Notifier + Clone + 'static>(clockId: i32, notifier: &T) -> RawTimer {
    return TIMER_MGR.NewTimer(clockId, notifier);
}

pub fn RemoveTimer(timer: &RawTimer) {
    TIMER_MGR.RemoveTimer(timer);
}

pub fn FireTimer(timerId: u64, seqNo: u64) {
    //error!("FireTimer timerId is {}, seqNo is {}", timerId, seqNo);
    TIMER_MGR.Fire(timerId, seqNo);
}

pub type ClockID = i32;

pub const REALTIME: ClockID = 0;
pub const MONOTONIC: ClockID = 1;

