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

use crate::qlib::mutex::*;
use alloc::sync::Arc;
use core::ops::Deref;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::Ordering;

use super::super::super::super::common::*;
use super::super::super::super::linux::time::*;
use super::super::super::kernel::time::*;
//use super::super::super::super::perf_tunning::*;
use super::super::vdso::*;
use super::calibratedClock::*;
use super::timer::Clock;
use super::timer::*;
use super::*;

use crate::GuestHostSharedAllocator;
use crate::GUEST_HOST_SHARED_ALLOCATOR;

#[derive(Clone)]
pub struct TimeKeeper(Arc<QRwLock<TimeKeeperInternal>, GuestHostSharedAllocator>);

impl Default for TimeKeeper {
    fn default() -> Self {
        return TimeKeeper(Arc::new_in(
            QRwLock::new(TimeKeeperInternal::default()),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
    }
}

impl Deref for TimeKeeper {
    type Target = Arc<QRwLock<TimeKeeperInternal>, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QRwLock<TimeKeeperInternal>, GuestHostSharedAllocator> {
        &self.0
    }
}

impl TimeKeeper {
    pub fn Initialization(&self, vdsoParamPageAddr: u64) {
        {
            let mut internal = self.write();
            internal.Init(vdsoParamPageAddr);
        }
        let timer = Timer::Period(
            &MONOTONIC_CLOCK,
            TimerListener::TimerUpdater(TimerUpdater {}),
            1 * SECOND,
        );

        {
            let mut internal = self.write();
            internal.timer = Some(timer);
        }
    }

    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn NewClock(&self, clockId: ClockID) -> Clock {
        let c = TimeKeeperClock {
            tk: self.clone(),
            c: clockId,
        };

        return Clock::TimeKeeperClock(Arc::new_in(c,GUEST_HOST_SHARED_ALLOCATOR));
    }

    pub fn Update(&self) {
        self.write().Update();
    }

    pub fn GetTime(&self, c: ClockID) -> Result<i64> {
        return self.read().GetTime(c);
    }

    pub fn BootTime(&self) -> Time {
        return self.read().BootTime();
    }
}

pub struct TimeKeeperInternal {
    // clocks are the clock sources.
    pub clocks: CalibratedClocks,

    // bootTime is the realtime when the system "booted". i.e., when
    // SetClocks was called in the initial (not restored) run.
    pub bootTime: Time,

    // monotonicOffset is the offset to apply to the monotonic clock output
    // from clocks.
    //
    // It is set only once, by SetClocks.
    pub monotonicOffset: i64,

    // params manages the parameter page.
    pub params: VDSOParamPage,

    pub inited: AtomicBool,

    pub timer: Option<Timer>,
}

impl Default for TimeKeeperInternal {
    fn default() -> Self {
        let clocks = CalibratedClocks::New();

        let res = Self {
            clocks: clocks,
            bootTime: Time::default(),
            monotonicOffset: 0,
            params: VDSOParamPage::default(),
            inited: AtomicBool::new(false),
            timer: None,
        };

        return res;
    }
}

impl TimeKeeperInternal {
    pub fn Init(&mut self, vdsoParamPageAddr: u64) {
        self.params.SetParamPageAddr(vdsoParamPageAddr);

        // Compute the offset of the monotonic clock from the base Clocks.
        //
        let wantMonotonic = 0;

        let nowMonotonic = self
            .clocks
            .GetTime(MONOTONIC)
            .expect("Unable to get current monotonic time");
        let nowRealtime = self
            .clocks
            .GetTime(REALTIME)
            .expect("Unable to get current realtime");

        self.monotonicOffset = wantMonotonic - nowMonotonic;
        self.bootTime = Time::FromNs(nowRealtime);
        self.inited.store(true, Ordering::SeqCst);
        self.Update();
    }

    pub fn MonotonicFrequency(&self) -> u64 {
        return self.params.vdsoParams.monotonicFrequency;
    }

    pub fn Update(&mut self) {
        //PerfPrint();
        //super::super::super::perflog::THREAD_COUNTS.lock().Print(true);
        //super::super::super::AllocatorPrint();

        assert!(self.inited.load(Ordering::Relaxed), "TimeKeeper not inited");
        let (monotonicParams, monotonicOk, realtimeParams, realtimeOk) = self.clocks.Update();

        let mut p: VdsoParams = VdsoParams::default();
        if monotonicOk {
            p.monotonicReady = 1;
            p.monotonicBaseCycles = monotonicParams.BaseCycles;
            p.monotonicBaseRef = monotonicParams.BaseRef + self.monotonicOffset;
            p.monotonicFrequency = monotonicParams.Frequency;
        }

        //error!("TimeKeeperInternal::Update monotonicParams is {:?}", &monotonicParams);

        if realtimeOk {
            p.realtimeReady = 1;
            p.realtimeBaseCycles = realtimeParams.BaseCycles;
            p.realtimeBaseRef = realtimeParams.BaseRef;
            p.realtimeFrequency = realtimeParams.Frequency;
        }

        match self.params.Write(&p) {
            Err(err) => info!("Unable to update VDSO parameter page: {:?}", err),
            _ => (),
        }
    }

    // GetTime returns the current time in nanoseconds.
    pub fn GetTime(&self, c: ClockID) -> Result<i64> {
        assert!(self.inited.load(Ordering::Relaxed), "TimeKeeper not inited");
        match self.clocks.GetTime(c) {
            Err(e) => return Err(e),
            Ok(mut now) => {
                if c == MONOTONIC {
                    now += self.monotonicOffset;
                }

                return Ok(now);
            }
        }
    }

    // BootTime returns the system boot real time.
    pub fn BootTime(&self) -> Time {
        assert!(self.inited.load(Ordering::Relaxed), "TimeKeeper not inited");
        return self.bootTime;
    }
}

#[derive(Clone)]
pub struct TimeKeeperClock {
    pub tk: TimeKeeper,
    pub c: ClockID,
}

impl TimeKeeperClock {
    pub fn Now(&self) -> Time {
        let now = self.tk.GetTime(self.c).expect("timekeeperClock Now fail");
        return Time::FromNs(now);
    }

    pub fn WallTimeUntil(&self, t: Time, now: Time) -> Duration {
        return t.Sub(now);
    }
}
