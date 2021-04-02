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

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::linux::time::*;
use super::super::super::SignalDef::*;
use super::super::super::task::*;
use super::super::super::threadmgr::thread_group::*;
use super::super::waiter::*;
use super::super::timer::raw_timer::*;
use super::super::time::*;
use super::*;

// ClockEventSet occurs when a Clock undergoes a discontinuous change.
pub const CLOCK_EVENT_SET: EventMask = 1 << 0;

// ClockEventRateIncrease occurs when the rate at which a Clock advances
// increases significantly, such that values returned by previous calls to
// Clock.WallTimeUntil may be too large.
pub const CLOCK_EVENT_RATE_INCREASE: EventMask = 1 << 0;
pub const TIMER_TICK_EVENTS: EventMask = CLOCK_EVENT_SET | CLOCK_EVENT_RATE_INCREASE;

pub trait Clock: Sync + Send {
    // Now returns the current time in nanoseconds according to the Clock.
    fn Now(&self) -> Time;

    // WallTimeUntil returns the estimated wall time until Now will return a
    // value greater than or equal to t, given that a recent call to Now
    // returned now. If t has already passed, WallTimeUntil may return 0 or a
    // negative value.
    //
    // WallTimeUntil must be abstract to support Clocks that do not represent
    // wall time (e.g. thread group execution timers). Clocks that represent
    // wall times may embed the WallRateClock type to obtain an appropriate
    // trivial implementation of WallTimeUntil.
    //
    // WallTimeUntil is used to determine when associated Timers should next
    // check for expirations. Returning too small a value may result in
    // spurious Timer goroutine wakeups, while returning too large a value may
    // result in late expirations. Implementations should usually err on the
    // side of underestimating.
    fn WallTimeUntil(&self, t: Time, now: Time) -> Duration;
}

pub struct ClockEventsQueue {
    pub queue: Queue,
}

impl Waitable for ClockEventsQueue {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        return 0
    }
}

pub trait TimerListener: Sync + Send {
    // Notify is called when its associated Timer expires. exp is the number of
    // expirations.
    //
    // Notify is called with the associated Timer's mutex locked, so Notify
    // must not take any locks that precede Timer.mu in lock order.
    //
    // Preconditions: exp > 0.
    fn Notify(&self, exp: u64);

    // Destroy is called when the timer is destroyed.
    fn Destroy(&self);
}

pub struct DummyTimerListener {}

impl TimerListener for DummyTimerListener {
    fn Notify(&self, _exp: u64) {}
    fn Destroy(&self) {}
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Setting {
    // Enabled is true if the timer is running.
    pub Enabled: bool,

    // Next is the time in nanoseconds of the next expiration.
    pub Next: Time,

    // Period is the time in nanoseconds between expirations. If Period is
    // zero, the timer will not automatically restart after expiring.
    //
    // Invariant: Period >= 0.
    pub Period: Duration,
}

impl Setting {
    pub fn FromSpec(value: Duration, interval: Duration, c: &Arc<Clock>) -> Result<Self> {
        return Self::FromSpecAt(value, interval, c.Now())
    }

    pub fn FromSpecAt(value: Duration, interval: Duration, now: Time) -> Result<Self> {
        if value < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if value == 0 {
            return Ok(Self {
                Enabled: false,
                Next: Time(0),
                Period: interval,
            })
        }

        return Ok(Self {
            Enabled: true,
            Next: now.Add(value),
            Period: interval,
        })
    }

    pub fn FromAbsSpec(value: Time, interval: Duration) -> Result<Self> {
        if value.Before(ZERO_TIME) {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if value.IsZero() {
            return Ok(Self {
                Enabled: false,
                Next: Time(0),
                Period: interval,
            })
        }

        return Ok(Self {
            Enabled: true,
            Next: value,
            Period: interval,
        })
    }

    pub fn FromItimerspec(its: &Itimerspec, abs: bool, c: &Arc<Clock>) -> Result<Self> {
        if abs {
            return Self::FromAbsSpec(Time::FromTimespec(&its.Value), its.Interval.ToDuration()?);
        }

        return Self::FromSpec(its.Value.ToDuration()?, its.Interval.ToDuration()?, c)
    }

    // At returns an updated Setting and a number of expirations after the
    // associated Clock indicates a time of now.
    //
    // Settings may be created by successive calls to At with decreasing
    // values of now (i.e. time may appear to go backward). Supporting this is
    // required to support non-monotonic clocks, as well as allowing
    // Timer.clock.Now() to be called without holding Timer.mu.
    pub fn At(mut self, now: Time) -> (Self, u64) {
        if !self.Enabled {
            return (self, 0)
        }

        if self.Next.After(now) {
            return (self, 0)
        }

        if self.Period == 0 {
            self.Enabled = false;
            return (self, 1)
        }

        let exp = 1 + (now.Sub(self.Next) as u64) / (self.Period as u64);
        self.Next = self.Next.Add(self.Period * exp as i64);
        return (self, exp)
    }
}

pub fn SpecFromSetting(now: Time, s: Setting) -> (Duration, Duration) {
    if !s.Enabled {
        return (0, s.Period)
    }

    return (s.Next.Sub(now), s.Period)
}

pub fn ItimerspecFromSetting(now: Time, s: Setting) -> Itimerspec {
    let (val, iv) = SpecFromSetting(now, s);
    return Itimerspec {
        Interval: Timespec::FromNs(iv),
        Value: Timespec::FromNs(val),
    }
}

// Timer is an optionally-periodic timer driven by sampling a user-specified
// Clock. Timer's semantics support the requirements of Linux's interval timers
// (setitimer(2), timer_create(2), timerfd_create(2)).
//
// Timers should be created using NewTimer and must be cleaned up by calling
// Timer.Destroy when no longer used.
pub struct TimerInternal {
    pub clockId: i32,
    // clock is the time source. clock is immutable.
    pub clock: Arc<Clock>,

    // listener is notified of expirations. listener is immutable.
    pub listener: Arc<TimerListener>,

    // setting is the timer setting. setting is protected by mu.
    pub setting: Setting,

    // paused is true if the Timer is paused. paused is protected by mu.
    pub paused: bool,
    pub kicker: Option<RawTimer>,
}

impl Default for TimerInternal {
    fn default() -> Self {
        return Self {
            clockId: CLOCK_REALTIME,
            clock: REALTIME_CLOCK.clone(),
            listener: Arc::new(DummyTimerListener {}),
            setting: Setting::default(),
            paused: true,
            kicker: None,
        }
    }
}


impl TimerInternal {
    fn NextExpire(&mut self) -> i64 {
        if self.setting.Enabled {
            let now = self.clock.Now();
            let expire = self.setting.Next;
            let mut delta = expire.0 - now.0 ;
            if delta <= 0 {
                delta = 0;
            }

            return delta
        } else {
            return 0
        }
    }

    pub fn Kicker(&self) -> RawTimer {
        return self.kicker.clone().unwrap();
    }
}

#[derive(Clone, Default)]
pub struct Timer(Arc<Mutex<TimerInternal>>);

impl Deref for Timer {
    type Target = Arc<Mutex<TimerInternal>>;

    fn deref(&self) -> &Arc<Mutex<TimerInternal>> {
        &self.0
    }
}

impl Notifier for Timer {
    fn Timeout(&self) -> i64 {
        let mut t = self.lock();

        let now = t.clock.Now();
        if t.paused {
            return 0;
        }

        let (s, exp) = t.setting.At(now);
        t.setting = s;
        if exp > 0 {
            t.listener.Notify(exp)
        }

        return t.NextExpire();
    }

    fn Reset(&self) {}
}

impl Timer {
    pub fn New<C: Clock + 'static, L: TimerListener + 'static>(clockId: i32, clock: &Arc<C>, listener: &Arc<L>) -> Self {
        let internal = TimerInternal {
            clockId: clockId,
            clock: clock.clone(),
            listener: listener.clone(),
            setting: Setting::default(),
            paused: true,
            kicker: None,
        };

        let mut res = Self(Arc::new(Mutex::new(internal)));
        res.Init();
        return res;
    }

    pub fn Period<C: Clock + 'static, L: TimerListener + 'static>(clockId: i32, clock: &Arc<C>, listener: &Arc<L>, duration: Duration) -> Self {
        let internal = TimerInternal {
            clockId: clockId,
            clock: clock.clone(),
            listener: listener.clone(),
            setting: Setting::default(),
            paused: false,
            kicker: None,
        };

        let mut res = Self(Arc::new(Mutex::new(internal)));
        res.Init();

        let now = clock.Now();
        res.Swap(&Setting {
            Enabled: true,
            Period: duration,
            Next: now.Add(duration)
        });

        return res;
    }

    pub fn After<C: Clock + 'static, L: TimerListener + 'static>(clockId: i32, clock: &Arc<C>, listener: &Arc<L>, duration: Duration) -> Self {
        let internal = TimerInternal {
            clockId: clockId,
            clock: clock.clone(),
            listener: listener.clone(),
            setting: Setting::default(),
            paused: false,
            kicker: None,
        };

        let mut res = Self(Arc::new(Mutex::new(internal)));
        res.Init();

        let now = clock.Now();
        res.Swap(&Setting {
            Enabled: true,
            Period: 0,
            Next: now.Add(duration)
        });

        return res;
    }

    fn Init(&mut self) {
        let mut t = self.lock();
        if t.kicker.is_some() {
            return
        }

        t.kicker = Some(NewRawTimer(t.clockId, self));
        t.Kicker().Reset(0);
        t.Kicker().lock().Timer = self.clone();
    }

    pub fn Destroy(&self) {
        let mut t = self.lock();

        t.setting.Enabled = false;
        if t.kicker.is_some() {
            t.Kicker().Drop();
        }

        t.listener.Destroy();
    }

    pub fn Pause(&self) {
        let mut t = self.lock();

        t.paused = true;
        if t.kicker.is_some() {
            t.Kicker().Stop();
        }
    }

    pub fn Resume(&self) {
        let mut t = self.lock();
        if !t.paused {
            return;
        }

        t.paused = false;
        let delta = t.NextExpire();
        t.Kicker().Reset(delta);
    }

    pub fn Cancel(&self) {
        //cancel current runtimer to stop it for unexpired fire
        self.lock().paused = true;
        self.lock().Kicker().Stop();
    }

    // Get returns a snapshot of the Timer's current Setting and the time
    // (according to the Timer's Clock) at which the snapshot was taken.
    //
    // Preconditions: The Timer must not be paused (since its Setting cannot
    // be advanced to the current time while it is paused.)
    pub fn Get(&self) -> (Time, Setting) {
        let mut t = self.lock();
        let now = t.clock.Now();
        if t.paused {
            panic!("Timer.Get called on paused Timer")
        }

        let (s, exp) = t.setting.At(now);
        t.setting = s;
        if exp > 0 {
            t.listener.Notify(exp)
        }

        let delta = t.NextExpire();
        t.Kicker().Reset(delta);
        return (now, s)
    }

    // Swap atomically changes the Timer's Setting and returns the Timer's previous
    // Setting and the time (according to the Timer's Clock) at which the snapshot
    // was taken. Setting s.Enabled to true starts the Timer, while setting
    // s.Enabled to false stops it.
    //
    // Preconditions: The Timer must not be paused.
    pub fn Swap(&self, s: &Setting) -> (Time, Setting) {
        return self.SwapAnd(s, || {});
    }

    // SwapAnd atomically changes the Timer's Setting, calls f if it is not nil,
    // and returns the Timer's previous Setting and the time (according to the
    // Timer's Clock) at which the Setting was changed. Setting s.Enabled to true
    // starts the timer, while setting s.Enabled to false stops it.
    //
    // Preconditions: The Timer must not be paused. f cannot call any Timer methods
    // since it is called with the Timer mutex locked.
    pub fn SwapAnd(&self, s: &Setting, mut f: impl FnMut()) -> (Time, Setting) {
        let mut t = self.lock();
        let now = t.clock.Now();

        let oldS = if !t.paused {
            let (oldS, oldExp) = t.setting.At(now);
            if oldExp > 0 {
                t.listener.Notify(oldExp)
            }

            oldS
        } else {
            t.paused = false;
            let (oldS, _oldExp) = t.setting.At(now);
            oldS
        };

        f();

        let (newS, newExp) = s.At(now);
        t.setting = newS;

        if newExp > 0 {
            t.listener.Notify(newExp);
        }

        let delta = t.NextExpire();
        t.Kicker().Reset(delta);
        return (now, oldS)
    }

    // Atomically invokes f atomically with respect to expirations of t; that is, t
    // cannot generate expirations while f is being called.
    //
    // Preconditions: f cannot call any Timer methods since it is called with the
    // Timer mutex locked.
    pub fn Atomically(&self, mut f: impl FnMut()) {
        let _t = self.lock();
        f();
    }

    pub fn Clock(&self) -> Arc<Clock> {
        return self.lock().clock.clone();
    }
}

#[derive(Clone)]
pub struct WaitEntryListener {
    pub entry: WaitEntry,
}

impl WaitEntryListener {
    pub fn New(e: &WaitEntry) -> Self {
        return Self {
            entry: e.clone(),
        }
    }
}

impl TimerListener for WaitEntryListener {
    fn Notify(&self, _exp: u64) {
        self.entry.Timeout();
    }

    fn Destroy(&self) {}
}

pub struct ITimerRealListener {
    pub tg: ThreadGroupWeak,
}

impl TimerListener for ITimerRealListener {
    fn Notify(&self, _exp: u64) {
        let tg = self.tg.Upgrade().expect("TimerListener::Notify upgrade fail");
        tg.SendSignal(&SignalInfoPriv(Signal::SIGALRM)).expect("TimerListener::Notify fail")
    }

    fn Destroy(&self) {}
}
