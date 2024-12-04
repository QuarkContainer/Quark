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

use crate::qlib::mutex::*;
use alloc::sync::Arc;
use alloc::sync::Weak;
use core::fmt;
use core::ops::Deref;

use super::super::super::super::common::*;
use super::super::super::super::linux::time::*;
use super::super::super::super::linux_def::*;
use super::super::super::fs::timerfd::*;
use super::super::super::task::*;
use super::super::super::threadmgr::task_sched::*;
use super::super::super::threadmgr::thread_group::*;
use super::super::super::uid::*;
use super::super::super::SignalDef::*;
use super::super::posixtimer::*;
use super::super::time::*;
use super::super::timer::TimerUpdater;
use super::super::waiter::*;
use super::timekeeper::*;
use super::timer_store::*;
use super::*;

use crate::qlib::mem::cc_allocator::GuestHostSharedAllocator;
use crate::GUEST_HOST_SHARED_ALLOCATOR;

// ClockEventSet occurs when a Clock undergoes a discontinuous change.
pub const CLOCK_EVENT_SET: EventMask = 1 << 0;

// ClockEventRateIncrease occurs when the rate at which a Clock advances
// increases significantly, such that values returned by previous calls to
// Clock.WallTimeUntil may be too large.
pub const CLOCK_EVENT_RATE_INCREASE: EventMask = 1 << 0;
pub const TIMER_TICK_EVENTS: EventMask = CLOCK_EVENT_SET | CLOCK_EVENT_RATE_INCREASE;

#[derive(Clone)]
pub enum Clock {
    TimeKeeperClock(Arc<TimeKeeperClock,GuestHostSharedAllocator>),
    TaskClock(TaskClock),
    ThreadGroupClock(ThreadGroupClock),
    Dummy,
}

impl Clock {
    // Now returns the current time in nanoseconds according to the Clock.
    pub fn Now(&self) -> Time {
        match self {
            Self::TimeKeeperClock(ref c) => c.Now(),
            Self::TaskClock(ref c) => c.Now(),
            Self::ThreadGroupClock(ref c) => c.Now(),
            Self::Dummy => panic!("Clock::Dummy Now..."),
        }
    }

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
    pub fn WallTimeUntil(&self, t: Time, now: Time) -> Duration {
        match self {
            Self::TimeKeeperClock(ref c) => c.WallTimeUntil(t, now),
            Self::TaskClock(ref c) => c.WallTimeUntil(t, now),
            Self::ThreadGroupClock(ref c) => c.WallTimeUntil(t, now),
            Self::Dummy => panic!("Clock::Dummy WallTimeUntil..."),
        }
    }
}

pub struct ClockEventsQueue {
    pub queue: Queue,
}

impl Waitable for ClockEventsQueue {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        return 0;
    }
}

#[derive(Clone)]
pub enum TimerListener {
    TimerOperations(Arc<TimerOperationsInternal>),
    IntervalTimer(IntervalTimer),
    TimerUpdater(TimerUpdater),
    DummyTimerListener(DummyTimerListener),
    WaitEntryListener(WaitEntryListener),
    ITimerRealListener(Arc<ITimerRealListener>),
    KernelCPUClockTicker(Arc<KernelCPUClockTicker>),
}

impl fmt::Debug for TimerListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimerOperations(_) => f.debug_struct("TimerOperations").finish(),
            Self::IntervalTimer(_) => f.debug_struct("IntervalTimer").finish(),
            Self::TimerUpdater(_) => f.debug_struct("TimerUpdater").finish(),
            Self::DummyTimerListener(_) => f.debug_struct("DummyTimerListener").finish(),
            Self::WaitEntryListener(_) => f.debug_struct("WaitEntryListener").finish(),
            Self::ITimerRealListener(_) => f.debug_struct("ITimerRealListener").finish(),
            Self::KernelCPUClockTicker(_) => f.debug_struct("KernelCPUClockTicker").finish(),
        }
    }
}

impl TimerListener {
    // Notify is called when its associated Timer expires. exp is the number of
    // expirations.
    //
    // Notify is called with the associated Timer's mutex locked, so Notify
    // must not take any locks that precede Timer.mu in lock order.
    //
    // Preconditions: exp > 0.
    pub fn Notify(&self, exp: u64) {
        match self {
            Self::TimerOperations(tl) => tl.Notify(exp),
            Self::IntervalTimer(tl) => tl.Notify(exp),
            Self::TimerUpdater(tl) => tl.Notify(exp),
            Self::DummyTimerListener(tl) => tl.Notify(exp),
            Self::WaitEntryListener(tl) => tl.Notify(exp),
            Self::ITimerRealListener(tl) => tl.Notify(exp),
            Self::KernelCPUClockTicker(tl) => tl.Notify(exp),
        }
    }

    pub fn Destroy(&self) {
        match self {
            Self::TimerOperations(tl) => tl.Destroy(),
            Self::IntervalTimer(tl) => tl.Destroy(),
            Self::TimerUpdater(tl) => tl.Destroy(),
            Self::DummyTimerListener(tl) => tl.Destroy(),
            Self::WaitEntryListener(tl) => tl.Destroy(),
            Self::ITimerRealListener(tl) => tl.Destroy(),
            Self::KernelCPUClockTicker(tl) => tl.Destroy(),
        }
    }
}

pub trait TimerListenerTrait: Sync + Send {
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

#[derive(Clone)]
pub struct DummyTimerListener {}

impl TimerListenerTrait for DummyTimerListener {
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
    pub fn FromSpec(value: Duration, interval: Duration, c: &Clock) -> Result<Self> {
        return Self::FromSpecAt(value, interval, c.Now());
    }

    pub fn FromSpecAt(value: Duration, interval: Duration, now: Time) -> Result<Self> {
        if value < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if value == 0 {
            return Ok(Self {
                Enabled: false,
                Next: Time(0),
                Period: interval,
            });
        }

        return Ok(Self {
            Enabled: true,
            Next: now.Add(value),
            Period: interval,
        });
    }

    pub fn FromAbsSpec(value: Time, interval: Duration) -> Result<Self> {
        if value.Before(ZERO_TIME) {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if value.IsZero() {
            return Ok(Self {
                Enabled: false,
                Next: Time(0),
                Period: interval,
            });
        }

        return Ok(Self {
            Enabled: true,
            Next: value,
            Period: interval,
        });
    }

    pub fn FromItimerspec(its: &Itimerspec, abs: bool, c: &Clock) -> Result<Self> {
        if abs {
            return Self::FromAbsSpec(Time::FromTimespec(&its.Value), its.Interval.ToDuration()?);
        }

        return Self::FromSpec(its.Value.ToDuration()?, its.Interval.ToDuration()?, c);
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
            return (self, 0);
        }

        if self.Next.After(now) {
            return (self, 0);
        }

        if self.Period == 0 {
            self.Enabled = false;
            return (self, 1);
        }

        let exp = 1 + (now.Sub(self.Next) as u64) / (self.Period as u64);
        self.Next = self.Next.Add(self.Period * exp as i64);
        return (self, exp);
    }
}

pub fn SpecFromSetting(now: Time, s: Setting) -> (Duration, Duration) {
    if !s.Enabled {
        return (0, s.Period);
    }

    return (s.Next.Sub(now), s.Period);
}

pub fn ItimerspecFromSetting(now: Time, s: Setting) -> Itimerspec {
    let (val, iv) = SpecFromSetting(now, s);
    return Itimerspec {
        Interval: Timespec::FromNs(iv),
        Value: Timespec::FromNs(val),
    };
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

// Timer is an optionally-periodic timer driven by sampling a user-specified
// Clock. Timer's semantics support the requirements of Linux's interval timers
// (setitimer(2), timer_create(2), timerfd_create(2)).
//
// Timers should be created using NewTimer and must be cleaned up by calling
// Timer.Destroy when no longer used.
pub struct TimerInternal {
    // clock is the time source. clock is immutable.
    pub clock: Clock,

    // listener is notified of expirations. listener is immutable.
    pub listener: TimerListener,

    // setting is the timer setting. setting is protected by mu.
    pub setting: Setting,

    // paused is true if the Timer is paused. paused is protected by mu.
    pub paused: bool,

    // RawTimer
    pub Id: u64,
    pub Expire: i64,
    pub State: TimerState,
}

impl Default for TimerInternal {
    fn default() -> Self {
        let id = NewUID();
        return Self {
            clock: REALTIME_CLOCK.clone(),
            listener: TimerListener::DummyTimerListener(DummyTimerListener {}),
            setting: Setting::default(),
            paused: true,

            Id: id,
            State: TimerState::default(),
            Expire: 0,
        };
    }
}

impl TimerInternal {
    pub fn Dummy() -> Self {
        let id = NewUID();
        let ret = Self {
            clock: Clock::Dummy,
            listener: TimerListener::DummyTimerListener(DummyTimerListener {}),
            setting: Setting::default(),
            paused: true,

            Id: id,
            State: TimerState::default(),
            Expire: 0,
        };

        return ret;
    }

    fn NextExpire(&mut self) -> i64 {
        if self.setting.Enabled {
            let now = self.clock.Now();
            let expire = self.setting.Next;
            let mut delta = self.clock.WallTimeUntil(expire, now);

            if delta < 0 {
                // need to trigger timeout immediately. Add 1000 to trigger the uring timeout
                // this is workaround
                // Todo: fix it
                delta = 10;
            }

            return delta;
        } else {
            return 0;
        }
    }

    pub fn TimerUnit(&self) -> TimerUnit {
        return TimerUnit {
            timerId: self.Id,
            expire: self.Expire,
        };
    }
}

pub struct TimerWeak(Weak<QMutex<TimerInternal>, GuestHostSharedAllocator>);

impl TimerWeak {
    pub fn Upgrade(&self) -> Option<Timer> {
        let f = match self.0.upgrade() {
            None => return None,
            Some(f) => f,
        };

        return Some(Timer(f));
    }
}

#[derive(Clone)]
pub struct Timer(Arc<QMutex<TimerInternal>, GuestHostSharedAllocator>);

impl Default for Timer {
    fn default() -> Self {
        return Timer(Arc::new_in(
            QMutex::new(TimerInternal::default()),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
    }
}

impl Deref for Timer {
    type Target = Arc<QMutex<TimerInternal>, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QMutex<TimerInternal>, GuestHostSharedAllocator> {
        &self.0
    }
}

/*
impl Drop for Timer {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            self.Drop();
        }
    }
}
*/

impl Timer {
    pub fn Dummy() -> Self {
        let ret = Self(Arc::new_in(
            QMutex::new(TimerInternal::Dummy()),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
        return ret;
    }

    pub fn Downgrade(&self) -> TimerWeak {
        return TimerWeak(Arc::downgrade(&self.0));
    }

    fn Timeout(&self) -> i64 {
        let mut t = self.lock();

        let now = t.clock.Now();
        if t.paused {
            return 0;
        }

        // +2000ns as process time
        let (s, exp) = t.setting.At(Time(now.0 + 20000));
        t.setting = s;
        if exp > 0 {
            t.listener.Notify(exp);
        }

        return t.NextExpire();
    }

    pub fn New(clock: &Clock, listener: TimerListener) -> Self {
        let id = NewUID();
        let internal = TimerInternal {
            clock: clock.clone(),
            listener: listener,
            setting: Setting::default(),
            paused: false,

            Id: id,
            State: TimerState::default(),
            Expire: 0,
        };

        let mut res = Self(Arc::new_in(
            QMutex::new(internal),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
        res.Init();
        return res;
    }

    pub fn Period(clock: &Clock, listener: TimerListener, duration: Duration) -> Self {
        let id = NewUID();
        let internal = TimerInternal {
            clock: clock.clone(),
            listener: listener,
            setting: Setting::default(),
            paused: false,

            Id: id,
            State: TimerState::default(),
            Expire: 0,
        };

        let mut res = Self(Arc::new_in(
            QMutex::new(internal),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
        res.Init();

        let now = clock.Now();
        res.Swap(&Setting {
            Enabled: true,
            Period: duration,
            Next: now.Add(duration),
        });

        return res;
    }

    pub fn After(clock: &Clock, listener: TimerListener, duration: Duration) -> Self {
        let id = NewUID();
        let internal = TimerInternal {
            clock: clock.clone(),
            listener: listener,
            setting: Setting::default(),
            paused: false,

            Id: id,
            State: TimerState::default(),
            Expire: 0,
        };

        let mut res = Self(Arc::new_in(
            QMutex::new(internal),
            GUEST_HOST_SHARED_ALLOCATOR,
        ));
        res.Init();

        let now = clock.Now();
        res.Swap(&Setting {
            Enabled: true,
            Period: 0,
            Next: now.Add(duration),
        });

        return res;
    }

    fn Init(&mut self) {
        self.Stop();
    }

    pub fn Destroy(&self) {
        {
            let mut t = self.lock();
            t.setting.Enabled = false;
            t.listener.Destroy();
        }

        self.Drop()
    }

    pub fn Pause(&self) {
        self.lock().paused = true;
        self.Stop();
    }

    pub fn Resume(&self) {
        let delta;
        {
            let mut t = self.lock();
            if !t.paused {
                return;
            }

            t.paused = false;
            delta = t.NextExpire();
        }

        self.Reset(delta);
    }

    pub fn Cancel(&self) {
        //cancel current runtimer to stop it for unexpired fire
        self.lock().paused = true;
        self.Stop();
    }

    // Get returns a snapshot of the Timer's current Setting and the time
    // (according to the Timer's Clock) at which the snapshot was taken.
    //
    // Preconditions: The Timer must not be paused (since its Setting cannot
    // be advanced to the current time while it is paused.)
    pub fn Get(&self) -> (Time, Setting) {
        let now;
        let s;
        let delta;
        {
            let mut t = self.lock();
            now = t.clock.Now();
            if t.paused {
                panic!("Timer.Get called on paused Timer")
            }

            let (setting, exp) = t.setting.At(Time(now.0 + 20000));
            t.setting = setting;
            if exp > 0 {
                t.listener.Notify(exp)
            }

            delta = t.NextExpire();
            s = setting;
        }

        self.Reset(delta);
        return (now, s);
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
        let mut now;
        let oldS;
        let delta;
        {
            let mut t = self.lock();
            now = t.clock.Now();
            now = Time(now.0 + 30000);

            oldS = if !t.paused {
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

            delta = t.NextExpire();
        }

        self.Reset(delta);
        return (now, oldS);
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

    pub fn Clock(&self) -> Clock {
        return self.lock().clock.clone();
    }

    // Stop prevents the Timer from firing.
    // It returns true if the call stops the timer, false if the timer has already
    // expired or been stopped.
    pub fn Stop(&self) -> bool {
        let state = self.lock().State;
        if state != TimerState::Running {
            return false;
        }

        self.lock().State = TimerState::Stopped;
        TIMER_STORE.CancelTimer(self);
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
        self.lock().State = TimerState::Expired;

        let delta = self.Timeout();
        if delta > 0 {
            ts.ResetTimer(self, delta);
            self.lock().State = TimerState::Running;
        }
    }

    pub fn Drop(&self) {
        self.Stop();
    }
}

#[derive(Clone)]
pub struct WaitEntryListener {
    pub entry: WaitEntry,
}

impl WaitEntryListener {
    pub fn New(e: &WaitEntry) -> Self {
        return Self { entry: e.clone() };
    }
}

impl TimerListenerTrait for WaitEntryListener {
    fn Notify(&self, _exp: u64) {
        self.entry.Timeout();
    }

    fn Destroy(&self) {}
}

pub struct ITimerRealListener {
    pub tg: ThreadGroupWeak,
}

impl TimerListenerTrait for ITimerRealListener {
    fn Notify(&self, _exp: u64) {
        let tg = self
            .tg
            .Upgrade()
            .expect("TimerListener::Notify upgrade fail");
        tg.SendSignal(&SignalInfoPriv(Signal::SIGALRM))
            .expect("TimerListener::Notify fail")
    }

    fn Destroy(&self) {}
}
