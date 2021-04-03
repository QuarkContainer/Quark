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
use core::fmt;

use super::super::SignalDef::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::linux::time::*;
use super::super::threadmgr::thread::*;
use super::timer::timer::Clock;
use super::timer::timer;

#[derive(Default)]
pub struct IntervalTimerInternal {
    pub timer: Option<timer::Timer>,

    // If target is not nil, it receives signo from timer expirations. If group
    // is true, these signals are thread-group-directed. These fields are
    // immutable.
    pub target: Option<Thread>,
    pub signo: Signal,
    pub id: TimerID,
    pub sigval: u64,
    pub group: bool,

    // If sigpending is true, a signal to target is already queued, and timer
    // expirations should increment overrunCur instead of sending another
    // signal. sigpending is protected by target's signal mutex. (If target is
    // nil, the timer will never send signals, so sigpending will be unused.)
    pub sigpending: bool,

    // If sigorphan is true, timer's setting has been changed since sigpending
    // last became true, such that overruns should no longer be counted in the
    // pending signals si_overrun. sigorphan is protected by target's signal
    // mutex.
    pub sigorphan: bool,

    // overrunCur is the number of overruns that have occurred since the last
    // time a signal was sent. overrunCur is protected by target's signal
    // mutex.
    pub overrunCur: u64,

    // Consider the last signal sent by this timer that has been dequeued.
    // overrunLast is the number of overruns that occurred between when this
    // signal was sent and when it was dequeued. Equivalently, overrunLast was
    // the value of overrunCur when this signal was dequeued. overrunLast is
    // protected by target's signal mutex.
    pub overrunLast: u64,
}

impl fmt::Debug for IntervalTimerInternal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntervalTimerInternal")
            .field("signo", &self.signo)
            .field("id", &self.id)
            .finish()
    }
}

impl IntervalTimerInternal {
    pub fn Timer(&self) -> timer::Timer {
        return self.timer.clone().unwrap();
    }

    pub fn Target(&self) -> Thread {
        return self.target.clone().unwrap();
    }

    pub fn timerSettingChanged(&mut self) {
        if self.target.is_none() {
            return
        }

        let target = self.target.clone().unwrap();
        let tg = target.ThreadGroup();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.ReadLock();

        let lock = tg.lock().signalLock.clone();
        let _l = lock.lock();

        self.sigorphan = true;
        self.overrunCur = 0;
        self.overrunLast = 0;
    }

    pub fn updateDequeuedSignalLocked(&mut self, si: &mut SignalInfo) {
        self.sigpending = false;
        if self.sigorphan {
            return
        }

        self.overrunLast = self.overrunCur;
        self.overrunCur = 0;
        si.SigTimer().overrun = saturateI32FromU64(self.overrunLast);
    }

    pub fn signalRejectedLocked(&mut self) {
        self.sigpending = false;
        if self.sigorphan {
            return
        }

        self.overrunCur += 1;
    }
}

#[derive(Debug, Clone)]
pub struct IntervalTimer(Arc<Mutex<IntervalTimerInternal>>);

impl Deref for IntervalTimer {
    type Target = Arc<Mutex<IntervalTimerInternal>>;

    fn deref(&self) -> &Arc<Mutex<IntervalTimerInternal>> {
        &self.0
    }
}

impl timer::TimerListener for IntervalTimer {
    fn Notify(&self, exp: u64) {
        let mut it = self.lock();

        if it.target.is_none() {
            return
        }

        let target = it.target.clone().unwrap();
        let tg = target.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.ReadLock();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        if it.sigpending {
            it.overrunCur += exp;
            return
        }

        // sigpending must be set before sendSignalTimerLocked() so that it can be
        // unset if the signal is discarded (in which case sendSignalTimerLocked()
        // will return nil).
        it.sigpending = true;
        it.sigorphan = false;
        it.overrunCur += exp - 1;

        let mut si = SignalInfo {
            Signo: it.signo.0,
            Code: SignalInfo::SIGNAL_INFO_TIMER,
            ..Default::default()
        };

        let timer = si.SigTimer();
        timer.tid = it.id;
        timer.sigval = it.sigval;

        // si_overrun is set when the signal is dequeued.
        let err = it.Target().sendSignalTimerLocked(&si, it.group, Some(self.clone()));
        match err {
            Err(_) => {
                it.signalRejectedLocked();
            }
            _ => (),
        }
    }

    fn Destroy(&self) {}
}

impl IntervalTimer {
    pub fn New(id: TimerID, sigval: u64) -> Self {
        let internal = IntervalTimerInternal {
            id: id,
            sigval: sigval,
            ..Default::default()
        };

        return Self(Arc::new(Mutex::new(internal)))
    }

    pub fn DestroyTimer(&self) {
        let mut t = self.lock();
        t.Timer().Destroy();
        t.timerSettingChanged();
        t.timer = None;
    }

    pub fn PauseTimer(&self) {
        self.lock().Timer().Pause();
    }

    pub fn ResumeTimer(&self) {
        self.lock().Timer().Resume();
    }
}

fn saturateI32FromU64(x: u64) -> i32 {
    if x > core::i32::MAX as u64 {
        return core::i32::MAX
    }

    return x as i32;
}

impl Thread {
    pub fn IntervalTimerCreate(&self, c: &Clock, sigev: &mut Sigevent) -> Result<TimerID> {
        let tg = self.lock().tg.clone();
        let mut tg = tg.lock();

        let mut id: TimerID;
        let end = tg.nextTimerID;
        loop {
            id = tg.nextTimerID;
            let ok = tg.timers.contains_key(&id);
            tg.nextTimerID += 1;
            if tg.nextTimerID < 0 {
                tg.nextTimerID = 0;
            }

            if !ok {
                break;
            }

            if tg.nextTimerID == end {
                return Err(Error::SysError(SysErr::EAGAIN))
            }
        }

        //todo: fix this
        // "The implementation of the default case where evp [sic] is NULL is
        // handled inside glibc, which invokes the underlying system call with a
        // suitably populated sigevent structure." - timer_create(2). This is
        // misleading; the timer_create syscall also handles a NULL sevp as
        // described by the man page
        // (kernel/time/posix-timers.c:sys_timer_create(), do_timer_create()). This
        // must be handled here instead of the syscall wrapper since sigval is the
        // timer ID, which isn't available until we allocate it in this function.

        //if sigev is none

        let it = IntervalTimer::New(id, sigev.Value);

        match sigev.Notify {
            SIGEV_NONE => (),
            SIGEV_SIGNAL | SIGEV_THREAD => {
                it.lock().target = tg.leader.Upgrade();
                it.lock().group = true;
            }
            SIGEV_THREAD_ID => {
                let pidns = tg.pidns.clone();

                {
                    let owner = pidns.lock().owner.clone();
                    let _r = owner.ReadLock();

                    match pidns.lock().tasks.get(&sigev.Tid).clone() {
                        None => return Err(Error::SysError(SysErr::EINVAL)),
                        Some(t) => it.lock().target = Some(t.clone()),
                    }
                };
            }
            _ => return Err(Error::SysError(SysErr::EINVAL)),
        }

        if sigev.Notify != SIGEV_NONE {
            it.lock().signo = Signal(sigev.Signo);
            if !it.lock().signo.IsValid() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        it.lock().timer = Some(timer::Timer::New(CLOCK_MONOTONIC, c, &Arc::new(it.clone())));
        tg.timers.insert(id, it);
        return Ok(id)
    }

    // IntervalTimerDelete implements timer_delete(2).
    pub fn IntervalTimerDelete(&self, id: TimerID) -> Result<()> {
        let tg = self.lock().tg.clone();
        let mut tg = tg.lock();

        let it = match tg.timers.remove(&id) {
            None => {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            Some(it) => it,
        };

        it.DestroyTimer();
        return Ok(())
    }

    // IntervalTimerSettime implements timer_settime(2).
    pub fn IntervalTimerSettime(&self, id: TimerID, its: &Itimerspec, abs: bool) -> Result<Itimerspec> {
        let tg = self.lock().tg.clone();
        let tg = tg.lock();

        let it = match tg.timers.get(&id) {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(ref it) => it.clone(),
        };

        let timer = it.lock().timer.clone().unwrap();
        let clock = timer.Clock();
        let newS = timer::Setting::FromItimerspec(its, abs, &clock)?;

        let (tm, oldS) = timer.SwapAnd(&newS, || {
            it.lock().timerSettingChanged()
        });
        let its = timer::ItimerspecFromSetting(tm, oldS);
        return Ok(its)
    }

    // IntervalTimerGettime implements timer_gettime(2).
    pub fn IntervalTimerGettime(&self, id: TimerID) -> Result<Itimerspec> {
        let tg = self.lock().tg.clone();
        let tg = tg.lock();

        match tg.timers.get(&id) {
            None => {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            Some(ref it) => {
                let (tm, s) = it.lock().timer.clone().unwrap().Get();
                let its = timer::ItimerspecFromSetting(tm, s);
                return Ok(its)
            }
        }
    }

    // IntervalTimerGetoverrun implements timer_getoverrun(2).
    //
    // Preconditions: The caller must be running on the task context.
    pub fn IntervalTimerGetoverrun(&self, id: TimerID) -> Result<i32> {
        let tg = self.lock().tg.clone();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        let tglock = tg.lock();
        match tglock.timers.get(&id).clone() {
            None => {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            Some(it) => {
                return Ok(saturateI32FromU64(it.lock().overrunLast))
            }
        }
    }
}