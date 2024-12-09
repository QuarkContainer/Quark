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

use super::super::super::common::*;
use super::super::super::linux::time::*;
use super::super::super::linux_def::*;
use super::super::kernel::time::*;
use super::super::kernel::timer::timer::Clock;
use super::super::kernel::timer::timer::Setting;
use super::super::kernel::timer::timer::WaitEntryListener;
use super::super::kernel::timer::timer::*;
use super::super::kernel::timer::*;
use super::super::kernel::waiter::*;
use super::super::threadmgr::thread::*;

impl Thread {
    pub fn Interrupted(&self, clear: bool) -> bool {
        let blocker = self.lock().blocker.clone();
        return blocker.Interrupted(clear);
    }
}

impl ThreadInternal {
    pub fn HasSignal(&self) -> bool {
        return self.pendingSignals.HasSignal(self.signalMask);
    }

    pub fn interrupt(&self) {
        self.interruptSelf();
    }

    pub fn interruptSelf(&self) {
        self.blocker.interruptSelf();
    }
}

#[derive(Clone)]
pub struct Blocker {
    pub waiter: Waiter,

    pub timerEntry: WaitEntry,
    pub timerListner: WaitEntryListener,
    pub realBlockTimer: Timer,
    pub monoBlockTimer: Timer,

    pub interruptEntry: WaitEntry,
    pub generalEntry: WaitEntry,
}

impl Drop for Blocker {
    fn drop(&mut self) {
        self.Drop();
    }
}

impl Default for Blocker {
    fn default() -> Self {
        info!("blocker::default");

        let waiter = Waiter::default();

        let timerEntry = waiter.NewWaitEntry(Waiter::TIMER_WAITID, 1);
        let listener = WaitEntryListener::New(&timerEntry);

        let interruptEntry = waiter.NewWaitEntry(Waiter::INTERRUPT_WAITID, 1);
        let generalEntry = waiter.NewWaitEntry(Waiter::GENERAL_WAITID, 0);

        let monoClock = MONOTONIC_CLOCK.clone();
        let monoTimer = Timer::New(
            &monoClock,
            TimerListener::WaitEntryListener(listener.clone()),
        );

        let realClock = REALTIME_CLOCK.clone();
        let realTimer = Timer::New(
            &realClock,
            TimerListener::WaitEntryListener(listener.clone()),
        );

        return Self {
            waiter: waiter,
            timerEntry: timerEntry,
            timerListner: listener,
            realBlockTimer: realTimer,
            monoBlockTimer: monoTimer,
            interruptEntry: interruptEntry,
            generalEntry: generalEntry,
        };
    }
}

impl Blocker {
    pub fn Dummy() -> Self {
        let waiter = Waiter::default();

        let timerEntry = waiter.NewWaitEntry(Waiter::TIMER_WAITID, 1);
        let listener = WaitEntryListener::New(&timerEntry);

        let waiter = Waiter::default();
        let interruptEntry = waiter.NewWaitEntry(Waiter::INTERRUPT_WAITID, 1);
        let generalEntry = waiter.NewWaitEntry(Waiter::GENERAL_WAITID, 0);

        let monoTimer = Timer::Dummy();
        let realTimer = Timer::Dummy();

        return Self {
            waiter: waiter,
            timerEntry: timerEntry,
            timerListner: listener,
            realBlockTimer: realTimer,
            monoBlockTimer: monoTimer,
            interruptEntry: interruptEntry,
            generalEntry: generalEntry,
        };
    }

    pub fn Drop(&mut self) {
        self.monoBlockTimer.Destroy();
        self.realBlockTimer.Destroy();
    }

    pub fn New(taskId: u64) -> Self {
        let waiter = Waiter::New(taskId);

        let timerEntry = waiter.NewWaitEntry(Waiter::TIMER_WAITID, 1);
        let listener = WaitEntryListener::New(&timerEntry);

        let interruptEntry = waiter.NewWaitEntry(Waiter::INTERRUPT_WAITID, 1);
        let generalEntry = waiter.NewWaitEntry(Waiter::GENERAL_WAITID, 0);

        let monoClock = MONOTONIC_CLOCK.clone();
        let monoTimer = Timer::New(
            &monoClock,
            TimerListener::WaitEntryListener(listener.clone()),
        );

        let realClock = REALTIME_CLOCK.clone();
        let realTimer = Timer::New(
            &realClock,
            TimerListener::WaitEntryListener(listener.clone()),
        );

        return Self {
            waiter: waiter,
            timerEntry: timerEntry,
            timerListner: listener,
            realBlockTimer: realTimer,
            monoBlockTimer: monoTimer,
            interruptEntry: interruptEntry,
            generalEntry: generalEntry,
        };
    }

    pub fn BlockWithTimeout(
        &self,
        timer: Timer,
        waitGeneral: bool,
        timeout: Option<Duration>,
    ) -> (Duration, Result<()>) {
        if timeout.is_none() || timeout.unwrap() == core::i64::MAX {
            return (-1, self.block(waitGeneral, None));
        }

        let timeout = timeout.unwrap();
        let adjustTimeout = timeout - 30_000; // 30 us is process time.

        if adjustTimeout <= 0 {
            // if timeout < 30 us, just timeout immediately as 30 us is process time.
            super::super::taskMgr::Yield();
            return (0, self.Check(waitGeneral));
        }

        let clock = timer.Clock();
        let start = clock.Now().0;

        let deadline = if core::i64::MAX - timeout > start {
            // avoid overflow
            Time(start + timeout)
        } else {
            Time(core::i64::MAX)
        };

        let res = self.BlockWithTimer(timer, waitGeneral, Some(deadline));
        match res {
            Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                return (0, Err(Error::SysError(SysErr::ETIMEDOUT)))
            }
            _ => (),
        }

        let end = clock.Now().0;
        let remain = adjustTimeout - (end - start);
        if remain < 0 {
            return (0, res);
        }

        return (remain, res);
    }

    pub fn BlockWithMonoTimeout(
        &self,
        waitGeneral: bool,
        timeout: Option<Duration>,
    ) -> (Duration, Result<()>) {
        let timer = self.GetTimer(MONOTONIC);
        return self.BlockWithTimeout(timer, waitGeneral, timeout);
    }

    pub fn BlockWithRealTimer(&self, waitGeneral: bool, deadline: Option<Time>) -> Result<()> {
        let timer = self.GetTimer(CLOCK_REALTIME);
        return self.BlockWithTimer(timer, waitGeneral, deadline);
    }

    pub fn BlockWithMonoTimer(&self, waitGeneral: bool, deadline: Option<Time>) -> Result<()> {
        let timer = self.GetTimer(CLOCK_MONOTONIC);
        return self.BlockWithTimer(timer, waitGeneral, deadline);
    }

    pub fn BlockWithTimer(
        &self,
        timer: Timer,
        waitGeneral: bool,
        deadline: Option<Time>,
    ) -> Result<()> {
        if deadline.is_none() {
            return self.block(waitGeneral, None);
        }

        let deadline = deadline.unwrap();
        timer.Swap(&Setting {
            Enabled: true,
            Next: deadline,
            Period: 0,
        });

        let err = self.block(waitGeneral, Some(timer));

        self.timerEntry.Clear();
        return err;
    }

    pub fn BlockInterrupt(&self) -> Result<()> {
        self.waiter.Wait(0b010);
        self.SleepFinish(false);
        return Err(Error::SysError(SysErr::ERESTARTNOHAND));
    }

    pub fn GetTimerWithClock(&self, clock: &Clock) -> Timer {
        return Timer::New(
            clock,
            TimerListener::WaitEntryListener(self.timerListner.clone()),
        );
    }

    pub fn GetTimer(&self, clockId: i32) -> Timer {
        return match clockId {
            CLOCK_REALTIME => self.realBlockTimer.clone(),
            CLOCK_MONOTONIC => self.monoBlockTimer.clone(),
            _ => panic!("BlockWithTimer invalid clockid {}", clockId),
        };
    }

    // check whether  the generalEntry or the interrupt ready
    // it is used for the scenario that the timeout is zero
    pub fn Check(&self, waitGeneral: bool) -> Result<()> {
        let bitmap = if waitGeneral { 0x011 } else { 0x010 };

        let id = self.waiter.Check(bitmap);
        match id {
            None => {
                self.SleepFinish(true);
                super::super::taskMgr::Yield();
                return Err(Error::SysError(SysErr::ETIMEDOUT));
            }
            Some(id) => {
                if id == Waiter::GENERAL_WAITID {
                    self.SleepFinish(true);
                    self.waiter.lock().bitmap &= !(1 << Waiter::GENERAL_WAITID);
                    return Ok(());
                } else {
                    //interrutpted
                    self.SleepFinish(false);
                    //self.waiter.lock().bitmap &= !(1<<Waiter::INTERRUPT_WAITID);
                    return Err(Error::ErrInterrupted);
                }
            }
        }
    }

    pub fn block(&self, waitGeneral: bool, waitTimer: Option<Timer>) -> Result<()> {
        self.SleepStart();

        let mut mask = 0b010; // interrupt is always enabled

        if waitTimer.is_some() {
            mask |= 0b100;
        }

        if waitGeneral {
            mask |= 0b001;
        }

        let id = self.waiter.Wait(mask);
        match id {
            Waiter::GENERAL_WAITID => {
                self.SleepFinish(true);
                match waitTimer {
                    Some(timer) => {
                        timer.Cancel();
                    }
                    _ => (),
                }
                self.waiter.lock().bitmap &= !(1 << Waiter::GENERAL_WAITID);
                return Ok(());
            }
            Waiter::TIMER_WAITID => {
                super::super::taskMgr::Yield();
                self.SleepFinish(true);
                self.waiter.lock().bitmap &= !(1 << Waiter::TIMER_WAITID);
                return Err(Error::SysError(SysErr::ETIMEDOUT));
            }
            Waiter::INTERRUPT_WAITID => {
                self.SleepFinish(false);
                match waitTimer {
                    Some(timer) => {
                        timer.Cancel();
                    }
                    _ => (),
                }
                return Err(Error::ErrInterrupted);
            }
            _ => {
                panic!("task block invalid block id {}", id)
            }
        }
    }

    // block on both generalentry and interrupt
    pub fn BlockGeneral(&self) -> Result<()> {
        let id = self.waiter.Wait(0b011);

        if id == Waiter::GENERAL_WAITID {
            self.waiter.lock().bitmap &= !(1 << Waiter::GENERAL_WAITID);
            return Ok(());
        } else {
            //interrutpted
            self.SleepFinish(false);
            return Err(Error::ErrInterrupted);
        }
    }

    // block on general entry
    pub fn BlockGeneralOnly(&self) {
        self.waiter.Wait(0b001);
        self.waiter.lock().bitmap &= !(1 << Waiter::GENERAL_WAITID);

        return;
    }

    pub fn SleepStart(&self) {}

    pub fn SleepFinish(&self, success: bool) {
        if !success {
            self.interruptSelf()
        }
    }

    pub fn Interrupted(&self, clear: bool) -> bool {
        return self.waiter.TryWait(&self.interruptEntry, clear);
    }

    pub fn interruptSelf(&self) {
        self.interruptEntry.Notify(1);
    }
}
