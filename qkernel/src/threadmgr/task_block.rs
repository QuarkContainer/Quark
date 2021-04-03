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

use super::super::kernel::waiter::*;
use super::super::kernel::timer::*;
use super::super::kernel::timer::timer::Setting;
use super::super::kernel::timer::timer::Timer;
use super::super::kernel::timer::timer::Clock;
use super::super::kernel::timer::timer::WaitEntryListener;
use super::super::threadmgr::thread::*;
use super::super::qlib::linux::time::*;
use super::super::kernel::time::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;

impl ThreadInternal {
    pub fn TryWaitInterrupt(&mut self) -> bool {
        return self.blocker.TryWaitInterrupt();
    }

    pub fn Interrupted(&self) -> bool {
        return self.blocker.Interrupted();
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

impl Default for Blocker {
    fn default() -> Self {
        info!("blocker::default");

        let waiter = Waiter::default();

        let timerEntry = waiter.NewWaitEntry(Waiter::TIMER_WAITID, 1);
        let listener = WaitEntryListener::New(&timerEntry);

        let interruptEntry = waiter.NewWaitEntry(Waiter::INTERRUPT_WAITID, 1);
        let generalEntry = waiter.NewWaitEntry(Waiter::GENERAL_WAITID, 0);

        let monoClock = MONOTONIC_CLOCK.clone();
        let monoTimer = Timer::New(&monoClock, &Arc::new(listener.clone()));

        let realClock = REALTIME_CLOCK.clone();
        let realTimer = Timer::New(&realClock, &Arc::new(listener.clone()));


        return Self {
            waiter: waiter,
            timerEntry: timerEntry,
            timerListner: listener,
            realBlockTimer: realTimer,
            monoBlockTimer: monoTimer,
            interruptEntry: interruptEntry,
            generalEntry: generalEntry,
        }
    }
}

impl Blocker {
    pub fn Drop(&mut self) {
        self.monoBlockTimer.Destroy();
    }

    pub fn New(taskId: u64) -> Self {
        let waiter = Waiter::New(taskId);

        let timerEntry = waiter.NewWaitEntry(Waiter::TIMER_WAITID, 1);
        let listener = WaitEntryListener::New(&timerEntry);

        let interruptEntry = waiter.NewWaitEntry(Waiter::INTERRUPT_WAITID, 1);
        let generalEntry = waiter.NewWaitEntry(Waiter::GENERAL_WAITID, 0);

        let monoClock = MONOTONIC_CLOCK.clone();
        let monoTimer = Timer::New(&monoClock, &Arc::new(listener.clone()));

        let realClock = REALTIME_CLOCK.clone();
        let realTimer = Timer::New(&realClock, &Arc::new(listener.clone()));

        return Self {
            waiter: waiter,
            timerEntry: timerEntry,
            timerListner: listener,
            realBlockTimer: realTimer,
            monoBlockTimer: monoTimer,
            interruptEntry: interruptEntry,
            generalEntry: generalEntry,
        }
    }

    pub fn BlockWithTimeout(&self, timer: Timer, waitGeneral: bool, timeout: Option<Duration>) -> (Duration, Result<()>) {
        if timeout.is_none() {
            return (0, self.block(waitGeneral, None));
        }

        let adjustTimeout = timeout.unwrap() - 30_000; // 30 us is process time.

        if adjustTimeout <= 0 { // if timeout < 30 us, just timeout immediately as 30 us is process time.
            return (0, Err(Error::SysError(SysErr::ETIMEDOUT)))
        }

        let clock = timer.Clock();
        let start = clock.Now().0;
        let deadline = Time(start + adjustTimeout);

        let res = self.BlockWithTimer(timer, waitGeneral, Some(deadline));
        match res {
            Err(Error::SysError(SysErr::ETIMEDOUT)) => return (0, Err(Error::SysError(SysErr::ETIMEDOUT))),
            _ => (),
        }

        let end = clock.Now().0;
        let remain = adjustTimeout - (end - start);
        if remain < 0 {
            return (0, res)
        }

        return (remain, res)
    }

    pub fn BlockWithMonoTimeout(&self, waitGeneral: bool, timeout: Option<Duration>) -> (Duration, Result<()>) {
        let timer = self.GetTimer(MONOTONIC);
        return self.BlockWithTimeout(timer, waitGeneral, timeout)
    }

    pub fn BlockWithRealTimer(&self, waitGeneral: bool, deadline: Option<Time>) -> Result<()> {
        let timer = self.GetTimer(CLOCK_REALTIME);
        return self.BlockWithTimer(timer, waitGeneral, deadline);
    }

    pub fn BlockWithMonoTimer(&self, waitGeneral: bool, deadline: Option<Time>) -> Result<()> {
        let timer = self.GetTimer(CLOCK_MONOTONIC);
        return self.BlockWithTimer(timer, waitGeneral, deadline);
    }

    pub fn BlockWithTimer(&self, timer: Timer, waitGeneral: bool, deadline: Option<Time>) -> Result<()> {
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
        let entries = [None, None, Some(self.interruptEntry.clone())];
        let _entry = self.waiter.Wait(&entries);
        self.SleepFinish(false);
        return Err(Error::SysError(SysErr::ERESTARTNOHAND));
    }

    pub fn GetTimerWithClock(&self, clock: &Clock) -> Timer {
        return Timer::New(clock, &Arc::new(self.timerListner.clone()));
    }

    pub fn GetTimer(&self, clockId: i32) -> Timer {
        return match clockId {
            CLOCK_REALTIME => self.realBlockTimer.clone(),
            CLOCK_MONOTONIC => self.monoBlockTimer.clone(),
            _ => panic!("BlockWithTimer invalid clockid {}", clockId)
        }
    }

    pub fn block(&self, waitGeneral: bool, waitTimer: Option<Timer>) -> Result<()> {
        if waitGeneral && self.waiter.TryWait(&self.generalEntry) {
            return Ok(())
        }

        self.SleepStart();

        let entries = if waitGeneral && waitTimer.is_some() {
            [Some(self.generalEntry.clone()), Some(self.timerEntry.clone()), Some(self.interruptEntry.clone())]
        } else if waitGeneral {
            [Some(self.generalEntry.clone()), None, Some(self.interruptEntry.clone())]
        } else if waitTimer.is_some() {
            [None, Some(self.timerEntry.clone()), Some(self.interruptEntry.clone())]
        } else {
            [None, None, Some(self.interruptEntry.clone())]
        };

        let entry = self.waiter.Wait(&entries);
        Task::Current().DoStop();
        if entry == self.generalEntry.clone() {
            self.SleepFinish(true);
            match waitTimer {
                Some(timer) => {
                    timer.Cancel();
                }
                _ => (),
            }
            self.waiter.lock().bitmap &= !(1<<Waiter::TIMER_WAITID);
            return Ok(())
        } else if entry == self.timerEntry.clone() {
            self.SleepFinish(true);
            return Err(Error::SysError(SysErr::ETIMEDOUT));
        } else {
            //interrutpted
            self.SleepFinish(false);
            match waitTimer {
                Some(timer) => {
                    timer.Cancel();
                }
                _ => (),
            }
            self.waiter.lock().bitmap &= !(1<<Waiter::TIMER_WAITID);
            return Err(Error::ErrInterrupted);
        }
    }

    // block on both generalentry and interrupt
    pub fn BlockGeneral(&self) -> Result<()> {
        let entries = [Some(self.generalEntry.clone()), None, Some(self.interruptEntry.clone())];
        let entry = self.waiter.Wait(&entries);

        if entry == self.generalEntry.clone() {
            return Ok(())
        } else {
            //interrutpted
            self.SleepFinish(false);
            return Err(Error::ErrInterrupted);
        }
    }

    // block on general entry
    pub fn BlockGeneralOnly(&self) {
        let entries = [Some(self.generalEntry.clone()), None, None];
        let _entry = self.waiter.Wait(&entries);

        return
    }

    pub fn SleepStart(&self) {}

    pub fn SleepFinish(&self, success: bool) {
        if !success {
            self.interruptSelf()
        }
    }

    pub fn Interrupted(&self) -> bool {
        return self.waiter.TryWait(&self.interruptEntry)
    }

    pub fn TryWaitInterrupt(&self) -> bool {
        return self.waiter.TryWait(&self.interruptEntry)
    }

    pub fn interruptSelf(&self) {
        self.interruptEntry.Notify(1);
    }
}
