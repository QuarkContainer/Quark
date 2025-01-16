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

use alloc::boxed::Box;
use alloc::sync::Arc;

use super::super::super::common::*;
#[cfg(target_arch="x86_64")]
use super::super::super::cpuid::*;
use super::super::super::linux::time::*;
use super::super::super::linux_def::*;
#[cfg(target_arch="x86_64")]
use super::super::arch::__arch::arch_def::ArchFPState;
use super::super::kernel::posixtimer::*;
use super::super::kernel::waiter::*;
use super::super::stack::*;
use super::super::task::*;
use super::super::threadmgr::thread::*;
use super::super::threadmgr::thread_group::*;
use super::super::SignalDef::*;
use super::task_exit::*;
use super::task_stop::*;
use super::task_syscall::*;
#[cfg(target_arch = "aarch64")]
use crate::qlib::kernel::loader::vdso::VDSO;

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct FPSoftwareFrame {
    pub Magic1: u32,
    pub ExtendedSize: u32,
    pub Xfeatures: u64,
    pub XstateSize: u32,
    pub Padding: [u32; 7],
}

#[derive(Copy, Clone, Default)]
pub struct SignalAction {}

impl SignalAction {
    pub const TERM: u64 = 0;
    pub const CORE: u64 = 1;
    pub const STOP: u64 = 2;
    pub const IGNORE: u64 = 3;
    pub const HANDLER: u64 = 4;
    //pub const CONT: u64 = 5;
}

pub static DEFAULT_ACTION: &'static [u64] = &[
    SignalAction::IGNORE, //0
    SignalAction::TERM,   //1
    SignalAction::TERM,   //2
    SignalAction::CORE,   //3
    SignalAction::CORE,   //4
    SignalAction::CORE,   //5
    SignalAction::CORE,   //6
    SignalAction::CORE,   //7
    SignalAction::CORE,   //8
    SignalAction::TERM,   //9
    SignalAction::TERM,   //10
    SignalAction::CORE,   //11
    SignalAction::TERM,   //12
    SignalAction::TERM,   //13
    SignalAction::TERM,   //14
    SignalAction::TERM,   //15
    SignalAction::TERM,   //16
    SignalAction::IGNORE, //17
    SignalAction::IGNORE, //18
    SignalAction::STOP,   //19
    SignalAction::STOP,   //20
    SignalAction::STOP,   //21
    SignalAction::STOP,   //22
    SignalAction::IGNORE, //23
    SignalAction::CORE,   //24
    SignalAction::CORE,   //25
    SignalAction::TERM,   //26
    SignalAction::TERM,   //27
    SignalAction::IGNORE, //28
    SignalAction::TERM,   //29
    SignalAction::CORE,   //30
    SignalAction::CORE,   //31
];

// UnblockableSignals contains the set of signals which cannot be blocked.
pub static UNBLOCKED_SIGNALS: SignalSet =
    SignalSet((1 << Signal::SIGKILL | 1 << Signal::SIGSTOP) >> 1);

// StopSignals is the set of signals whose default action is SignalActionStop.
pub static STOP_SIGNALS: SignalSet = SignalSet(
    (1 << Signal::SIGSTOP | 1 << Signal::SIGTSTP | 1 << Signal::SIGTTIN | 1 << Signal::SIGTTOU)
        >> 1,
);

// computeAction figures out what to do given a signal number
// and an arch.SignalAct. SIGSTOP always results in a SignalActionStop,
// and SIGKILL always results in a SignalActionTerm.
// Signal 0 is always ignored as many programs use it for various internal functions
// and don't expect it to do anything.
//
// In the event the signal is not one of these, act.Handler determines what
// happens next.
// If act.Handler is:
// 0, the default action is taken;
// 1, the signal is ignored;
// anything else, the function returns SignalActionHandler.
pub fn ComputeAction(sig: Signal, act: &SigAct) -> u64 {
    match sig.0 {
        Signal::SIGSTOP => SignalAction::STOP,
        Signal::SIGKILL => SignalAction::TERM,
        0 => SignalAction::IGNORE,
        _ => {
            if act.handler == SigAct::SIGNAL_ACT_DEFAULT {
                // todo: what's default action for realtime signal?
                if sig.0 >= 32 {
                    return SignalAction::HANDLER;
                }
                return DEFAULT_ACTION[sig.0 as usize];
            } else if act.handler == SigAct::SIGNAL_ACT_IGNORE {
                return SignalAction::IGNORE;
            } else {
                return SignalAction::HANDLER;
            }
        }
    }
}

impl ThreadInternal {
    // dequeueSignalLocked returns a pending signal that is *not* included in mask.
    // If there are no pending unmasked signals, dequeueSignalLocked returns nil.
    //
    // Preconditions: t.tg.signalHandlers.mu must be locked.
    pub fn dequeueSignalLocked(&mut self, mask: SignalSet) -> Option<Box<SignalInfo>> {
        let info = self.pendingSignals.Deque(mask);
        match info {
            Some(si) => return Some(si),
            None => (),
        };

        return self.tg.lock().pendingSignals.Deque(mask);
    }

    // participateGroupStopLocked is called to handle thread group side effects
    // after t unsets t.groupStopPending. The caller must handle task side effects
    // (e.g. placing the task goroutine into the group stop). It returns true if
    // the caller must notify t.tg.leader's parent of a completed group stop (which
    // participateGroupStopLocked cannot do due to holding the wrong locks).
    //
    // Preconditions: The signal mutex must be locked.
    pub fn participateGroupStopLocked(&mut self) -> bool {
        if self.groupStopAcknowledged {
            return false;
        }

        self.groupStopAcknowledged = true;
        let mut tg = self.tg.lock();
        tg.groupStopPendingCount -= 1;

        if tg.groupStopPendingCount != 0 {
            return false;
        }

        if tg.groupStopComplete {
            return false;
        }

        tg.groupStopComplete = true;
        tg.groupStopWaitable = true;
        tg.groupContNotify = false;
        tg.groupContWaitable = false;
        return true;
    }
}

impl Thread {
    // canReceiveSignalLocked returns true if t should be interrupted to receive
    // the given signal. canReceiveSignalLocked is analogous to Linux's
    // kernel/signal.c:wants_signal(), but see below for divergences.
    //
    // Preconditions: The signal mutex must be locked.
    pub fn canReceiveSignalLocked(&self, sig: Signal) -> bool {
        let queue = self.lock().SignalQueue.clone();
        queue.Notify(SignalSet::MakeSignalSet(&[sig]).0 as EventMask);

        // - Do not choose tasks that are blocking the signal.
        if SignalSet::New(sig).0 & self.lock().signalMask.0 != 0 {
            return false;
        }

        // - No need to check Task.exitState, as the exit path sets every bit in the
        // signal mask when it transitions from TaskExitNone to TaskExitInitiated.
        // - No special case for SIGKILL: SIGKILL already interrupted all tasks in the
        // task group via applySignalSideEffects => killLocked.
        // - Do not choose stopped tasks, which cannot handle signals.
        if self.lock().stop.is_some() {
            return false;
        }

        // - Do not choose tasks that have already been interrupted, as they may be
        // busy handling another signal.
        if self.Interrupted(false) {
            return false;
        }

        return true;
    }

    // forceSignal ensures that the task is not ignoring or blocking the given
    // signal. If unconditional is true, forceSignal takes action even if the
    // signal isn't being ignored or blocked.
    pub fn forceSignal(&self, sig: Signal, unconditional: bool) {
        let tg = self.ThreadGroup();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();

        let _r = owner.read();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();
        let sh = tg.lock().signalHandlers.clone();

        let mut sh = sh.lock();

        let blocked = SignalSet::New(sig).0 & self.lock().signalMask.0 != 0;
        let mut act = sh.GetAct(sig);
        let ignored = act.handler == SigAct::SIGNAL_ACT_IGNORE;

        if blocked || ignored || unconditional {
            act.handler = SigAct::SIGNAL_ACT_DEFAULT;
            sh.actions.insert(sig.0, act);
            if blocked {
                self.setSignalMaskLocked(SignalSet(
                    self.lock().signalMask.0 & !SignalSet::New(sig).0,
                ))
            }
        }
    }

    // Preconditions: The signal mutex must be locked.
    pub fn setSignalMaskLocked(&self, mask: SignalSet) {
        let mask = SignalSet(mask.0 & !UNMASKABLE_MASK);

        let oldMask = self.lock().signalMask;
        self.lock().signalMask = mask;

        // If the new mask blocks any signals that were not blocked by the old
        // mask, and at least one such signal is pending in tg.pendingSignals, and
        // t has been woken, it could be the case that t was woken to handle that
        // signal, but will no longer do so as a result of its new signal mask, so
        // we have to pick a replacement.
        let blocked = mask.0 & !oldMask.0;
        let tg = self.ThreadGroup();
        let blockedGroupPending = SignalSet(blocked & tg.lock().pendingSignals.pendingSet.0);
        if blockedGroupPending.0 != 0 && self.Interrupted(true) {
            blockedGroupPending.ForEachSignal(|sig| {
                let nt = tg.lock().findSignalReceiverLocked(sig);
                if nt.is_some() {
                    nt.unwrap().lock().interrupt();
                }
            });

            // We have to re-issue the interrupt consumed by t.interrupted() since
            // it might have been for a different reason.
            self.lock().interruptSelf();
        }

        // Conversely, if the new mask unblocks any signals that were blocked by
        // the old mask, and at least one such signal is pending, we may now need
        // to handle that signal.
        let unblocked = oldMask.0 & !mask.0;
        let pendingSet = self.lock().pendingSignals.pendingSet.0;
        let tgPendingSet = tg.lock().pendingSignals.pendingSet.0;
        let unblockedPending = unblocked & (pendingSet | tgPendingSet);
        if unblockedPending != 0 {
            self.lock().interruptSelf();
        }
    }

    // initiateGroupStop attempts to initiate a group stop based on a
    // previously-dequeued stop signal.
    //
    // Preconditions: The caller must be running on the task goroutine.
    pub fn initiateGroupStop(&self, info: &SignalInfo) {
        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        if self.lock().groupStopPending {
            info!(
                "Signal {}: not stopping thread group: lost to racing stop signal",
                info.Signo
            );
            return;
        }

        let mut tg = tg.lock();
        if !tg.groupStopDequeued {
            info!(
                "Signal {}: not stopping thread group: lost to racing SIGCONT",
                info.Signo
            );
            return;
        }

        if tg.exiting {
            info!(
                "Signal {}: not stopping thread group: lost to racing group exit",
                info.Signo
            );
            return;
        }

        if tg.execing.Upgrade().is_some() {
            info!(
                "Signal {}: not stopping thread group: lost to racing execve",
                info.Signo
            );
            return;
        }

        if !tg.groupStopComplete {
            tg.groupStopSignal = Signal(info.Signo);
        }

        tg.groupStopPendingCount = 0;

        let mut add = 0;

        for t2 in &tg.tasks {
            let mut t2 = t2.lock();

            if t2.killedLocked() || t2.exitState >= TaskExitState::TaskExitInitiated {
                t2.groupStopPending = false;
                continue;
            }

            t2.groupStopPending = true;
            t2.groupStopAcknowledged = false;
            t2.interrupt();

            add += 1;
        }

        tg.groupStopPendingCount += add;

        info!(
            "Signal {}: stopping {} threads in thread group",
            info.Signo, tg.groupStopPendingCount
        );
    }

    // SetSignalMask sets t's signal mask.
    //
    // Preconditions: SetSignalMask can only be called by the task goroutine.
    // t.exitState < TaskExitZombie.
    pub fn SetSignalMask(&self, mask: SignalSet) {
        let tg = self.lock().tg.clone();
        let lock = tg.lock().signalLock.clone();
        // By precondition, t prevents t.tg from completing an execve and mutating
        // t.tg.signalHandlers, so we can skip the TaskSet mutex.
        let _s = lock.lock();
        self.setSignalMaskLocked(mask);
    }

    // signalStop sends a signal to t's thread group of a new group stop, group
    // continue, or ptrace stop, if appropriate. code and status are set in the
    // signal sent to tg, if any.
    //
    // Preconditions: The TaskSet mutex must be locked (for reading or writing).
    pub fn signalStop(&self, target: &Thread, code: i32, status: i32) {
        let tg = self.lock().tg.clone();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        let sh = tg.lock().signalHandlers.clone();
        match sh.lock().actions.get(&Signal::SIGCHLD) {
            None => (),
            Some(act) => {
                if !(act.handler != SigAct::SIGNAL_ACT_IGNORE
                    && act.flags.0 & SigFlag::SIGNAL_FLAG_NO_CLD_STOP == 0)
                {
                    return;
                }
            }
        };

        let mut sigchldInfo = SignalInfo {
            Signo: Signal::SIGCHLD,
            Code: code,
            ..Default::default()
        };

        let sigchld = sigchldInfo.SigChld();
        let pidns = tg.PIDNamespace();
        let creds = self.lock().creds.clone();
        let userns = creds.lock().UserNamespace.clone();

        sigchld.pid = *pidns.lock().tids.get(target).unwrap();
        let realKUID = target.Credentials().lock().RealKUID;
        sigchld.uid = realKUID.In(&userns).OrOverflow().0;
        sigchld.status = status;
        self.sendSignalLocked(&sigchldInfo, true).unwrap();
    }

    pub fn sendSignalLocked(&self, info: &SignalInfo, group: bool) -> Result<()> {
        info!("sendsignalLocked, signal:{:?}, group:{}", info, group);
        return self.sendSignalTimerLocked(info, group, None);
    }

    pub fn sendSignalTimerLocked(
        &self,
        info: &SignalInfo,
        group: bool,
        timer: Option<IntervalTimer>,
    ) -> Result<()> {
        info!("sendsignalTimerLocked, signal:{:?}", info);
        if self.lock().exitState == TaskExitState::TaskExitDead {
            return Err(Error::SysError(SysErr::ESRCH));
        }

        let sig = Signal(info.Signo);
        if sig.0 == 0 {
            return Ok(());
        }

        if !sig.IsValid() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // Signal side effects apply even if the signal is ultimately discarded.
        let tg = self.lock().tg.clone();
        tg.lock().applySignalSideEffectsLocked(sig);

        // TODO: "Only signals for which the "init" process has established a
        // signal handler can be sent to the "init" process by other members of the
        // PID namespace. This restriction applies even to privileged processes,
        // and prevents other members of the PID namespace from accidentally
        // killing the "init" process." - pid_namespaces(7). We don't currently do
        // this for child namespaces, though we should; we also don't do this for
        // the root namespace (the same restriction applies to global init on
        // Linux), where whether or not we should is much murkier. In practice,
        // most sandboxed applications are not prepared to function as an init
        // process.

        // Unmasked, ignored signals are discarded without being queued, unless
        // they will be visible to a tracer. Even for group signals, it's the
        // originally-targeted task's signal mask and tracer that matter; compare
        // Linux's kernel/signal.c:__send_signal() => prepare_signal() =>
        // sig_ignored().

        let mut sh = tg.lock().signalHandlers.clone();
        let ignored = ComputeAction(sig, &sh.GetAct(sig)) == SignalAction::IGNORE;
        let sigset = SignalSet::New(sig);
        let signalMask = self.lock().signalMask;
        let realSignalMask = self.lock().realSignalMask;
        if sigset.0 & signalMask.0 == 0 && sigset.0 & realSignalMask.0 == 0 && ignored {
            info!("Discarding ignored signal {:?}", sig);
            if timer.is_some() {
                timer.unwrap().lock().signalRejectedLocked();
            }

            return Ok(());
        }

        let res = if !group {
            self.lock()
                .pendingSignals
                .Enque(Box::new(*info), timer.clone())?
        } else {
            tg.lock()
                .pendingSignals
                .Enque(Box::new(*info), timer.clone())?
        };

        if !res {
            if sig.IsRealtime() {
                return Err(Error::SysError(SysErr::EAGAIN));
            }

            if timer.is_some() {
                timer.clone().unwrap().lock().signalRejectedLocked();
            }

            return Ok(());
        }

        // Find a receiver to notify. Note that the task we choose to notify, if
        // any, may not be the task that actually dequeues and handles the signal;
        // e.g. a racing signal mask change may cause the notified task to become
        // ineligible, or a racing sibling task may dequeue the signal first.
        let canReceiveSignalLocked = self.canReceiveSignalLocked(sig);
        if canReceiveSignalLocked {
            info!("Thread[{}] Notified of signal {:?}", self.lock().id, sig);
            self.lock().interrupt();
            return Ok(());
        }

        if group {
            let nt = tg.lock().findSignalReceiverLocked(sig);
            if nt.is_some() {
                nt.unwrap().lock().interrupt();
                return Ok(());
            }
        }

        info!("No task notified of signal {:?}", sig);
        return Ok(());
    }

    // PendingSignals returns the set of pending signals.
    pub fn PendingSignals(&self) -> SignalSet {
        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        let pendingset = self.lock().pendingSignals.pendingSet.0;
        return SignalSet(pendingset | tg.lock().pendingSignals.pendingSet.0);
    }

    // PendingSignals returns the set of pending signals without lock. Just for signalfd readiness check.
    pub fn PendingSignalsNolock(&self) -> SignalSet {
        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        // it is readonly pendingSignals for readiness check. Even there is inconsistent state, but it is acceptable
        //let _s = lock.lock();

        let pendingset = self.lock().pendingSignals.pendingSet.0;
        return SignalSet(pendingset | tg.lock().pendingSignals.pendingSet.0);
    }

    // SendSignal sends the given signal to t.
    //
    // The following errors may be returned:
    //
    //	syserror.ESRCH - The task has exited.
    //	syserror.EINVAL - The signal is not valid.
    //	syserror.EAGAIN - THe signal is realtime, and cannot be queued.
    //
    pub fn SendSignal(&self, info: &SignalInfo) -> Result<()> {
        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        return self.sendSignalLocked(info, false);
    }

    // SendGroupSignal sends the given signal to t's thread group.
    pub fn SendGroupSignal(&self, info: &SignalInfo) -> Result<()> {
        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        return self.sendSignalLocked(info, true);
    }

    // Sigtimedwait implements the semantics of sigtimedwait(2).
    //
    // Preconditions: The caller must be running on the task context. t.exitState
    // < TaskExitZombie.
    pub fn Sigtimedwait(&self, set: SignalSet, timeout: Duration) -> Result<Box<SignalInfo>> {
        // set is the set of signals we're interested in; invert it to get the set
        // of signals to block.
        let mask = SignalSet(!(set.0 & !UNBLOCKED_SIGNALS.0));

        let tg = self.lock().tg.clone();
        let lock = tg.lock().signalLock.clone();

        {
            let _s = lock.lock();

            let info = self.lock().dequeueSignalLocked(mask);
            if info.is_some() {
                return Ok(info.unwrap());
            }

            if timeout == 0 {
                return Err(Error::SysError(SysErr::EAGAIN));
            }

            // Unblock signals we're waiting for. Remember the original signal mask so
            // that Task.sendSignalTimerLocked doesn't discard ignored signals that
            // we're temporarily unblocking.
            let signalMask = self.lock().signalMask;
            self.lock().realSignalMask = signalMask;
            self.setSignalMaskLocked(SignalSet(signalMask.0 & mask.0));
        }

        let blocker = self.lock().blocker.clone();
        let (_, err) = blocker.BlockWithMonoTimeout(false, Some(timeout));

        {
            let _s = lock.lock();

            let realSignalMask = self.lock().realSignalMask;
            self.setSignalMaskLocked(realSignalMask);
            self.lock().realSignalMask = SignalSet(0);

            let info = self.lock().dequeueSignalLocked(mask);
            if info.is_some() {
                return Ok(info.unwrap());
            }

            match err {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    return Err(Error::SysError(SysErr::EAGAIN));
                }
                Err(Error::ErrInterrupted) => {
                    return Err(Error::SysError(SysErr::ERESTARTNOHAND));
                }
                Err(e) => return Err(e),
                e => panic!("TaskExitZombie, unknow return {:?}", e),
            }
        }
    }

    // SignalMask returns a copy of t's signal mask.
    pub fn SignalMask(&self) -> SignalSet {
        let mask = self.lock().signalMask;
        return mask;
    }

    // SetSavedSignalMask sets the saved signal mask (see Task.savedSignalMask's
    // comment).
    //
    // Preconditions: SetSavedSignalMask can only be called by the task goroutine.
    pub fn SetSavedSignalMask(&self, mask: SignalSet) {
        let mut t = self.lock();

        t.savedSignalMask = mask;
        t.haveSavedSignalMask = true;
    }

    pub fn SignalRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let tg = self.ThreadGroup();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        self.lock().SignalQueue.EventRegister(task, e, mask)
    }

    pub fn SignalUnregister(&self, task: &Task, e: &WaitEntry) {
        let tg = self.ThreadGroup();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        self.lock().SignalQueue.EventUnregister(task, e)
    }
}

impl ThreadGroupInternal {
    // discardSpecificLocked removes all instances of the given signal from all
    // signal queues in tg.
    //
    // Preconditions: The signal mutex must be locked.
    pub fn discardSpecificLocked(&mut self, sig: Signal) {
        self.pendingSignals.Discard(sig);
        for t in &self.tasks {
            t.lock().pendingSignals.Discard(sig);
        }
    }

    pub fn applySignalSideEffectsLocked(&mut self, sig: Signal) {
        if SignalSet::New(sig).0 & STOP_SIGNALS.0 != 0 {
            // Stop signals cause all prior SIGCONT to be discarded. (This is
            // despite the fact this has little effect since SIGCONT's most
            // important effect is applied when the signal is sent in the branch
            // below, not when the signal is delivered.)
            self.discardSpecificLocked(Signal(Signal::SIGCONT));
        } else if sig.0 == Signal::SIGCONT {
            // "The SIGCONT signal has a side effect of waking up (all threads of)
            // a group-stopped process. This side effect happens before
            // signal-delivery-stop. The tracer can't suppress this side effect (it
            // can only suppress signal injection, which only causes the SIGCONT
            // handler to not be executed in the tracee, if such a handler is
            // installed." - ptrace(2)
            self.endGroupStopLocked(true);
        } else if sig.0 == Signal::SIGKILL {
            // "SIGKILL does not generate signal-delivery-stop and therefore the
            // tracer can't suppress it. SIGKILL kills even within system calls
            // (syscall-exit-stop is not generated prior to death by SIGKILL)." -
            // ptrace(2)
            //
            // Note that this differs from ThreadGroup.requestExit in that it
            // ignores tg.execing.
            if !self.exiting {
                self.exiting = true;
                self.exitStatus = ExitStatus {
                    Signo: Signal::SIGKILL,
                    ..Default::default()
                }
            }

            for t in &self.tasks {
                t.lock().killLocked();
            }
        }
    }

    // findSignalReceiverLocked returns a task in tg that should be interrupted to
    // receive the given signal. If no such task exists, findSignalReceiverLocked
    // returns nil.
    //
    // Linux actually records curr_target to balance the group signal targets.
    //
    // Preconditions: The signal mutex must be locked.
    pub fn findSignalReceiverLocked(&self, sig: Signal) -> Option<Thread> {
        for t in &self.tasks {
            if t.canReceiveSignalLocked(sig) {
                return Some(t.clone());
            }
        }

        return None;
    }

    // endGroupStopLocked ensures that all prior stop signals received by tg are
    // not stopping tg and will not stop tg in the future. If broadcast is true,
    // parent and tracer notification will be scheduled if appropriate.
    //
    // Preconditions: The signal mutex must be locked.
    pub fn endGroupStopLocked(&mut self, broadcast: bool) {
        STOP_SIGNALS.ForEachSignal(|sig| {
            self.discardSpecificLocked(sig);
        });

        if self.groupStopPendingCount == 0 && !self.groupStopComplete {
            return;
        }

        for t in &self.tasks {
            let mut t = t.lock();
            t.groupStopPending = false;
            if t.stop.is_some() && t.stop.clone().unwrap().Type() == TaskStopType::GROUPSTOP {
                t.endInternalStopLocked();
            }
        }

        if broadcast {
            // Instead of notifying the parent here, set groupContNotify so that
            // one of the continuing tasks does so. (Linux does something similar.)
            // The reason we do this is to keep locking sane. In order to send a
            // signal to the parent, we need to lock its signal mutex, but we're
            // already holding tg's signal mutex, and the TaskSet mutex must be
            // locked for writing for us to hold two signal mutexes. Since we don't
            // want to require this for endGroupStopLocked (which is called from
            // signal-sending paths), nor do we want to lose atomicity by releasing
            // the mutexes we're already holding, just let the continuing thread
            // group deal with it.
            self.groupContNotify = true;
            self.groupContInterrupted = !self.groupStopComplete;
            self.groupContWaitable = true;
        }

        // Unsetting groupStopDequeued will cause racing calls to initiateGroupStop
        // to recognize that the group stop has been cancelled.
        self.groupStopDequeued = false;
        self.groupStopSignal = Signal(0);
        self.groupStopPendingCount = 0;
        self.groupStopComplete = false;
        self.groupStopWaitable = false;
    }
}

impl ThreadGroup {
    pub fn SendSignal(&self, info: &SignalInfo) -> Result<()> {
        debug!("sendSignal {:?}", &info);
        let pidns = self.PIDNamespace();
        let owner = pidns.lock().owner.clone();

        let _r = owner.read();
        let lock = self.lock().signalLock.clone();
        let _s = lock.lock();

        let leader = self.lock().leader.Upgrade();
        match leader {
            None => return Err(Error::NotExist),
            Some(l) => return l.sendSignalLocked(info, true),
        }
    }

    // SetSignalAct atomically sets the thread group's signal action for signal sig
    // to *actptr (if actptr is not nil) and returns the old signal action.
    pub fn SetSignalAct(&self, sig: Signal, actptr: &Option<SigAct>) -> Result<SigAct> {
        if !sig.IsValid() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let pidns = self.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();

        let lock = self.lock().signalLock.clone();
        let _s = lock.lock();
        let sh = self.lock().signalHandlers.clone();

        let mut sh = sh.lock();

        let oldact = sh.GetAct(sig);

        if (sig.0 == Signal::SIGKILL || sig.0 == Signal::SIGSTOP) && actptr.is_some() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        match actptr {
            None => {}
            Some(actptr) => {
                let mut act = *actptr;
                act.mask &= !UNBLOCKED_SIGNALS.0;
                sh.actions.insert(sig.0, act.clone());

                // From POSIX, by way of Linux:
                //
                // "Setting a signal action to SIG_IGN for a signal that is pending
                // shall cause the pending signal to be discarded, whether or not it is
                // blocked."
                //
                // "Setting a signal action to SIG_DFL for a signal that is pending and
                // whose default action is to ignore the signal (for example, SIGCHLD),
                // shall cause the pending signal to be discarded, whether or not it is
                // blocked."
                if ComputeAction(sig, &act) == SigAct::SIGNAL_ACT_IGNORE {
                    self.lock().discardSpecificLocked(sig);
                }
            }
        }

        return Ok(oldact);
    }
}

// groupStop is a TaskStop placed on tasks that have received a stop signal
// (SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU). (The term "group-stop" originates from
// the ptrace man page.)
pub struct GroupStop {}

impl TaskStop for GroupStop {
    fn Type(&self) -> TaskStopType {
        return TaskStopType::GROUPSTOP;
    }

    fn Killable(&self) -> bool {
        return true;
    }
}

pub fn SignalInfoNoInfo(signo: Signal, sender: &Thread, receiver: &Thread) -> SignalInfo {
    let info = SignalInfo {
        Signo: signo.0,
        Code: SignaCode::SI_USER,
        ..Default::default()
    };

    let senderTg = sender.ThreadGroup();
    let senderPid = senderTg.PIDNamespace().IDOfThreadGroup(&senderTg);
    let recUserns = receiver.UserNamespace();
    let receiverUid = sender
        .Credentials()
        .lock()
        .RealKUID
        .In(&recUserns)
        .OrOverflow();

    info.Kill().pid = senderPid;
    info.Kill().uid = receiverUid.0 as _;
    return info;
}

impl Task {
    pub fn HandleExceedsFileSizeLimit(&self) -> Result<()> {
        let thread = self.Thread();
        let info = SignalInfoNoInfo(Signal(Signal::SIGXFSZ), &thread, &thread);
        self.Thread().SendSignal(&info)?;

        return Err(Error::SysError(SysErr::EFBIG));
    }

    pub fn RunInterrupt(&mut self) -> TaskRunState {
        let task = self;
        // Interrupts are de-duplicated (if t is interrupted twice before
        // t.interrupted() is called, t.interrupted() will only return true once),
        // so early exits from this function must re-enter the runInterrupt state
        // to check for more interrupt-signaled conditions.

        let t = task.Thread();
        let tg = t.lock().tg.clone();
        let pidns = tg.PIDNamespace();

        let owner = pidns.lock().owner.clone();
        let lock = tg.lock().signalLock.clone();
        let locker = lock.lock();

        // Did we just leave a group stop?
        let groupContNotify = tg.lock().groupContNotify;
        if groupContNotify {
            tg.lock().groupContNotify = false;
            let sig = tg.lock().groupStopSignal;
            let intr = tg.lock().groupContInterrupted;
            core::mem::drop(locker);

            let _r = owner.read();
            // For consistency with Linux, if the parent and (thread group
            // leader's) tracer are in the same thread group, deduplicate
            // notifications.
            let leader = tg.lock().leader.Upgrade().unwrap();
            let notifyParent = leader.lock().parent.is_some();
            if notifyParent {
                // If groupContInterrupted, do as Linux does and pretend the group
                // stop completed just before it ended. The theoretical behavior in
                // this case would be to send a SIGCHLD indicating the completed
                // stop, followed by a SIGCHLD indicating the continue. However,
                // SIGCHLD is a standard signal, so the latter would always be
                // dropped. Hence sending only the former is equivalent.
                let parent = leader.lock().parent.clone().unwrap();
                if intr {
                    parent.signalStop(&leader, SignalInfo::CLD_STOPPED, sig.0);
                    let ptg = parent.lock().tg.clone();
                    ptg.lock()
                        .eventQueue
                        .Notify(EVENT_GROUP_CONTINUE | EVENT_CHILD_GROUP_STOP);
                } else {
                    parent.signalStop(&leader, SignalInfo::CLD_CONTINUED, sig.0);
                    let ptg = parent.lock().tg.clone();
                    ptg.lock().eventQueue.Notify(EVENT_GROUP_CONTINUE);
                }
            }

            return TaskRunState::RunInterrupt;
        }

        // Do we need to enter a group stop or related ptrace stop? This path is
        // analogous to Linux's kernel/signal.c:get_signal() => do_signal_stop()
        // (with ptrace enabled) and do_jobctl_trap().
        let groupStopPending = t.lock().groupStopPending;
        let trapStopPending = t.lock().trapStopPending;
        let trapNotifyPending = t.lock().trapNotifyPending;
        if groupStopPending || trapStopPending || trapNotifyPending {
            let sig = tg.lock().groupStopSignal;
            let mut notifyParent = false;
            if groupStopPending {
                t.lock().groupStopPending = false;
                // We care about t.tg.groupStopSignal (for tracer notification)
                // even if this doesn't complete a group stop, so keep the
                // value of sig we've already read.
                notifyParent = t.lock().participateGroupStopLocked();
            }

            t.lock().trapStopPending = false;
            t.lock().trapNotifyPending = false;
            // Drop the signal mutex so we can take the TaskSet mutex.
            core::mem::drop(locker);

            let _r = owner.read();
            let leader = tg.lock().leader.Upgrade().unwrap();
            if leader.lock().parent.is_none() {
                notifyParent = false;
            }

            {
                let _s = lock.lock();
                let killedLocked = t.lock().killedLocked();
                if !killedLocked {
                    t.lock().beginInternalStopLocked(&Arc::new(GroupStop {}));
                }
            }

            if notifyParent {
                let parent = leader.lock().parent.clone().unwrap();
                parent.signalStop(&leader, SignalInfo::CLD_STOPPED, sig.0);
                let ptg = parent.lock().tg.clone();
                ptg.lock().eventQueue.Notify(EVENT_CHILD_GROUP_STOP);
            }

            return TaskRunState::RunInterrupt;
        }

        // Are there signals pending?
        let signalMask = t.lock().signalMask;
        let info = match t.lock().dequeueSignalLocked(signalMask) {
            Some(info) => info,
            None => {
                return TaskRunState::RunApp;
            }
        };

        if SignalSet::New(Signal(info.Signo)).0 & STOP_SIGNALS.0 != 0 {
            // Indicate that we've dequeued a stop signal before unlocking the
            // signal mutex; initiateGroupStop will check for races with
            // endGroupStopLocked after relocking it.
            tg.lock().groupStopDequeued = true;
        }

        let sh = tg.lock().signalHandlers.clone();
        let act = sh.DequeAct(Signal(info.Signo));
        core::mem::drop(locker);
        return task.ThreadDeliverSignal(&info, &act);
    }

    // deliverSignal delivers the given signal and returns the following run state.
    pub fn ThreadDeliverSignal(&mut self, info: &SignalInfo, act: &SigAct) -> TaskRunState {
        let sigact = ComputeAction(Signal(info.Signo), act);
        debug!("task_signals: thread deliver signal {:#?}", act);

        if self.haveSyscallReturn {
            let ret = self.Return();
            let (sre, ok) = SyscallRestartErrnoFromReturn(ret);
            if ok {
                // Signals that are ignored, cause a thread group stop, or
                // terminate the thread group do not interact with interrupted
                // syscalls; in Linux terms, they are never returned to the signal
                // handling path from get_signal => get_signal_to_deliver. The
                // behavior of an interrupted syscall is determined by the first
                // signal that is actually handled (by userspace).
                if sigact == SignalAction::HANDLER {
                    if sre == SysErr::ERESTARTNOHAND
                        || sre == SysErr::ERESTART_RESTARTBLOCK && !act.flags.IsRestart()
                        || sre == SysErr::ERESTARTSYS && !act.flags.IsRestart()
                    {
                        self.SetReturn(-SysErr::EINTR as u64)
                    } else if sre == SysErr::ERESTART_RESTARTBLOCK {
                        self.RestartSyscallWithRestartBlock();
                    } else {
                        self.RestartSyscall();
                    }
                }
            }
        }

        match sigact {
            SignalAction::TERM | SignalAction::CORE => {
                info!("Signal {}: terminating thread group", info.Signo);
                //todo: fix this
                //let tid = t.k.TaskSet().root.IDOfTask(self)
                //let tid = 0xabcd;
                //let pid = 0xabcd;
                /*let mut ucs = UncaughtSignal {
                    Tid: tid,
                    Pid: pid,
                    SignalNumber: info.Signo,
                    FaultAddr: 0,
                };

                match info.Signo {
                    Signal::SIGSEGV
                    | Signal::SIGFPE
                    | Signal::SIGILL
                    | Signal::SIGTRAP
                    | Signal::SIGBUS => {
                        //ucs.FaultAddr = info.SigFault().addr;
                    }
                    _ => (),
                }*/
                //Emit(&Event::UncaughtSignal(ucs)).unwrap();
                self.Thread().PrepareGroupExit(ExitStatus {
                    Code: 0,
                    Signo: info.Signo,
                });

                return TaskRunState::RunExit;
            }
            SignalAction::STOP => self.Thread().initiateGroupStop(info),
            SignalAction::IGNORE => {
                info!("Signal {}: ignored", info.Signo)
            }
            SignalAction::HANDLER => {
                info!("Signal {}: delivering to handler", info.Signo);
                let res = self.deliverSignalToHandler(info, &act);
                match res {
                    Err(e) => {
                        info!(
                            "Failed to deliver signal {:?} to user handler: {:?}",
                            info, e
                        );

                        self.Thread()
                            .forceSignal(Signal(Signal::SIGSEGV), info.Signo == Signal::SIGSEGV);
                        self.Thread()
                            .SendSignal(&SignalInfoPriv(Signal::SIGSEGV))
                            .unwrap();
                    }
                    Ok(()) => {
                        return TaskRunState::RunSyscallRet;
                    }
                }
            }
            _ => {
                //todo: fix this
                panic!(
                    "Unknown signal action {:?}, {}",
                    &info, "todo...................."
                )
            }
        }

        return TaskRunState::RunInterrupt;
    }

    pub const FP_XSTATE_MAGIC1: u32 = 0x46505853;
    pub const FP_XSTATE_MAGIC2: u32 = 0x46505845;
    pub const FP_XSTATE_MAGIC2_SIZE: usize = 4;
    pub const UC_FP_XSTATE: u64 = 1;
    // xsave features that are always enabled in signal frame fpstate.
    pub const XFEATURE_MASK_FPSSE: u64 = 0x3;

    #[cfg(target_arch="x86_64")]
    pub fn deliverSignalToHandler(&mut self, info: &SignalInfo, sigAct: &SigAct) -> Result<()> {
        let pt = self.GetPtRegs();
        let mut userStack = Stack::New(pt.get_stack_pointer() - 128); // red zone


        if sigAct.flags.IsOnStack() && self.signalStack.IsEnable() {
            self.signalStack.SetOnStack();
            if !self.signalStack.Contains(pt.get_stack_pointer()) {
                userStack = Stack::New(self.signalStack.Top());
            }
        }

        let (mut fpSize, fpAlign) = HostFeatureSet().ExtendedStateSize();
        fpSize += Self::FP_XSTATE_MAGIC2_SIZE as u32;
        let fpStart = (userStack.sp - fpSize as u64) & !(fpAlign as u64 - 1);

        userStack.sp = fpStart + fpSize as u64;
        userStack.PushU32(self, Self::FP_XSTATE_MAGIC2)?;
        self.SaveFp();
        let fpstate = self.archfpstate.as_ref().unwrap().Slice();
        if fpstate.len() > 512 {
            userStack.PushSlice(
                self,
                &self.archfpstate.as_ref().unwrap().Slice()[512..],
            )?;
        }

        let fpsw = FPSoftwareFrame {
            Magic1: Self::FP_XSTATE_MAGIC1,
            ExtendedSize: fpSize as u32,
            Xfeatures: Self::XFEATURE_MASK_FPSSE | HostFeatureSet().ValidXCR0Mask(),
            XstateSize: fpSize as u32 - Self::FP_XSTATE_MAGIC2_SIZE as u32,
            ..Default::default()
        };

        userStack.PushType::<FPSoftwareFrame>(self, &fpsw)?;
        let fpstateAddr = userStack.PushSlice(self, &fpstate[..464])?;

        self.archfpstate = Some(Box::new(ArchFPState::default()));

        let t = self.Thread();
        let mut mask = t.lock().signalMask;
        let haveSavedSignalMask = t.lock().haveSavedSignalMask;
        if haveSavedSignalMask {
            mask = t.lock().savedSignalMask;
            t.lock().haveSavedSignalMask = false;
        }

        let mut newMask = t.lock().signalMask;
        newMask.0 |= sigAct.mask;
        if !sigAct.flags.IsNoDefer() {
            newMask.0 |= SignalSet::New(Signal(info.Signo)).0;
        }
        t.SetSignalMask(newMask);

        let mut cr2 = 0;
        if info.Signo == Signal::SIGBUS || info.Signo == Signal::SIGSEGV {
            let fault = info.SigFault();
            cr2 = fault.addr;
        }

        let mut ctx = UContext::New(pt, mask.0, cr2, fpstateAddr, &self.signalStack);
        ctx.Flags |= Self::UC_FP_XSTATE;

        let sigInfoAddr = userStack.PushType::<SignalInfo>(self, info)?;
        let sigCtxAddr = userStack.PushType::<UContext>(self, &ctx)?;

        let signo = info.Signo as u64;
        let rsp = userStack.PushU64(self, sigAct.restorer)?;
        info!(
            "start enter user, the address is {:?}, rsp is {:x}, signo is {}",
            sigAct, rsp, signo
        );
        let currTask = Task::Current();
        let regs = currTask.GetPtRegs();
        *regs = PtRegs::default();
        regs.rsp = rsp;
        regs.rcx = sigAct.handler;  // x86 return address
        regs.r11 = 0x2;             // x86 rflags
        regs.rdi = signo;           // paras to handler
        regs.rsi = sigInfoAddr;
        regs.rdx = sigCtxAddr;
        regs.rax = 0;
        regs.rip = regs.rcx;
        regs.eflags = regs.r11;

        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    pub fn SignalReturn(&mut self, _rt: bool) -> Result<i64> {
        // sigret from user signal handler
        // sys_sigreturn
        let pt = self.GetPtRegs();

        let mut userStack = Stack::New(pt.get_stack_pointer());
        let mut uc = UContext::default();
        userStack.PopType::<UContext>(self, &mut uc)?;
        let mut sigInfo = SignalInfo::default();
        userStack.PopType::<SignalInfo>(self, &mut sigInfo)?;

        if uc.MContext.fpstate == 0 {
            self.archfpstate = Some(Box::new(ArchFPState::default()));
        } else {
            userStack.sp = uc.MContext.fpstate;
            let slice = self.archfpstate.as_ref().unwrap().Slice();
            userStack.PopSlice(self, slice)?;
            self.archfpstate.as_ref().unwrap().SanitizeUser();
            self.savefpsate = true;
        }

        // restore user stack?
        let alt = uc.Stack;

        self.SetSignalStack(alt);

        let cEflags = pt.eflags;
        let nEflags = uc.MContext.eflags;

        pt.Set(&uc.MContext);

        let oldMask = uc.MContext.oldmask & !(UNBLOCKED_SIGNALS.0);
        let t = self.Thread();
        t.SetSignalMask(SignalSet(oldMask));

        pt.eflags =
            (cEflags & !EflagsDef::EFLAGS_RESTOREABLE) | (nEflags & EflagsDef::EFLAGS_RESTOREABLE);
        pt.orig_rax = core::u64::MAX;

        if t.lock().HasSignal() {
            t.lock().interruptSelf();
        }

        return Err(Error::SysCallRetCtrl(TaskRunState::RunSyscallRet));
    }


    #[cfg(target_arch = "aarch64")]
    pub fn deliverSignalToHandler(&mut self, info: &SignalInfo, sigAct: &SigAct) -> Result<()> {
        let pt = self.GetPtRegs();
        let mut userStack = Stack::New(pt.get_stack_pointer() - 128); // red zone

        // if user specifies their own stack
        if sigAct.flags.IsOnStack() && self.signalStack.IsEnable() {
            self.signalStack.SetOnStack();
            if !self.signalStack.Contains(pt.get_stack_pointer()) {
                userStack = Stack::New(self.signalStack.Top());
            }
        }
        let t = self.Thread();
        let mut mask = t.lock().signalMask;
        let haveSavedSignalMask = t.lock().haveSavedSignalMask;
        if haveSavedSignalMask {
            mask = t.lock().savedSignalMask;
            t.lock().haveSavedSignalMask = false;
        }

        let mut newMask = t.lock().signalMask;
        newMask.0 |= sigAct.mask;
        if !sigAct.flags.IsNoDefer() {
            newMask.0 |= SignalSet::New(Signal(info.Signo)).0;
        }
        t.SetSignalMask(newMask);

        let mut fault_addr = 0;
        if info.Signo == Signal::SIGBUS || info.Signo == Signal::SIGSEGV {
            fault_addr = info.SigFault().addr;
        }

        // set up signal frame

        // let's forget about the fp units for the moment
        let uc = UContext::New(pt, mask.0, fault_addr, 0, &self.signalStack);
        // TODO if SA_SIGINFO is not requested, the handler takes the form
        //      void handler(int signo)
        // we should push the siginfo and sigctx conditionally
        let sigInfoAddr = userStack.PushType::<SignalInfo>(self, info)?;
        let sigCtxAddr = userStack.PushType::<UContext>(self, &uc)?;
        let signo = info.Signo as u64;
        let currTask = Task::Current();
        let regs = currTask.GetPtRegs();
        regs.sp = sigCtxAddr;
        regs.pc = sigAct.handler;
        regs.pstate = 0x0;

        // parameters to the handler
        // handler(int signo, siginfo_t *info, void *context)
        regs.regs[0] = signo;
        regs.regs[1] = sigInfoAddr;
        regs.regs[2] = sigCtxAddr;
        // TODO set frame pointer regs[29] if there are other sig frames
        if sigAct.flags.HasRestorer() {
            regs.regs[30] = sigAct.restorer;
        } else {
            let vdsoAddr = Task::Current().mm.GetUserVDSOBase();
            assert!(vdsoAddr != 0);
            regs.regs[30] = vdsoAddr + VDSO.get_symbol_page_offset("__kernel_rt_sigreturn")
                                           .expect("aarch64: __kernel_rt_sigreturn not set.");
        }
        return Ok(());
    }

    #[cfg(target_arch = "aarch64")]
    pub fn SignalReturn(&mut self, _rt: bool) -> Result<i64> {
        let pt = self.GetPtRegs();
        // pop the signal frame
        let mut userStack = Stack::New(pt.get_stack_pointer());
        let mut uc = UContext::default();
        userStack.PopType::<UContext>(self, &mut uc)?;
        let mut sigInfo = SignalInfo::default();
        userStack.PopType::<SignalInfo>(self, &mut sigInfo)?;

        let alt = uc.Stack;
        self.SetSignalStack(alt);
        // restore user context
        // ignore restoring fpstate for now.
        pt.Set(&uc.MContext);

        let oldMask = uc.MContext.oldmask & !(UNBLOCKED_SIGNALS.0);
        let t = self.Thread();
        t.SetSignalMask(SignalSet(oldMask));
        // TODO restart syscall if needed
        if t.lock().HasSignal() {
            t.lock().interruptSelf();
        }

        // TODO Segfault if badframe
        return Err(Error::SysCallRetCtrl(TaskRunState::RunSyscallRet));
    }
}
