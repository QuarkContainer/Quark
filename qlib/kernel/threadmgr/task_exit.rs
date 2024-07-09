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

use core::sync::atomic::AtomicU64;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::qlib::kernel::PAGE_MGR;
use crate::qlib::proxy::ProxyCommand;

use super::super::super::auth::id::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::boot::controller::WriteWaitAllResponse;
use super::super::threadmgr::pid_namespace::*;
use super::super::threadmgr::thread::*;
use super::super::threadmgr::thread_group::*;
use super::super::threadmgr::threads::*;
use super::super::ExecID;
use super::super::LOADER;
//use super::super::Common::*;
use super::super::super::perf_tunning::*;
use super::super::task::*;
use super::super::SignalDef::*;
use super::task_stop::*;

lazy_static! {
    pub static ref SYS_CALL_TIME: Vec<AtomicU64> = {
        let mut tbl = Vec::with_capacity(500);
        for _ in 0..500 {
            tbl.push(AtomicU64::new(0));
        }
        tbl
    };
    pub static ref QUARK_SYSCALL_TIME: Vec<AtomicU64> = {
        let mut tbl = Vec::with_capacity(10);
        for _ in 0..10 {
            tbl.push(AtomicU64::new(0));
        }
        tbl
    };
    pub static ref SYSPROXY_CALL_TIME: Vec<AtomicU64> = {
        let mut tbl = Vec::with_capacity(500);
        for _ in 0..500 {
            tbl.push(AtomicU64::new(0));
        }
        tbl
    };
}

// An ExitStatus is a value communicated from an exiting task or thread group
// to the party that reaps it.
#[derive(Clone, Copy, Default, Debug)]
pub struct ExitStatus {
    // Code is the numeric value passed to the call to exit or exit_group that
    // caused the exit. If the exit was not caused by such a call, Code is 0.
    pub Code: i32,

    // Signo is the signal that caused the exit. If the exit was not caused by
    // a signal, Signo is 0.
    pub Signo: i32,
}

impl ExitStatus {
    pub fn New(code: i32, signo: i32) -> Self {
        return ExitStatus {
            Code: code,
            Signo: signo,
        };
    }

    // Signaled returns true if the ExitStatus indicates that the exiting task or
    // thread group was killed by a signal.
    pub fn Signaled(&self) -> bool {
        return self.Signo != 0;
    }

    // Status returns the numeric representation of the ExitStatus returned by e.g.
    // the wait4() system call.
    pub fn Status(&self) -> u32 {
        return (((self.Code as u32) & 0xff) << 8) | ((self.Signo as u32) & 0xff);
    }

    // ShellExitCode returns the numeric exit code that Bash would return for an
    // exit status of es.
    pub fn ShellExitCode(&self) -> i32 {
        if self.Signaled() {
            return 128 + self.Signo;
        }

        return self.Code;
    }
}

// TaskExitState represents a step in the task exit path.
//
// "Exiting" and "exited" are often ambiguous; prefer to name specific states.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskExitState {
    // TaskExitNone indicates that the task has not begun exiting.
    TaskExitNone,

    // TaskExitInitiated indicates that the task goroutine has entered the exit
    // path, and the task is no longer eligible to participate in group stops
    // or group signal handling. TaskExitInitiated is analogous to Linux's
    // PF_EXITING.
    TaskExitInitiated,

    // TaskExitZombie indicates that the task has released its resources, and
    // the task no longer prevents a sibling thread from completing execve.
    TaskExitZombie,

    // TaskExitDead indicates that the task's thread IDs have been released,
    // and the task no longer prevents its thread group leader from being
    // reaped. ("Reaping" refers to the transitioning of a task from
    // TaskExitZombie to TaskExitDead.)
    TaskExitDead,
}

impl core::default::Default for TaskExitState {
    fn default() -> Self {
        return Self::TaskExitNone;
    }
}

impl TaskExitState {
    pub fn String(&self) -> &'static str {
        match self {
            TaskExitState::TaskExitNone => return "TaskExitNone",
            TaskExitState::TaskExitInitiated => return "TaskExitInitiated",
            TaskExitState::TaskExitZombie => return "TaskExitZombie",
            TaskExitState::TaskExitDead => return "TaskExitDead",
        }
    }
}

impl ThreadInternal {
    // killLocked marks t as killed by enqueueing a SIGKILL, without causing the
    // thread-group-affecting side effects SIGKILL usually has.
    //
    // Preconditions: The signal mutex must be locked.
    pub fn killLocked(&mut self) {
        if self.stop.is_some() && self.stop.clone().unwrap().Killable() {
            self.endInternalStopLocked();
        }

        self.pendingSignals
            .Enque(
                Box::new(SignalInfo {
                    Signo: Signal::SIGKILL,
                    Code: SignalInfo::SIGNAL_INFO_USER,
                    ..Default::default()
                }),
                None,
            )
            .expect("killLocked fail");

        self.interrupt();
    }

    // killed returns true if t has a SIGKILL pending. killed is analogous to
    // Linux's fatal_signal_pending().
    //
    // Preconditions: The caller must be running on the task goroutine.
    pub fn killed(&self) -> bool {
        let lock = self.tg.lock().signalLock.clone();
        let _s = lock.lock();
        return self.killedLocked();
    }

    pub fn killedLocked(&self) -> bool {
        return self.pendingSignals.pendingSet.0 & SignalSet::New(Signal(Signal::SIGKILL)).0 != 0;
    }

    // advanceExitStateLocked checks that t's current exit state is oldExit, then
    // sets it to newExit. If t's current exit state is not oldExit,
    // advanceExitStateLocked panics.
    //
    // Preconditions: The TaskSet mutex must be locked.
    pub fn advanceExitStateLocked(&mut self, oldExit: TaskExitState, newExit: TaskExitState) {
        info!(
            "advanceExitStateLocked[{}] {:?}=>{:?}",
            self.id, oldExit, newExit
        );
        if self.exitState != oldExit {
            panic!(
                "Transitioning from exit state {:?} to {:?}: unexpected preceding state {:?}",
                oldExit, newExit, self.exitState
            )
        }

        self.exitState = newExit;
    }
}

// Task events that can be waited for.

// EventExit represents an exit notification generated for a child thread
// group leader or a tracee under the conditions specified in the comment
// above runExitNotify.
pub const EVENT_EXIT: EventMask = 1;

// EventChildGroupStop occurs when a child thread group completes a group
// stop (i.e. all tasks in the child thread group have entered a stopped
// state as a result of a group stop).
pub const EVENT_CHILD_GROUP_STOP: EventMask = 1 << 1;

// EventTraceeStop occurs when a task that is ptraced by a task in the
// notified thread group enters a ptrace stop (see ptrace(2)).
pub const EVENT_TRACEE_STOP: EventMask = 1 << 2;

// EventGroupContinue occurs when a child thread group, or a thread group
// whose leader is ptraced by a task in the notified thread group, that had
// initiated or completed a group stop leaves the group stop, due to the
// child thread group or any task in the child thread group being sent
// SIGCONT.
pub const EVENT_GROUP_CONTINUE: EventMask = 1 << 3;

impl Thread {
    // findReparentTargetLocked returns the task to which t's children should be
    // reparented. If no such task exists, findNewParentLocked returns nil.
    //
    // Preconditions: The TaskSet mutex must be locked.
    pub fn findReparentTargetLocked(&self) -> Option<Thread> {
        let tg = self.lock().tg.clone();
        // Reparent to any sibling in the same thread group that hasn't begun
        // exiting.
        match tg.anyNonExitingTaskLocked() {
            Some(t2) => return Some(t2),
            None => (),
        }

        // "A child process that is orphaned within the namespace will be
        // reparented to [the init process for the namespace] ..." -
        // pid_namespaces(7)
        let pidns = tg.PIDNamespace();
        let init = match pidns.lock().tasks.get(&INIT_TID) {
            Some(init) => init.clone(),
            None => return None,
        };

        let inittg = init.lock().tg.clone();
        return inittg.anyNonExitingTaskLocked();
    }

    // PrepareExit indicates an exit with status es.
    //
    // Preconditions: The caller must be running on the task goroutine.
    pub fn PrepareExit(&self, es: ExitStatus) {
        let mut t = self.lock();
        let lock = t.tg.lock().signalLock.clone();
        let _s = lock.lock();
        t.exitStatus = es;
    }

    // PrepareGroupExit indicates a group exit with status es to t's thread group.
    //
    // PrepareGroupExit is analogous to Linux's do_group_exit(), except that it
    // does not tail-call do_exit(), except that it *does* set Task.exitStatus.
    // (Linux does not do so until within do_exit(), since it reuses exit_code for
    // ptrace.)
    //
    // Preconditions: The caller must be running on the task goroutine.
    pub fn PrepareGroupExit(&self, es: ExitStatus) {
        let tg = self.lock().tg.clone();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        let exiting = tg.lock().exiting;
        let execing = tg.lock().execing.Upgrade();
        if exiting || execing.is_some() {
            // Note that if t.tg.exiting is false but t.tg.execing is not nil, i.e.
            // this "group exit" is being executed by the killed sibling of an
            // execing task, then Task.Execve never set t.tg.exitStatus, so it's
            // still the zero value. This is consistent with Linux, both in intent
            // ("all other threads ... report death as if they exited via _exit(2)
            // with exit code 0" - ptrace(2), "execve under ptrace") and in
            // implementation (compare fs/exec.c:de_thread() =>
            // kernel/signal.c:zap_other_threads() and
            // kernel/exit.c:do_group_exit() =>
            // include/linux/sched.h:signal_group_exit()).
            self.lock().exitStatus = tg.lock().exitStatus;
            return;
        }

        tg.lock().exiting = true;
        tg.lock().exitStatus = es;
        self.lock().exitStatus = es;
        let tasks: Vec<Thread> = tg.lock().tasks.iter().cloned().collect();
        for sibling in &tasks {
            if *sibling != *self {
                sibling.lock().killLocked();
            }
        }
    }

    // exitThreadGroup transitions t to TaskExitInitiated, indicating to t's thread
    // group that it is no longer eligible to participate in group activities. It
    // returns true if t is the last task in its thread group to call
    // exitThreadGroup.
    pub fn exitThreadGroup(&self) -> bool {
        let tg = self.ThreadGroup();
        let pidns = tg.PIDNamespace();
        let _owner = pidns.lock().owner.clone();
        let lock = tg.lock().signalLock.clone();

        let sig;
        let notifyParent;
        let last;
        {
            let _s = lock.lock();

            self.lock().advanceExitStateLocked(
                TaskExitState::TaskExitNone,
                TaskExitState::TaskExitInitiated,
            );
            tg.lock().activeTasks -= 1;
            last = tg.lock().activeTasks == 0;

            // Ensure that someone will handle the signals we can't.
            self.setSignalMaskLocked(SignalSet(!0));

            if !self.lock().groupStopPending {
                return last;
            }

            self.lock().groupStopPending = false;
            sig = tg.lock().groupStopSignal;
            notifyParent = self.lock().participateGroupStopLocked();
        }

        let leader = tg.lock().leader.Upgrade().unwrap();
        if notifyParent && leader.lock().parent.is_some() {
            let parent = leader.lock().parent.clone().unwrap();
            parent.signalStop(self, SignalInfo::CLD_STOPPED, sig.0);
            let tgOfParent = parent.lock().tg.clone();
            tgOfParent.lock().eventQueue.Notify(EVENT_CHILD_GROUP_STOP);
        }

        return last;
    }

    pub fn exitChildren(&self) {
        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _l = owner.WriteLock();

        let newParent = self.findReparentTargetLocked();
        if newParent.is_none() {
            // "If the init process of a PID namespace terminates, the kernel
            // terminates all of the processes in the namespace via a SIGKILL
            // signal." - pid_namespaces(7)
            pidns.lock().exiting = true;
            let tgids: Vec<ThreadGroup> = pidns.lock().tgids.keys().cloned().collect();
            for other in &tgids {
                if *other == self.lock().tg.clone() {
                    continue;
                }
                let lock = other.lock().signalLock.clone();
                let _s = lock.lock();

                let leader = other.lock().leader.Upgrade().unwrap();
                leader
                    .sendSignalLocked(
                        &SignalInfo {
                            Signo: Signal::SIGKILL,
                            ..Default::default()
                        },
                        true,
                    )
                    .unwrap();
            }
        }

        // This is correct even if newParent is nil (it ensures that children don't
        // wait for a parent to reap them.)
        let creds = self.lock().creds.clone();
        let mut children = Vec::new();
        for c in &self.lock().children {
            children.push(c.clone());
        }
        for c in &children {
            let sig = c.ParentDeathSignal();
            if sig.0 != 0 {
                let mut sigInfo = SignalInfo {
                    Signo: sig.0,
                    Code: SignalInfo::SIGNAL_INFO_USER,
                    ..Default::default()
                };

                let sigchild = sigInfo.SigChld();
                let tg = c.lock().tg.clone();
                let pidns = tg.PIDNamespace();
                let userns = c.UserNamespace();

                sigchild.pid = *pidns.lock().tids.get(self).unwrap();
                sigchild.uid = creds.lock().RealKUID.In(&userns).OrOverflow().0;

                let lock = tg.lock().signalLock.clone();
                let _s = lock.lock();

                c.sendSignalLocked(&sigInfo, true).unwrap();
            }

            c.reparentLocked(&newParent);
            if newParent.is_some() {
                newParent.clone().unwrap().lock().children.insert(c.clone());
            }
        }
    }

    pub fn reparentLocked(&self, parent: &Option<Thread>) {
        let oldParent = self.lock().parent.clone();
        self.lock().parent = parent.clone();
        /*{
            let oldid = match oldParent.clone() {
                Some(t) => t.lock().id,
                None => 0
            };

            let parent = self.lock().parent.clone();
            if parent.is_some() {
                error!("reparentLocked set {} old parent is {} parent to {}", self.lock().id, oldid, parent.unwrap().lock().id);
            } else {
                error!("reparentLocked set {} old parent is {} parent None", self.lock().id, oldid);
            }

        }*/

        // If a thread group leader's parent changes, reset the thread group's
        // termination signal to SIGCHLD and re-check exit notification. (Compare
        // kernel/exit.c:reparent_leader().)
        let tg = self.lock().tg.clone();
        let leader = tg.lock().leader.Upgrade();
        if Some(self.clone()) != leader {
            return;
        }

        if oldParent.is_none() && parent.is_none() {
            return;
        }

        if oldParent.is_some() && parent.is_some() {
            let oldtg = oldParent.clone().unwrap().lock().tg.clone();
            let parenttg = parent.clone().unwrap().lock().tg.clone();
            if oldtg == parenttg {
                return;
            }
        }

        tg.lock().terminationSignal = Signal(Signal::SIGCHLD);
        let exitParentNotified = self.lock().exitParentNotified;
        let exitParentAcked = self.lock().exitParentAcked;
        if exitParentNotified && !exitParentAcked {
            self.lock().exitParentNotified = false;
            self.exitNotifyLocked()
        }
    }

    pub fn waitOnce(&self, opts: &WaitOptions) -> Result<WaitResult> {
        let mut anyWaitableTasks = false;

        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _lock = owner.WriteLock();

        if opts.SiblingChildren {
            // We can wait on the children and tracees of any task in the
            // same thread group.
            let parents: Vec<Thread> = tg.lock().tasks.iter().cloned().collect();
            for parent in &parents {
                let (wr, any) = self.waitParentLocked(opts, parent);
                if wr.is_some() {
                    return Ok(wr.unwrap());
                }

                anyWaitableTasks = anyWaitableTasks || any;
            }
        } else {
            // We can only wait on this task.
            let (wr, any) = self.waitParentLocked(opts, self);
            anyWaitableTasks = any;
            if wr.is_some() {
                return Ok(wr.unwrap());
            }
        }

        if anyWaitableTasks {
            return Err(Error::ErrNoWaitableEvent);
        }

        return Err(Error::SysError(SysErr::ECHILD));
    }

    pub fn waitParentLocked(
        &self,
        opts: &WaitOptions,
        parent: &Thread,
    ) -> (Option<WaitResult>, bool) {
        let mut anyWaitableTasks = false;

        let parenttg = parent.lock().tg.clone();
        let pidns = parenttg.PIDNamespace();

        let children: Vec<Thread> = parent.lock().children.iter().cloned().collect();
        for child in &children {
            let child = child.clone();
            if !opts.matchesTask(&child, &pidns) {
                continue;
            }

            // Non-leaders don't notify parents on exit and aren't eligible to
            // be waited on.
            let childtg = child.lock().tg.clone();
            let childleader = childtg.lock().leader.Upgrade();
            if opts.Events & EVENT_EXIT != 0
                && Some(child.clone()) == childleader
                && !child.lock().exitParentAcked
            {
                anyWaitableTasks = true;
                let wr = self.waitCollectZombieLocked(&child, opts);
                if wr.is_some() {
                    return (wr, anyWaitableTasks);
                }
            }

            // Check for group stops and continues. Tasks that have passed
            // TaskExitInitiated can no longer participate in group stops.
            if opts.Events & (EVENT_CHILD_GROUP_STOP | EVENT_GROUP_CONTINUE) == 0 {
                continue;
            }

            if child.lock().exitState >= TaskExitState::TaskExitInitiated {
                continue;
            }

            anyWaitableTasks = true;
            if opts.Events & EVENT_CHILD_GROUP_STOP != 0 {
                let wr = self.waitCollectChildGroupStopLocked(&child, opts);
                if wr.is_some() {
                    return (wr, anyWaitableTasks);
                }
            }

            if opts.Events & EVENT_GROUP_CONTINUE != 0 {
                let wr = self.waitCollectGroupContinueLocked(&child, opts);
                if wr.is_some() {
                    return (wr, anyWaitableTasks);
                }
            }
        }

        return (None, anyWaitableTasks);
    }

    pub fn waitCollectZombieLocked(
        &self,
        target: &Thread,
        opts: &WaitOptions,
    ) -> Option<WaitResult> {
        if !target.lock().exitParentNotified {
            return None;
        }

        let tg = self.ThreadGroup();
        let target = target.clone();
        let targetTg = target.ThreadGroup();
        let targetLead = targetTg.lock().leader.Upgrade();

        // Zombied thread group leaders are never waitable until their thread group
        // is otherwise empty. Usually this is caught by the
        // target.exitParentNotified check above, but if t is both (in the thread
        // group of) target's tracer and parent, asPtracer may be true.
        if targetLead.is_some() && target == targetLead.unwrap() && targetTg.lock().tasksCount != 1
        {
            return None;
        }

        let pidns = targetTg.PIDNamespace();
        let pid = pidns.IDOfTaskLocked(&target); // .lock().tids.get(&target).unwrap();

        let creds = target.Credentials();
        let userns = self.UserNamespace();
        let uid = creds.lock().RealKUID.In(&userns).OrOverflow();

        let mut status = target.lock().exitStatus.Status();

        if !opts.ConsumeEvent {
            return Some(WaitResult {
                Thread: target.clone(),
                TID: pid,
                UID: uid,
                Event: EVENT_EXIT,
                Status: status,
            });
        }

        // Surprisingly, the exit status reported by a non-consuming wait can
        // differ from that reported by a consuming wait; the latter will return
        // the group exit code if one is available.

        if targetTg.lock().exiting {
            status = targetTg.lock().exitStatus.Status();
        }

        let targetParent = target.lock().parent.clone();
        let exitParentNotified = target.lock().exitParentNotified;

        assert!(
            targetParent.is_some(),
            "waitCollectZombieLocked parent should not be none"
        );
        let parentTg = targetParent.unwrap().lock().tg.clone();
        let targetLead = targetTg.lock().leader.Upgrade();
        if parentTg != targetTg && exitParentNotified {
            target.lock().exitParentAcked = true;
            if targetLead.is_some() && target == targetLead.unwrap() {
                // target.tg.exitedCPUStats doesn't include target.CPUStats() yet,
                // and won't until after target.exitNotifyLocked() (maybe). Include
                // target.CPUStats() explicitly. This is consistent with Linux,
                // which accounts an exited task's cputime to its thread group in
                // kernel/exit.c:release_task() => __exit_signal(), and uses
                // thread_group_cputime_adjusted() in wait_task_zombie().
                let mut tglock = tg.lock();
                let targettglock = targetTg.lock();
                tglock.childCPUStats.Accumulate(&target.CPUStats());
                tglock
                    .childCPUStats
                    .Accumulate(&targettglock.exitedCPUStats);
                tglock.childCPUStats.Accumulate(&targettglock.childCPUStats);

                // Update t's child max resident set size. The size will be the maximum
                // of this thread's size and all its childrens' sizes.
                let maxRSS = tglock.maxRSS;
                if maxRSS < targettglock.maxRSS {
                    tglock.childMaxRSS = targettglock.maxRSS;
                }

                let childMaxRSS = tglock.childMaxRSS;
                if childMaxRSS < targettglock.childMaxRSS {
                    tglock.childMaxRSS = targettglock.childMaxRSS;
                }
            }
        }

        target.exitNotifyLocked();
        return Some(WaitResult {
            Thread: target,
            TID: pid,
            UID: uid,
            Event: EVENT_EXIT,
            Status: status,
        });
    }

    // updateRSSLocked updates t.tg.maxRSS.
    //
    // Preconditions: The TaskSet mutex must be locked for writing.
    pub fn updateRSSLocked(&self) {
        let mm = self.lock().memoryMgr.clone();
        let mmMaxRSS = mm.MaxResidentSetSize();
        let tg = self.lock().tg.clone();
        if tg.lock().maxRSS < mmMaxRSS {
            tg.lock().maxRSS = mmMaxRSS;
        }
    }

    pub fn waitCollectChildGroupStopLocked(
        &self,
        target: &Thread,
        opts: &WaitOptions,
    ) -> Option<WaitResult> {
        let targetTg = target.ThreadGroup();
        let lock = targetTg.lock().signalLock.clone();
        let _s = lock.lock();

        if !targetTg.lock().groupStopWaitable {
            return None;
        }

        let pidns = targetTg.PIDNamespace();
        let pid = pidns.IDOfTaskLocked(target);

        let creds = target.Credentials();
        let userns = self.UserNamespace();
        let uid = creds.lock().RealKUID.In(&userns).OrOverflow();

        let signal = targetTg.lock().groupStopSignal;
        if opts.ConsumeEvent {
            targetTg.lock().groupStopWaitable = false;
        }

        return Some(WaitResult {
            Thread: target.clone(),
            TID: pid,
            UID: uid,
            Event: EVENT_CHILD_GROUP_STOP,
            Status: ((signal.0 as u32) & 0xff) << 8 | 0x7f,
        });
    }

    pub fn waitCollectGroupContinueLocked(
        &self,
        target: &Thread,
        opts: &WaitOptions,
    ) -> Option<WaitResult> {
        let tg = target.lock().tg.clone();
        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        if !tg.lock().groupContWaitable {
            return None;
        }

        let pidns = self.PIDNamespace();
        let pid = *pidns.lock().tids.get(target).unwrap();

        let creds = target.Credentials();
        let userns = self.UserNamespace();
        let uid = creds.lock().RealKUID.In(&userns).OrOverflow();

        if opts.ConsumeEvent {
            tg.lock().groupContWaitable = false;
        }

        return Some(WaitResult {
            Thread: target.clone(),
            TID: pid,
            UID: uid,
            Event: EVENT_GROUP_CONTINUE,
            Status: 0xffff,
        });
    }

    // exitNotifyLocked is called after changes to t's state that affect exit
    // notification.
    //
    //
    // Preconditions: The TaskSet mutex must be locked for writing.
    pub fn exitNotifyLocked(&self) {
        let t = self.clone();
        if t.lock().exitState != TaskExitState::TaskExitZombie {
            return;
        }

        let exitTracerNotified = t.lock().exitTracerNotified;
        if !exitTracerNotified {
            t.lock().exitTracerNotified = true;
            t.lock().exitTracerAcked = true;
        }

        let exitTracerAcked = t.lock().exitTracerAcked;
        let exitParentNotified = t.lock().exitParentNotified;
        if exitTracerAcked && !exitParentNotified {
            let tg = t.lock().tg.clone();
            let leader = tg.lock().leader.Upgrade();

            if Some(t.clone()) != leader {
                t.lock().exitParentNotified = true;
                t.lock().exitParentAcked = true;
            } else if tg.lock().tasksCount == 1 {
                t.lock().exitParentNotified = true;
                let parent = t.lock().parent.clone();
                if parent.is_none() {
                    t.lock().exitParentAcked = true;
                } else {
                    // "POSIX.1-2001 specifies that if the disposition of SIGCHLD is
                    // set to SIG_IGN or the SA_NOCLDWAIT flag is set for SIGCHLD (see
                    // sigaction(2)), then children that terminate do not become
                    // zombies and a call to wait() or waitpid() will block until all
                    // children have terminated, and then fail with errno set to
                    // ECHILD. (The original POSIX standard left the behavior of
                    // setting SIGCHLD to SIG_IGN unspecified. Note that even though
                    // the default disposition of SIGCHLD is "ignore", explicitly
                    // setting the disposition to SIG_IGN results in different
                    // treatment of zombie process children.) Linux 2.6 conforms to
                    // this specification." - wait(2)
                    //
                    // Some undocumented Linux-specific details:
                    //
                    // - All of the above is ignored if the termination signal isn't
                    // SIGCHLD.
                    //
                    // - SA_NOCLDWAIT causes the leader to be immediately reaped, but
                    // does not suppress the SIGCHLD.

                    let signal = tg.lock().terminationSignal;
                    let mut signalParent = signal.IsValid();
                    let parentTg = parent.unwrap().lock().tg.clone();
                    let lock = parentTg.lock().signalLock.clone();

                    {
                        let _s = lock.lock();
                        let sh = parentTg.lock().signalHandlers.clone();
                        if signal.0 == Signal::SIGCHLD {
                            match sh.lock().actions.get(&signal.0) {
                                None => (),
                                Some(act) => {
                                    if act.handler == SigAct::SIGNAL_ACT_IGNORE {
                                        t.lock().exitParentAcked = true;
                                        signalParent = false;
                                    } else if act.flags.IsNoCldWait() {
                                        t.lock().exitParentAcked = true;
                                    }
                                }
                            }
                        }

                        if signalParent {
                            let leader = parentTg.lock().leader.Upgrade();
                            let terminationSignal = tg.lock().terminationSignal;
                            let parent = t.lock().parent.clone().unwrap();
                            let signalInfo = t.exitNotificationSignal(terminationSignal, &parent);
                            leader.unwrap().sendSignalLocked(&signalInfo, true).unwrap();
                        }
                    }

                    // If a task in the parent was waiting for a child group stop
                    // or continue, it needs to be notified of the exit, because
                    // there may be no remaining eligible tasks (so that wait
                    // should return ECHILD).
                    parentTg
                        .lock()
                        .eventQueue
                        .Notify(EVENT_EXIT | EVENT_CHILD_GROUP_STOP | EVENT_GROUP_CONTINUE);
                }
            }
        }

        let exitTracerAcked = t.lock().exitTracerAcked;
        let exitParentAcked = t.lock().exitParentAcked;
        if exitTracerAcked && exitParentAcked {
            t.lock()
                .advanceExitStateLocked(TaskExitState::TaskExitZombie, TaskExitState::TaskExitDead);

            let tg = t.lock().tg.clone();
            let mut pidns = tg.PIDNamespace();

            loop {
                let tid = *pidns.lock().tids.get(&t).unwrap();
                pidns.lock().tasks.remove(&tid);
                pidns.lock().tids.remove(&t);
                let leader = tg.lock().leader.Upgrade();
                if Some(t.clone()) == leader {
                    pidns.lock().tgids.remove(&tg);
                }

                let tmp = pidns.lock().parent.clone();
                if tmp.is_none() {
                    break;
                }

                pidns = tmp.unwrap();
            }

            let cpuStatus = t.lock().CPUStats();
            tg.lock().exitedCPUStats.Accumulate(&cpuStatus);
            //tg.lock().ioUsage.Accumulate(t.lock().ioUsage);
            let tc = {
                let lock = tg.lock().signalLock.clone();
                let _s = lock.lock();

                tg.lock().tasks.remove(&t);
                tg.lock().tasksCount -= 1;
                let tc = tg.lock().tasksCount;
                tc
            };

            let leader = tg.lock().leader.Upgrade();
            if tc == 1 && Some(t.clone()) != leader {
                // Our fromPtraceDetach doesn't matter here (in Linux terms, this
                // is via a call to release_task()).
                leader.unwrap().exitNotifyLocked();
            } else if tc == 0 {
                let processGroup = tg.lock().processGroup.clone();
                let parentPg = tg.parentPG();
                processGroup.unwrap().decRefWithParent(parentPg);
            }

            let parent = t.lock().parent.clone();
            if parent.is_some() {
                parent.unwrap().lock().children.remove(&t);
                t.lock().parent = None;
            }
        }
    }

    pub fn exitNotificationSignal(&self, sig: Signal, receiver: &Thread) -> SignalInfo {
        let mut info = SignalInfo {
            Signo: sig.0,
            ..Default::default()
        };

        let tg = receiver.lock().tg.clone();
        let pidns = tg.lock().pidns.clone();
        let tid = *pidns.lock().tids.get(self).unwrap();
        info.SigChld().pid = tid;

        let creds = self.Credentials();
        let kuid = creds.lock().RealKUID;
        #[cfg(not(feature = "cc"))]
        let userns = receiver.UserNamespace();
        #[cfg(feature = "cc")]
        let userns = creds.lock().UserNamespace.clone();

        info.SigChld().uid = kuid.In(&userns).OrOverflow().0;

        let signaled = self.lock().exitStatus.Signaled();
        if signaled {
            info.Code = SignalInfo::CLD_KILLED;
            info.SigChld().status = self.lock().exitStatus.Signo;
        } else {
            info.Code = SignalInfo::CLD_EXITED;
            info.SigChld().status = self.lock().exitStatus.Code;
        }

        return info;
    }

    pub fn ExitStatus(&self) -> ExitStatus {
        let ts = self.TaskSet();
        let tg = self.lock().tg.clone();
        let lock = tg.lock().signalLock.clone();
        let _r = ts.read();
        let _s = lock.lock();

        return self.lock().exitStatus;
    }

    pub fn ExitState(&self) -> TaskExitState {
        return self.lock().exitState;
    }

    pub fn ParentDeathSignal(&self) -> Signal {
        return self.lock().parentDeathSignal;
    }

    pub fn SetParentDeathSignal(&self, sig: Signal) {
        self.lock().parentDeathSignal = sig;
    }

    pub fn Signaled(&self) -> bool {
        let tg = self.lock().tg.clone();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();
        let tglock = tg.lock();
        return tglock.exiting && tglock.exitStatus.Signaled();
    }

    pub fn ExitMain(&self) {
        let lastExiter = self.exitThreadGroup();
        let tg = self.lock().tg.clone();

        {
            let pidns = tg.PIDNamespace();
            let owner = pidns.lock().owner.clone();
            let _l = owner.WriteLock();
            self.updateRSSLocked();
        }

        // todo: fix this
        let task = Task::Current();
        // Handle the robust futex list.
        self.ExitRobustList(task);

        self.UnstopVforkParent();

        // If this is the last task to exit from the thread group, release the
        // thread group's resources.
        if lastExiter {
            tg.release();
        }

        self.exitChildren();
        //self.ExitNotify();
    }

    pub fn ExitNotify(&self) {
        let tg = self.lock().tg.clone();
        let cid = tg.lock().containerID.clone();
        let execId = tg.lock().execId.clone();
        let isRootProcess = tg.lock().root;
        let tid = tg.ID();

        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let ownerlock = owner.WriteLock();

        self.lock().advanceExitStateLocked(
            TaskExitState::TaskExitInitiated,
            TaskExitState::TaskExitZombie,
        );

        {
            let mut tglock = tg.lock();
            tglock.liveTasks -= 1;
            tglock.liveThreads.Add(-1);

            info!("living task count:{}", tglock.liveTasks);
            // Check if this completes a sibling's execve.
            if tglock.execing.Upgrade().is_some() && tglock.liveTasks == 1 {
                // execing blocks the addition of new tasks to the thread group, so
                // the sole living task must be the execing one.
                let t = tglock.execing.Upgrade().unwrap();

                tglock.signalLock.lock();

                let mut tlock = t.lock();
                match &tlock.stop {
                    None => (),
                    Some(ref stop) => {
                        if stop.Type() == TaskStopType::EXECSTOP {
                            tlock.endInternalStopLocked();
                        }
                    }
                }
            }
        }

        self.exitNotifyLocked();
        if isRootProcess && tg.lock().liveTasks == 0 {
            let execId = execId.unwrap_or_default();
            info!(
                " sending exit notification for CID:{}, execID:{}",
                &cid, &execId
            );
            WriteWaitAllResponse(cid.clone(), execId.clone(), tg.ExitStatus().Status() as i32);
            let curr = Task::Current();
            LOADER
                .Lock(curr)
                .unwrap()
                .processes
                .remove(&ExecID { cid: cid, pid: tid });
        }
        let taskCnt = owner.write().DecrTaskCount1();
        // error!(
        //     "ExitNotify 4 [{:x}], taskcnt is {}",
        //     self.lock().taskId,
        //     taskCnt
        // );
        if taskCnt == 0 {
            info!("ExitNotify shutdown");
            PAGE_MGR.Clear();

            if super::super::SHARESPACE.config.read().PerfDebug {
                use crate::qlib::kernel::Scale;
                use crate::qlib::SysCallID;

                #[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
                struct PerfStruct {
                    time: u64,
                    callId: SysCallID,
                }

                let mut perfVec = Vec::new();

                for nr in 0..450 {
                    let callId: SysCallID = unsafe { core::mem::transmute(nr as u64) };
                    if SYS_CALL_TIME[nr].load(core::sync::atomic::Ordering::Relaxed) > 0 {
                        let perfStruct = PerfStruct {
                            time: Scale(
                                SYS_CALL_TIME[nr].load(core::sync::atomic::Ordering::Relaxed) as _,
                            ) as u64,
                            callId: callId,
                        };

                        perfVec.push(perfStruct);
                    }
                }

                for i in 0..QUARK_SYSCALL_TIME.len() {
                    let nr = i + 10001; //crate::qlib::syscalls::syscalls::EXTENSION_CALL_OFFSET;
                    let callId: SysCallID = unsafe { core::mem::transmute(nr as u64) };
                    if QUARK_SYSCALL_TIME[i].load(core::sync::atomic::Ordering::Relaxed) > 0 {
                        let perfStruct = PerfStruct {
                            time: Scale(
                                QUARK_SYSCALL_TIME[i].load(core::sync::atomic::Ordering::Relaxed)
                                    as _,
                            ) as u64,
                            callId: callId,
                        };

                        perfVec.push(perfStruct);
                    }
                }

                perfVec.sort();

                error!("syscall time is {:#?}", &perfVec);

                #[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
                struct ProxyPerfStruct {
                    time: u64,
                    callId: ProxyCommand,
                }

                let mut total = 0;
                let mut perfVec = Vec::new();
                for i in 0..SYSPROXY_CALL_TIME.len() {
                    if SYSPROXY_CALL_TIME[i].load(core::sync::atomic::Ordering::Relaxed) > 0 {
                        let gap = Scale(
                            SYSPROXY_CALL_TIME[i].load(core::sync::atomic::Ordering::Relaxed)
                                as i64,
                        ) as u64;
                        let cmd: ProxyCommand = unsafe { core::mem::transmute(i as u64) };
                        let perfStruct = ProxyPerfStruct {
                            time: gap,
                            callId: cmd,
                        };

                        total += gap;

                        perfVec.push(perfStruct);
                    }
                }
                perfVec.sort();

                error!("sys_proxy timeis  {:#?}  total is {} ", &perfVec, total);
            }

            super::super::SHARESPACE.StoreShutdown();
            //PerfStop();
            PerfPrint();
            super::super::perflog::THREAD_COUNTS.lock().Print(false);
            //super::super::AllocatorPrint();
            core::mem::drop(ownerlock);
            let exitStatus = tg.ExitStatus();
            //super::super::PAGE_MGR.PrintRefs();
            super::super::EXIT_CODE.store(exitStatus.ShellExitCode(), QOrdering::SEQ_CST);
        }
    }
}

impl ThreadGroup {
    pub fn anyNonExitingTaskLocked(&self) -> Option<Thread> {
        let tasks: Vec<_> = self.lock().tasks.iter().cloned().collect();
        for t in &tasks {
            if t.lock().exitState == TaskExitState::TaskExitNone {
                return Some(t.clone());
            }
        }

        return None;
    }

    pub fn ExitStatus(&self) -> ExitStatus {
        let ts = self.TaskSet();
        let lock = self.lock().signalLock.clone();
        ts.read();
        let _s = lock.lock();

        {
            let tglock = self.lock();
            if tglock.exiting {
                return tglock.exitStatus;
            }
        }

        //todo: there is chance that the leader is none, need fix
        let leader = self.lock().leader.Upgrade().unwrap();
        return leader.lock().exitStatus;
    }

    pub fn TerminationSignal(&self) -> Signal {
        let ts = self.TaskSet();
        let _r = ts.ReadLock();
        return self.lock().terminationSignal;
    }
}

impl TaskSet {
    // Kill requests that all tasks in ts exit as if group exiting with status es.
    // Kill does not wait for tasks to exit.
    //
    // Kill has no analogue in Linux; it's provided for save/restore only.
    pub fn Kill(&self, es: ExitStatus) {
        let _r = self.ReadLock();
        let ts = self.read();

        let pidns = ts.root.clone().unwrap();
        pidns.lock().exiting = true;

        let threads: Vec<_> = pidns.lock().tids.keys().cloned().collect();
        for t in &threads {
            let mut t = t.lock();
            let lock = t.tg.lock().signalLock.clone();
            let _s = lock.lock();

            if !t.tg.lock().exiting {
                t.tg.lock().exiting = true;
                t.tg.lock().exitStatus = es;
            }

            t.killLocked();
        }
    }
}

// WaitOptions controls the behavior of Task.Wait.
#[derive(Debug, Clone, Default)]
pub struct WaitOptions {
    // If SpecificTID is non-zero, only events from the task with thread ID
    // SpecificTID are eligible to be waited for. SpecificTID is resolved in
    // the PID namespace of the waiter (the method receiver of Task.Wait). If
    // no such task exists, or that task would not otherwise be eligible to be
    // waited for by the waiting task, then there are no waitable tasks and
    // Wait will return ECHILD.
    pub SpecificTID: ThreadID,

    // If SpecificPGID is non-zero, only events from ThreadGroups with a
    // matching ProcessGroupID are eligible to be waited for. (Same
    // constraints as SpecificTID apply.)
    pub SpecificPGID: ProcessGroupID,

    // Terminology note: Per waitpid(2), "a clone child is one which delivers
    // no signal, or a signal other than SIGCHLD to its parent upon
    // termination." In Linux, termination signal is technically a per-task
    // property rather than a per-thread-group property. However, clone()
    // forces no termination signal for tasks created with CLONE_THREAD, and
    // execve() resets the termination signal to SIGCHLD, so all
    // non-group-leader threads have no termination signal and are therefore
    // "clone tasks".

    // If NonCloneTasks is true, events from non-clone tasks are eligible to be
    // waited for.
    pub NonCloneTasks: bool,

    // If CloneTasks is true, events from clone tasks are eligible to be waited
    // for.
    pub CloneTasks: bool,

    // If SiblingChildren is true, events from children tasks of any task
    // in the thread group of the waiter are eligible to be waited for.
    pub SiblingChildren: bool,

    // Events is a bitwise combination of the events defined above that specify
    // what events are of interest to the call to Wait.
    pub Events: EventMask,

    // If ConsumeEvent is true, the Wait should consume the event such that it
    // cannot be returned by a future Wait. Note that if a task exit is
    // consumed in this way, in most cases the task will be reaped.
    pub ConsumeEvent: bool,

    // If BlockInterruptErr is not nil, Wait will block until either an event
    // is available or there are no tasks that could produce a waitable event;
    // if that blocking is interrupted, Wait returns BlockInterruptErr. If
    // BlockInterruptErr is nil, Wait will not block.
    pub BlockInterruptErr: Option<Error>,
}

impl WaitOptions {
    // Preconditions: The TaskSet mutex must be locked (for reading or writing).
    pub fn matchesTask(&self, t: &Thread, pidns: &PIDNamespace) -> bool {
        if self.SpecificTID != 0 {
            // && self.SpecificTID != *pidns.lock().tids.get(t).unwrap() {
            let id = match pidns.lock().tids.get(t) {
                None => {
                    return false;
                }
                Some(id) => *id,
            };

            if id != self.SpecificTID {
                return false;
            }
        }

        let tg = t.lock().tg.clone();
        let pg = tg.lock().processGroup.clone();
        if self.SpecificPGID != 0
            && pg.is_some()
            && self.SpecificPGID != *pidns.lock().pgids.get(&pg.unwrap()).unwrap()
        {
            return false;
        }

        let leader = tg.lock().leader.Upgrade();
        if Some(t.clone()) == leader && tg.lock().terminationSignal.0 == Signal::SIGCHLD {
            return self.NonCloneTasks;
        }

        return self.CloneTasks;
    }
}

// WaitResult contains information about a waited-for event.
#[derive(Clone)]
pub struct WaitResult {
    pub Thread: Thread,

    // TID is the thread ID of Task in the PID namespace of the task that
    // called Wait (that is, the method receiver of the call to Task.Wait). TID
    // is provided because consuming exit waits cause the thread ID to be
    // deallocated.
    pub TID: ThreadID,

    // UID is the real UID of Task in the user namespace of the task that
    // called Wait.
    pub UID: UID,

    // Event is exactly one of the events defined above.
    pub Event: EventMask,

    // Status is the numeric status associated with the event.
    pub Status: u32,
}

impl Task {
    pub fn RunExit(&mut self) -> TaskRunState {
        let t = self.Thread();
        t.ExitMain();
        return TaskRunState::RunExitNotify;
    }

    pub fn RunExitNotify(&mut self) -> TaskRunState {
        let t = self.Thread();

        //info!("RunExitNotify 1 [{:x}]", self.taskId);
        self.Exit();

        //info!("RunExitNotify 2 [{:x}]", self.taskId);
        t.ExitNotify();

        //info!("RunExitNotify 3 [{:x}]", self.taskId);
        //won't reach here
        return TaskRunState::RunExitDone;
    }

    pub fn RunThreadExit(&mut self) -> TaskRunState {
        let t = self.Thread();
        t.ExitMain();
        return TaskRunState::RunThreadExitNotify;
    }

    pub fn RunThreadExitNotify(&mut self) -> TaskRunState {
        let t = self.Thread();
        if self.isWaitThread {
            panic!("Exit from wait thread!")
        }

        if !t.Signaled() {
            match self.tidInfo.clear_child_tid {
                None => {
                    //println!("there is no clear_child_tid");
                }
                Some(addr) => {
                    let val: u32 = 0;
                    self.CopyOutObj(&val, addr).expect(&format!(
                        "RunThreadExitNotify clear_child_tid copy fail {:x}",
                        addr
                    ));
                    self.futexMgr.Wake(self, addr, false, !0, 1).ok();
                    //.expect(&format!("RunThreadExitNotify futexMgrm wake fail {:x}, oldmap is {}", addr, map));
                }
            }
        }

        t.ExitNotify();

        //won't reach here
        return TaskRunState::RunExitDone;
    }
}
