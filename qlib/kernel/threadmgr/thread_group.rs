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

use alloc::sync::Arc;
use alloc::sync::Weak;
use crate::qlib::mutex::*;
use core::ops::Deref;
use alloc::collections::btree_map::BTreeMap;
use alloc::collections::btree_set::BTreeSet;
use alloc::vec::Vec;
use core::cmp::*;
use alloc::string::String;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::SignalDef::*;
use super::super::super::limits::*;
use super::super::kernel::timer::timer::Timer;
use super::super::kernel::timer::timer::Setting;
use super::super::kernel::posixtimer::*;
use super::super::threadmgr::task_exit::*;
use super::super::super::linux;
use super::super::super::usage::cpu::*;
use super::super::super::usage::io::*;
use super::super::kernel::signal_handler::*;
use super::super::kernel::waiter::queue::*;
use super::super::kernel::waiter::waitgroup::*;

use super::thread::*;
use super::threads::*;
use super::session::*;
use super::processgroup::*;
use super::pid_namespace::*;

#[derive(Default)]
pub struct ThreadGroupInternal {
    // pidns is the PID namespace containing the thread group and all of its
    // member tasks. The pidns pointer is immutable.
    pub pidns: PIDNamespace,

    pub eventQueue: Queue,

    // leader is the thread group's leader, which is the oldest task in the
    // thread group; usually the last task in the thread group to call
    // execve(), or if no such task exists then the first task in the thread
    // group, which was created by a call to fork() or clone() without
    // CLONE_THREAD. Once a thread group has been made visible to the rest of
    // the system by TaskSet.newTask, leader is never nil.
    //
    // Note that it's possible for the leader to exit without causing the rest
    // of the thread group to exit; in such a case, leader will still be valid
    // and non-nil, but leader will not be in tasks.
    //
    // leader is protected by the TaskSet mutex.
    pub leader: ThreadWeak,

    // If execing is not nil, it is a task in the thread group that has killed
    // all other tasks so that it can become the thread group leader and
    // perform an execve. (execing may already be the thread group leader.)
    //
    // execing is analogous to Linux's signal_struct::group_exit_task.
    //
    // execing is protected by the TaskSet mutex.
    pub execing: ThreadWeak,

    pub tasks: BTreeSet<Thread>,

    // tasksCount is the number of tasks in the thread group that have not yet
    // been reaped; equivalently, tasksCount is the number of tasks in tasks.
    //
    // tasksCount is protected by both the TaskSet mutex and the signal mutex,
    // as with tasks.
    pub tasksCount: i32,

    // liveTasks is the number of tasks in the thread group that have not yet
    // reached TaskExitZombie.
    //
    // liveTasks is protected by the TaskSet mutex (NOT the signal mutex).
    pub liveTasks: i32,

    // activeTasks is the number of tasks in the thread group that have not yet
    // reached TaskExitInitiated.
    //
    // activeTasks is protected by both the TaskSet mutex and the signal mutex,
    // as with tasks.
    pub activeTasks: i32,

    // processGroup is the processGroup for this thread group.
    //
    // processGroup is protected by the TaskSet mutex.
    pub processGroup: Option<ProcessGroup>,

    pub signalLock: Arc<QMutex<()>>,
    pub signalHandlers: SignalHandlers,

    // pendingSignals is the set of pending signals that may be handled by any
    // task in this thread group.
    //
    // pendingSignals is protected by the signal mutex.
    pub pendingSignals: PendingSignals,

    // If groupStopDequeued is true, a task in the thread group has dequeued a
    // stop signal, but has not yet initiated the group stop.
    //
    // groupStopDequeued is analogous to Linux's JOBCTL_STOP_DEQUEUED.
    //
    // groupStopDequeued is protected by the signal mutex.
    pub groupStopDequeued: bool,

    // groupStopSignal is the signal that caused a group stop to be initiated.
    //
    // groupStopSignal is protected by the signal mutex.
    pub groupStopSignal: Signal,

    // groupStopPendingCount is the number of active tasks in the thread group
    // for which Task.groupStopPending is set.
    //
    // groupStopPendingCount is analogous to Linux's
    // signal_struct::group_stop_count.
    //
    // groupStopPendingCount is protected by the signal mutex.
    pub groupStopPendingCount: i32,

    // If groupStopComplete is true, groupStopPendingCount transitioned from
    // non-zero to zero without an intervening SIGCONT.
    //
    // groupStopComplete is analogous to Linux's SIGNAL_STOP_STOPPED.
    //
    // groupStopComplete is protected by the signal mutex.
    pub groupStopComplete: bool,

    // If groupStopWaitable is true, the thread group is indicating a waitable
    // group stop event (as defined by EventChildGroupStop).
    //
    // Linux represents the analogous state as SIGNAL_STOP_STOPPED being set
    // and group_exit_code being non-zero.
    //
    // groupStopWaitable is protected by the signal mutex.
    pub groupStopWaitable: bool,

    // If groupContNotify is true, then a SIGCONT has recently ended a group
    // stop on this thread group, and the first task to observe it should
    // notify its parent. groupContInterrupted is true iff SIGCONT ended an
    // incomplete group stop. If groupContNotify is false, groupContInterrupted is
    // meaningless.
    //
    // Analogues in Linux:
    //
    // - groupContNotify && groupContInterrupted is represented by
    // SIGNAL_CLD_STOPPED.
    //
    // - groupContNotify && !groupContInterrupted is represented by
    // SIGNAL_CLD_CONTINUED.
    //
    // - !groupContNotify is represented by neither flag being set.
    //
    // groupContNotify and groupContInterrupted are protected by the signal
    // mutex.
    pub groupContNotify: bool,
    pub groupContInterrupted: bool,

    // If groupContWaitable is true, the thread group is indicating a waitable
    // continue event (as defined by EventGroupContinue).
    //
    // groupContWaitable is analogous to Linux's SIGNAL_STOP_CONTINUED.
    //
    // groupContWaitable is protected by the signal mutex.
    pub groupContWaitable: bool,

    // exiting is true if all tasks in the ThreadGroup should exit. exiting is
    // analogous to Linux's SIGNAL_GROUP_EXIT.
    //
    // exiting is protected by the signal mutex. exiting can only transition
    // from false to true.
    pub exiting: bool,

    // exitStatus is the thread group's exit status.
    //
    // While exiting is false, exitStatus is protected by the signal mutex.
    // When exiting becomes true, exitStatus becomes immutable.
    pub exitStatus: ExitStatus,

    // terminationSignal is the signal that this thread group's leader will
    // send to its parent when it exits.
    //
    // terminationSignal is protected by the TaskSet mutex.
    pub terminationSignal: Signal,

    //liveThreads is the number of non-exited thread
    pub liveThreads: WaitGroup,

    // itimerRealTimer implements ITIMER_REAL for the thread group.
    pub itimerRealTimer: Timer,

    // itimerVirtSetting is the ITIMER_VIRTUAL setting for the thread group.
    //
    // itimerVirtSetting is protected by the signal mutex.
    pub itimerVirtSetting: Setting,

    // itimerProfSetting is the ITIMER_PROF setting for the thread group.
    //
    // itimerProfSetting is protected by the signal mutex.
    pub itimerProfSetting: Setting,

    // rlimitCPUSoftSetting is the setting for RLIMIT_CPU soft limit
    // notifications for the thread group.
    //
    // rlimitCPUSoftSetting is protected by the signal mutex.
    pub rlimitCPUSoftSetting: Setting,

    // cpuTimersEnabled is non-zero if itimerVirtSetting.Enabled is true,
    // itimerProfSetting.Enabled is true, rlimitCPUSoftSetting.Enabled is true,
    // or limits.Get(CPU) is finite.
    //
    // cpuTimersEnabled is protected by the signal mutex. cpuTimersEnabled is
    // accessed using atomic memory operations.
    pub cpuTimersEnabled: u32,

    // timers is the thread group's POSIX interval timers. nextTimerID is the
    // TimerID at which allocation should begin searching for an unused ID.
    //
    // timers and nextTimerID are protected by timerMu.
    pub timers: BTreeMap<linux::TimeID, IntervalTimer>,
    pub nextTimerID: linux::TimeID,

    // exitedCPUStats is the CPU usage for all exited tasks in the thread
    // group. exitedCPUStats is protected by the TaskSet mutex.
    pub exitedCPUStats: CPUStats,

    // childCPUStats is the CPU usage of all joined descendants of this thread
    // group. childCPUStats is protected by the TaskSet mutex.
    pub childCPUStats: CPUStats,

    // ioUsage is the I/O usage for all exited tasks in the thread group.
    // The ioUsage pointer is immutable.
    pub ioUsage: IO,

    // maxRSS is the historical maximum resident set size of the thread group, updated when:
    //
    // - A task in the thread group exits, since after all tasks have
    // exited the MemoryManager is no longer reachable.
    //
    // - The thread group completes an execve, since this changes
    // MemoryManagers.
    //
    // maxRSS is protected by the TaskSet mutex.
    pub maxRSS: u64,

    // childMaxRSS is the maximum resident set size in bytes of all joined
    // descendants of this thread group.
    //
    // childMaxRSS is protected by the TaskSet mutex.
    pub childMaxRSS: u64,

    // Resource limits for this ThreadGroup. The limits pointer is immutable.
    pub limits: LimitSet,

    // execed indicates an exec has occurred since creation. This will be
    // set by finishExec, and new TheadGroups will have this field cleared.
    // When execed is set, the processGroup may no longer be changed.
    //
    // execed is protected by the TaskSet mutex.
    pub execed: bool,

    pub containerID: String,

    pub timerMu: Arc<QMutex<()>>,
    // todo: handle tty
    //pub tty: Option<TTY>
}

#[derive(Default)]
pub struct ThreadGroupWeak {
    pub uid: UniqueID,
    pub data: Weak<QMutex<ThreadGroupInternal>>,
}

impl ThreadGroupWeak {
    pub fn Upgrade(&self) -> Option<ThreadGroup> {
        let t = match self.data.upgrade() {
            None => return None,
            Some(t) => t,
        };

        return Some(ThreadGroup {
            uid: self.uid,
            data: t
        })
    }
}

#[derive(Clone, Default)]
pub struct ThreadGroup {
    pub uid: UniqueID,
    pub data: Arc<QMutex<ThreadGroupInternal>>
}

impl Deref for ThreadGroup {
    type Target = Arc<QMutex<ThreadGroupInternal>>;

    fn deref(&self) -> &Arc<QMutex<ThreadGroupInternal>> {
        &self.data
    }
}

impl Ord for ThreadGroup {
    fn cmp(&self, other: &Self) -> Ordering {
        let id1 = self.uid;
        let id2 = other.uid;
        id1.cmp(&id2)
    }
}

impl PartialOrd for ThreadGroup {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ThreadGroup {
    fn eq(&self, other: &Self) -> bool {
        let id1 = self.uid;
        let id2 = other.uid;
        return id1 == id2;
    }
}

impl Eq for ThreadGroup {}

/*impl Drop for ThreadGroup {
    fn drop(&mut self) {
        info!("todo: threadgroup drop**");
        //self.release();
    }
}*/

impl ThreadGroup {
    pub fn Downgrade(&self) -> ThreadGroupWeak {
        return ThreadGroupWeak {
            uid: self.uid,
            data: Arc::downgrade(&self.data),
        }
    }

    pub fn TimerMu(&self) -> Arc<QMutex<()>> {
        return self.lock().timerMu.clone();
    }

    pub fn PIDNamespace(&self) -> PIDNamespace {
        return self.lock().pidns.clone();
    }

    pub fn TaskSet(&self) -> TaskSet {
        let pidns = self.PIDNamespace();
        return pidns.lock().owner.clone();
    }

    pub fn Leader(&self) -> Option<Thread> {
        let ts = self.TaskSet();
        let _ts = ts.ReadLock();
        return self.lock().leader.Upgrade();
    }

    pub fn Count(&self) -> usize {
        let ts = self.TaskSet();
        let _ts = ts.ReadLock();
        return self.lock().tasks.len();
    }

    pub fn MemberIDs(&self, pidns: &PIDNamespace) -> Vec<ThreadID> {
        let ts = self.TaskSet();
        let _ts = ts.ReadLock();

        let mut tasks = Vec::new();
        for thread in &self.lock().tasks {
            let tid = thread.lock().id;
            if pidns.lock().tasks.contains_key(&tid) {
                tasks.push(tid)
            }
        }

        return tasks;
    }

    // ID returns tg's leader's thread ID in its own PID namespace. If tg's leader
    // is dead, ID returns 0.
    pub fn ID(&self) -> ThreadID {
        let ts = self.TaskSet();
        let _ts = ts.ReadLock();

        let pidns = self.PIDNamespace();
        let tid = match pidns.lock().tgids.get(self) {
            None => 0,
            Some(tid) => *tid,
        };

        return tid;
    }

    pub fn parentPG(&self) -> Option<ProcessGroup> {
        let lead = match &self.lock().leader.Upgrade() {
            None => return None,
            Some(ref l) => l.clone(),
        };

        let res = match &lead.lock().parent {
            None => None,
            Some(p) => p.lock().tg.lock().processGroup.clone(),
        };

        return res;
    }

    pub fn forEachChildThreadGroupLocked(&self, f: impl Fn(ThreadGroup)) {
        for t in &self.lock().tasks {
            for child in &t.lock().children {
                let tg = child.lock().tg.clone();
                let leader = match &tg.lock().leader.Upgrade() {
                    None => continue,
                    Some(ref l) => l.clone(),
                };

                if child.clone() == leader {
                    let tg = child.lock().tg.clone();
                    f(tg)
                }
            }
        }
    }

    pub fn CreateProcessGroup(&self) -> Result<()> {
        let ts = self.TaskSet();
        let _l = ts.WriteLock();
        let ts = ts.write();

        let pidns = self.PIDNamespace();
        let id = match pidns.lock().tgids.get(self) {
            None => 0,
            Some(tid) => *tid,
        };

        for s in &ts.sessions {
            let leader = s.lock().leader.clone();

            if leader.lock().pidns.clone() != pidns {
                continue;
            }

            if leader == self.clone() {
                return Err(Error::SysError(SysErr::EPERM));
            }

            for pg in &s.lock().processGroups {
                if pg.lock().id == id {
                    return Err(Error::SysError(SysErr::EPERM));
                }
            }
        }

        let pg = ProcessGroup::New(id, self.clone(), self.lock().processGroup.clone().unwrap().lock().session.clone());

        let leader = self.lock().leader.Upgrade().unwrap();
        if let Some(ref parent) = &leader.lock().parent {
            let session = pg.lock().session.clone();
            let tgTmp = parent.lock().tg.clone();
            let pgTmp = tgTmp.lock().processGroup.clone().unwrap();
            let sessionTmp = pgTmp.lock().session.clone();
            if sessionTmp == session {
                pg.lock().ancestors += 1;
            }
        }

        let oldParentPG = self.parentPG();
        self.forEachChildThreadGroupLocked(|childTG: ThreadGroup| {
            let currentPg = childTG.lock().processGroup.clone().unwrap();
            currentPg.incRefWithParent(Some(pg.clone()));
            currentPg.decRefWithParent(oldParentPG.clone());
        });
        self.lock().processGroup.clone().unwrap().decRefWithParent(oldParentPG);
        self.lock().processGroup = Some(pg.clone());

        let sessionTmp = pg.lock().session.clone();
        sessionTmp.lock().processGroups.insert(pg.clone());

        let mut ns = pidns.clone();
        loop {
            let local = *ns.lock().tgids.get(self).unwrap();
            ns.lock().pgids.insert(pg.clone(), local);
            ns.lock().processGroups.insert(local, pg.clone());

            let tmp = match &ns.lock().parent {
                None => break,
                Some(ref p) => p.clone(),
            };

            ns = tmp;
        }

        return Ok(())
    }

    pub fn JoinProcessGroup(&self, pidns: &PIDNamespace, pgid: ProcessGroupID, checkExec: bool) -> Result<()> {
        let owner = pidns.lock().owner.clone();
        let _r = owner.ReadLock();

        let pg = match pidns.lock().processGroups.get(&pgid) {
            None => return Err(Error::SysError(SysErr::EPERM)),
            Some(pg) => pg.clone(),
        };

        if checkExec && self.lock().execed {
            return Err(Error::SysError(SysErr::EACCES))
        }

        let session = pg.lock().session.clone();
        let currentPg = self.lock().processGroup.clone().unwrap();
        if session != currentPg.lock().session {
            return Err(Error::SysError(SysErr::EPERM))
        }

        let parentPG = self.parentPG();
        pg.incRefWithParent(parentPG.clone());

        let pgCurr = self.lock().processGroup.clone().unwrap();
        pg.incRefWithParent(Some(pgCurr.clone()));
        self.forEachChildThreadGroupLocked(|childTG: ThreadGroup| {
            let pgTmp = childTG.lock().processGroup.clone().unwrap();
            pgTmp.incRefWithParent(Some(pg.clone()));
            pgTmp.decRefWithParent(Some(pgCurr.clone()));
        });

        pgCurr.decRefWithParent(parentPG);
        self.lock().processGroup = Some(pg);
        return Ok(())
    }

    pub fn Session(&self) -> Option<Session> {
        let ts = self.TaskSet();
        let _r = ts.ReadLock();

        match self.lock().processGroup.clone() {
            None => None,
            Some(pg) => Some(pg.lock().session.clone()),
        }
    }

    pub fn ProcessGroup(&self) -> Option<ProcessGroup> {
        let ts = self.TaskSet();
        let _r = ts.ReadLock();
        return self.lock().processGroup.clone();
    }

    pub fn SignalHandlers(&self) -> SignalHandlers {
        return self.lock().signalHandlers.clone();
    }

    pub fn Limits(&self) -> LimitSet {
        return self.lock().limits.clone();
    }

    pub fn release(&self) {
        // Timers must be destroyed without holding the TaskSet or signal mutexes
        // since timers send signals with Timer.mu locked.
        let timer = self.lock().itimerRealTimer.clone();
        timer.Destroy();
        let mut its = Vec::new();
        let ts = self.TaskSet();
        {
            let _w = ts.write();
            let lock = self.lock().signalLock.clone();
            let _s = lock.lock();
            for (_, it) in &self.lock().timers {
                its.push(it.clone())
            }

            self.lock().timers.clear();
        }

        for it in its {
            it.DestroyTimer();
        }
    }

    pub fn CreateSessoin(&self) -> Result<()> {
        let ts = self.TaskSet();
        let _w = ts.WriteLock();
        return self.createSession();
    }

    pub fn createSession(&self) -> Result<()> {
        let pidns = self.PIDNamespace();
        let ts = pidns.lock().owner.clone();

        let id = match pidns.lock().tgids.get(self) {
            None => 0,
            Some(id) => *id,
        };

        let sessions: Vec<Session> = ts.read().sessions.iter().cloned().collect();
        for s in &sessions {
            if s.lock().leader.lock().pidns != pidns {
                continue;
            }

            if s.lock().leader == self.clone() {
                return Err(Error::SysError(SysErr::EPERM))
            }

            if s.lock().id == id {
                return Err(Error::SysError(SysErr::EPERM))
            }

            for pg in &s.lock().processGroups {
                if pg.lock().id == id {
                    return Err(Error::SysError(SysErr::EPERM))
                }
            }
        }

        let s = Session::New(id, self.clone());
        let pg = ProcessGroup::New(id, self.clone(), s.clone());
        s.lock().processGroups.insert(pg.clone());
        ts.write().sessions.insert(s.clone());

        if self.lock().processGroup.clone().is_some() {
            let oldParentPG = self.parentPG();
            self.forEachChildThreadGroupLocked(|childTG: ThreadGroup| {
                let pgTmp = childTG.lock().processGroup.clone().unwrap();
                pgTmp.incRefWithParent(Some(pg.clone()));
                pgTmp.decRefWithParent(oldParentPG.clone());
            });

            let oldPg = self.lock().processGroup.clone().unwrap();
            oldPg.decRefWithParent(oldParentPG);
            self.lock().processGroup = Some(pg.clone());
        } else {
            self.lock().processGroup = Some(pg.clone());
            pg.lock().ancestors += 1;
        }

        let mut ns = pidns.clone();

        loop {
            let local = match ns.lock().tgids.get(self) {
                None => 0,
                Some(id) => *id,
            };

            ns.lock().sids.insert(s.clone(), local);
            ns.lock().sessions.insert(local, s.clone());
            ns.lock().pgids.insert(pg.clone(), local);
            ns.lock().processGroups.insert(local, pg.clone());

            let tmp = match ns.lock().parent.clone() {
                None => break,
                Some(ns) => ns,
            };

            ns = tmp;
        }

        return Ok(())
    }
}