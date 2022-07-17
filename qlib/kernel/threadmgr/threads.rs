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
use alloc::collections::btree_set::BTreeSet;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use spin::*;

use super::super::super::auth::userns::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::usage::io::*;
use super::super::kernel::kernel::*;
use super::super::kernel::waiter::queue::*;
use super::super::kernel::waiter::waitgroup::*;
use super::super::task::*;
use super::super::threadmgr::task_start::*;
use super::super::uid::NewUID;
use super::super::SignalDef::*;
use super::pid_namespace::*;
use super::session::*;
use super::task_exit::*;
use super::task_sched::*;
use super::thread::*;
use super::thread_group::*;

#[derive(Clone, Default)]
pub struct TaskSetInternal {
    pub root: Option<PIDNamespace>,
    pub sessions: BTreeSet<Session>,
    pub stopCount: i32,
    pub taskCount: i32,
}

impl TaskSetInternal {
    pub fn AssignTids(&mut self, t: &Thread) -> Result<()> {
        struct AllocatedTID {
            ns: PIDNamespace,
            tid: ThreadID,
        }

        let tg = t.lock().tg.clone();
        let mut pidns = tg.PIDNamespace();

        let mut allocatedTIDs: Vec<AllocatedTID> = Vec::new();

        loop {
            let tid = match pidns.AllocateTID() {
                Err(e) => {
                    for a in allocatedTIDs {
                        let tns = a.ns.clone();
                        tns.lock().tasks.remove(&a.tid);
                        //error!("AssignTids remove tid {}", a.tid);
                        tns.lock().tids.remove(&t);
                        if tg.lock().leader.Upgrade().is_none() {
                            pidns.lock().tgids.remove(&tg);
                        }
                    }

                    return Err(e);
                }
                Ok(id) => id,
            };

            t.lock().id = tid;
            //error!("AssignTids add tid {}", tid);
            pidns.lock().tasks.insert(tid, t.clone());
            pidns.lock().tids.insert(t.clone(), tid);
            if tg.lock().leader.Upgrade().is_none() {
                pidns.lock().tgids.insert(tg.clone(), tid);
            }

            allocatedTIDs.push(AllocatedTID {
                ns: pidns.clone(),
                tid: tid,
            });

            let tmp = match &pidns.lock().parent {
                None => break,
                Some(ref ns) => ns.clone(),
            };

            pidns = tmp;
        }

        return Ok(());
    }

    pub fn IncrTaskCount(&mut self) -> i32 {
        self.taskCount += 1;
        return self.taskCount;
    }

    pub fn DecrTaskCount1(&mut self) -> i32 {
        self.taskCount -= 1;
        /*if self.taskCount == 0 {
            info!("start to exit vm...");
            super::super::super::Kernel::HostSpace::ExitVM().unwrap();
        }*/
        return self.taskCount;
    }
}

#[derive(Clone, Default)]
pub struct TaskSet(Arc<RwLock<TaskSetInternal>>, Arc<RwLock<()>>);

impl Deref for TaskSet {
    type Target = Arc<RwLock<TaskSetInternal>>;

    fn deref(&self) -> &Arc<RwLock<TaskSetInternal>> {
        &self.0
    }
}

impl TaskSet {
    pub fn New() -> Self {
        let ts = Self(
            Arc::new(RwLock::new(TaskSetInternal {
                root: None,
                sessions: BTreeSet::new(),
                stopCount: 0,
                taskCount: 0,
            })),
            Arc::new(RwLock::new(())),
        );

        let userns = UserNameSpace::NewRootUserNamespace();
        ts.write().root = Some(PIDNamespace::New(&ts, None, &userns));

        return ts;
    }

    pub fn ReadLock(&self) -> RwLockReadGuard<()> {
        return self.1.read();
    }

    pub fn WriteLock(&self) -> RwLockWriteGuard<()> {
        return self.1.write();
    }

    pub fn Root(&self) -> PIDNamespace {
        return self.read().root.as_ref().unwrap().clone();
    }

    // forEachThreadGroupLocked applies f to each thread group in ts.
    //
    // Preconditions: ts.mu must be locked (for reading or writing).
    pub fn forEachThreadGroupLocked(&self, mut f: impl FnMut(&ThreadGroup)) {
        let root = self.Root();
        let tgids: Vec<ThreadGroup> = root.lock().tgids.keys().cloned().collect();
        for tg in &tgids {
            f(tg)
        }
    }

    pub fn NewTask(&self, cfg: &TaskConfig, fromContext: bool, kernel: &Kernel) -> Result<Thread> {
        let tg = cfg.ThreadGroup.clone();

        let internal = ThreadInternal {
            id: 0,
            name: "".to_string(),
            taskId: cfg.TaskId,
            blocker: cfg.Blocker.clone(),
            k: kernel.clone(),
            memoryMgr: cfg.MemoryMgr.clone(),
            fsc: cfg.FSContext.clone(),
            fdTbl: cfg.Fdtbl.clone(),
            vforkParent: None,
            creds: cfg.Credentials.clone(),
            utsns: cfg.UTSNamespace.clone(),
            ipcns: cfg.IPCNamespace.clone(),
            SignalQueue: Queue::default(),
            tg: tg.clone(),
            parent: cfg.Parent.clone(),
            children: BTreeSet::new(),
            childPIDNamespace: None,
            SysCallReturn: None,
            //scedSeq: SeqCount::default(),
            sched: TaskSchedInfo::default(),
            yieldCount: 0,
            pendingSignals: PendingSignals::default(),
            signalMask: cfg.SignalMask.clone(),
            realSignalMask: SignalSet::default(),
            haveSavedSignalMask: false,
            savedSignalMask: SignalSet::default(),
            signalStack: SignalStack::default(),
            groupStopPending: false,
            groupStopAcknowledged: false,
            trapStopPending: false,
            trapNotifyPending: false,
            allowedCPUMask: cfg.AllowedCPUMask.Copy(),
            cpu: 0,
            niceness: cfg.Niceness,
            numaPolicy: 0,
            numaNodeMask: 0,
            netns: false,
            parentDeathSignal: Signal::default(),
            stop: None,
            stopCount: WaitGroup::default(),
            exitStatus: ExitStatus::default(),
            exitState: TaskExitState::default(),
            exitTracerNotified: false,
            exitTracerAcked: false,
            exitParentNotified: false,
            exitParentAcked: false,
            startTime: Task::RealTimeNow(),
            containerID: cfg.ContainerID.to_string(),
            ioUsage: IO::default(),
            robust_list_head: 0,
        };

        let t = Thread {
            uid: NewUID(),
            data: Arc::new(QMutex::new(internal)),
        };

        if fromContext {
            let task = Task::Current();
            let ioUsage = t.lock().ioUsage.clone();
            task.thread = Some(t.clone());
            task.ioUsage = ioUsage;
        }

        {
            let mut tslock = self.write();

            let lock = tg.lock().signalLock.clone();
            let _s = lock.lock();

            {
                let tglock = tg.lock();
                if tglock.exiting || tglock.execing.Upgrade().is_some() {
                    // If the caller is in the same thread group, then what we return
                    // doesn't matter too much since the caller will exit before it returns
                    // to userspace. If the caller isn't in the same thread group, then
                    // we're in uncharted territory and can return whatever we want.
                    return Err(Error::SysError(SysErr::EINTR));
                }
            }

            tslock.AssignTids(&t)?;
            tslock.IncrTaskCount();
        }

        if cfg.InheritParent.is_some() {
            t.lock().parent = cfg.InheritParent.clone().unwrap().lock().parent.clone();
        }

        t.lock().yieldCount = 456;
        let parent = t.lock().parent.clone();
        match &parent {
            None => (),
            Some(p) => {
                p.lock().children.insert(t.clone());
            }
        }

        let leader = tg.lock().leader.Upgrade();
        if leader.is_none() {
            tg.lock().leader = t.Downgrade();
            let parentPG = tg.parentPG();
            if parentPG.is_none() {
                tg.createSession().unwrap();
            } else {
                parentPG
                    .as_ref()
                    .unwrap()
                    .incRefWithParent(parentPG.clone());
                tg.lock().processGroup = parentPG;
            }
        }

        tg.lock().tasks.insert(t.clone());
        tg.lock().tasksCount += 1;
        tg.lock().liveTasks += 1;
        tg.lock().activeTasks += 1;

        t.lock().stopCount.Add(self.read().stopCount);

        return Ok(t);
    }
}
