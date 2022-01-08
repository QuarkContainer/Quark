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
use ::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use core::ops::Deref;
use alloc::vec::Vec;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::auth::userns::*;
use super::session::*;
use super::thread::*;
use super::threads::*;
use super::thread_group::*;
use super::processgroup::*;

const TASKS_LIMIT: ThreadID = 1 << 16;
const INIT_TID: ThreadID = 1;

#[derive(Default)]
pub struct PIDNamespaceInternal {
    pub owner: TaskSet,
    pub parent: Option<PIDNamespace>,
    pub userns: UserNameSpace,
    pub last: ThreadID,

    pub tasks: BTreeMap<ThreadID, Thread>,
    pub tids: BTreeMap<Thread, ThreadID>,
    //Thread unique id to thread id of this namespace

    pub tgids: BTreeMap<ThreadGroup, ThreadID>,
    //Threadgroup uid to Threadgroup id of this namespace

    pub sessions: BTreeMap<SessionID, Session>,
    pub sids: BTreeMap<Session, SessionID>,
    //Session uid to Session id of this namespace

    pub processGroups: BTreeMap<ProcessGroupID, ProcessGroup>,
    pub pgids: BTreeMap<ProcessGroup, ProcessGroupID>,
    //ProcessGroup uid to ProcessGroup id of this namespace

    pub exiting: bool,
}

#[derive(Clone, Default)]
pub struct PIDNamespace(pub Arc<QMutex<PIDNamespaceInternal>>);

impl Deref for PIDNamespace {
    type Target = Arc<QMutex<PIDNamespaceInternal>>;

    fn deref(&self) -> &Arc<QMutex<PIDNamespaceInternal>> {
        &self.0
    }
}

impl PartialEq for PIDNamespace {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for PIDNamespace {}

impl PIDNamespace {
    pub fn New(ts: &TaskSet, parent: Option<PIDNamespace>, userns: &UserNameSpace) -> Self {
        let internal = PIDNamespaceInternal {
            owner: ts.clone(),
            parent: parent,
            userns: userns.clone(),
            last: 0,
            tasks: BTreeMap::new(),
            tids: BTreeMap::new(),
            tgids: BTreeMap::new(),
            sessions: BTreeMap::new(),
            sids: BTreeMap::new(),
            processGroups: BTreeMap::new(),
            pgids: BTreeMap::new(),
            exiting: false,
        };

        return Self(Arc::new(QMutex::new(internal)))
    }

    pub fn Count(&self) -> usize {
        return Arc::strong_count(&self.0);
    }

    pub fn Owner(&self) -> TaskSet {
        return self.lock().owner.clone();
    }

    pub fn NewChild(&self, userns: &UserNameSpace) -> Self {
        let owner = self.lock().owner.clone();
        return Self::New(&owner, Some(self.clone()), userns)
    }

    // TaskWithID returns the task with thread ID tid in PID namespace ns. If no
    // task has that TID, TaskWithID returns nil.
    pub fn TaskWithID(&self, tid: ThreadID) -> Option<Thread> {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();
        match me.tasks.get(&tid) {
            None => None,
            Some(t) => Some(t.clone()),
        }
    }

    // ThreadGroupWithID returns the thread group lead by the task with thread ID
    // tid in PID namespace ns. If no task has that TID, or if the task with that
    // TID is not a thread group leader, ThreadGroupWithID returns nil.
    pub fn ThreadGroupWithID(&self, tid: ThreadID) -> Option<ThreadGroup> {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();
        let t = match me.tasks.get(&tid) {
            None => return None,
            Some(t) => t.clone(),
        };

        let tg = t.lock().tg.clone();

        let leader = match &tg.lock().leader.Upgrade() {
            None => return None,
            Some(ref l) => l.clone(),
        };

        if leader == t {
            return Some(tg);
        }

        return None;
    }

    // IDOfTask returns the TID assigned to the given task in PID namespace ns. If
    // the task is not visible in that namespace, IDOfTask returns 0. (This return
    // value is significant in some cases, e.g. getppid() is documented as
    // returning 0 if the caller's parent is in an ancestor namespace and
    // consequently not visible to the caller.) If the task is nil, IDOfTask returns
    // 0.
    pub fn IDOfTask(&self, t: &Thread) -> ThreadID {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        return self.IDOfTaskLocked(t);
    }

    pub fn IDOfTaskLocked(&self, t: &Thread) -> ThreadID {
        match self.lock().tids.get(t) {
            None => 0,
            Some(tid) => *tid
        }
    }

    // IDOfThreadGroup returns the TID assigned to tg's leader in PID namespace ns.
    // If the task is not visible in that namespace, IDOfThreadGroup returns 0.
    pub fn IDOfThreadGroup(&self, tg: &ThreadGroup) -> ThreadID {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        return match self.lock().tgids.get(tg) {
            None => 0,
            Some(id) => *id,
        }
    }

    // Tasks returns a snapshot of the tasks in ns.
    pub fn Tasks(&self) -> Vec<Thread> {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();

        let mut tasks = Vec::with_capacity(me.tasks.len());
        for (_, task) in &me.tasks {
            tasks.push(task.clone())
        }

        return tasks;
    }

    // ThreadGroups returns a snapshot of the thread groups in ns.
    pub fn ThreadGroups(&self) -> Vec<ThreadGroup> {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();
        let mut tgs = Vec::with_capacity(me.tgids.len());
        for (tg, _) in &me.tgids {
            tgs.push(tg.clone())
        }

        return tgs;
    }

    pub fn UserNamespace(&self) -> UserNameSpace {
        return self.lock().userns.clone();
    }

    pub fn IDOfSession(&self, s: &Session) -> SessionID {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();
        match me.sids.get(s) {
            None => 0,
            Some(id) => *id,
        }
    }

    pub fn SessionWithID(&self, id: SessionID) -> Option<Session> {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();
        return match me.sessions.get(&id) {
            None => None,
            Some(s) => Some(s.clone()),
        }
    }

    pub fn IDOfProcessGroup(&self, pg: &ProcessGroup) -> ProcessGroupID {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();
        match me.pgids.get(pg) {
            None => 0,
            Some(id) => *id,
        }
    }

    pub fn ProcessGroupWithID(&self, id: ProcessGroupID) -> Option<ProcessGroup> {
        let owner = self.lock().owner.clone();
        let _r = owner.ReadLock();

        let me = self.lock();
        let keys : Vec<i32> = me.processGroups.keys().cloned().collect();
        info!("ProcessGroupWithID key is {:?}", keys);

        return match me.processGroups.get(&id) {
            None => None,
            Some(pg) => Some(pg.clone()),
        }
    }

    // allocateTID returns an unused ThreadID from ns.
    pub fn AllocateTID(&self) -> Result<ThreadID> {
        let mut me = self.lock();

        if me.exiting {
            return Err(Error::SysError(SysErr::ENOMEM))
        }

        let mut tid = me.last;

        loop {
            tid += 1;
            if tid > TASKS_LIMIT {
                tid = INIT_TID;
            }

            if !me.tasks.contains_key(&tid) {
                me.last = tid;
                return Ok(tid)
            }

            if tid == me.last {
                return Err(Error::SysError(SysErr::EAGAIN))
            }
        }
    }
}