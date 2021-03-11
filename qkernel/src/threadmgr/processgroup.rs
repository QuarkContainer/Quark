// Copyright (c) 2021 QuarkSoft LLC
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
use core::cmp::*;

use super::super::uid::NewUID;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::SignalDef::*;
use super::thread_group::*;
use super::session::*;
use super::thread::*;
use super::refcounter::*;

#[derive(Default)]
pub struct ProcessGroupInternal {
    // id is the cached identifier in the originator's namespace.
    //
    // The id is immutable.
    pub id: ProcessGroupID,

    pub refs: AtomicRefCount,
    // originator is the originator of the group.
    //
    // See note re: leader in Session. The same applies here.
    //
    // The originator is immutable.
    pub originator: ThreadGroup,

    // Session is the parent Session.
    //
    // The session is immutable.
    pub session: Session,

    // ancestors is the number of thread groups in this process group whose
    // parent is in a different process group in the same session.
    //
    // The name is derived from the fact that process groups where
    // ancestors is zero are considered "orphans".
    //
    // ancestors is protected by TaskSet.mu.
    pub ancestors: u32,
}

#[derive(Clone, Default)]
pub struct ProcessGroup {
    pub uid: UniqueID,
    pub data: Arc<Mutex<ProcessGroupInternal>>
}

impl Deref for ProcessGroup {
    type Target = Arc<Mutex<ProcessGroupInternal>>;

    fn deref(&self) -> &Arc<Mutex<ProcessGroupInternal>> {
        &self.data
    }
}

impl Ord for ProcessGroup {
    fn cmp(&self, other: &Self) -> Ordering {
        let id1 = self.uid;
        let id2 = other.uid;
        id1.cmp(&id2)
    }
}

impl PartialOrd for ProcessGroup {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ProcessGroup {
    fn eq(&self, other: &Self) -> bool {
        return self.Uid() == other.Uid()
    }
}

impl Eq for ProcessGroup {}

impl ProcessGroup {
    pub fn Uid(&self) -> UniqueID {
        return self.uid
    }

    pub fn New(id: ProcessGroupID, orginator: ThreadGroup, session: Session) -> Self {
        let pg = ProcessGroupInternal {
            id: id,
            refs: AtomicRefCount::default(),
            originator: orginator,
            session: session,
            ancestors: 0,
        };

        return Self {
            uid: NewUID(),
            data: Arc::new(Mutex::new(pg)),
        }
    }

    pub fn Originator(&self) -> ThreadGroup {
        return self.lock().originator.clone();
    }

    pub fn IsOrphan(&self) -> bool {
        let taskSet = self.Originator().TaskSet();
        let _r = taskSet.ReadLock();
        return self.lock().ancestors == 0;
    }

    pub fn incRefWithParent(&self, parentPG: Option<ProcessGroup>) {
        let add = if Some(self.clone()) != parentPG {
            match parentPG {
                None => true,
                Some(ppg) => {
                    let sid = self.lock().session.uid;
                    let psid = ppg.lock().session.uid;
                    sid == psid
                }
            }
        } else {
            false
        };

        if add {
            self.lock().ancestors += 1;
        }

        self.lock().refs.IncRef();
    }

    pub fn decRefWithParent(&self, parentPG: Option<ProcessGroup>) {
        let ok = if Some(self.clone()) != parentPG {
            match parentPG {
                None => true,
                Some(ppg) => {
                    let sid = self.lock().session.uid;
                    let psid = ppg.lock().session.uid;
                    sid == psid
                }
            }
        } else {
            false
        };

        if ok {
            // if this pg was parentPg. But after reparent become non-parentpg. the ancestor will be sub to negtive
            // todo: this is bug. fix it.

            let ancestors = self.lock().ancestors;
            if ancestors > 0 {
                self.lock().ancestors = ancestors - 1;
            }
        }

        let mut alive = true;
        let originator = self.lock().originator.clone();

        let mut needRemove = false;
        self.lock().refs.DecRefWithDesctructor(|| {
            needRemove = true;
        });

        if needRemove {
            alive = false;

            let mut ns = originator.PIDNamespace();
            loop {
                {
                    let mut nslock = ns.lock();
                    let id = match nslock.pgids.get(self) {
                        None => 0,
                        Some(id) => {
                            *id
                        }
                    };

                    nslock.processGroups.remove(&id);
                    nslock.pgids.remove(self);
                }

                let tmp = match ns.lock().parent {
                    None => break,
                    Some(ref ns) => ns.clone(),
                };

                ns = tmp;
            }
        }

        if alive {
            self.handleOrphan();
        };
    }

    // handleOrphan checks whether the process group is an orphan and has any
    // stopped jobs. If yes, then appropriate signals are delivered to each thread
    // group within the process group.
    //
    // Precondition: callers must hold TaskSet.mu for writing.
    pub fn handleOrphan(&self) {
        if self.lock().ancestors != 0 {
            return;
        }

        let mut hasStopped = false;
        let originator = self.lock().originator.clone();
        let pidns = originator.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        owner.forEachThreadGroupLocked(|tg: &ThreadGroup| {
            match tg.lock().processGroup.clone() {
                None => return,
                Some(pg) => {
                    if pg != self.clone() {
                        return
                    }
                }
            }

            {
                let lock = tg.lock().signalLock.clone();
                let _s = lock.lock();
                if tg.lock().groupStopComplete {
                    hasStopped = true;
                }
            }
        });

        if !hasStopped {
            return
        }

        owner.forEachThreadGroupLocked(|tg: &ThreadGroup| {
            match tg.lock().processGroup.clone() {
                None => return,
                Some(pg) => {
                    if pg != self.clone() {
                        return
                    }
                }
            }

            {
                let lock = tg.lock().signalLock.clone();
                let _s = lock.lock();
                let leader = tg.lock().leader.Upgrade().unwrap();
                leader.sendSignalLocked(&SignalInfoPriv(Signal::SIGHUP), true).unwrap();
                leader.sendSignalLocked(&SignalInfoPriv(Signal::SIGCONT), true).unwrap();
            }
        });
    }

    pub fn Session(&self) -> Session {
        return self.lock().session.clone();
    }

    pub fn SendSignal(&self, info: &SignalInfo) -> Result<()> {
        let ts = self.lock().originator.TaskSet();
        let mut lastError: Result<()> = Ok(());
        let rootns = ts.Root();

        let _r = ts.ReadLock();
        for (tg, _) in &rootns.lock().tgids {
            if tg.ProcessGroup() == Some(self.clone()) {
                let lock = tg.lock().signalLock.clone();
                let _s = lock.lock();
                let leader = tg.lock().leader.Upgrade().unwrap();
                let infoCopy = *info;
                match leader.sendSignalLocked(&infoCopy, true) {
                    Err(e) => lastError = Err(e),
                    _ => ()
                };
            }
        }

        return lastError
    }
}
