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
use alloc::sync::Arc;
use core::cmp::*;
use core::ops::Deref;

use super::super::uid::NewUID;
use super::processgroup::*;
use super::thread::*;
use super::thread_group::*;
use crate::qlib::kernel::threadmgr::refcounter::AtomicRefCount;
use crate::qlib::kernel::threadmgr::pid_namespace::PIDNamespace;

#[derive(Default)]
pub struct SessionInternal {
    pub id: SessionID,
    pub leader: ThreadGroupWeak,
    pub pidns: PIDNamespace,
    pub refs: AtomicRefCount,
    pub processGroups: BTreeSet<ProcessGroup>,
}

#[derive(Clone, Default)]
pub struct Session {
    pub uid: UniqueID,
    pub data: Arc<QMutex<SessionInternal>>,
}

impl Deref for Session {
    type Target = Arc<QMutex<SessionInternal>>;

    fn deref(&self) -> &Arc<QMutex<SessionInternal>> {
        &self.data
    }
}

impl Ord for Session {
    fn cmp(&self, other: &Self) -> Ordering {
        let id1 = self.uid;
        let id2 = other.uid;
        id1.cmp(&id2)
    }
}

impl PartialOrd for Session {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Session {
    fn eq(&self, other: &Self) -> bool {
        return self.uid == other.uid;
    }
}

impl Eq for Session {}

impl Session {
    pub fn New(id: SessionID, leader: ThreadGroup) -> Self {
        let internal = SessionInternal {
            id: id,
            leader: leader.Downgrade(),
            pidns: leader.PIDNamespace(),
            refs: Default::default(),
            processGroups: BTreeSet::new(),
        };
        return Self {
            uid: NewUID(),
            data: Arc::new(QMutex::new(internal)),
        };
    }

    pub fn DecRef(&self) {
        let mut needRemove = false;
        self.data
            .lock()
            .refs
            .DecRefWithDesctructor(|| needRemove = true);
        let id = self.data.lock().id;
        let mut pidns = self.data.lock().pidns.clone();
        if needRemove {
            loop {
                pidns.lock().sids.remove(self);
                pidns.lock().sessions.remove(&id);
                let parentNs = match pidns.lock().parent.clone() {
                    None => break,
                    Some(ns) => ns,
                };
                pidns = parentNs
            }
            pidns.lock().owner.write().sessions.remove(self);
        }
    }

    pub fn IncRef(&self) {
        self.data.lock().refs.IncRef();
    }
}
