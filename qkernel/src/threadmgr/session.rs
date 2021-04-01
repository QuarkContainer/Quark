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
use alloc::collections::btree_set::BTreeSet;
use core::cmp::*;

use super::super::uid::NewUID;
use super::thread::*;
use super::thread_group::*;
use super::processgroup::*;

#[derive(Default)]
pub struct SessionInternal {
    pub id: SessionID,
    pub leader: ThreadGroup,

    pub processGroups: BTreeSet<ProcessGroup>,
}

#[derive(Clone, Default)]
pub struct Session {
    pub uid: UniqueID,
    pub data: Arc<Mutex<SessionInternal>>
}

impl Deref for Session {
    type Target = Arc<Mutex<SessionInternal>>;

    fn deref(&self) -> &Arc<Mutex<SessionInternal>> {
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
        return self.uid == other.uid
    }
}

impl Eq for Session {}

impl Session {
    pub fn New(id: SessionID, leader: ThreadGroup) -> Self {
        let internal = SessionInternal {
            id: id,
            leader: leader,
            processGroups: BTreeSet::new(),
        };

        return Self {
            uid: NewUID(),
            data: Arc::new(Mutex::new(internal)),
        }
    }
}
