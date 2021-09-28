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

use alloc::collections::btree_map::BTreeMap;
use core::ops::Bound::Included;
use alloc::vec::Vec;
use alloc::sync::Arc;
use core::ops::Deref;
use super::super::mutex::*;

use super::super::common::*;
use super::super::linux_def::*;
use super::id::*;

#[derive(Default, Debug)]
pub struct UserNameSpaceInternal {
    pub parent: Option<UserNameSpace>,
    pub owner: KUID,

    pub uidMapFromParent: IdMap,
    pub uidMapToParent: IdMap,
    pub gidMapFromParent: IdMap,
    pub gidMapToParent: IdMap,
}

impl UserNameSpaceInternal {
    pub fn GetIDMap(&self, m: &IdMap) -> Vec<IdMapEntry> {
        let mut ret = Vec::with_capacity(m.map.len());
        for (_, entry) in &m.map {
            ret.push(IdMapEntry {
                FirstFromId: entry.FirstFromId,
                FirstToId: entry.FirstToId,
                Len: entry.Len,
            });
        }

        return ret;
    }

    pub fn UIPMap(&self) -> Vec<IdMapEntry> {
        return self.GetIDMap(&self.uidMapToParent)
    }

    pub fn GIDMap(&self) -> Vec<IdMapEntry> {
        return self.GetIDMap(&self.gidMapToParent)
    }
}

#[derive(Clone, Default, Debug)]
pub struct UserNameSpace(pub Arc<QMutex<UserNameSpaceInternal>>);

impl Deref for UserNameSpace {
    type Target = Arc<QMutex<UserNameSpaceInternal>>;

    fn deref(&self) -> &Arc<QMutex<UserNameSpaceInternal>> {
        &self.0
    }
}

impl PartialEq for UserNameSpace {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for UserNameSpace {}

impl UserNameSpace {
    pub fn NewRootUserNamespace() -> Self {
        let internal = UserNameSpaceInternal {
            parent: None,
            owner: KUID::default(),
            uidMapFromParent: IdMap::All(),
            uidMapToParent: IdMap::All(),
            gidMapFromParent: IdMap::All(),
            gidMapToParent: IdMap::All(),
        };

        return Self(Arc::new(QMutex::new(internal)))
    }

    /*pub fn SetUIDMap(&mut self, task: &Task, entries: &Vec<IdMapEntry>) -> Result<()> {
        let creds = &task.creds;

        if self.uidMapFromParent.IsEmpty() {
            return Err(Error::SysError(SysErr::EPERM))
        }

        if entries.len() == 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if !creds.
    }*/

    pub fn trySetUidMap(&mut self, entries: &Vec<IdMapEntry>) -> Result<()> {
        let mut me = self.lock();
        for entry in entries {
            me.uidMapToParent.AddEntry(entry.FirstFromId, entry.FirstToId, entry.Len)?;
            me.uidMapFromParent.AddEntry(entry.FirstToId, entry.FirstFromId, entry.Len)?
        }

        return Ok(())
    }

    pub fn trySetGidMap(&mut self, entries: &Vec<IdMapEntry>) -> Result<()> {
        let mut me = self.lock();
        for entry in entries {
            me.gidMapToParent.AddEntry(entry.FirstFromId, entry.FirstToId, entry.Len)?;
            me.gidMapFromParent.AddEntry(entry.FirstToId, entry.FirstFromId, entry.Len)?
        }

        return Ok(())
    }

    pub fn MapFromKUID(&self, kuid: KUID) -> UID {
        let me = self.lock();
        match &me.parent {
            None => return UID(kuid.0),
            Some(parent) => return UID(me.uidMapFromParent.Map(parent.MapFromKUID(kuid).0))
        }
    }

    pub fn MapFromKGID(&self, kgid: KGID) -> GID {
        let me = self.lock();
        match &me.parent {
            None => return GID(kgid.0),
            Some(parent) => return GID(me.gidMapFromParent.Map(parent.MapFromKGID(kgid).0))
        }
    }

    pub fn MapToKUID(&self, uid: UID) -> KUID {
        let me = self.lock();
        match &me.parent {
            None => return KUID(uid.0),
            Some(parent) => return KUID(me.uidMapToParent.Map(parent.MapToKUID(uid).0))
        }
    }

    pub fn MapToKGID(&self, gid: GID) -> KGID {
        let me = self.lock();
        match &me.parent {
            None => return KGID(gid.0),
            Some(parent) => return KGID(me.gidMapToParent.Map(parent.MapToKGID(gid).0))
        }
    }

    pub fn Depth(&self) -> usize {
        let mut i = 0;
        let mut ns = self.clone();
        loop {
            i += 1;
            let parent = ns.lock().parent.clone();
            match parent {
                None => break,
                Some(ref n) => ns = n.clone(),
            }
        }

        return i;
    }

    pub fn UIDMap(&self) -> Vec<IdMapEntry> {
        return self.lock().UIPMap();
    }

    pub fn GIDMap(&self) -> Vec<IdMapEntry> {
        return self.lock().GIDMap();
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct IdMapEntry {
    pub FirstFromId: u32,
    pub FirstToId: u32,
    pub Len: u32,
}

#[derive(Debug)]
pub struct IdMap {
    pub map: BTreeMap<u32, IdMapEntry>,
}

impl Default for IdMap {
    fn default() -> Self {
        return Self {
            map: BTreeMap::new(),
        }
    }
}

impl IdMap {
    pub fn All() -> Self {
        let mut res = Self::default();
        res.AddEntry(0, 0, core::u32::MAX).unwrap();
        return res;
    }

    pub fn AddEntry(&mut self, FirstFromId: u32, FirstToId: u32, Len: u32) -> Result<()> {
        let mut id = FirstFromId;
        if id < FirstToId {
            id = FirstToId
        }

        if core::u32::MAX - Len < id {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        //todo: check whether there is overlap

        self.map.insert(FirstFromId, IdMapEntry {
            FirstFromId,
            FirstToId,
            Len,
        });

        return Ok(())
    }

    pub fn RemoveAll(&mut self) {
        self.map.clear();
    }

    pub fn Map(&self, id: u32) -> u32 {
        for (_, &val) in self.map.range((Included(0), Included(id))).rev() {
            if id < val.FirstFromId || id >= val.FirstFromId + val.Len {
                return NO_ID
            }

            return id - val.FirstFromId + val.FirstToId
        }

        return NO_ID
    }

    pub fn IsEmpty(&self) -> bool {
        self.map.len() == 0
    }
}