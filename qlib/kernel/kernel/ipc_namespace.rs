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
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::super::auth::id::*;
use super::super::super::auth::userns::*;
use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::linux::ipc::*;
use super::super::super::linux_def::*;
use super::super::task::*;
use super::msgqueue;
use super::semaphore;
use super::shm;
use crate::qlib::mutex::*;

#[derive(Clone)]
pub struct IPCNamespace {
    pub userNS: UserNameSpace,
    pub semphores: semaphore::SemRegistry,
    pub shms: shm::ShmRegistry,
    pub queues: msgqueue::MQRegistry,
}

impl Default for IPCNamespace {
    fn default() -> Self {
        return Self::New(&UserNameSpace::default());
    }
}

impl IPCNamespace {
    pub fn New(userNS: &UserNameSpace) -> Self {
        return Self {
            userNS: userNS.clone(),
            semphores: semaphore::SemRegistry::New(userNS),
            shms: shm::ShmRegistry::New(userNS),
            queues: msgqueue::MQRegistry::New(userNS),
        };
    }

    pub fn SemaphoreRegistry(&self) -> semaphore::SemRegistry {
        return self.semphores.clone();
    }

    pub fn ShmRegistry(&self) -> shm::ShmRegistry {
        return self.shms.clone();
    }

    pub fn MsgqueueRegistry(&self) -> msgqueue::MQRegistry {
        return self.queues.clone();
    }
}

// Key is a user-provided identifier for IPC objects.
pub type Key = i32;

// ID is a kernel identifier for IPC objects.
pub type ID = i32;

pub struct MechanismIntern<T: Object> {
    // UserNS owning the IPC namespace this registry belongs to. Immutable.
    pub userNS: UserNameSpace,

    // ID is a kernel identifier for the IPC object. Immutable.
    pub id: ID,

    // Key is a user-provided identifier for the IPC object. Immutable.
    pub key: Key,

    // Creator is the user who created the IPC object. Immutable.
    pub creator: FileOwner,

    // Owner is the current owner of the IPC object.
    pub owner: FileOwner,

    // Perms is the access permissions the IPC object.
    pub perms: FilePermissions,

    pub obj: T,
}

#[derive(Clone)]
pub struct Mechanism<T: Object>(Arc<QMutex<MechanismIntern<T>>>);

impl<T: Object> From<Arc<QMutex<MechanismIntern<T>>>> for Mechanism<T> {
    fn from(intern: Arc<QMutex<MechanismIntern<T>>>) -> Self {
        Mechanism(intern)
    }
}

impl<T: Object> Deref for Mechanism<T> {
    type Target = Arc<QMutex<MechanismIntern<T>>>;

    fn deref(&self) -> &Arc<QMutex<MechanismIntern<T>>> {
        &self.0
    }
}

impl<'a, T: Object> MechanismIntern<T> {
    pub fn checkOwnership(&self, creds: &Credentials) -> bool {
        let effectiveKUID = creds.lock().EffectiveKUID;

        if self.owner.UID == effectiveKUID || self.creator.UID == effectiveKUID {
            return true;
        }

        // Tasks with CAP_SYS_ADMIN may bypass ownership checks. Strangely, Linux
        // doesn't use CAP_IPC_OWNER for this despite CAP_IPC_OWNER being documented
        // for use to "override IPC ownership checks".
        return creds.HasCapabilityIn(Capability::CAP_SYS_ADMIN, &self.userNS);
    }

    // CheckPermissions verifies whether an IPC object is accessible using creds for
    // access described by req. See ipc/util.c:ipcperms() in Linux.
    pub fn checkPermission(&self, creds: &Credentials, req: &PermMask) -> bool {
        let mut p = self.perms.Other;
        if self.owner.UID == creds.lock().EffectiveKUID {
            p = self.perms.User;
        } else if creds.InGroup(self.owner.GID) {
            p = self.perms.Group;
        }

        if p.SupersetOf(req) {
            return true;
        }

        return creds.HasCapabilityIn(Capability::CAP_IPC_OWNER, &self.userNS);
    }

    // Set modifies attributes for an IPC object. See *ctl(IPC_SET).
    pub fn Set(&mut self, task: &Task, perm: &IPCPerm) -> Result<()> {
        let creds = task.creds.clone();
        let userns = creds.lock().UserNamespace.clone();
        let uid = userns.MapToKUID(UID(perm.UID));
        let gid = userns.MapToKGID(GID(perm.GID));
        if !uid.Ok() || !gid.Ok() {
            // The man pages don't specify an errno for invalid uid/gid, but EINVAL
            // is generally used for invalid arguments.
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if !self.checkOwnership(&creds) {
            // "The argument cmd has the value IPC_SET or IPC_RMID, but the
            //  effective user ID of the calling process is not the creator (as
            //  found in msg_perm.cuid) or the owner (as found in msg_perm.uid)
            //  of the message queue, and the caller is not privileged (Linux:
            //  does not have the CAP_SYS_ADMIN capability)."
            return Err(Error::SysError(SysErr::EPERM));
        }

        let mode = FileMode(perm.Mode & 0x1ff);
        self.perms = FilePermissions::FromMode(mode);

        self.owner.UID = uid;
        self.owner.GID = gid;
        return Ok(());
    }
}

impl<T: Object> Mechanism<T> {
    pub fn New(
        userns: UserNameSpace,
        key: Key,
        creator: &FileOwner,
        owner: &FileOwner,
        perms: &FilePermissions,
        obj: T,
    ) -> Self {
        let intern = MechanismIntern {
            userNS: userns,
            id: 0,
            key: key,
            creator: *creator,
            owner: *owner,
            perms: *perms,
            obj: obj,
        };

        return Self(Arc::new(QMutex::new(intern)));
    }

    pub fn Id(&self) -> ID {
        return self.lock().id;
    }
}

pub trait Object {
    fn Destory(&mut self);
}

#[derive(Default)]
pub struct RegistryInternal<T: Object> {
    // UserNS owning the IPC namespace this registry belongs to. Immutable.
    pub userNS: UserNameSpace,

    // objects is a map of IDs to IPC mechanisms.
    pub objects: BTreeMap<ID, Mechanism<T>>,

    // keyToID maps a lookup key to an ID.
    pub keyToID: BTreeMap<Key, ID>,

    // lastIDUsed is used to find the next available ID for object creation.
    pub lastIDUsed: i32,
}

impl<T: Object> RegistryInternal<T> {
    pub fn New(userNS: &UserNameSpace) -> Self {
        return Self {
            userNS: userNS.clone(),
            objects: BTreeMap::new(),
            keyToID: BTreeMap::new(),
            lastIDUsed: 0,
        };
    }

    // newID finds the first unused ID in the registry, and returns an error if
    // non is found.
    pub fn NewId(&mut self) -> Result<ID> {
        let mut id = self.lastIDUsed.wrapping_add(1);
        while id != self.lastIDUsed {
            if !self.objects.contains_key(&id) {
                self.lastIDUsed = id;
                return Ok(id);
            }
            id = id.wrapping_add(1);
        }

        error!("ids exhausted, they may be leaking");

        // The man pages for shmget(2) mention that ENOSPC should be used if "All
        // possible shared memory IDs have been taken (SHMMNI)". Other SysV
        // mechanisms don't have a specific errno for running out of IDs, but they
        // return ENOSPC if the max number of objects is exceeded, so we assume that
        // it's the same case.
        return Err(Error::SysError(SysErr::ENOSPC));
    }

    // Register adds the given object into Registry.Objects, and assigns it a new
    // ID. It returns an error if all IDs are exhausted.
    pub fn Register(&mut self, m: Mechanism<T>) -> Result<()> {
        let id = self.NewId()?;

        m.lock().id = id;
        let key = m.lock().key;
        self.objects.insert(id, m);
        self.keyToID.insert(key, id);
        return Ok(());
    }

    // Remove removes the mechanism with the given id from the registry, and calls
    // mechanism.Destroy to perform mechanism-specific removal.
    pub fn Remove(&mut self, id: ID, creds: &Credentials) -> Result<()> {
        let mech = match self.objects.get(&id) {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(m) => Mechanism::from(m.0.clone()),
        };

        let mut mechLock = mech.lock();
        if !mechLock.checkOwnership(&creds) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        let key = mechLock.key;
        self.keyToID.remove(&key);
        self.objects.remove(&id);

        mechLock.obj.Destory();

        return Ok(());
    }

    // Find uses key to search for and return a SysV mechanism. Find returns an
    // error if an object is found by shouldn't be, or if the user doesn't have
    // permission to use the object. If no object is found, Find checks create
    // flag, and returns an error only if it's false.
    pub fn Find(
        &self,
        task: &Task,
        key: Key,
        mode: FileMode,
        create: bool,
        exclusive: bool,
    ) -> Result<Option<Mechanism<T>>> {
        let me = self;
        match me.keyToID.get(&key) {
            Some(id) => {
                let mech: Mechanism<T> = match me.objects.get(id) {
                    None => panic!("abc"),
                    Some(m) => Mechanism::from(m.0.clone()),
                };
                {
                    let mechlock = mech.lock();

                    let creds = task.creds.clone();
                    if !mechlock.checkPermission(&creds, &PermMask::FromMode(mode)) {
                        // The [calling process / user] does not have permission to access
                        // the set, and does not have the CAP_IPC_OWNER capability in the
                        // user namespace that governs its IPC namespace.
                        return Err(Error::SysError(SysErr::EINVAL));
                    }
                }

                if create && exclusive {
                    // IPC_CREAT and IPC_EXCL were specified, but an object already
                    // exists for key.
                    return Err(Error::SysError(SysErr::EEXIST));
                }

                return Ok(Some(mech));
            }
            None => (),
        }

        if !create {
            // No object exists for key and msgflg did not specify IPC_CREAT.
            return Err(Error::SysError(SysErr::ENOENT));
        }

        return Ok(None);
    }

    // FindByID returns the mechanism with the given ID, nil if non exists.
    pub fn FindById(&self, id: ID) -> Option<Mechanism<T>> {
        return match self.objects.get(&id) {
            None => None,
            Some(m) => Some(Mechanism::from(m.0.clone())),
        };
    }

    // ForAllObjects executes a given function for all given objects.
    pub fn ForAllObjects(&self, f: &mut FnMut(&Mechanism<T>)) {
        for (_, o) in &self.objects {
            f(o)
        }
    }

    // DissociateKey removes the association between a mechanism and its key
    // (deletes it from r.keysToIDs), preventing it from being discovered by any new
    // process, but not necessarily destroying it. If the given key doesn't exist,
    // nothing is changed.
    pub fn DissociateKey(&mut self, key: Key) {
        self.keyToID.remove(&key);
    }

    // DissociateID removes the association between a mechanism and its ID (deletes
    // it from r.objects). An ID can't be removed unless the associated key is
    // removed already, this is done to prevent the users from acquiring nil a
    // Mechanism.
    //
    // Precondition: must be preceded by a call to r.DissociateKey.
    pub fn DissociateId(&mut self, id: ID) {
        self.objects.remove(&id);
    }

    // ObjectCount returns the number of registered objects.
    pub fn ObjectCount(&self) -> usize {
        return self.objects.len();
    }

    // LastIDUsed returns the last used ID.
    pub fn LastIDUsed(&self) -> ID {
        return self.lastIDUsed;
    }
}

#[derive(Clone)]
pub struct Registry<T: Object>(Arc<QMutex<RegistryInternal<T>>>);

impl<T: Object> Deref for Registry<T> {
    type Target = Arc<QMutex<RegistryInternal<T>>>;

    fn deref(&self) -> &Arc<QMutex<RegistryInternal<T>>> {
        &self.0
    }
}

impl<T: Object> Registry<T> {
    pub fn New(userNS: &UserNameSpace) -> Self {
        let intern = RegistryInternal {
            userNS: userNS.clone(),
            objects: BTreeMap::new(),
            keyToID: BTreeMap::new(),
            lastIDUsed: 0,
        };

        return Self(Arc::new(QMutex::new(intern)));
    }
}
