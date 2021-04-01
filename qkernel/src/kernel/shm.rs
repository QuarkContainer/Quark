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
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use core::fmt::Debug;

use super::super::PAGE_MGR;
use super::super::qlib::auth::userns::*;
use super::super::qlib::auth::*;
use super::super::qlib::auth::id::*;
use super::super::qlib::addr::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;
use super::super::qlib::range::*;
use super::super::qlib::device::*;
use super::super::memmgr::*;
use super::super::qlib::linux::ipc::*;
use super::super::qlib::linux::shm::*;
use super::time::*;

type Key = i32;
type ID = i32;

#[derive(Default, Debug)]
pub struct RegistryInternal {
    pub userNS: UserNameSpace,
    pub shms: BTreeMap<ID, Shm>,
    pub keysToShms: BTreeMap<Key, Shm>,
    pub totalPages: u64,
    pub lastIDUsed: ID,
}

impl RegistryInternal {}

#[derive(Clone, Default, Debug)]
pub struct Registry(Arc<Mutex<RegistryInternal>>);

impl Deref for Registry {
    type Target = Arc<Mutex<RegistryInternal>>;

    fn deref(&self) -> &Arc<Mutex<RegistryInternal>> {
        &self.0
    }
}

impl Registry {
    pub fn New(userNS: &UserNameSpace) -> Self {
        let internal = RegistryInternal {
            userNS: userNS.clone(),
            shms: BTreeMap::new(),
            keysToShms: BTreeMap::new(),
            totalPages: 0,
            lastIDUsed: 0,
        };

        return Self(Arc::new(Mutex::new(internal)))
    }

    pub fn FindByID(&self, id: ID) -> Option<Shm> {
        let me = self.lock();
        return match me.shms.get(&id) {
            None => None,
            Some(shm) => Some(shm.clone()),
        }
    }

    fn dissociateKey(&self, shm: &Shm) {
        let mut me = self.lock();
        let mut s = shm.lock();

        if s.key != IPC_PRIVATE {
            me.keysToShms.remove(&s.key);
            s.key = IPC_PRIVATE;
        }
    }

    pub fn FindOrCreate(&self, task: &Task, pid: i32, key: Key, size: u64,
                        mode: &FileMode, private: bool, create: bool, exclusive: bool) -> Result<Shm> {
        if (create || private) && (size < SHMMIN || size > SHMMAX) {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        {
            let me = self.lock();
            if me.shms.len() > SHMMNI as usize {
                return Err(Error::SysError(SysErr::ENOSPC))
            }

            if !private {
                match me.keysToShms.get(&key) {
                    None => (),
                    Some(s) => {
                        {
                            let shm = s.lock();

                            if !shm.checkPermission(task, &PermMask::FromMode(mode.clone())) {
                                return Err(Error::SysError(SysErr::EACCES))
                            }

                            if size > shm.size {
                                return Err(Error::SysError(SysErr::EINVAL))
                            }

                            if create && exclusive {
                                return Err(Error::SysError(SysErr::EEXIST))
                            }
                        }

                        return Ok(s.clone());
                    }
                };

                if !create {
                    return Err(Error::SysError(SysErr::ENOENT))
                }
            }

            let sizeAligned = match Addr(size).RoundUp() {
                Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
                Ok(addr) => addr.0,
            };

            let numPages = sizeAligned / MemoryDef::PAGE_SIZE;
            if me.totalPages + numPages > SHMALL {
                return Err(Error::SysError(SysErr::ENOSPC));
            }
        }

        let creator = task.FileOwner();
        let perms = FilePermissions::FromMode(*mode);

        return self.newShm(task, pid, key, &creator, &perms, size);
    }

    fn newShm(&self, task: &Task, pid: i32, key: Key, creator: &FileOwner, perms: &FilePermissions, size: u64) -> Result<Shm> {
        let effectiveSize = Addr(size).MustRoundUp().0;
        let addr = PAGE_MGR.MapAnon(effectiveSize)?;
        let fr = Range::New(addr, effectiveSize);

        PAGE_MGR.RefRange(&fr)?;

        let shm = Shm(Arc::new(Mutex::new(ShmInternal {
            registry: self.clone(),
            id: 0,
            creator: *creator,
            size: size,
            effectiveSize: effectiveSize,
            fr: fr,
            key: key,
            perms: *perms,
            owner: *creator,
            attachTime: Time(0),
            detachTime: Time(0),
            changeTime: task.Now(),
            creatorPID: pid,
            lastAttachDetachPID: 0,
            pendingDestruction: false,
            refCount: 0,
        })));

        let mut me = self.lock();

        let mut id = me.lastIDUsed + 1;
        while id != me.lastIDUsed {
            if id < 0 {
                id = 0;
            }

            if me.shms.get(&id).is_some() {
                id += 1;
                continue;
            }

            me.lastIDUsed = id;
            shm.lock().id = id;
            me.shms.insert(id, shm.clone());
            me.keysToShms.insert(key, shm.clone());
            me.totalPages += effectiveSize / MemoryDef::PAGE_SIZE;
            return Ok(shm)
        }

        info!("Shm ids exhuasted, they may be leaking");
        return Err(Error::SysError(SysErr::ENOSPC));
    }

    pub fn IPCInfo(&self) -> ShmParams {
        return ShmParams {
            ShmMax: SHMMAX,
            ShmMin: SHMMIN,
            ShmMni: SHMMNI,
            ShmSeg: SHMSEG,
            ShmAll: SHMALL,
        }
    }

    pub fn ShmInfo(&self) -> ShmInfo {
        let me = self.lock();
        return ShmInfo {
            UsedIDs: me.lastIDUsed,
            ShmTot: me.totalPages,
            ShmRss: me.totalPages,
            // We could probably get a better estimate from memory accounting.
            ShmSwp: 0,
            ..Default::default()
        }
    }

    fn remove(&self, s: &Shm) {
        let mut me = self.lock();
        let s = s.lock();

        if s.key == IPC_PRIVATE {
            panic!("Attempted to remove {:?} from the registry whose key is still associated", s);
        }

        me.shms.remove(&s.id);
        me.totalPages -= s.effectiveSize / MemoryDef::PAGE_SIZE;
    }
}

#[derive(Clone, Debug)]
pub struct ShmInternal {
    pub registry: Registry,
    pub id: ID,
    pub creator: FileOwner,
    pub size: u64,
    pub effectiveSize: u64,
    pub fr: Range,
    pub key: Key,
    pub perms: FilePermissions,
    pub owner: FileOwner,
    pub attachTime: Time,
    pub detachTime: Time,
    pub changeTime: Time,
    pub creatorPID: i32,
    pub lastAttachDetachPID: i32,
    pub pendingDestruction: bool,

    pub refCount: i64,
}

impl ShmInternal {
    pub fn checkOwnership(&self, task: &Task) -> bool {
        let creds = task.creds.clone();
        let effectiveKUID = creds.lock().EffectiveKUID;

        if self.owner.UID == effectiveKUID || self.creator.UID == effectiveKUID {
            return true
        }

        let userns = self.registry.lock().userNS.clone();

        return creds.HasCapabilityIn(Capability::CAP_SYS_ADMIN, &userns)
    }

    pub fn checkPermission(&self, task: &Task, req: &PermMask) -> bool {
        let creds = task.creds.clone();

        let mut p = self.perms.Other;
        if self.owner.UID == creds.lock().EffectiveKUID {
            p = self.perms.User;
        } else if creds.InGroup(self.owner.GID) {
            p = self.perms.Group;
        }

        if p.SupersetOf(req) {
            return true
        }

        let ns = self.registry.lock().userNS.clone();
        return creds.HasCapabilityIn(Capability::CAP_IPC_OWNER, &ns)
    }
}

#[derive(Clone, Debug)]
pub struct Shm(Arc<Mutex<ShmInternal>>);

impl Deref for Shm {
    type Target = Arc<Mutex<ShmInternal>>;

    fn deref(&self) -> &Arc<Mutex<ShmInternal>> {
        &self.0
    }
}

impl Mapping for Shm {
    fn MappedName(&self, _task: &Task) -> String {
        return format!("SYSV{}", self.lock().key)
    }

    fn DeviceID(&self) -> u64 {
        return SHM_DEVICE.lock().DeviceID();
    }

    fn InodeID(&self) -> u64 {
        return self.lock().id as u64;
    }
}

impl Shm {
    pub fn DecRef(&self) {
        {
            let mut me = self.lock();
            me.refCount -= 1;
            if me.refCount != 0 {
                return
            }
        }

        self.destroy()
    }

    pub fn EffectiveSize(&self) -> u64 {
        return self.lock().effectiveSize;
    }

    pub fn IPCStat(&self, task: &Task) -> Result<ShmidDS> {
        let me = self.lock();

        if !me.checkPermission(task, &PermMask { read: true, ..Default::default() }) {
            return Err(Error::SysError(SysErr::EACCES))
        }

        let mut mode: u16 = 0;
        if me.pendingDestruction {
            mode |= SHM_DEST;
        }

        let userns = task.creds.lock().UserNamespace.clone();
        let ds = ShmidDS {
            ShmPerm: IPCPerm {
                Key: me.key as u32,
                UID: userns.MapFromKUID(me.owner.UID).0,
                GID: userns.MapFromKGID(me.owner.GID).0,
                CUID: userns.MapFromKUID(me.creator.UID).0,
                CGID: userns.MapFromKGID(me.creator.GID).0,
                Mode: mode | me.perms.LinuxMode() as u16,
                Seq: 0,
                ..Default::default()
            },
            ShmSegsz: me.size,
            ShmAtime: me.attachTime.TimeT(),
            ShmDtime: me.detachTime.TimeT(),
            ShmCtime: me.changeTime.TimeT(),
            ShmCpid: me.creatorPID,
            ShmLpid: me.lastAttachDetachPID,
            ShmNattach: me.refCount,
            ..Default::default()
        };

        return Ok(ds)
    }

    pub fn Set(&self, task: &Task, ds: &ShmidDS) -> Result<()> {
        let mut me = self.lock();

        if !me.checkOwnership(task) {
            return Err(Error::SysError(SysErr::EPERM))
        }

        let creds = task.creds.clone();
        let userns = creds.lock().UserNamespace.clone();

        let uid = userns.MapToKUID(UID(ds.ShmPerm.UID));
        let gid = userns.MapToKGID(GID(ds.ShmPerm.GID));
        if !uid.Ok() || !gid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mode = FileMode(ds.ShmPerm.Mode & 0x1ff);
        me.perms = FilePermissions::FromMode(mode);

        me.owner.UID = uid;
        me.owner.GID = gid;

        me.changeTime = task.Now();
        return Ok(())
    }

    pub fn MarkDestroyed(&self) {
        self.lock().registry.dissociateKey(self);

        let needDec = {
            let mut me = self.lock();
            if !me.pendingDestruction {
                me.pendingDestruction = true;
                true
            } else {
                false
            }
        };

        if needDec {
            self.DecRef()
        }
    }

    pub fn destroy(&self) {
        let me = self.lock();
        PAGE_MGR.DerefRange(&me.fr).unwrap();
        let registry = self.lock().registry.clone();
        registry.remove(self)
    }
}

pub struct AttachOpts {
    pub Execute: bool,
    pub ReadOnly: bool,
    pub Remap: bool,
}
