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
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::super::addr::*;
use super::super::super::auth::id::*;
use super::super::super::auth::userns::*;
use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::device::SHM_DEVICE;
use super::super::super::linux::ipc::*;
use super::super::super::linux::shm::*;
use super::super::super::linux_def::*;
use super::super::super::range::*;
use super::super::fs::host::hostinodeop::*;
use super::super::memmgr::mm::MemoryManager;
use super::super::memmgr::*;
use super::super::task::*;
use super::ipc_namespace::*;
use super::time::*;

#[derive(Default)]
pub struct ShmRegistryInternal {
    pub userNS: UserNameSpace,
    pub shms: BTreeMap<ID, Shm>,
    pub keysToShms: BTreeMap<Key, Shm>,
    pub lastIDUsed: ID,
    pub totalPages: u64,
}

impl ShmRegistryInternal {}

#[derive(Clone, Default)]
pub struct ShmRegistry(Arc<QMutex<ShmRegistryInternal>>);

impl Deref for ShmRegistry {
    type Target = Arc<QMutex<ShmRegistryInternal>>;

    fn deref(&self) -> &Arc<QMutex<ShmRegistryInternal>> {
        &self.0
    }
}

impl ShmRegistry {
    pub fn New(userNS: &UserNameSpace) -> Self {
        let internal = ShmRegistryInternal {
            userNS: userNS.clone(),
            shms: BTreeMap::new(),
            keysToShms: BTreeMap::new(),
            totalPages: 0,
            lastIDUsed: 0,
        };

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn FindByID(&self, id: ID) -> Option<Shm> {
        let me = self.lock();
        return match me.shms.get(&id) {
            None => None,
            Some(shm) => Some(shm.clone()),
        };
    }

    fn dissociateKey(&self, shm: &Shm) {
        let mut me = self.lock();
        let mut s = shm.lock();

        if s.key != IPC_PRIVATE {
            me.keysToShms.remove(&s.key);
            s.key = IPC_PRIVATE;
        }
    }

    pub fn FindOrCreate(
        &self,
        task: &Task,
        pid: i32,
        key: Key,
        size: u64,
        mode: &FileMode,
        private: bool,
        create: bool,
        exclusive: bool,
    ) -> Result<Shm> {
        if (create || private) && (size < SHMMIN || size > SHMMAX) {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        {
            let me = self.lock();
            if me.shms.len() > SHMMNI as usize {
                return Err(Error::SysError(SysErr::ENOSPC));
            }

            if !private {
                match me.keysToShms.get(&key) {
                    None => (),
                    Some(s) => {
                        {
                            let shm = s.lock();

                            if !shm.checkPermission(task, &PermMask::FromMode(mode.clone())) {
                                return Err(Error::SysError(SysErr::EACCES));
                            }

                            if size > shm.size {
                                return Err(Error::SysError(SysErr::EINVAL));
                            }

                            if create && exclusive {
                                return Err(Error::SysError(SysErr::EEXIST));
                            }
                        }

                        return Ok(s.clone());
                    }
                };

                if !create {
                    return Err(Error::SysError(SysErr::ENOENT));
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

    fn newShm(
        &self,
        task: &Task,
        pid: i32,
        key: Key,
        creator: &FileOwner,
        perms: &FilePermissions,
        size: u64,
    ) -> Result<Shm> {
        let effectiveSize = Addr(size).MustRoundUp().0;
        let fr = Range::New(0, effectiveSize);

        let memfdIops = HostInodeOp::NewMemfdIops(size as _)?;

        let shm = Shm(Arc::new(QMutex::new(ShmInternal {
            memfdIops: memfdIops,
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
            return Ok(shm);
        }

        return Err(Error::SysError(SysErr::ENOSPC));
    }

    pub fn IPCInfo(&self) -> ShmParams {
        return ShmParams {
            ShmMax: SHMMAX,
            ShmMin: SHMMIN,
            ShmMni: SHMMNI,
            ShmSeg: SHMSEG,
            ShmAll: SHMALL,
        };
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
        };
    }

    fn remove(&self, s: &Shm) {
        let mut me = self.lock();
        let s = s.lock();

        if s.key == IPC_PRIVATE {
            //panic!("Attempted to remove {:?} from the registry whose key is still associated", s);
            panic!("Attempted to remove Shm from the registry whose key is still associated");
        }

        me.shms.remove(&s.id);
        me.totalPages -= s.effectiveSize / MemoryDef::PAGE_SIZE;
    }
}

#[derive(Clone)]
pub struct ShmInternal {
    pub memfdIops: HostInodeOp,

    pub registry: ShmRegistry,
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
}

impl ShmInternal {
    // checkOwnership verifies whether a segment may be accessed by ctx as an
    // owner. See ipc/util.c:ipcctl_pre_down_nolock() in Linux.
    pub fn checkOwnership(&self, task: &Task) -> bool {
        let creds = task.creds.clone();
        let effectiveKUID = creds.lock().EffectiveKUID;

        if self.owner.UID == effectiveKUID || self.creator.UID == effectiveKUID {
            return true;
        }

        let userns = self.registry.lock().userNS.clone();

        return creds.HasCapabilityIn(Capability::CAP_SYS_ADMIN, &userns);
    }

    // checkPermissions verifies whether a segment is accessible by ctx for access
    // described by req. See ipc/util.c:ipcperms() in Linux.
    pub fn checkPermission(&self, task: &Task, req: &PermMask) -> bool {
        let creds = task.creds.clone();

        let mut p = self.perms.Other;
        if self.owner.UID == creds.lock().EffectiveKUID {
            p = self.perms.User;
        } else if creds.InGroup(self.owner.GID) {
            p = self.perms.Group;
        }

        if p.SupersetOf(req) {
            return true;
        }

        let ns = self.registry.lock().userNS.clone();
        return creds.HasCapabilityIn(Capability::CAP_IPC_OWNER, &ns);
    }
}

#[derive(Clone)]
pub struct Shm(Arc<QMutex<ShmInternal>>);

impl Deref for Shm {
    type Target = Arc<QMutex<ShmInternal>>;

    fn deref(&self) -> &Arc<QMutex<ShmInternal>> {
        &self.0
    }
}

impl PartialEq for Shm {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl Eq for Shm {}

impl Mapping for Shm {
    fn MappedName(&self, _task: &Task) -> String {
        return format!("SYSV{}", self.lock().key);
    }

    fn DeviceID(&self) -> u64 {
        return SHM_DEVICE.lock().DeviceID();
    }

    fn InodeID(&self) -> u64 {
        return self.lock().id as u64;
    }
}

impl Shm {
    pub fn Id(&self) -> ID {
        return self.lock().id;
    }

    pub fn HostIops(&self) -> HostInodeOp {
        return self.lock().memfdIops.clone();
    }

    pub fn EffectiveSize(&self) -> u64 {
        return self.lock().effectiveSize;
    }

    pub fn AttachCount(&self) -> usize {
        let mut attachCount = Arc::strong_count(&self.0) - 1; //sub the current reference;
        let me = self.lock();
        if !me.pendingDestruction {
            attachCount -= 1; 
        }
        return attachCount / 2; // one for mappable, one for mapping
    }

    pub fn IPCStat(&self, task: &Task) -> Result<ShmidDS> {
        let attachCount = self.AttachCount();
        let me = self.lock();

        if !me.checkPermission(
            task,
            &PermMask {
                read: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
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
            ShmNattach: attachCount as _,
            ..Default::default()
        };

        return Ok(ds);
    }

    pub fn Set(&self, task: &Task, ds: &ShmidDS) -> Result<()> {
        let mut me = self.lock();

        if !me.checkOwnership(task) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        let creds = task.creds.clone();
        let userns = creds.lock().UserNamespace.clone();

        let uid = userns.MapToKUID(UID(ds.ShmPerm.UID));
        let gid = userns.MapToKGID(GID(ds.ShmPerm.GID));
        if !uid.Ok() || !gid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let mode = FileMode(ds.ShmPerm.Mode & 0x1ff);
        me.perms = FilePermissions::FromMode(mode);

        me.owner.UID = uid;
        me.owner.GID = gid;

        me.changeTime = task.Now();
        return Ok(());
    }

    pub fn MarkDestroyed(&self) {
        let registry = self.lock().registry.clone();
        registry.dissociateKey(self);

        let _needDec = {
            let mut me = self.lock();
            if !me.pendingDestruction {
                me.pendingDestruction = true;
                true
            } else {
                false
            }
        };
    }

    pub fn destroy(&self) {
        let registry = self.lock().registry.clone();
        registry.remove(self)
    }

    // ConfigureAttach creates an mmap configuration for the segment with the
    // requested attach options.
    //
    // Postconditions: The returned MMapOpts are valid only as long as a reference
    // continues to be held on s.
    pub fn ConfigureAttach(&self, task: &Task, addr: u64, opts: &AttachOpts) -> Result<MMapOpts> {
        let attachCount = self.AttachCount();
        let me = self.lock();
        if me.pendingDestruction && attachCount == 0 {
            return Err(Error::SysError(SysErr::EIDRM));
        }

        if !me.checkPermission(
            task,
            &PermMask {
                read: true,
                write: !opts.ReadOnly,
                execute: opts.Execute,
            },
        ) {
            // "The calling process does not have the required permissions for the
            // requested attach type, and does not have the CAP_IPC_OWNER capability
            // in the user namespace that governs its IPC namespace." - man shmat(2)
            return Err(Error::SysError(SysErr::EACCES));
        }

        let mmapOpts = MMapOpts {
            Length: me.size,
            Addr: addr,
            Offset: 0,
            Fixed: opts.Remap,
            Unmap: opts.Remap,
            Map32Bit: false,
            Perms: AccessType::New(true, !opts.ReadOnly, opts.Execute),
            MaxPerms: AccessType::AnyAccess(),
            Private: false,
            VDSO: false,
            GrowsDown: false,
            Precommit: false,
            MLockMode: MLockMode::MlockNone,
            Kernel: false,
            Mapping: Some(Arc::new(self.clone())),
            Mappable: MMappable::FromShm(self.clone()),
            Hint: format!(""),
        };

        return Ok(mmapOpts);
    }
}

pub struct AttachOpts {
    pub Execute: bool,
    pub ReadOnly: bool,
    pub Remap: bool,
}

impl MemoryManager {
    // DetachShm unmaps a sysv shared memory segment.
    pub fn DetachShm(&self, task: &Task, addr: u64) -> Result<()> {
        if addr != Addr(addr).RoundDown().unwrap().0 {
            // "... shmaddr is not aligned on a page boundary." - man shmdt(2)
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let _ml = self.MappingWriteLock();
        let mut mapping = self.mapping.lock();
        let (mut vseg, vgap) = mapping.vmas.Find(addr);
        if vgap.Ok() {
            vseg = vgap.NextSeg();
        }

        let mut detached = None;
        while vseg.Ok() {
            let vma = vseg.Value();
            match vma.mappable {
                MMappable::Shm(shm) => {
                    if vseg.Range().Start() - addr == vma.offset {
                        detached = Some(shm.clone());
                        break;
                    }
                }
                _ => (),
            }
            vseg = vseg.NextSeg();
        }

        let detached = match detached {
            None => {
                // There is no shared memory segment attached at addr.
                return Err(Error::SysError(SysErr::EINVAL));
            }
            Some(shm) => shm,
        };

        detached.lock().detachTime = task.Now();

        let end = addr + detached.EffectiveSize();
        while vseg.Ok() && vseg.Range().End() <= end {
            let vma = vseg.Value();
            if vma.mappable == MMappable::FromShm(detached.clone())
                && vseg.Range().Start() - addr == vma.offset
            {
                let r = vseg.Range();
                mapping.usageAS -= r.Len();
                if vma.mlockMode != MLockMode::MlockNone {
                    mapping.lockedAS -= r.Len();
                }

                let mut pt = self.pagetable.write();

                pt.pt.MUnmap(r.Start(), r.Len())?;
                pt.curRSS -= r.Len();
                let vgap = mapping.vmas.Remove(&vseg);
                vseg = vgap.NextSeg();
            } else {
                vseg = vseg.NextSeg();
            }
        }

        self.TlbShootdown();
        return Ok(());
    }
}
