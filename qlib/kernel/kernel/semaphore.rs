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
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use core::sync::atomic::AtomicU64;

use super::super::super::auth::userns::*;
use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::linux::ipc::*;
use super::super::super::linux::sem::*;
use super::super::super::linux_def::*;
use super::super::super::singleton::*;
use super::super::task::*;
use super::super::uid::*;
use super::waiter::*;
//use super::super::fs::attr::*;
use super::time::*;

pub static WAITER_ID: Singleton<AtomicU64> = Singleton::<AtomicU64>::New();
pub unsafe fn InitSingleton() {
    WAITER_ID.Init(AtomicU64::new(1));
}

pub const VALUE_MAX: i16 = SEMMNI as i16; // SEMVMX

// setMax is "system-wide limit on the number of semaphore sets" (SEMMNI).
pub const SETS_MAX: u32 = SEMMNI;

// semaphoresMax is "maximum number of semaphores per semaphore ID" (SEMMSL).
pub const SEMAPHORES_MAX: u32 = SEMMSL;

// semaphoresTotalMax is "system-wide limit on the number of semaphores"
// (SEMMNS = SEMMNI*SEMMSL).
pub const SEMAPHORES_TOTAL_MAX: u32 = SEMMNS;

#[derive(Default)]
pub struct SemRegistryInternal {
    pub userNS: UserNameSpace,
    pub semaphores: BTreeMap<i32, Set>,
    pub lastIDUsed: i32,
}

impl SemRegistryInternal {
    pub fn findByID(&self, id: i32) -> Option<Set> {
        match self.semaphores.get(&id) {
            None => None,
            Some(s) => Some(s.clone()),
        }
    }

    fn findByKey(&self, key: i32) -> Option<Set> {
        for (_, v) in &self.semaphores {
            if v.lock().key == key {
                return Some(v.clone());
            }
        }

        return None;
    }

    pub fn HighestIndex(&mut self) -> i32 {
        let mut last = 0;
        error!("HighestIndex {:?}", self.semaphores.keys());
        for (i, _) in self.semaphores.iter().rev() {
            last = *i;
            break;
        }

        return last;
    }

    fn totalSems(&self) -> usize {
        let mut totalSems = 0;

        for (_, v) in &self.semaphores {
            totalSems += v.Size();
        }

        return totalSems;
    }

    fn newSet(
        &mut self,
        task: &Task,
        myself: SemRegistry,
        key: i32,
        owner: &FileOwner,
        creator: &FileOwner,
        perms: &FilePermissions,
        nsems: i32,
    ) -> Result<Set> {
        let set = Set::New(
            task,
            myself,
            key,
            owner.clone(),
            creator.clone(),
            perms.clone(),
            nsems,
        );
        let me = self;

        let mut id = me.lastIDUsed + 1;
        while id != me.lastIDUsed {
            if id < 0 {
                id = 0;
            }

            if !me.semaphores.contains_key(&id) {
                me.lastIDUsed = id;
                set.lock().id = id;
                me.semaphores.insert(id, set.clone());
                return Ok(set);
            }

            id += 1;
        }

        info!("Semaphore map is full, they must be leaking");
        return Err(Error::SysError(SysErr::ENOMEM));
    }
}

#[derive(Clone, Default)]
pub struct SemRegistry(Arc<QMutex<SemRegistryInternal>>);

impl Deref for SemRegistry {
    type Target = Arc<QMutex<SemRegistryInternal>>;

    fn deref(&self) -> &Arc<QMutex<SemRegistryInternal>> {
        &self.0
    }
}

impl SemRegistry {
    pub fn New(userNS: &UserNameSpace) -> Self {
        let internal = SemRegistryInternal {
            userNS: userNS.clone(),
            semaphores: BTreeMap::new(),
            lastIDUsed: 0,
        };

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn FindOrCreate(
        &self,
        task: &Task,
        key: i32,
        nsems: i32,
        mode: FileMode,
        private: bool,
        create: bool,
        exclusive: bool,
    ) -> Result<Set> {
        if nsems < 0 || nsems > SEMAPHORES_MAX as i32 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let clone = self.clone();

        let mut me = self.lock();

        if !private {
            match me.findByKey(key) {
                None => (),
                Some(set) => {
                    {
                        let set = set.lock();

                        let creds = task.creds.clone();
                        if !set.checkPerms(&creds, &PermMask::FromMode(mode.clone())) {
                            return Err(Error::SysError(SysErr::EACCES));
                        }

                        if nsems > set.sems.len() as i32 {
                            return Err(Error::SysError(SysErr::EINVAL));
                        }

                        if create && exclusive {
                            return Err(Error::SysError(SysErr::EEXIST));
                        }
                    }

                    return Ok(set);
                }
            }

            if !create {
                return Err(Error::SysError(SysErr::ENOENT));
            }
        }

        if nsems == 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if me.semaphores.len() >= SETS_MAX as usize {
            return Err(Error::SysError(SysErr::ENOSPC));
        }

        if me.totalSems() > SEMAPHORES_TOTAL_MAX as usize - nsems as usize {
            return Err(Error::SysError(SysErr::ENOSPC));
        }

        let owner = task.FileOwner();
        let perms = FilePermissions::FromMode(mode);
        return me.newSet(task, clone, key, &owner, &owner, &perms, nsems);
    }

    pub fn IPCInfo(&self) -> SemInfo {
        return SemInfo {
            SemMap: SEMMAP,
            SemMni: SEMMNI,
            SemMns: SEMMNS,
            SemMnu: SEMMNU,
            SemMsl: SEMMSL,
            SemOpm: SEMOPM,
            SemUme: SEMUME,
            SemUsz: SEMUSZ,
            SemVmx: SEMVMX,
            SemAem: SEMAEM,
        };
    }

    pub fn SemInfo(&self) -> SemInfo {
        let me = self.lock();
        let mut info = self.IPCInfo();
        info.SemUsz = me.semaphores.len() as _;
        info.SemAem = me.totalSems() as _;
        return info;
    }

    pub fn RemoveId(&self, id: i32, creds: &Credentials) -> Result<()> {
        let mut me = self.lock();

        let set = match me.semaphores.get(&id) {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(s) => s.clone(),
        };

        let mut set = set.lock();

        if !set.checkCredentials(creds) && !set.checkCapability(creds) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        me.semaphores.remove(&set.id);
        set.destroy();

        return Ok(());
    }
}

pub struct SetInternal {
    pub registry: SemRegistry,
    pub id: i32,
    pub key: i32,
    pub creator: FileOwner,
    pub owner: FileOwner,
    pub perms: FilePermissions,
    pub opTime: Time,
    pub changeTime: Time,

    pub sems: Vec<Sem>,
    pub dead: bool,
}

impl<'a> SetInternal {
    fn findSem_mut(&mut self, num: i32) -> Option<&mut Sem> {
        if num < 0 || num as usize > self.sems.len() {
            return None;
        }

        return Some(&mut self.sems[num as usize]);
    }

    fn findSem(&self, num: i32) -> Option<&Sem> {
        if num < 0 || num as usize >= self.sems.len() {
            return None;
        }

        return Some(&self.sems[num as usize]);
    }

    fn Size(&self) -> i32 {
        return self.sems.len() as _;
    }

    fn checkCredentials(&self, creds: &Credentials) -> bool {
        let creds = creds.lock();

        return self.owner.UID == creds.EffectiveKUID
            || self.owner.GID == creds.EffectiveKGID
            || self.creator.UID == creds.EffectiveKUID
            || self.creator.GID == creds.EffectiveKGID;
    }

    fn checkCapability(&self, creds: &Credentials) -> bool {
        let res = creds.HasCapabilityIn(Capability::CAP_IPC_OWNER, &self.registry.lock().userNS);
        let userns = creds.lock().UserNamespace.clone();
        return res && userns.MapFromKUID(self.owner.UID).Ok();
    }

    fn checkPerms(&self, creds: &Credentials, reqPerms: &PermMask) -> bool {
        let mut p: PermMask = self.perms.Other;
        if self.owner.UID == creds.lock().EffectiveKUID {
            p = self.perms.User;
        } else if creds.InGroup(self.owner.GID) {
            p = self.perms.Group;
        }

        if p.SupersetOf(reqPerms) {
            return true;
        }

        return self.checkCapability(creds);
    }

    fn destroy(&mut self) {
        self.dead = true;
        for s in &mut self.sems {
            s.queue.Notify(EVENT_IN);
        }
    }

    // ret Ok(n): if n >= 0, waiting on sems[n], if n == -1 success
    fn executeOps(
        &mut self,
        task: &Task,
        ops: &[Sembuf],
        e: &WaitEntry,
        pid: i32,
    ) -> Result<(u64, i32)> {
        // Did it race with a removal operation?
        if self.dead {
            return Err(Error::SysError(SysErr::EIDRM));
        }

        let mut tmpVals = Vec::with_capacity(self.sems.len());
        for i in 0..self.sems.len() {
            tmpVals.push(self.sems[i].value)
        }

        for op in ops {
            let sem = &mut self.sems[op.SemNum as usize];
            if op.SemOp == 0 {
                if tmpVals[op.SemNum as usize] != 0 {
                    if op.SemFlag & IPC_NOWAIT != 0 {
                        return Err(Error::SysError(SysErr::EWOULDBLOCK));
                    }

                    let waiterId = NewUID();
                    sem.waiters.insert(waiterId, op.SemOp);

                    sem.queue.EventRegister(task, e, EVENT_READ);
                    return Ok((waiterId, op.SemNum as _));
                }
            } else {
                if op.SemOp < 0 {
                    if -op.SemOp > VALUE_MAX {
                        return Err(Error::SysError(SysErr::ERANGE));
                    }

                    if -op.SemOp > tmpVals[op.SemNum as usize] {
                        if op.SemFlag & IPC_NOWAIT != 0 {
                            return Err(Error::SysError(SysErr::EWOULDBLOCK));
                        }

                        let waiterId = NewUID();
                        sem.waiters.insert(waiterId, op.SemOp);
                        sem.queue.EventRegister(task, e, READABLE_EVENT);
                        return Ok((waiterId, op.SemNum as _));
                    }
                } else {
                    if tmpVals[op.SemNum as usize] > VALUE_MAX - op.SemOp {
                        return Err(Error::SysError(SysErr::ERANGE));
                    }
                }

                tmpVals[op.SemNum as usize] += op.SemOp;
            }
        }

        for i in 0..tmpVals.len() {
            self.sems[i].value = tmpVals[i];
            self.sems[i].wakeWaiters();
            self.sems[i].pid = pid;
        }

        self.opTime = task.Now();

        // -1 means success
        return Ok((0, -1));
    }
}

#[derive(Clone)]
pub struct Set(Arc<QMutex<SetInternal>>);

impl Deref for Set {
    type Target = Arc<QMutex<SetInternal>>;

    fn deref(&self) -> &Arc<QMutex<SetInternal>> {
        &self.0
    }
}

impl Set {
    pub fn New(
        task: &Task,
        r: SemRegistry,
        key: i32,
        owner: FileOwner,
        creator: FileOwner,
        perms: FilePermissions,
        nsems: i32,
    ) -> Self {
        let mut internal = SetInternal {
            registry: r.clone(),
            id: 0,
            key: key,
            creator: creator,
            owner: owner,
            perms: perms,
            opTime: Time::default(),
            changeTime: task.Now(),
            sems: Vec::with_capacity(nsems as usize),
            dead: false,
        };

        for _i in 0..nsems as usize {
            internal.sems.push(Sem::default())
        }

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn Id(&self) -> i32 {
        return self.lock().id;
    }

    pub fn Size(&self) -> usize {
        return self.lock().sems.len();
    }

    pub fn GetStat(&self, creds: &Credentials) -> Result<SemidDS> {
        return self.semStat(
            creds,
            &PermMask {
                read: true,
                ..Default::default()
            },
        );
    }

    pub fn GetStatAny(&self, creds: &Credentials) -> Result<SemidDS> {
        return self.semStat(
            creds,
            &PermMask {
                ..Default::default()
            },
        );
    }

    pub fn semStat(&self, creds: &Credentials, permMask: &PermMask) -> Result<SemidDS> {
        let me = self.lock();

        // "The calling process must have read permission on the semaphore set."
        if !me.checkPerms(creds, permMask) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        let UID = creds.lock().UserNamespace.MapFromKUID(me.owner.UID).0;
        let GID = creds.lock().UserNamespace.MapFromKGID(me.owner.GID).0;
        let CUID = creds.lock().UserNamespace.MapFromKUID(me.creator.UID).0;
        let CGID = creds.lock().UserNamespace.MapFromKGID(me.creator.GID).0;
        let ds = SemidDS {
            SemPerm: IPCPerm {
                Key: me.key as u32,
                UID: UID,
                GID: GID,
                CUID: CUID,
                CGID: CGID,
                Mode: me.perms.LinuxMode() as _,
                Seq: 0,
                ..Default::default()
            },
            SemOTime: me.opTime.TimeT(),
            SemCTime: me.changeTime.TimeT(),
            SemNSems: me.Size() as _,
            ..Default::default()
        };

        return Ok(ds);
    }

    pub fn Change(
        &self,
        task: &Task,
        creds: &Credentials,
        owner: &FileOwner,
        perms: &FilePermissions,
    ) -> Result<()> {
        let mut me = self.lock();

        if !me.checkCredentials(creds) && !me.checkCapability(creds) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        me.owner = owner.clone();
        me.perms = perms.clone();
        me.changeTime = task.Now();

        return Ok(());
    }

    pub fn SetVal(
        &self,
        task: &Task,
        num: i32,
        val: i16,
        creds: &Credentials,
        pid: i32,
    ) -> Result<()> {
        if val < 0 || val > VALUE_MAX {
            return Err(Error::SysError(SysErr::ERANGE));
        }

        let mut me = self.lock();

        if !me.checkPerms(
            creds,
            &PermMask {
                write: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        match me.findSem_mut(num) {
            None => return Err(Error::SysError(SysErr::ERANGE)),
            Some(sem) => {
                sem.value = val;
                sem.pid = pid;
                sem.wakeWaiters();
            }
        }

        me.changeTime = task.Now();
        return Ok(());
    }

    pub fn SetValAll(
        &self,
        task: &Task,
        vals: &[u16],
        creds: &Credentials,
        pid: i32,
    ) -> Result<()> {
        if vals.len() != self.Size() {
            panic!(
                "vals length ({}) different that Set.Size() ({})",
                vals.len(),
                self.Size()
            );
        }

        for val in vals {
            if *val > VALUE_MAX as _ {
                return Err(Error::SysError(SysErr::ERANGE));
            }
        }

        let mut me = self.lock();

        if !me.checkPerms(
            creds,
            &PermMask {
                write: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        for i in 0..vals.len() {
            let sem = &mut me.sems[i];
            sem.value = vals[i] as i16;
            sem.pid = pid;
            sem.wakeWaiters();
        }

        me.changeTime = task.Now();
        return Ok(());
    }

    pub fn GetVal(&self, num: i32, creds: &Credentials) -> Result<i16> {
        let me = self.lock();

        if !me.checkPerms(
            creds,
            &PermMask {
                read: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        match me.findSem(num) {
            None => return Err(Error::SysError(SysErr::ERANGE)),
            Some(v) => return Ok(v.value),
        }
    }

    pub fn GetValAll(&self, creds: &Credentials) -> Result<Vec<i16>> {
        let me = self.lock();

        if !me.checkPerms(
            creds,
            &PermMask {
                read: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        let mut vals = Vec::with_capacity(me.sems.len());
        for i in 0..me.sems.len() {
            vals.push(me.sems[i].value);
        }

        return Ok(vals);
    }

    pub fn GetPID(&self, num: i32, creds: &Credentials) -> Result<i32> {
        let me = self.lock();

        if !me.checkPerms(
            creds,
            &PermMask {
                read: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        match me.findSem(num) {
            None => return Err(Error::SysError(SysErr::ERANGE)),
            Some(v) => return Ok(v.pid),
        }
    }

    pub fn GetZeroWaiters(&self, num: i32, creds: &Credentials) -> Result<u16> {
        let me = self.lock();

        if !me.checkPerms(
            creds,
            &PermMask {
                read: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        let sem = match me.findSem(num) {
            None => return Err(Error::SysError(SysErr::ERANGE)),
            Some(v) => v,
        };

        let mut semzcnt = 0;
        for (_key, val) in &sem.waiters {
            if *val == 0 {
                semzcnt += 1;
            }
        }

        return Ok(semzcnt);
    }

    pub fn GetNegativeWaiters(&self, num: i32, creds: &Credentials) -> Result<u16> {
        let me = self.lock();

        if !me.checkPerms(
            creds,
            &PermMask {
                read: true,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        let sem = match me.findSem(num) {
            None => return Err(Error::SysError(SysErr::ERANGE)),
            Some(v) => v,
        };

        let mut semzcnt = 0;
        for (_key, val) in &sem.waiters {
            if *val < 0 {
                semzcnt += 1;
            }
        }

        return Ok(semzcnt);
    }

    pub fn ExecuteOps(
        &self,
        task: &Task,
        ops: &[Sembuf],
        creds: &Credentials,
        e: &WaitEntry,
        pid: i32,
    ) -> Result<(u64, i32)> {
        let mut me = self.lock();

        if me.dead {
            return Err(Error::SysError(SysErr::EIDRM));
        }

        let mut readonly = true;

        for op in ops {
            if me.findSem(op.SemNum as i32).is_none() {
                return Err(Error::SysError(SysErr::EFBIG));
            }

            if op.SemOp != 0 {
                readonly = false;
            }
        }

        if !me.checkPerms(
            creds,
            &PermMask {
                read: readonly,
                write: !readonly,
                ..Default::default()
            },
        ) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        return me.executeOps(task, ops, e, pid);
    }

    pub fn AbortWait(&self, task: &Task, waiterid: u64, num: i32, e: &WaitEntry) {
        let mut me = self.lock();
        me.sems[num as usize].waiters.remove(&waiterid);
        let queue = me.sems[num as usize].queue.clone();
        queue.EventUnregister(task, e)
    }
}

pub struct Sem {
    pub value: i16,
    pub pid: i32,
    pub queue: Queue,
    pub waiters: BTreeMap<u64, i16>,
}

impl Default for Sem {
    fn default() -> Self {
        return Self {
            value: 0,
            pid: 0,
            queue: Queue::default(),
            waiters: BTreeMap::new(),
        };
    }
}

impl Sem {
    // wakeWaiters goes over all waiters and checks which of them can be notified.
    pub fn wakeWaiters(&mut self) {
        self.queue.Notify(EVENT_IN);
    }
}
