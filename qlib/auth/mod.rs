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

pub mod id;
pub mod cap_set;
pub mod userns;

use alloc::vec::Vec;

use super::common::*;
use super::linux_def::*;
//use super::fs::inode::*;
use self::id::*;
use self::cap_set::*;
use self::userns::*;

#[derive(Debug)]
pub struct CredentialsInternal {
    pub RealKUID: KUID,
    pub EffectiveKUID: KUID,
    pub SavedKUID: KUID,
    pub RealKGID: KGID,
    pub EffectiveKGID: KGID,
    pub SavedKGID: KGID,

    pub ExtraKGIDs: Vec<KGID>,

    pub PermittedCaps: CapSet,
    pub InheritableCaps: CapSet,
    pub EffectiveCaps: CapSet,
    pub BoundingCaps: CapSet,

    pub KeepCaps: bool,
    pub UserNamespace: UserNameSpace,
}

impl CredentialsInternal {
    pub fn InGroup(&self, kgid: KGID) -> bool {
        if self.EffectiveKGID == kgid {
            return true;
        }

        for extraKGID in &self.ExtraKGIDs {
            if *extraKGID == kgid {
                return true
            }
        }

        return false
    }

    pub fn HasCapabilityIn(&self, cp: u64, ns: &UserNameSpace) -> bool {
        let mut ns = ns.clone();
        loop {
            if Arc::ptr_eq(&self.UserNamespace, &ns) {
                return CapSet::New(cp).0 & self.EffectiveCaps.0 != 0
            }

            let tmp: UserNameSpace;
            {
                let nsInternal = ns.lock();

                if let Some(parent) = &nsInternal.parent {
                    if Arc::ptr_eq(&self.UserNamespace, &parent.clone()) && self.EffectiveKUID.0 == nsInternal.owner.0 {
                        return true;
                    } else {
                        tmp = parent.clone();
                    }
                } else {
                    return false
                }
            }

            ns = tmp
        }
    }

    pub fn HasCapability(&self, cp: u64) -> bool {
        return self.HasCapabilityIn(cp, &self.UserNamespace.clone())
    }

    fn copy(&self) -> Self {
        let mut extraKGIDs = Vec::with_capacity(self.ExtraKGIDs.len());
        for i in 0..self.ExtraKGIDs.len() {
            extraKGIDs.push(self.ExtraKGIDs[i])
        }

        let internal = Self {
            RealKUID: self.RealKUID,
            EffectiveKUID: self.EffectiveKUID,
            SavedKUID: self.SavedKUID,
            RealKGID: self.RealKGID,
            EffectiveKGID: self.EffectiveKGID,
            SavedKGID: self.SavedKGID,
            ExtraKGIDs: extraKGIDs,
            PermittedCaps: self.PermittedCaps,
            InheritableCaps: self.InheritableCaps,
            EffectiveCaps: self.EffectiveCaps,
            BoundingCaps: self.BoundingCaps,
            KeepCaps: self.KeepCaps,
            UserNamespace: self.UserNamespace.clone(),
        };

        return internal
    }
}

#[derive(Clone, Debug)]
pub struct Credentials(Arc<Mutex<CredentialsInternal>>);

impl PartialEq for Credentials {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for Credentials {
    type Target = Arc<Mutex<CredentialsInternal>>;

    fn deref(&self) -> &Arc<Mutex<CredentialsInternal>> {
        &self.0
    }
}

impl Default for Credentials {
    fn default() -> Self {
        return Self::NewRootCredentials(UserNameSpace::NewRootUserNamespace())
    }
}

impl Credentials {
    pub fn NewAnaoCredentials() -> Self {
        let internal = CredentialsInternal {
            RealKUID: NOBODY_KUID,
            EffectiveKUID: NOBODY_KUID,
            SavedKUID: NOBODY_KUID,
            RealKGID: NOBODY_KGID,
            EffectiveKGID: NOBODY_KGID,
            SavedKGID: NOBODY_KGID,

            ExtraKGIDs: Vec::new(),

            PermittedCaps: CapSet::New(0),
            InheritableCaps: CapSet::New(0),
            EffectiveCaps: CapSet::New(0),
            BoundingCaps: CapSet::New(0),
            KeepCaps: false,
            UserNamespace: UserNameSpace::NewRootUserNamespace(),
        };

        return Self(Arc::new(Mutex::new(internal)))
    }

    pub fn NewRootCredentials(userns: UserNameSpace) -> Self {
        let internal = CredentialsInternal {
            RealKUID: ROOT_KUID,
            EffectiveKUID: ROOT_KUID,
            SavedKUID: ROOT_KUID,
            RealKGID: ROOT_KGID,
            EffectiveKGID: ROOT_KGID,
            SavedKGID: ROOT_KGID,

            ExtraKGIDs: Vec::new(),

            PermittedCaps: ALL_CAP,
            InheritableCaps: CapSet::New(0),
            EffectiveCaps: ALL_CAP,
            BoundingCaps: ALL_CAP,
            KeepCaps: false,
            UserNamespace: userns,
        };

        return Self(Arc::new(Mutex::new(internal)))
    }

    pub fn NewUserCredentials(kuid: KUID, kgid: KGID, extraKGIDs: &[KGID], caps: Option<&TaskCaps>, userns: &UserNameSpace) -> Self {
        let res = Self::NewRootCredentials(userns.clone());

        {
            let mut creds = res.lock();

            creds.RealKUID = kuid;
            creds.EffectiveKUID = kuid;
            creds.SavedKUID = kuid;

            creds.RealKGID = kgid;
            creds.EffectiveKGID = kgid;
            creds.SavedKGID = kgid;

            for gid in extraKGIDs {
                creds.ExtraKGIDs.push(*gid);
            }

            match caps {
                Some(caps) => {
                    creds.PermittedCaps = caps.PermittedCaps;
                    creds.EffectiveCaps = caps.EffectiveCaps;
                    creds.BoundingCaps = caps.BoundingCaps;
                    creds.InheritableCaps = caps.InheritableCaps;
                }
                None => {
                    if kuid.0 == ROOT_KUID.0 {
                        creds.PermittedCaps = ALL_CAP;
                        creds.EffectiveCaps = ALL_CAP;
                    } else {
                        creds.PermittedCaps = CapSet::New(0);
                        creds.EffectiveCaps = CapSet::New(0);
                    }

                    creds.BoundingCaps = ALL_CAP;
                }
            }
        }

        return res;
    }

    fn copy(&self) -> Self {
        let internal = self.lock().copy();

        return Self(Arc::new(Mutex::new(internal)))
    }

    pub fn Fork(&self) -> Self {
        return self.copy()
    }

    pub fn FileOwner(&self) -> FileOwner {
        let me = self.lock();
        return FileOwner {
            UID: me.EffectiveKUID,
            GID: me.EffectiveKGID,
        }
    }

    pub fn InGroup(&self, kgid: KGID) -> bool {
        let me = self.lock();
        return me.InGroup(kgid)
    }

    pub fn HasCapabilityIn(&self, cp: u64, ns: &UserNameSpace) -> bool {
        let me = self.lock();
        return me.HasCapabilityIn(cp, ns)
    }

    pub fn HasCapability(&self, cp: u64) -> bool {
        let me = self.lock();
        return me.HasCapability(cp)
    }

    pub fn UseUID(&self, uid: UID) -> Result<KUID> {
        let me = self.lock();

        let kuid = me.UserNamespace.MapToKUID(uid);
        if !kuid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if me.HasCapability(Capability::CAP_SETUID) {
            return Ok(kuid)
        }

        if kuid == me.RealKUID || kuid == me.EffectiveKUID || kuid == me.SavedKUID {
            return Ok(kuid)
        }

        return Err(Error::SysError(SysErr::EPERM))
    }

    pub fn UseGID(&self, gid: GID) -> Result<KGID> {
        let me = self.lock();

        let kgid = me.UserNamespace.MapToKGID(gid);
        if !kgid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if me.HasCapability(Capability::CAP_SETGID) {
            return Ok(kgid)
        }

        if kgid == me.RealKGID || kgid == me.EffectiveKGID || kgid == me.SavedKGID {
            return Ok(kgid)
        }

        return Err(Error::SysError(SysErr::EPERM))
    }

    pub fn NewChildUserNamespace(&self) -> Result<UserNameSpace> {
        let ns = self.lock().UserNamespace.clone();

        if ns.Depth() >= MAX_USER_NAMESPACE_DEPTH {
            // "... Calls to unshare(2) or clone(2) that would cause this limit to
            // be exceeded fail with the error EUSERS." - user_namespaces(7)
            return Err(Error::SysError(SysErr::EUSERS))
        }

        // "EPERM: CLONE_NEWUSER was specified in flags, but either the effective
        // user ID or the effective group ID of the caller does not have a mapping
        // in the parent namespace (see user_namespaces(7))." - clone(2)
        // "CLONE_NEWUSER requires that the user ID and group ID of the calling
        // process are mapped to user IDs and group IDs in the user namespace of
        // the calling process at the time of the call." - unshare(2)
        if !self.lock().EffectiveKUID.In(&ns).Ok() {
            return Err(Error::SysError(SysErr::EPERM))
        }

        let internal = UserNameSpaceInternal {
            parent: Some(ns),
            owner: self.lock().EffectiveKUID,
            ..Default::default()
        };

        return Ok(UserNameSpace(Arc::new(Mutex::new(internal))))
    }
}

const MAX_USER_NAMESPACE_DEPTH: usize = 32;

#[derive(Debug, Default, Copy, Clone)]
pub struct FileOwner {
    pub UID: KUID,
    pub GID: KGID,
}

pub const ROOT_OWNER: FileOwner = FileOwner {
    UID: ROOT_KUID,
    GID: ROOT_KGID,
};
