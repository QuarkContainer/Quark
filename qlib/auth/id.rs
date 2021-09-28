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

use alloc::vec::Vec;
use spin::Mutex;

use super::userns::*;
use super::super::singleton::*;

pub static HOST_AUTH_ID : Singleton<Mutex<HostAuthID>> = Singleton::<Mutex<HostAuthID>>::New();

pub unsafe fn InitSingleton() {
    HOST_AUTH_ID.Init(Mutex::new(HostAuthID::New()));
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct UID(pub u32);

impl UID {
    pub fn Ok(&self) -> bool {
        return self.0 != NO_ID
    }

    pub fn OrOverflow(&self) -> Self {
        if self.Ok() {
            return self.clone()
        }

        return OVERFLOW_UID.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct GID(pub u32);

impl GID {
    pub fn Ok(&self) -> bool {
        return self.0 != NO_ID
    }

    pub fn OrOverflow(&self) -> Self {
        if self.Ok() {
            return self.clone()
        }

        return OVERFLOW_GID.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
pub struct KUID(pub u32);

impl KUID {
    pub fn Ok(&self) -> bool {
        return self.0 != NO_ID
    }

    pub fn In(&self, ns: &UserNameSpace) -> UID {
        return ns.MapFromKUID(*self)
    }
}

impl Default for KUID {
    fn default() -> Self {
        return NOBODY_KUID
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
pub struct KGID(pub u32);

impl KGID {
    pub fn Ok(&self) -> bool {
        return self.0 != NO_ID
    }

    pub fn In(&self, ns: &UserNameSpace) -> GID {
        return ns.MapFromKGID(*self)
    }
}

impl Default for KGID {
    fn default() -> Self {
        return NOBODY_KGID
    }
}

pub const NO_ID: u32 = core::u32::MAX;
pub const OVERFLOW_UID: UID = UID(65534);
pub const OVERFLOW_GID: GID = GID(65534);
pub const OVERFLOW_KUID: KUID = KUID(65534);
pub const OVERFLOW_KGID: KGID = KGID(65534);
pub const NOBODY_KUID: KUID = KUID(65534);
pub const NOBODY_KGID: KGID = KGID(65534);

pub const ROOT_UID: UID = UID(0);
pub const ROOT_GID: GID = GID(0);
pub const ROOT_KUID: KUID = KUID(0);
pub const ROOT_KGID: KGID = KGID(0);

#[derive(Debug)]
pub struct HostAuthID {
    pub uid: u32,
    pub gids: Vec<u32>,
}

impl HostAuthID {
    pub fn New() -> Self {
        return Self {
            uid: 0,
            gids: Vec::new(),
        }
    }
}
