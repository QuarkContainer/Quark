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

pub mod fdinfo;
//pub mod file_range_mgr;

use std::collections::BTreeMap;

use self::fdinfo::*;
use super::super::qlib::common::*;

pub struct IOMgr {
    pub osMap: BTreeMap<i32, FdInfo>,
}

unsafe impl Send for IOMgr {}

impl IOMgr {
    pub fn Print(&self) {
        for (osfd, fdInfo) in &self.osMap {
            info!("osfd[{}]->{:?}", osfd, fdInfo)
        }
    }

    pub fn Init() -> Result<Self> {
        let mut res = Self {
            osMap: BTreeMap::new(),
        };

        res.osMap.insert(0, FdInfo::New(0));
        res.osMap.insert(1, FdInfo::New(1));
        res.osMap.insert(2, FdInfo::New(2));
        return Ok(res);
    }

    pub fn AddFd(&mut self, osfd: i32) -> i32 {
        let fdInfo = FdInfo::New(osfd);
        self.osMap.insert(osfd, fdInfo.clone());

        return fdInfo.osfd;
    }

    //ret: true: exist, false: not exist
    pub fn RemoveFd(&mut self, hostfd: i32) -> Option<FdInfo> {
        return self.osMap.remove(&hostfd);
    }

    pub fn GetFdByHost(&self, hostfd: i32) -> Option<i32> {
        if self.osMap.contains_key(&hostfd) {
            return Some(hostfd)
        }

        return None
    }

    pub fn GetByHost(&self, hostfd: i32) -> Option<FdInfo> {
        match self.osMap.get(&hostfd) {
            None => {
                None
            }
            Some(fdInfo) => Some(fdInfo.clone()),
        }
    }
}
