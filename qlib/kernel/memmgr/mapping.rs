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

use alloc::string::String;
use alloc::string::ToString;

use super::super::task::*;
use super::*;

pub struct SpecialMapping {
    pub name: String,
}

impl SpecialMapping {
    pub fn New(name: String) -> Self {
        return Self { name: name };
    }
}

impl Mapping for SpecialMapping {
    fn MappedName(&self, _task: &Task) -> String {
        return self.name.to_string();
    }

    // DeviceID returns the device number shown in /proc/[pid]/maps.
    fn DeviceID(&self) -> u64 {
        return 0;
    }

    // InodeID returns the inode number shown in /proc/[pid]/maps.
    fn InodeID(&self) -> u64 {
        return 0;
    }
}
