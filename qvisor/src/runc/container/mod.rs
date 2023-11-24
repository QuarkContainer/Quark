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

pub mod cgroups;
pub mod container;
pub mod exec_hook;
pub mod hook;
pub mod mounts;
pub mod nix_ext;
pub mod status;
pub mod nvidia;

use lazy_static;
use nix::sched::CloneFlags;
use std::collections::HashMap;

use super::oci::LinuxDevice;
use super::oci::LinuxDeviceType;

lazy_static! {
    pub static ref DEFAULT_DEVICES: Vec<LinuxDevice> = {
        let mut v = Vec::new();
        v.push(LinuxDevice {
            path: "/dev/null".to_string(),
            typ: LinuxDeviceType::c,
            major: 1,
            minor: 3,
            file_mode: Some(0o066),
            uid: None,
            gid: None,
        });
        v.push(LinuxDevice {
            path: "/dev/zero".to_string(),
            typ: LinuxDeviceType::c,
            major: 1,
            minor: 5,
            file_mode: Some(0o066),
            uid: None,
            gid: None,
        });
        v.push(LinuxDevice {
            path: "/dev/full".to_string(),
            typ: LinuxDeviceType::c,
            major: 1,
            minor: 7,
            file_mode: Some(0o066),
            uid: None,
            gid: None,
        });
        v.push(LinuxDevice {
            path: "/dev/tty".to_string(),
            typ: LinuxDeviceType::c,
            major: 5,
            minor: 0,
            file_mode: Some(0o066),
            uid: None,
            gid: None,
        });
        v.push(LinuxDevice {
            path: "/dev/urandom".to_string(),
            typ: LinuxDeviceType::c,
            major: 1,
            minor: 9,
            file_mode: Some(0o066),
            uid: None,
            gid: None,
        });
        v.push(LinuxDevice {
            path: "/dev/random".to_string(),
            typ: LinuxDeviceType::c,
            major: 1,
            minor: 8,
            file_mode: Some(0o066),
            uid: None,
            gid: None,
        });
        v
    };
}

lazy_static! {
    static ref NAMESPACES: HashMap<CloneFlags, &'static str> = {
        let mut result = HashMap::new();
        result.insert(CloneFlags::CLONE_NEWIPC, "ipc");
        result.insert(CloneFlags::CLONE_NEWUTS, "uts");
        result.insert(CloneFlags::CLONE_NEWNET, "net");
        result.insert(CloneFlags::CLONE_NEWPID, "pid");
        result.insert(CloneFlags::CLONE_NEWNS, "mnt");
        result.insert(CloneFlags::CLONE_NEWCGROUP, "cgroup");
        result.insert(CloneFlags::CLONE_NEWUSER, "user");
        result
    };
}
