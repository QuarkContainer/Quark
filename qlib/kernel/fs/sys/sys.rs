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
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::auth::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::task::*;
use super::super::attr::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::ramfs::dir::*;
use super::devices::*;

pub fn NewFile(
    iops: Iops,
    msrc: &Arc<QMutex<MountSource>>,
) -> Inode {
    let deviceId = SYS_DEVICE.lock().id.DeviceID();
    let inodeId = SYS_DEVICE.lock().NextIno();

    let sattr = StableAttr {
        Type: InodeType::SpecialFile,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: 4096,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    return Inode::New(iops, msrc, &sattr);
}

pub fn NewDir(
    task: &Task,
    msrc: &Arc<QMutex<MountSource>>,
    contents: BTreeMap<String, Inode>,
) -> Inode {
    let d = Dir::New(
        task,
        contents,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o0555)),
    );

    let deviceId = SYS_DEVICE.lock().id.DeviceID();
    let inodeId = SYS_DEVICE.lock().NextIno();

    let sattr = StableAttr {
        Type: InodeType::SpecialDirectory,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: 4096,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    return Inode::New(d.into(), msrc, &sattr);
}

pub fn NewSys(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let mut content = BTreeMap::new();
    content.insert("block".to_string(), NewDir(task, msrc, BTreeMap::new()));
    content.insert("bus".to_string(), NewDir(task, msrc, BTreeMap::new()));

    let mut classContent = BTreeMap::new();
    classContent.insert(
        "power_supply".to_string(),
        NewDir(task, msrc, BTreeMap::new()),
    );
    content.insert("class".to_string(), NewDir(task, msrc, classContent));

    content.insert("dev".to_string(), NewDir(task, msrc, BTreeMap::new()));
    content.insert("devices".to_string(), NewDevicesDir(task, msrc));
    content.insert("firmware".to_string(), NewDir(task, msrc, BTreeMap::new()));
    content.insert("fs".to_string(), NewDir(task, msrc, BTreeMap::new()));
    content.insert("kernel".to_string(), NewDir(task, msrc, BTreeMap::new()));
    content.insert("module".to_string(), NewDir(task, msrc, BTreeMap::new()));
    content.insert("power".to_string(), NewDir(task, msrc, BTreeMap::new()));

    return NewDir(task, msrc, content);
}
