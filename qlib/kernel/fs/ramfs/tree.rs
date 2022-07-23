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
use alloc::vec::Vec;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::path::*;
use super::super::super::task::*;
use super::super::super::uid::NewUID;
use super::super::attr::*;
use super::super::inode::*;
use super::super::mount::*;
use super::dir::*;

fn emptyDir(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let dir = Dir::New(
        task,
        BTreeMap::new(),
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o777)),
    );
    let deviceId = PSEUDO_DEVICE.lock().id.DeviceID();
    let inodeId = PSEUDO_DEVICE.lock().NextIno();
    let stableAttr = StableAttr {
        Type: InodeType::Directory,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: Arc::new(dir),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn makeSubdir(task: &Task, msrc: &Arc<QMutex<MountSource>>, root: &Dir, subDir: &str) {
    let mut root = root.clone();

    let arr: Vec<&str> = subDir.split('/').collect();
    for c in arr {
        if c.len() == 0 {
            continue;
        }

        let name = c.to_string();

        let child = match root.FindChild(&name) {
            None => {
                let mut child = emptyDir(task, msrc);
                root.AddChild(task, &name, &mut child);
                child
            }
            Some(c) => c,
        };

        root = child
            .lock()
            .InodeOp
            .as_any()
            .downcast_ref::<Dir>()
            .unwrap()
            .clone();
    }
}

pub fn MakeDirectoryTree(
    task: &Task,
    msrc: &Arc<QMutex<MountSource>>,
    subDirs: &Vec<String>,
) -> Result<Inode> {
    let root = emptyDir(task, msrc);
    let dir = root
        .lock()
        .InodeOp
        .as_any()
        .downcast_ref::<Dir>()
        .unwrap()
        .clone();

    for subdir in subDirs {
        let subdir = subdir.clone();
        if Clean(&subdir) != subdir {
            return Err(Error::Common(
                "cannot add subdir at an unclean path".to_string(),
            ));
        }

        if subdir == "" || subdir == "/" {
            return Err(Error::Common("cannot add subdir at".to_string()));
        }

        makeSubdir(task, msrc, &dir, &subdir);
    }

    return Ok(root);
}
