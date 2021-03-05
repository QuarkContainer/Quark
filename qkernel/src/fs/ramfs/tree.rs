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

use alloc::string::String;
use alloc::string::ToString;
use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::Mutex;

use super::dir::*;
use super::super::mount::*;
use super::super::inode::*;
use super::super::super::qlib::path::*;
use super::super::attr::*;
use super::super::super::task::*;
use super::super::super::qlib::auth::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::device::*;
use super::super::super::qlib::common::*;

fn emptyDir(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let dir = Dir::New(task, BTreeMap::new(), &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o777)));
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
        InodeOp: Arc::new(dir),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(Mutex::new(inodeInternal)))
}

fn makeSubdir(task: &Task, msrc: &Arc<Mutex<MountSource>>, root: &Dir, subDir: &str) {
    let mut root = root.clone();

    let arr: Vec<&str> = subDir.split('/').collect();
    for c in arr {
        if c.len() == 0 {
            continue
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

        root = child.lock().InodeOp.as_any().downcast_ref::<Dir>().unwrap().clone();
    }
}

pub fn MakeDirectoryTree(task: &Task, msrc: &Arc<Mutex<MountSource>>, subDirs: &Vec<String>) -> Result<Inode> {
    let root = emptyDir(task, msrc);
    let dir = root.lock().InodeOp.as_any().downcast_ref::<Dir>().unwrap().clone();

    for subdir in subDirs {
        let subdir = subdir.clone();
        if Clean(&subdir) != subdir {
            return Err(Error::Common("cannot add subdir at an unclean path".to_string()));
        }

        if subdir == "" || subdir == "/" {
            return Err(Error::Common("cannot add subdir at".to_string()));
        }

        makeSubdir(task, msrc, &dir, &subdir);
    }

    return Ok(root)
}

#[cfg(test1)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_abs_path() {
        let subdirs = vec!["/tmp".to_string(),
                           "/tmp/a".to_string(),
                           "/tmp/a/b".to_string(),
                           "/tmp/a/c/d".to_string(),
                           "/tmp/c".to_string(),
                           "/proc".to_string(),
                           "/dev/a/b".to_string(),
                           "/tmp".to_string(), ];

        let task = Task::default();
        let mount = MountSource::NewPseudoMountSource();
        let tree = MakeDirectoryTree(&task, &Arc::new(Mutex::new(mount)), &subdirs).unwrap();

        let mm = MountNs::New(&tree);
        let root = mm.root.clone();

        for p in &subdirs {
            let mut maxTraversals = 0;
            mm.FindInode(&task, &root, None, p, &mut maxTraversals).unwrap();
        }
    }

    #[test]
    fn test_rel_path() {
        let subdirs = vec!["tmp".to_string(),
                           "tmp/a/b".to_string(),
                           "tmp/a/c/d".to_string(),
                           "tmp/c".to_string(),
                           "proc".to_string(),
                           "dev/a/b".to_string(),
                           "tmp".to_string(), ];

        let task = Task::default();
        let mount = MountSource::NewPseudoMountSource();
        let tree = MakeDirectoryTree(&task, &Arc::new(Mutex::new(mount)), &subdirs).unwrap();

        let mm = MountNs::New(&tree);
        let root = mm.root.clone();

        for p in &subdirs {
            let mut maxTraversals = 0;
            mm.FindInode(&task, &root, None, p, &mut maxTraversals).unwrap();
        }
    }
}