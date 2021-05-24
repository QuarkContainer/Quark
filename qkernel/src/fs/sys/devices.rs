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

use alloc::sync::Arc;
use spin::Mutex;
use alloc::vec::Vec;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::auth::*;
use super::super::super::kernel::kernel::*;
use super::super::super::task::*;
use super::super::fsutil::file::readonly_file::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::dirent::*;
use super::super::mount::*;
use super::super::inode::*;
use super::sys::*;

pub fn NewPossible(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let v = NewPossibleSimpleFileInode(task, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC);
    return NewFile(&Arc::new(v), msrc)

}

pub fn NewPossibleSimpleFileInode(task: &Task,
                                 owner: &FileOwner,
                                 perms: &FilePermissions,
                                 typ: u64)
                                 -> SimpleFileInode<PossibleData> {
    let fs = PossibleData{};
    return SimpleFileInode::New(task, owner, perms, typ, false, fs)
}

pub struct PossibleData {
}

impl PossibleData {
    pub fn GenSnapshot(&self, _task: &Task) -> Vec<u8> {
        let kernel = GetKernel();
        let maxCore = kernel.applicationCores - 1;

        let ret = format!("0-{}\n", maxCore);
        return ret.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for PossibleData {
    fn GetFile(&self, task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}

pub fn NewCPU(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let mut m = BTreeMap::new();

    m.insert("online".to_string(), NewPossible(task, msrc));
    m.insert("possible".to_string(), NewPossible(task, msrc));
    m.insert("present".to_string(), NewPossible(task, msrc));

    let kernel = GetKernel();
    let cores = kernel.applicationCores;
    for i in 0..cores {
        let name = format!("cpu{}", i);
        m.insert(name, NewDir(task, msrc, BTreeMap::new()));
    }

    return NewDir(task, msrc, m)
}

pub fn NewSystemDir(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let mut m = BTreeMap::new();

    m.insert("cpu".to_string(), NewCPU(task, msrc));
    //m.insert("node".to_string(), NewCPU(task, msrc));
    return NewDir(task, msrc, m)
}

pub fn NewDevicesDir(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let mut m = BTreeMap::new();

    m.insert("system".to_string(), NewSystemDir(task, msrc));
    return NewDir(task, msrc, m)
}