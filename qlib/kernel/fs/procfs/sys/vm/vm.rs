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
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::super::super::auth::*;
use super::super::super::super::super::super::common::*;
use super::super::super::super::super::super::linux_def::*;
use super::super::super::super::super::task::*;
use super::super::super::super::attr::*;
use super::super::super::super::dirent::*;
use super::super::super::super::file::*;
use super::super::super::super::flags::*;
use super::super::super::super::inode::*;
use super::super::super::super::mount::*;
use super::super::super::super::ramfs::dir::*;
use super::super::super::dir_proc::*;
use super::super::super::inode::*;
use super::mmap_min_addr::*;
use super::overcommit::*;

// ProcSysDirNode represents a /proc/sys directory.
#[derive(Clone)]
pub struct ProcSysVMDirNode {}

impl DirDataNodeTrait for ProcSysVMDirNode {
    fn Lookup(&self, d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return d.Lookup(task, dir, name);
    }

    fn GetFile(
        &self,
        d: &Dir,
        task: &Task,
        dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        return d.GetFile(task, dir, dirent, flags);
    }
}

pub fn NewVm(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let mut contents = BTreeMap::new();
    contents.insert("mmap_min_addr".to_string(), NewMinAddrData(task, msrc));
    contents.insert("overcommit_memory".to_string(), NewOvercommit(task, msrc));

    let taskDir = DirNode {
        dir: Dir::New(
            task,
            contents,
            &ROOT_OWNER,
            &FilePermissions::FromMode(FileMode(0o0555)),
        ),
        data: ProcSysVMDirNode {}.into(),
    };

    return NewProcInode(taskDir.into(), msrc, InodeType::SpecialDirectory, None);
}
