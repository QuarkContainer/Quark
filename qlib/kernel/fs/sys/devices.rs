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

use crate::qlib::kernel::fs::procfs::inode::NewStaticProcInodeWithHostFile;
use crate::qlib::kernel::fs::procfs::inode::NewStaticProcInodeWithString;
use crate::qlib::mutex::*;
use alloc::borrow::ToOwned;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::kernel::*;
use super::super::super::task::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::fsutil::file::readonly_file::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::inode::*;
use super::super::mount::*;
use super::sys::*;

pub fn NewPossible(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewPossibleSimpleFileInode(
        task,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o444)),
        FSMagic::PROC_SUPER_MAGIC,
    );
    return NewFile(v.into(), msrc);
}

pub fn NewPossibleSimpleFileInode(
    task: &Task,
    owner: &FileOwner,
    perms: &FilePermissions,
    typ: u64,
) -> SimpleFileInode {
    let fs: PossibleData = PossibleData {};
    return SimpleFileInode::New(task, owner, perms, typ, false, fs.into());
}

pub struct PossibleData {}

impl PossibleData {
    pub fn GenSnapshot(&self, _task: &Task) -> Vec<u8> {
        let kernel = GetKernel();
        let maxCore = kernel.applicationCores - 1;

        let ret = format!("0-{}\n", maxCore);
        return ret.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for PossibleData {
    fn GetFile(
        &self,
        task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops.into());
        return Ok(file);
    }
}

pub fn NewCpuTopo(task: &Task, msrc: &Arc<QMutex<MountSource>>, cpuId: usize) -> Inode {
    let mut m = BTreeMap::new();

    let folderName = format!("/sys/devices/system/cpu/cpu{}/topology/", cpuId);

    m.insert(
        "core_id".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "core_id")).unwrap(),
    );

    m.insert(
        "cluster_id".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "cluster_id")).unwrap(),
    );

    m.insert(
        "core_cpus".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "core_cpus")).unwrap(),
    );
    m.insert(
        "cluster_cpus".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "cluster_cpus")).unwrap(),
    );
    m.insert(
        "thread_siblings".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "thread_siblings"))
            .unwrap(),
    );

    m.insert(
        "die_cpus".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "die_cpus")).unwrap(),
    );
    m.insert(
        "package_cpus".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "package_cpus")).unwrap(),
    );
    m.insert(
        "core_siblings".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "core_siblings"))
            .unwrap(),
    );

    m.insert(
        "physical_package_id".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "physical_package_id"))
            .unwrap(),
    );
    m.insert(
        "die_id".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "die_id")).unwrap(),
    );

    m.insert(
        "die_cpus_list".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "die_cpus_list"))
            .unwrap(),
    );
    m.insert(
        "package_cpus_list".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "package_cpus_list"))
            .unwrap(),
    );
    m.insert(
        "core_siblings_list".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "core_siblings_list"))
            .unwrap(),
    );

    m.insert(
        "thread_siblings_list".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "thread_siblings_list"))
            .unwrap(),
    );
    m.insert(
        "cluster_cpus_list".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "cluster_cpus_list"))
            .unwrap(),
    );
    m.insert(
        "core_cpus_list".to_string(),
        NewStaticProcInodeWithHostFile(task, msrc, &(folderName.clone() + "core_cpus_list"))
            .unwrap(),
    );

    return NewDir(task, msrc, m);
}

pub fn NewCPU(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let mut m = BTreeMap::new();

    m.insert("online".to_string(), NewPossible(task, msrc));
    m.insert("possible".to_string(), NewPossible(task, msrc));
    m.insert("present".to_string(), NewPossible(task, msrc));

    let kernel = GetKernel();
    let cores = kernel.applicationCores;
    for i in 0..cores {
        let name = format!("cpu{}", i);

        let mut cpuMap = BTreeMap::new();
        cpuMap.insert("topology".to_owned(), NewCpuTopo(task, msrc, i));
        cpuMap.insert(
            "online".to_string(),
            NewStaticProcInodeWithString(task, msrc, &format!("1\n")),
        );

        m.insert(name, NewDir(task, msrc, cpuMap));
    }

    return NewDir(task, msrc, m);
}

pub fn NewSystemDir(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let mut m = BTreeMap::new();

    m.insert("cpu".to_string(), NewCPU(task, msrc));
    //m.insert("node".to_string(), NewCPU(task, msrc));
    return NewDir(task, msrc, m);
}

pub fn NewDevicesDir(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let mut m = BTreeMap::new();

    m.insert("system".to_string(), NewSystemDir(task, msrc));
    return NewDir(task, msrc, m);
}
