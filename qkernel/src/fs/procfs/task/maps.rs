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

use super::super::super::super::qlib::common::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::qlib::auth::*;
use super::super::super::super::memmgr::mm::*;
use super::super::super::fsutil::file::readonly_file::*;
use super::super::super::fsutil::inode::simple_file_inode::*;
use super::super::super::super::task::*;
use super::super::super::attr::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::dirent::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::inode::*;

pub fn NewMaps(task: &Task, thread: &Thread, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let v = NewMapsSimpleFileInode(task, thread, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC);
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, Some(thread.clone()))

}

pub fn NewMapsSimpleFileInode(task: &Task,
                              thread: &Thread,
                            owner: &FileOwner,
                            perms: &FilePermissions,
                            typ: u64)
                            -> SimpleFileInode<MapsData> {
    let io = MapsData{mm: thread.lock().memoryMgr.clone()};
    return SimpleFileInode::New(task, owner, perms, typ, false, io)
}

pub struct MapsData {
    mm: MemoryManager,
}

impl MapsData {
    pub fn GenSnapshot(&self, task: &Task) -> Vec<u8> {
        return self.mm.GenMapsSnapshot(task)
    }
}

impl SimpleFileTrait for MapsData {
    fn GetFile(&self, task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}