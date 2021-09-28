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
use ::qlib::mutex::*;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::auth::*;
use super::super::super::task::*;
use super::super::fsutil::file::readonly_file::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::attr::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::dirent::*;
use super::super::mount::*;
use super::super::inode::*;
use super::super::filesystems::*;
use super::inode::*;

pub fn NewFileSystem(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewFileSystemSimpleFileInode(task, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC);
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, None)

}

pub fn NewFileSystemSimpleFileInode(task: &Task,
                              owner: &FileOwner,
                              perms: &FilePermissions,
                              typ: u64)
                              -> SimpleFileInode<FileSystemData> {
    let fs = FileSystemData{};
    return SimpleFileInode::New(task, owner, perms, typ, false, fs)
}

pub struct FileSystemData {
}

impl FileSystemData {
    pub fn GenSnapshot(&self, _task: &Task) -> Vec<u8> {
        let mut ret = "".to_string();
        for sys in GetFilesystems() {
            if !sys.lock().AllowUserList() {
                continue;
            }

            let nodev = if sys.lock().Flags() & FILESYSTEM_REQUIRES_DEV != 0 {
                ""
            } else {
                "nodev"
            };

            ret += &format!("{}\t{}\n", nodev, sys.lock().Name());
        }

        return ret.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for FileSystemData {
    fn GetFile(&self, task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}