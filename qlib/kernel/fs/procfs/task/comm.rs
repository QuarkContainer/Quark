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
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::super::auth::*;
use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::task::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::super::attr::*;
use super::super::super::dirent::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::fsutil::file::readonly_file::*;
use super::super::super::fsutil::inode::simple_file_inode::*;
use super::super::super::inode::*;
use super::super::super::mount::*;
use super::super::inode::*;

pub fn NewComm(task: &Task, thread: &Thread, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewCommSimpleFileInode(
        task,
        thread,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o400)),
        FSMagic::PROC_SUPER_MAGIC,
    );
    return NewProcInode(
        &Arc::new(v),
        msrc,
        InodeType::SpecialFile,
        Some(thread.clone()),
    );
}

pub fn NewCommSimpleFileInode(
    task: &Task,
    thread: &Thread,
    owner: &FileOwner,
    perms: &FilePermissions,
    typ: u64,
) -> SimpleFileInode<CommSimpleFileTrait> {
    return SimpleFileInode::New(
        task,
        owner,
        perms,
        typ,
        false,
        CommSimpleFileTrait {
            thread: thread.clone(),
        },
    );
}

pub struct CommSimpleFileTrait {
    pub thread: Thread,
}

impl SimpleFileTrait for CommSimpleFileTrait {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = NewCommReadonlyFileOperations(&self.thread);
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}

pub fn NewCommReadonlyFileOperations(
    thread: &Thread,
) -> ReadonlyFileOperations<CommReadonlyFileNode> {
    return ReadonlyFileOperations {
        node: CommReadonlyFileNode {
            thread: thread.clone(),
        },
    };
}

pub struct CommReadonlyFileNode {
    pub thread: Thread,
}

impl ReadonlyFileNode for CommReadonlyFileNode {
    fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let buf = self.thread.lock().name.to_string() + "\n";
        info!("CommReadonlyFileNode buf is {}", &buf);
        if offset as usize > buf.len() {
            return Ok(0);
        }

        let n = task.CopyDataOutToIovs(&buf.as_bytes()[offset as usize..], dsts, true)?;

        return Ok(n as i64);
    }
}
