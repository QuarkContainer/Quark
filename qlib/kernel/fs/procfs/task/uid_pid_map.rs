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

pub fn NewIdMap(
    task: &Task,
    thread: &Thread,
    msrc: &Arc<QMutex<MountSource>>,
    gids: bool,
) -> Inode {
    let v = NewIdMapSimpleFileInode(
        task,
        thread,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o400)),
        FSMagic::PROC_SUPER_MAGIC,
        gids,
    );
    return NewProcInode(
        &Arc::new(v),
        msrc,
        InodeType::SpecialFile,
        Some(thread.clone()),
    );
}

pub fn NewIdMapSimpleFileInode(
    task: &Task,
    thread: &Thread,
    owner: &FileOwner,
    perms: &FilePermissions,
    typ: u64,
    gids: bool,
) -> SimpleFileInode<IdMapSimpleFileTrait> {
    return SimpleFileInode::New(
        task,
        owner,
        perms,
        typ,
        false,
        IdMapSimpleFileTrait {
            thread: thread.clone(),
            gids: gids,
        },
    );
}

pub struct IdMapSimpleFileTrait {
    pub thread: Thread,
    pub gids: bool,
}

impl SimpleFileTrait for IdMapSimpleFileTrait {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = NewIdMapReadonlyFileOperations(&self.thread, self.gids);
        let file = File::New(dirent, &flags, fops.into());
        return Ok(file);
    }
}

pub fn NewIdMapReadonlyFileOperations(
    thread: &Thread,
    gids: bool,
) -> ReadonlyFileOperations {
    return ReadonlyFileOperations {
        node: IdMapReadonlyFileNode {
            thread: thread.clone(),
            gids: gids,
        }.into(),
    };
}

pub struct IdMapReadonlyFileNode {
    pub thread: Thread,
    pub gids: bool,
}

//todo: shall we support Write?
impl ReadonlyFileNodeTrait for IdMapReadonlyFileNode {
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

        let userns = self.thread.UserNamespace();
        let entries = if self.gids {
            userns.GIDMap()
        } else {
            userns.UIDMap()
        };

        let mut buf = "".to_string();
        for e in &entries {
            buf += &format!("{} {} {}\n", e.FirstFromId, e.FirstToId, e.Len);
        }

        if offset as usize >= buf.len() {
            return Ok(0);
        }

        let n = task.CopyDataOutToIovs(&buf.as_bytes()[offset as usize..], dsts, true)?;

        return Ok(n as i64);
    }
}
