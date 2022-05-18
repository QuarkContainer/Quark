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
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::slice;

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

pub fn NewAUXVec(task: &Task, thread: &Thread, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewAUXVecSimpleFileInode(
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

pub fn NewAUXVecSimpleFileInode(
    task: &Task,
    thread: &Thread,
    owner: &FileOwner,
    perms: &FilePermissions,
    typ: u64,
) -> SimpleFileInode<AUXVecSimpleFileTrait> {
    return SimpleFileInode::New(
        task,
        owner,
        perms,
        typ,
        false,
        AUXVecSimpleFileTrait {
            thread: thread.clone(),
        },
    );
}

pub struct AUXVecSimpleFileTrait {
    pub thread: Thread,
}

impl SimpleFileTrait for AUXVecSimpleFileTrait {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = NewAUXVecReadonlyFileOperations(&self.thread);
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}

pub fn NewAUXVecReadonlyFileOperations(
    thread: &Thread,
) -> ReadonlyFileOperations<AUXVecReadonlyFileNode> {
    return ReadonlyFileOperations {
        node: AUXVecReadonlyFileNode {
            thread: thread.clone(),
        },
    };
}

pub struct AUXVecReadonlyFileNode {
    pub thread: Thread,
}

impl ReadonlyFileNode for AUXVecReadonlyFileNode {
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

        let mm = self.thread.lock().memoryMgr.clone();
        let metadata = mm.metadata.lock();
        let auxvlen = metadata.auxv.len();

        // Space for buffer with AT_NULL (0) terminator at the end.
        let size = (auxvlen + 1) * 16 - 16;
        if offset >= size as i64 {
            return Ok(0);
        }

        let mut buf: Vec<u64> = Vec::with_capacity(auxvlen + 1);
        for i in 1..auxvlen {
            let e = &metadata.auxv[i];
            buf.push(e.Key as u64);
            buf.push(e.Val);
        }

        buf.push(0);
        buf.push(0);

        let ptr = &buf[0] as *const _ as u64 as *const u8;
        assert!(buf.len() * 8 >= size);
        let slice = unsafe { slice::from_raw_parts(ptr, size) };

        let n = task.CopyDataOutToIovs(slice, dsts, true)?;

        return Ok(n as i64);
    }
}
