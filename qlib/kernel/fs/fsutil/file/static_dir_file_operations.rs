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

use core::any::Any;
use alloc::string::String;

use super::super::super::file::*;
use super::super::super::attr::*;
use super::super::super::dentry::*;
use super::super::super::dirent::*;
use super::super::super::super::kernel::waiter::*;
use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::task::*;
use super::super::super::host::hostinodeop::*;
use super::*;

pub struct StaticDirFileOperations {
    pub dentryMap: DentMap,
    pub dirCursor: String,
}

impl Waitable for StaticDirFileOperations {
    fn Readiness(&self, _task: &Task,_mask: EventMask) -> EventMask {
        panic!("StaticFileOps doesn't support Waitable::Readiness");
    }

    fn EventRegister(&self, _task: &Task,_e: &WaitEntry, _mask: EventMask) {
        panic!("StaticFileOps doesn't support Waitable::EventRegister");
    }

    fn EventUnregister(&self, _task: &Task,_e: &WaitEntry) {
        panic!("StaticFileOps doesn't support Waitable::EventUnregister");
    }
}

impl SpliceOperations for StaticDirFileOperations {}

impl FileOperations for StaticDirFileOperations {
    fn as_any(&self) -> &Any {
        return self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::StaticDirFileOperations
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, None)
    }

    fn ReadAt(&self, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR))
    }

    fn WriteAt(&self, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR))
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0))
    }

    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Ok(())
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(())
    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY))
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn IterateDir(&self, task: &Task, _d: &Dirent, dirCtx: &mut DirCtx, offset: i32) -> (i32, Result<i64>) {
        let n = match dirCtx.ReadDir(task, &self.dentryMap) {
            Err(e) => return (offset, Err(e)),
            Ok(n) => n,
        };

        return (offset + n as i32, Ok(n as i64))
    }

    fn ReadDir(&self, task: &Task, f: &File, _offset: i64, serializer: &mut DentrySerializer) -> Result<i64> {
        let kernel = task.Thread().lock().k.clone();
        let root = kernel.RootDir();

        let mut dirCtx = DirCtx {
            Serializer: serializer,
            DirCursor: self.dirCursor.to_string(),
        };

        return DirentReadDir(task, &f.Dirent, self, &root, &mut dirCtx, f.Offset(task)?)
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

impl SockOperations for StaticDirFileOperations {}