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
use alloc::string::String;
use alloc::string::ToString;
use core::any::Any;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::waiter::*;
use super::super::super::task::*;

use super::super::attr::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::fsutil::file::*;
use super::super::host::diriops::*;

pub struct HostDirFops {
    pub DirOp: HostDirOp,
    pub DirCursor: QMutex<String>,
    //pub Buf: HostFileBuf,
}

impl Waitable for HostDirFops {}

impl SpliceOperations for HostDirFops {}

impl FileOperations for HostDirFops {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::HostDirOp;
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        let mut dirCursor = self.DirCursor.lock();
        let mut cursor = "".to_string();
        let newOffset = SeekWithDirCursor(task, f, whence, current, offset, Some(&mut cursor))?;
        *dirCursor = cursor;
        return Ok(newOffset);
    }

    fn ReadDir(
        &self,
        task: &Task,
        file: &File,
        offset: i64,
        serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        let root = task.Root();
        let mut dirCursor = self.DirCursor.lock();

        let mut dirCtx = DirCtx {
            Serializer: serializer,
            DirCursor: (*dirCursor).to_string(),
        };

        let res = DirentReadDir(task, &file.Dirent, self, &root, &mut dirCtx, offset)?;
        *dirCursor = dirCtx.DirCursor;
        return Ok(res);
    }

    fn ReadAt(
        &self,
        _task: &Task,
        _f: &File,
        _dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn WriteAt(
        &self,
        _task: &Task,
        _f: &File,
        _srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn Append(&self, _task: &Task, _f: &File, _srcs: &[IoVec]) -> Result<(i64, i64)> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Ok(());
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY));
    }

    fn IterateDir(
        &self,
        task: &Task,
        _d: &Dirent,
        dirCtx: &mut DirCtx,
        offset: i32,
    ) -> (i32, Result<i64>) {
        return self.DirOp.lock().IterateDir(task, dirCtx, offset);
    }

    fn Mappable(&self) -> Result<HostIopsMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for HostDirFops {}

