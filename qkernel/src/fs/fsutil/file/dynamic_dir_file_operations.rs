// Copyright (c) 2021 QuarkSoft LLC
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

use super::super::super::file::*;
use super::super::super::attr::*;
use super::super::super::dentry::*;
use super::super::super::dirent::*;
use super::super::super::super::kernel::waiter::*;
use super::super::super::super::qlib::common::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::task::*;
use super::super::super::host::hostinodeop::*;
use super::*;

pub trait DynamicDirFileNode : Send + Sync {
    fn ReadDir(&self, _task: &Task, _f: &File, _offset: i64, _serializer: &mut DentrySerializer) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn IterateDir(&self, _task: &Task, _d: &Dirent, _dirCtx: &mut DirCtx, _offset: i32) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)))
    }
}

pub struct DynamicDirFileOperations<T: 'static + DynamicDirFileNode> {
    pub node: T,
}

impl <T: 'static + DynamicDirFileNode> Waitable for DynamicDirFileOperations <T> {}

impl <T: 'static + DynamicDirFileNode> SpliceOperations for DynamicDirFileOperations <T> {}

impl <T: 'static + DynamicDirFileNode> FileOperations for DynamicDirFileOperations <T> {
    fn as_any(&self) -> &Any {
        return self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::DynamicDirFileOperations
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

    fn IterateDir(&self, task: &Task,d: &Dirent, dirCtx: &mut DirCtx, offset: i32) -> (i32, Result<i64>) {
        return self.node.IterateDir(task, d, dirCtx, offset)
    }

    fn ReadDir(&self, task: &Task, f: &File, offset: i64, serializer: &mut DentrySerializer) -> Result<i64> {
        return self.node.ReadDir(task, f, offset, serializer)
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

impl <T: 'static + DynamicDirFileNode> SockOperations for DynamicDirFileOperations <T> {}