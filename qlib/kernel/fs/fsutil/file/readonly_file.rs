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
use alloc::vec::Vec;
use core::any::Any;
use enum_dispatch::enum_dispatch;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::kernel::waiter::*;
use super::super::super::super::task::*;
use super::super::super::attr::*;
use super::super::super::dentry::*;
use super::super::super::dirent::*;
use super::super::super::file::*;
use super::super::super::host::hostinodeop::*;
use super::*;

use crate::qlib::kernel::fs::procfs::meminfo::MeminfoFileNode;
use crate::qlib::kernel::fs::procfs::net::NetTCPReadonlyFileNode;
use crate::qlib::kernel::fs::procfs::task::auxvec::AUXVecReadonlyFileNode;
use crate::qlib::kernel::fs::procfs::task::comm::CommReadonlyFileNode;
use crate::qlib::kernel::fs::procfs::task::exec_args::ExecArgReadonlyFileNode;
use crate::qlib::kernel::fs::procfs::uptime::UptimeFileNode;
use crate::qlib::kernel::fs::procfs::task::uid_pid_map::IdMapReadonlyFileNode;
use crate::qlib::kernel::fs::procfs::net::NetUnixReadonlyFileNode;
use crate::qlib::kernel::fs::procfs::net::NetUDPReadonlyFileNode;

pub fn NewSnapshotReadonlyFileOperations(
    data: Vec<u8>,
) -> ReadonlyFileOperations {
    let node = SnapshotReadonlyFileNode {
        data: Arc::new(data),
    };

    return ReadonlyFileOperations { node: node.into() };
}

pub struct SnapshotReadonlyFileNode {
    pub data: Arc<Vec<u8>>,
}

impl ReadonlyFileNodeTrait for SnapshotReadonlyFileNode {
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

        if offset as usize > self.data.len() {
            return Ok(0);
        }

        let n = task.CopyDataOutToIovs(&self.data[offset as usize..], dsts, true)?;

        return Ok(n as i64);
    }
}

#[enum_dispatch]
pub enum ReadonlyFileNode {
    SnapshotReadonlyFileNode(SnapshotReadonlyFileNode),
    MeminfoFileNode(MeminfoFileNode),
    NetTCPReadonlyFileNode(NetTCPReadonlyFileNode),
    NetUDPReadonlyFileNode(NetUDPReadonlyFileNode),
    NetUnixReadonlyFileNode(NetUnixReadonlyFileNode),
    AUXVecReadonlyFileNode(AUXVecReadonlyFileNode),
    CommReadonlyFileNode(CommReadonlyFileNode),
    ExecArgReadonlyFileNode(ExecArgReadonlyFileNode),
    IdMapReadonlyFileNode(IdMapReadonlyFileNode),
    UptimeFileNode(UptimeFileNode),
}

#[enum_dispatch(ReadonlyFileNode)]
pub trait ReadonlyFileNodeTrait: Send + Sync {
    fn ReadAt(
        &self,
        _task: &Task,
        _f: &File,
        _dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }
}

pub struct ReadonlyFileOperations {
    pub node: ReadonlyFileNode,
}

impl Waitable for ReadonlyFileOperations {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        return mask;
    }

    fn EventRegister(&self, _task: &Task, _e: &WaitEntry, _mask: EventMask) {}

    fn EventUnregister(&self, _task: &Task, _e: &WaitEntry) {}
}

impl SpliceOperations for ReadonlyFileOperations {}

impl FileOperations for ReadonlyFileOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::ReadonlyFileOperations;
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, None);
    }

    fn ReadDir(
        &self,
        _task: &Task,
        _f: &File,
        _offset: i64,
        _serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn ReadAt(
        &self,
        task: &Task,
        f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        blocking: bool,
    ) -> Result<i64> {
        return self.node.ReadAt(task, f, dsts, offset, blocking);
    }

    fn WriteAt(
        &self,
        _task: &Task,
        _f: &File,
        _srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0));
    }

    fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        _syncType: SyncType,
    ) -> Result<()> {
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
        _task: &Task,
        _d: &Dirent,
        _dirCtx: &mut DirCtx,
        _offset: i32,
    ) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for ReadonlyFileOperations {}
