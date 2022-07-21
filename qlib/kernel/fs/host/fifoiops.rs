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

use alloc::string::String;
use alloc::vec::Vec;
use core::any::Any;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::time::*;
pub use super::super::super::memmgr::vma::MMappable;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::kernel::pipe::node::*;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::inode::*;
use super::hostinodeop::*;
use super::*;

#[derive(Clone)]
pub struct FifoIops {
    pub fifoiops: PipeIops,
    pub hosttiops: HostInodeOp,
}

impl InodeOperations for FifoIops {
    fn as_any(&self) -> &Any {
        self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::FifoIops;
    }

    fn InodeType(&self) -> InodeType {
        return self.hosttiops.InodeType();
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::Pipe;
    }

    fn WouldBlock(&self) -> bool {
        return self.fifoiops.WouldBlock();
    }

    fn Lookup(&self, _task: &Task, _parent: &Inode, _name: &str) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Create(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _flags: &FileFlags,
        _perm: &FilePermissions,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateDirectory(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _oldname: &str,
        _newname: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateHardLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _target: &Inode,
        _name: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateFifo(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Rename(
        &self,
        task: &Task,
        dir: &mut Inode,
        oldParent: &Inode,
        oldname: &str,
        newParent: &Inode,
        newname: &str,
        replacement: bool,
    ) -> Result<()> {
        return self.hosttiops.Rename(task, dir, oldParent, oldname, newParent, newname, replacement);
    }

    fn Bind(
        &self,
        _task: &Task,
        _dir: &Inode,
        _name: &str,
        _data: &BoundEndpoint,
        _perms: &FilePermissions,
    ) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return None;
    }

    fn GetFile(
        &self,
        task: &Task,
        dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        return self.fifoiops.GetFile(task, dir, dirent, flags);
    }

    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        return self.fifoiops.UnstableAttr(task)
    }

    //fn StableAttr(&self) -> &StableAttr;
    fn Getxattr(&self, dir: &Inode, name: &str, size: usize) -> Result<Vec<u8>> {
        return self.fifoiops.Getxattr(dir, name, size);
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        return self.fifoiops.Setxattr(dir, name, value, flags)
    }

    fn Listxattr(&self, dir: &Inode, size: usize) -> Result<Vec<String>> {
        return self.fifoiops.Listxattr(dir, size);
    }

    fn Removexattr(&self, dir: &Inode, name: &str) -> Result<()> {
        return self.fifoiops.Removexattr(dir, name)
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return self.hosttiops.Check(task, inode, reqPerms)
    }

    fn SetPermissions(&self, task: &Task, dir: &mut Inode, f: FilePermissions) -> bool {
        return self.hosttiops.SetPermissions(task, dir, f)
    }

    fn SetOwner(&self, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        return self.hosttiops.SetOwner(task, dir, owner);
    }

    fn SetTimestamps(&self, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return self.hosttiops.SetTimestamps(task, dir, ts);
    }

    fn Truncate(&self, task: &Task, dir: &mut Inode, size: i64) -> Result<()> {
        return self.fifoiops.Truncate(task, dir, size);
    }

    fn Allocate(&self, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()> {
        return self.fifoiops.Allocate(task, dir, offset, length)
    }

    fn ReadLink(&self, task: &Task, dir: &Inode) -> Result<String> {
        return self.fifoiops.ReadLink(task, dir);
    }

    fn GetLink(&self, task: &Task, dir: &Inode) -> Result<Dirent> {
        return self.fifoiops.GetLink(task, dir);
    }

    fn AddLink(&self, _task: &Task) {
        //return Err(Error::None)
    }

    fn DropLink(&self, _task: &Task) {
        //return Err(Error::None)
    }

    fn IsVirtual(&self) -> bool {
        false
    }

    fn Sync(&self) -> Result<()> {
        return self.fifoiops.Sync();
    }

    fn StatFS(&self, task: &Task) -> Result<FsInfo> {
        return self.hosttiops.StatFS(task)
    }

    fn Mappable(&self) -> Result<MMappable> {
        return self.fifoiops.Mappable();
    }
}