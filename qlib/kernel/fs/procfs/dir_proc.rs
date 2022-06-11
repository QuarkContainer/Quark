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
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::ramfs::dir::*;

pub trait DirDataNode: Send + Sync {
    fn Lookup(&self, d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent>;
    fn GetFile(
        &self,
        d: &Dir,
        task: &Task,
        dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File>;

    fn Check(&self, d: &Dir, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return d.Check(task, inode, reqPerms);
    }
}

#[derive(Clone)]
pub struct DirNode<T: 'static + DirDataNode> {
    pub dir: Dir,
    pub data: T,
}

impl<T: 'static + DirDataNode> InodeOperations for DirNode<T> {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::DirNode;
    }

    fn InodeType(&self) -> InodeType {
        return self.dir.InodeType();
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::DirNode;
    }

    fn WouldBlock(&self) -> bool {
        return false;
    }

    fn Lookup(&self, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return self.data.Lookup(&self.dir, task, dir, name);
    }

    fn Create(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        flags: &FileFlags,
        perms: &FilePermissions,
    ) -> Result<File> {
        return self.dir.Create(task, dir, name, flags, perms);
    }

    fn CreateDirectory(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perms: &FilePermissions,
    ) -> Result<()> {
        return self.dir.CreateDirectory(task, dir, name, perms);
    }

    fn CreateLink(&self, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()> {
        return self.dir.CreateLink(task, dir, oldname, newname);
    }

    fn CreateHardLink(
        &self,
        task: &Task,
        dir: &mut Inode,
        target: &Inode,
        name: &str,
    ) -> Result<()> {
        return self.dir.CreateHardLink(task, dir, target, name);
    }

    fn CreateFifo(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perms: &FilePermissions,
    ) -> Result<()> {
        return self.dir.CreateFifo(task, dir, name, perms);
    }

    //fn RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<QMutex<Dirent>>) -> Result<()> ;
    fn Remove(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.dir.Remove(task, dir, name);
    }

    fn RemoveDirectory(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.dir.RemoveDirectory(task, dir, name);
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
        return self.dir.Rename(
            task,
            dir,
            oldParent,
            oldname,
            newParent,
            newname,
            replacement,
        );
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

    fn GetFile(&self, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        return self.data.GetFile(&self.dir, task, dir, dirent, flags);
    }

    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        return self.dir.UnstableAttr(task);
    }

    fn Getxattr(&self, dir: &Inode, name: &str) -> Result<String> {
        return self.dir.Getxattr(dir, name);
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &str, flags: u32) -> Result<()> {
        return self.dir.Setxattr(dir, name, value, flags);
    }

    fn Listxattr(&self, dir: &Inode) -> Result<Vec<String>> {
        return self.dir.Listxattr(dir);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return self.data.Check(&self.dir, task, inode, reqPerms);
    }

    fn SetPermissions(&self, task: &Task, dir: &mut Inode, p: FilePermissions) -> bool {
        return self.dir.SetPermissions(task, dir, p);
    }

    fn SetOwner(&self, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        return self.dir.SetOwner(task, dir, owner);
    }

    fn SetTimestamps(&self, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return self.dir.SetTimestamps(task, dir, ts);
    }

    fn Truncate(&self, task: &Task, dir: &mut Inode, size: i64) -> Result<()> {
        return self.dir.Truncate(task, dir, size);
    }

    fn Allocate(&self, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()> {
        return self.dir.Allocate(task, dir, offset, length);
    }

    fn ReadLink(&self, task: &Task, dir: &Inode) -> Result<String> {
        return self.dir.ReadLink(task, dir);
    }

    fn GetLink(&self, task: &Task, dir: &Inode) -> Result<Dirent> {
        return self.dir.GetLink(task, dir);
    }

    fn AddLink(&self, task: &Task) {
        self.dir.AddLink(task);
    }

    fn DropLink(&self, task: &Task) {
        self.dir.DropLink(task);
    }

    fn IsVirtual(&self) -> bool {
        return self.dir.IsVirtual();
    }

    fn Sync(&self) -> Result<()> {
        return self.dir.Sync();
    }

    fn StatFS(&self, task: &Task) -> Result<FsInfo> {
        return self.dir.StatFS(task);
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return self.dir.Mappable();
    }
}
