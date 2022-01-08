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
use alloc::string::String;
use alloc::vec::Vec;
use core::any::Any;
use ::qlib::mutex::*;

use socket::unix::transport::unix::BoundEndpoint;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::auth::*;
use super::super::super::threadmgr::thread::*;
use super::super::super::kernel::time::*;
use super::super::super::task::*;
use super::super::host::hostinodeop::*;
use super::super::attr::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::dirent::*;
use super::super::mount::*;
use super::super::inode::*;
use super::super::ramfs::symlink::*;
use super::inode::*;

pub trait ReadLinkNode : Send + Sync {
    fn ReadLink(&self, link: &Symlink, task: &Task, _dir: &Inode) -> Result<String>;
    fn GetLink(&self, link: &Symlink, task: &Task, dir: &Inode) -> Result<Dirent>;
    fn GetFile(&self, link: &Symlink, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        return link.GetFile(task, dir, dirent, flags);
    }
}

pub struct SymlinkNode<T: 'static + ReadLinkNode> {
    pub link: Symlink,
    pub node: T,
}

impl <T: 'static + ReadLinkNode> SymlinkNode<T> {
    pub fn New(task: &Task, msrc: &Arc<QMutex<MountSource>>, node: T, thread: Option<Thread>) -> Inode {
        let link = Self {
            link: Symlink::New(task, &ROOT_OWNER, ""),
            node: node,
        };

        return NewProcInode(&Arc::new(link), msrc, InodeType::Symlink, thread)
    }
}

impl <T: 'static + ReadLinkNode> InodeOperations for SymlinkNode<T> {
    fn as_any(&self) -> &Any {
        return self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::SymlinkNode;
    }

    fn InodeType(&self) -> InodeType {
        return self.link.InodeType();
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::SymlinkNode;
    }

    fn WouldBlock(&self) -> bool {
        return false;
    }

    fn Lookup(&self, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return self.link.Lookup(task, dir, name);
    }

    fn Create(&self, task: &Task, dir: &mut Inode, name: &str, flags: &FileFlags, perms: &FilePermissions) -> Result<File> {
        return self.link.Create(task, dir, name, flags, perms);
    }

    fn CreateDirectory(&self, task: &Task, dir: &mut Inode, name: &str, perms: &FilePermissions) -> Result<()> {
        return self.link.CreateDirectory(task, dir, name, perms);
    }

    fn CreateLink(&self, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()> {
        return self.link.CreateLink(task, dir, oldname, newname);
    }

    fn CreateHardLink(&self, task: &Task, dir: &mut Inode, target: &Inode, name: &str) -> Result<()> {
        return self.link.CreateHardLink(task, dir, target, name);
    }

    fn CreateFifo(&self, task: &Task, dir: &mut Inode, name: &str, perms: &FilePermissions) -> Result<()> {
        return self.link.CreateFifo(task, dir, name, perms);
    }

    //fn RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<QMutex<Dirent>>) -> Result<()> ;
    fn Remove(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.link.Remove(task, dir, name);
    }

    fn RemoveDirectory(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.link.RemoveDirectory(task, dir, name);
    }

    fn Rename(&self, task: &Task, dir: &mut Inode, oldParent: &Inode, oldname: &str, newParent: &Inode, newname: &str, replacement: bool) -> Result<()> {
        return self.link.Rename(task, dir, oldParent, oldname, newParent, newname, replacement);
    }

    fn Bind(&self, _task: &Task, _dir: &Inode, _name: &str, _data: &BoundEndpoint, _perms: &FilePermissions) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return None
    }

    fn GetFile(&self, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        return self.node.GetFile(&self.link, task, dir, dirent, flags);
        //return self.link.GetFile(task, dir, dirent, flags);
    }

    fn UnstableAttr(&self, task: &Task, dir: &Inode) -> Result<UnstableAttr> {
        return self.link.UnstableAttr(task, dir);
    }

    fn Getxattr(&self, dir: &Inode, name: &str) -> Result<String> {
        return self.link.Getxattr(dir, name);
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &str) -> Result<()> {
        return self.link.Setxattr(dir, name, value);
    }

    fn Listxattr(&self, dir: &Inode) -> Result<Vec<String>> {
        return self.link.Listxattr(dir);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return self.link.Check(task, inode, reqPerms);
    }

    fn SetPermissions(&self, task: &Task, dir: &mut Inode, p: FilePermissions) -> bool {
        return self.link.SetPermissions(task, dir, p);
    }

    fn SetOwner(&self, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        return self.link.SetOwner(task, dir, owner);
    }

    fn SetTimestamps(&self, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return self.link.SetTimestamps(task, dir, ts);
    }

    fn Truncate(&self, task: &Task, dir: &mut Inode, size: i64) -> Result<()> {
        return self.link.Truncate(task, dir, size);
    }

    fn Allocate(&self, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()> {
        return self.link.Allocate(task, dir, offset, length);
    }

    fn ReadLink(&self, task: &Task, dir: &Inode) -> Result<String> {
        return self.node.ReadLink(&self.link, task, dir)
    }

    fn GetLink(&self, task: &Task, dir: &Inode) -> Result<Dirent> {
        return self.node.GetLink(&self.link, task, dir);
    }

    fn AddLink(&self, task: &Task) {
        self.link.AddLink(task);
    }

    fn DropLink(&self, task: &Task) {
        self.link.DropLink(task);
    }

    fn IsVirtual(&self) -> bool {
        return self.link.IsVirtual();
    }

    fn Sync(&self) -> Result<()> {
        return self.link.Sync();
    }

    fn StatFS(&self, task: &Task) -> Result<FsInfo> {
        return self.link.StatFS(task);
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return self.link.Mappable();
    }
}