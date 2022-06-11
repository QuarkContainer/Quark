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
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::super::auth::*;
use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::kernel::time::*;
use super::super::super::super::socket::unix::transport::unix::*;
use super::super::super::super::task::*;
use super::super::super::attr::*;
use super::super::super::dirent::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::host::hostinodeop::*;
use super::super::super::inode::*;
use super::super::super::mount::*;

pub trait SimpleFileTrait: Send + Sync {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        _dirent: &Dirent,
        _flags: FileFlags,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::ENXIO));
    }
}

pub struct SimpleFileNode {}

impl SimpleFileTrait for SimpleFileNode {}

pub struct SimpleFileInodeInternal<T: 'static + SimpleFileTrait> {
    pub fsType: u64,
    pub unstable: UnstableAttr,
    pub wouldBlock: bool,
    pub data: T,
}

pub struct SimpleFileInode<T: 'static + SimpleFileTrait>(QRwLock<SimpleFileInodeInternal<T>>);

impl<T: 'static + SimpleFileTrait> Deref for SimpleFileInode<T> {
    type Target = QRwLock<SimpleFileInodeInternal<T>>;

    fn deref(&self) -> &QRwLock<SimpleFileInodeInternal<T>> {
        &self.0
    }
}

impl<T: 'static + SimpleFileTrait> SimpleFileInode<T> {
    pub fn New(
        task: &Task,
        owner: &FileOwner,
        perms: &FilePermissions,
        typ: u64,
        wouldBlock: bool,
        data: T,
    ) -> Self {
        let unstable = WithCurrentTime(
            task,
            &UnstableAttr {
                Owner: *owner,
                Perms: *perms,
                ..Default::default()
            },
        );

        return Self::NewWithUnstable(&unstable, typ, wouldBlock, data);
    }

    pub fn NewWithUnstable(u: &UnstableAttr, typ: u64, wouldBlock: bool, data: T) -> Self {
        let internal = SimpleFileInodeInternal {
            fsType: typ,
            unstable: *u,
            wouldBlock: wouldBlock,
            data: data,
        };

        return Self(QRwLock::new(internal));
    }
}

impl<T: 'static + SimpleFileTrait> InodeOperations for SimpleFileInode<T> {
    fn as_any(&self) -> &Any {
        self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::SimpleFileInode;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::SpecialFile;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::SimpleFileInode;
    }

    fn WouldBlock(&self) -> bool {
        return self.read().wouldBlock;
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }

    fn Getxattr(&self, _dir: &Inode, _name: &str) -> Result<String> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Setxattr(&self, _dir: &mut Inode, _name: &str, _value: &str, _flags: u32) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Listxattr(&self, _dir: &Inode) -> Result<Vec<String>> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Lookup(&self, _task: &Task, _dir: &Inode, _name: &str) -> Result<Dirent> {
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
        _task: &Task,
        _dir: &mut Inode,
        _oldParent: &Inode,
        _oldname: &str,
        _newParent: &Inode,
        _newname: &str,
        _replacement: bool,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
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
        return self.read().data.GetFile(task, dir, dirent, flags);
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn IsVirtual(&self) -> bool {
        return false;
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let u = self.read().unstable;
        return Ok(u);
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, p: FilePermissions) -> bool {
        self.write().unstable.SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.write().unstable.SetOwner(task, owner);
        return Ok(());
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.write().unstable.SetTimestamps(task, ts);
        return Ok(());
    }

    fn AddLink(&self, _task: &Task) {
        self.write().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.write().unstable.Links -= 1;
    }

    fn Sync(&self) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        if self.read().fsType == 0 {
            return Err(Error::SysError(SysErr::ENOSYS));
        }

        return Ok(FsInfo {
            Type: self.read().fsType,
            ..Default::default()
        });
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

#[derive(Clone, Default, Debug)]
pub struct InodeSimpleAttributesInternal {
    pub fsType: u64,
    pub unstable: UnstableAttr,
}

impl InodeSimpleAttributesInternal {
    pub fn New(task: &Task, owner: &FileOwner, perms: &FilePermissions, typ: u64) -> Self {
        let unstable = WithCurrentTime(
            task,
            &UnstableAttr {
                Owner: *owner,
                Perms: *perms,
                ..Default::default()
            },
        );

        let internal = InodeSimpleAttributesInternal {
            fsType: typ,
            unstable: unstable,
        };

        return internal;
    }
}

pub struct InodeSimpleAttributes(pub QRwLock<InodeSimpleAttributesInternal>);

impl Default for InodeSimpleAttributes {
    fn default() -> Self {
        return Self(QRwLock::new(Default::default()));
    }
}

impl Deref for InodeSimpleAttributes {
    type Target = QRwLock<InodeSimpleAttributesInternal>;

    fn deref(&self) -> &QRwLock<InodeSimpleAttributesInternal> {
        &self.0
    }
}

impl InodeSimpleAttributes {
    pub fn New(task: &Task, owner: &FileOwner, perms: &FilePermissions, typ: u64) -> Self {
        let unstable = WithCurrentTime(
            task,
            &UnstableAttr {
                Owner: *owner,
                Perms: *perms,
                ..Default::default()
            },
        );

        return Self::NewWithUnstable(&unstable, typ);
    }

    fn NewWithUnstable(u: &UnstableAttr, typ: u64) -> Self {
        let internal = InodeSimpleAttributesInternal {
            fsType: typ,
            unstable: *u,
        };

        return Self(QRwLock::new(internal));
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let u = self.read().unstable;
        return Ok(u);
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, p: FilePermissions) -> bool {
        self.write().unstable.SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.write().unstable.SetOwner(task, owner);
        return Ok(());
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.write().unstable.SetTimestamps(task, ts);
        return Ok(());
    }

    fn AddLink(&self, _task: &Task) {
        self.write().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.write().unstable.Links -= 1;
    }
}
