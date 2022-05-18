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

pub mod iopsutil;
pub mod simple_file_inode;

pub use self::simple_file_inode::*;

use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::waiter::qlock::*;
use super::super::super::task::*;
use super::super::super::uid::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::inode::*;
use super::file::*;

pub struct InodeSimpleExtendedAttributesInternal {
    pub xattrs: BTreeMap<String, String>,
}

impl Default for InodeSimpleExtendedAttributesInternal {
    fn default() -> Self {
        return Self {
            xattrs: BTreeMap::new(),
        };
    }
}

pub struct InodeSimpleExtendedAttributes(pub QRwLock<InodeSimpleExtendedAttributesInternal>);

impl Default for InodeSimpleExtendedAttributes {
    fn default() -> Self {
        return Self(QRwLock::new(Default::default()));
    }
}

impl Deref for InodeSimpleExtendedAttributes {
    type Target = QRwLock<InodeSimpleExtendedAttributesInternal>;

    fn deref(&self) -> &QRwLock<InodeSimpleExtendedAttributesInternal> {
        &self.0
    }
}

impl InodeSimpleExtendedAttributes {
    pub fn Getxattr(&self, _dir: &Inode, name: &str) -> Result<String> {
        match self.read().xattrs.get(name) {
            None => Err(Error::SysError(SysErr::ENOATTR)),
            Some(s) => Ok(s.clone()),
        }
    }

    pub fn Setxattr(&self, _dir: &mut Inode, name: &str, value: &str) -> Result<()> {
        self.write()
            .xattrs
            .insert(name.to_string(), value.to_string());
        return Ok(());
    }

    pub fn Listxattr(&self, _dir: &Inode) -> Result<Vec<String>> {
        let mut res = Vec::new();
        for (name, _) in &self.read().xattrs {
            res.push(name.clone());
        }

        return Ok(res);
    }
}

pub struct InodeStaticFileGetterInternal {
    pub content: Arc<Vec<u8>>,
}

impl Default for InodeStaticFileGetterInternal {
    fn default() -> Self {
        return Self {
            content: Arc::new(Vec::new()),
        };
    }
}

pub struct InodeStaticFileGetter(pub QRwLock<InodeStaticFileGetterInternal>);

impl Default for InodeStaticFileGetter {
    fn default() -> Self {
        return Self(QRwLock::new(Default::default()));
    }
}

impl Deref for InodeStaticFileGetter {
    type Target = QRwLock<InodeStaticFileGetterInternal>;

    fn deref(&self) -> &QRwLock<InodeStaticFileGetterInternal> {
        &self.0
    }
}

impl InodeStaticFileGetter {
    fn GetFile(&self, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        return Ok(File(Arc::new(FileInternal {
            UniqueId: NewUID(),
            Dirent: dirent.clone(),
            flags: QMutex::new((flags.clone(), None)),
            offset: QLock::New(0),
            FileOp: Arc::new(StaticFile {
                content: self.read().content.clone(),
            }),
        })));
    }
}

pub struct InodeNotDirectoryInternal {}

impl InodeNotDirectoryInternal {
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
}

pub struct InodeNotTruncatable {}

impl InodeNotTruncatable {
    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }
}

pub struct InodeIsDirTruncate {}

impl InodeIsDirTruncate {
    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR));
    }
}

pub struct InodeNoopTruncate {}

impl InodeNoopTruncate {
    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Ok(());
    }
}

pub struct InodeNotRenameable {}

impl InodeNotRenameable {
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
}

pub struct InodeNotOpenable {}

impl InodeNotOpenable {
    fn GetFile(
        &self,
        _dir: &Inode,
        _dirent: &Dirent,
        _flags: FileFlags,
    ) -> Result<Arc<QMutex<File>>> {
        return Err(Error::SysError(SysErr::EIO));
    }
}

pub struct InodeNotVirtual {}

impl InodeNotVirtual {
    fn IsVirtual(&self) -> bool {
        return false;
    }
}

pub struct InodeVirtual {}

impl InodeVirtual {
    fn IsVirtual(&self) -> bool {
        return true;
    }
}

pub struct InodeNotSymlink {}

impl InodeNotSymlink {
    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }
}

pub struct InodeNoExtendedAttributes {}

impl InodeNoExtendedAttributes {
    fn Getxattr(&self, _dir: &Inode, _name: &str) -> Result<String> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Setxattr(&self, _dir: &mut Inode, _name: &str, _value: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Listxattr(&self, _dir: &Inode) -> Result<Vec<String>> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }
}

pub struct InodeGenericChecker {}

impl InodeGenericChecker {
    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }
}

pub struct InodeDenyWriteChecker {}

impl InodeDenyWriteChecker {
    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        if reqPerms.write {
            return Ok(false);
        }

        return ContextCanAccessFile(task, inode, reqPerms);
    }
}

pub struct InodeNotAllocatable {}

impl InodeNotAllocatable {
    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }
}

pub struct InodeNoopAllocate {}

impl InodeNoopAllocate {
    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Ok(());
    }
}

pub struct InodeIsDirAllocate {}

impl InodeIsDirAllocate {
    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR));
    }
}

pub struct InodeNotMappable {}

impl InodeNotMappable {
    fn Mmap(
        &self,
        _task: &Task,
        _len: u64,
        _hugePage: bool,
        _offset: u64,
        _share: bool,
    ) -> Result<u64> {
        return Err(Error::SysError(SysErr::EACCES));
    }
}
