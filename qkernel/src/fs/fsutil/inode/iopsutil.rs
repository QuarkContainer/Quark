// Copyright (c) 2021 Quark Container Authors
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
//use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::qlib::common::*;
use super::super::super::super::kernel::time::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::qlib::auth::*;
use super::super::super::super::task::*;

use super::super::super::inode::*;
use super::super::super::dirent::*;
use super::super::super::file::*;
use super::super::super::attr::*;
use super::super::super::flags::*;
use super::super::super::mount::*;
use super::super::super::host::hostinodeop::*;
use super::super::file::fileopsutil::*;

pub enum InodeOpsData {
    None,
}

pub type GetFileOp = fn(_data: &InodeOpsData, task: &Task) -> Result<Arc<FileOperations>>;
pub type InodeTypeFn = fn(_data: &InodeOpsData) -> InodeType;
pub type WouldBlock = fn(_data: &InodeOpsData) -> bool;
pub type Lookup = fn(_data: &InodeOpsData, task: &Task, dir: &Inode, name: &str) -> Result<Dirent>;
pub type Create = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, name: &str, flags: &FileFlags, perm: &FilePermissions) -> Result<File>;
pub type CreateDirectory = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, name: &str, perm: &FilePermissions) -> Result<()>;
pub type CreateLink = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()>;
pub type CreateHardLink = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, target: &Inode, name: &str) -> Result<()>;
pub type CreateFifo = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, name: &str, perm: &FilePermissions) -> Result<()>;
//pub type RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<Mutex<Dirent>>) -> Result<()> ;
pub type Remove = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, name: &str) -> Result<()>;
pub type RemoveDirectory = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, name: &str) -> Result<()>;
pub type Rename = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, oldParent: &Inode, oldname: &str, newParent: &Inode, newname: &str, replacement: bool) -> Result<()>;
pub type GetFile = fn(_data: &InodeOpsData, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File>;
pub type UnstableAttrFn= fn(_data: &InodeOpsData, task: &Task, dir: &Inode) -> Result<UnstableAttr>;
pub type Getxattr = fn(_data: &InodeOpsData, dir: &Inode, name: &str) -> Result<String>;
pub type Setxattr = fn(_data: &InodeOpsData, dir: &mut Inode, name: &str, value: &str) -> Result<()>;
pub type Listxattr = fn(_data: &InodeOpsData, dir: &Inode) -> Result<Vec<String>>;
pub type Check = fn(_data: &InodeOpsData, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool>;
pub type SetPermissions = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, f: FilePermissions) -> bool;
pub type SetOwner = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()>;
pub type SetTimestamps = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()>;
pub type Truncate = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, size: i64) -> Result<()>;
pub type Allocate = fn(_data: &InodeOpsData, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()>;
pub type ReadLink = fn(_data: &InodeOpsData, _task: &Task,dir: &Inode) -> Result<String>;
pub type GetLink = fn(_data: &InodeOpsData, _task: &Task, dir: &Inode) -> Result<Dirent>;
pub type AddLink = fn(_data: &InodeOpsData, _task: &Task);
pub type DropLink = fn(_data: &InodeOpsData, _task: &Task);
pub type IsVirtual = fn(_data: &InodeOpsData) -> bool;
pub type Sync = fn(_data: &InodeOpsData) -> Result<()>;
pub type StatFS = fn(_data: &InodeOpsData, task: &Task) -> Result<FsInfo>;
pub type Mmap = fn(_data: &InodeOpsData, task: &Task, len: u64, hugePage: bool, offset: u64, share: bool, prot: u64) -> Result<u64>;
pub type Mappable = fn(_data: &InodeOpsData) -> Option<HostInodeOp>;

fn InodeNotDirectory_Lookup(_data: &InodeOpsData, _task: &Task, _dir: &Inode, _name: &str) -> Result<Dirent> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_Create(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _name: &str, _flags: &FileFlags, _perm: &FilePermissions) -> Result<File> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_CreateDirectory(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_CreateLink(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _oldname: &str, _newname: &str) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_CreateHardLink(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _target: &Inode, _name: &str) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_CreateFifo(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_Remove(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_RemoveDirectory(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

fn InodeNotDirectory_Rename(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _oldParent: &Inode, _oldname: &str, _newParent: &Inode, _newname: &str, _replacement: bool) -> Result<()> {
    return Err(Error::SysError(SysErr::EINVAL))
}

fn InodeNotTruncatable_Truncate(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
    return Err(Error::SysError(SysErr::EINVAL))
}

fn InodeIsDirTruncate_Truncate(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
    return Err(Error::SysError(SysErr::EISDIR))
}

fn InodeNoopTruncate_Truncate(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
    return Ok(())
}

fn InodeNotRenameable_Rename(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _oldParent: &Inode, _oldname: &str, _newParent: &Inode, _newname: &str, _replacement: bool) -> Result<()> {
    return Err(Error::SysError(SysErr::EINVAL))
}

fn InodeNotOpenable_GetFile(_data: &InodeOpsData, _task: &Task, _dir: &Inode, _dirent: &Dirent, _flags: FileFlags) -> Result<File> {
    return Err(Error::SysError(SysErr::EIO))
}

fn InodeNotVirtual_IsVirtual(_data: &InodeOpsData) -> bool {
    return false
}

fn InodeVirtual_IsVirtual(_data: &InodeOpsData) -> bool {
    return true
}

fn InodeNotSymlink_ReadLink(_data: &InodeOpsData, _task: &Task,_dir: &Inode) -> Result<String> {
    return Err(Error::SysError(SysErr::ENOLINK))
}

fn InodeNotSymlink_GetLink(_data: &InodeOpsData, _task: &Task, _dir: &Inode) -> Result<Dirent> {
    return Err(Error::SysError(SysErr::ENOLINK))
}

fn InodeNotSymlink_AddLink(_data: &InodeOpsData, _task: &Task) {
}

fn InodeNotSymlink_DropLink(_data: &InodeOpsData, _task: &Task) {
}

fn InodeNoExtendedAttributes_Getxattr(_data: &InodeOpsData, _dir: &Inode, _name: &str) -> Result<String> {
    return Err(Error::SysError(SysErr::EOPNOTSUPP))
}

fn InodeNoExtendedAttributes_Setxattr(_data: &InodeOpsData, _dir: &mut Inode, _name: &str, _value: &str) -> Result<()> {
    return Err(Error::SysError(SysErr::EOPNOTSUPP))
}

fn InodeNoExtendedAttributes_Listxattr(_data: &InodeOpsData, _dir: &Inode) -> Result<Vec<String>> {
    return Err(Error::SysError(SysErr::EOPNOTSUPP))
}

fn InodeGenericChecker_Check(_data: &InodeOpsData, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
    return ContextCanAccessFile(task, inode, reqPerms)
}

fn InodeDenyWriteChecker_Check(_data: &InodeOpsData, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
    if reqPerms.write {
        return Ok(false)
    }

    return ContextCanAccessFile(task, inode, reqPerms)
}

fn InodeNotAllocatable_Allocate(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
    return Err(Error::SysError(SysErr::EOPNOTSUPP))
}

fn InodeNoopAllocate_Allocate(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
    return Ok(())
}

fn InodeIsDirAllocate_Allocate(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
    return Err(Error::SysError(SysErr::EISDIR))
}

fn InodeNotMappable_Mmap(_data: &InodeOpsData, _task: &Task, _len: u64, _hugePage: bool, _offset: u64, _share: bool, _prot: u64) -> Result<u64> {
    return Err(Error::SysError(SysErr::EACCES))
}

fn InodeNotMappable_Mappable(_data: &InodeOpsData) -> Option<HostInodeOp> {
    return None;
}

fn InodeNotInodeType_InodeType(_data: &InodeOpsData) -> InodeType {
    return InodeType::None
}

fn InodeWouldBlock_WouldBlock(_data: &InodeOpsData) -> bool {
    return false
}

fn InodeDefault_UnstableAttr(_data: &InodeOpsData, _task: &Task, _dir: &Inode) -> Result<UnstableAttr> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

fn InodeNoop_SetPermissions(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _f: FilePermissions) -> bool {
    return true
}

fn InodeDefault_SetOwner(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _owner: &FileOwner) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

fn InodeDefault_SetTimestamps(_data: &InodeOpsData, _task: &Task, _dir: &mut Inode, _ts: &InterTimeSpec) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

fn InodeDefault_StatFS(_data: &InodeOpsData, _task: &Task) -> Result<FsInfo> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

fn InodeDefault_Sync(_data: &InodeOpsData) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

fn InodeDefault_GetFileOp(_data: &InodeOpsData, _task: &Task) -> Result<Arc<FileOperations>> {
    return Ok(Arc::new(FileOptionsUtil::default()))
}

pub struct InodeOpsUtil {
    pub data: InodeOpsData,

    pub getFileOp: GetFileOp,
    pub inodeType: InodeTypeFn,
    pub wouldBlock: WouldBlock,
    pub lookup: Lookup,
    pub create: Create,
    pub createDirectory: CreateDirectory,
    pub createLink: CreateLink,
    pub createHardLink: CreateHardLink,
    pub createFifo: CreateFifo,
    pub remove: Remove,
    pub removeDirectory: RemoveDirectory,
    pub rename: Rename,
    pub getFile: GetFile,
    pub unstableAttr: UnstableAttrFn,
    pub getxattr: Getxattr,
    pub setxattr: Setxattr,
    pub listxattr: Listxattr,
    pub check: Check,
    pub setPermissions: SetPermissions,
    pub setOwner: SetOwner,
    pub setTimestamps: SetTimestamps,
    pub truncate: Truncate,
    pub allocate: Allocate,
    pub readLink: ReadLink,
    pub getLink: GetLink,
    pub addLink: AddLink,
    pub dropLink: DropLink,
    pub isVirtual: IsVirtual,
    pub sync: Sync,
    pub statFS: StatFS,
    pub mmap: Mmap,
    pub mappable: Mappable,
}

impl Default for InodeOpsUtil {
    fn default() -> Self {
        return Self {
            data: InodeOpsData::None,

            getFileOp: InodeDefault_GetFileOp,
            inodeType: InodeNotInodeType_InodeType,
            wouldBlock: InodeWouldBlock_WouldBlock,
            lookup: InodeNotDirectory_Lookup,
            create: InodeNotDirectory_Create,
            createDirectory: InodeNotDirectory_CreateDirectory,
            createLink: InodeNotDirectory_CreateLink,
            createHardLink: InodeNotDirectory_CreateHardLink,
            createFifo: InodeNotDirectory_CreateFifo,
            remove: InodeNotDirectory_Remove,
            removeDirectory: InodeNotDirectory_RemoveDirectory,
            rename: InodeNotDirectory_Rename,
            getFile: InodeNotOpenable_GetFile,
            unstableAttr: InodeDefault_UnstableAttr,
            getxattr: InodeNoExtendedAttributes_Getxattr,
            setxattr: InodeNoExtendedAttributes_Setxattr,
            listxattr: InodeNoExtendedAttributes_Listxattr,
            check: InodeGenericChecker_Check,
            setPermissions: InodeNoop_SetPermissions,
            setOwner: InodeDefault_SetOwner,
            setTimestamps: InodeDefault_SetTimestamps,
            truncate: InodeNotTruncatable_Truncate,
            allocate: InodeNotAllocatable_Allocate,
            readLink: InodeNotSymlink_ReadLink,
            getLink: InodeNotSymlink_GetLink,
            addLink: InodeNotSymlink_AddLink,
            dropLink: InodeNotSymlink_DropLink,
            isVirtual: InodeVirtual_IsVirtual,
            sync: InodeDefault_Sync,
            statFS: InodeDefault_StatFS,
            mmap: InodeNotMappable_Mmap,
            mappable: InodeNotMappable_Mappable,
        }
    }
}

impl InodeOpsUtil {
    pub fn SetInodeNotDirectory(&mut self) {
        self.lookup = InodeNotDirectory_Lookup;
        self.create = InodeNotDirectory_Create;
        self.createDirectory = InodeNotDirectory_CreateDirectory;
        self.createLink = InodeNotDirectory_CreateLink;
        self.createHardLink = InodeNotDirectory_CreateHardLink;
        self.createFifo = InodeNotDirectory_CreateFifo;
        self.remove = InodeNotDirectory_Remove;
        self.removeDirectory = InodeNotDirectory_RemoveDirectory;
        self.rename = InodeNotDirectory_Rename;
    }

    pub fn SetInodeNotTruncatable(&mut self) {
        self.truncate = InodeNotTruncatable_Truncate;
    }

    pub fn SetInodeNoopTruncate(&mut self) {
        self.truncate = InodeNoopTruncate_Truncate;
    }

    pub fn SetInodeNotRenameable(&mut self) {
        self.rename = InodeNotRenameable_Rename;
    }

    pub fn SetInodeNotOpenable(&mut self) {
        self.getFile = InodeNotOpenable_GetFile;
    }

    pub fn SetInodeNotVirtual(&mut self) {
        self.isVirtual = InodeNotVirtual_IsVirtual;
    }

    pub fn SetInodeVirtual(&mut self) {
        self.isVirtual = InodeVirtual_IsVirtual;
    }

    pub fn SetInodeNotSymlink(&mut self) {
        self.readLink = InodeNotSymlink_ReadLink;
        self.getLink = InodeNotSymlink_GetLink;
    }

    pub fn SetInodeNoExtendedAttributes(&mut self) {
        self.getxattr = InodeNoExtendedAttributes_Getxattr;
        self.setxattr = InodeNoExtendedAttributes_Setxattr;
        self.listxattr = InodeNoExtendedAttributes_Listxattr;
    }

    pub fn SetInodeGenericChecker(&mut self) {
        self.check = InodeGenericChecker_Check;
    }

    pub fn SetInodeDenyWriteChecker(&mut self) {
        self.check = InodeDenyWriteChecker_Check;
    }

    pub fn SetInodeNotAllocatable(&mut self) {
        self.allocate = InodeNotAllocatable_Allocate;
    }

    pub fn SetInodeNoopAllocate(&mut self) {
        self.allocate = InodeNoopAllocate_Allocate;
    }

    pub fn SetInodeNotMappable(&mut self) {
        self.mmap = InodeNotMappable_Mmap;
        self.mappable = InodeNotMappable_Mappable;
    }

    pub fn SetInodeIsDir(&mut self) {
        self.allocate = InodeIsDirAllocate_Allocate;
        self.truncate = InodeIsDirTruncate_Truncate;
    }
}