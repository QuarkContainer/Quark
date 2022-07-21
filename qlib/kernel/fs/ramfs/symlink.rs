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
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::qlock::*;
use super::super::super::kernel::waiter::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::uid::*;
use super::super::attr::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::super::mount::*;

pub struct SymlinkInternal {
    pub Target: String,

    pub fsType: u64,
    pub unstable: UnstableAttr,

    pub xattrs: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone)]
pub struct Symlink(pub Arc<QRwLock<SymlinkInternal>>);

impl Deref for Symlink {
    type Target = Arc<QRwLock<SymlinkInternal>>;

    fn deref(&self) -> &Arc<QRwLock<SymlinkInternal>> {
        &self.0
    }
}

impl Symlink {
    pub fn New(task: &Task, owner: &FileOwner, target: &str) -> Self {
        let unstable = WithCurrentTime(
            task,
            &UnstableAttr {
                Owner: *owner,
                Perms: FilePermissions::FromMode(FileMode(0o777)),
                ..Default::default()
            },
        );

        let internal = SymlinkInternal {
            Target: target.to_string(),
            fsType: FSMagic::RAMFS_MAGIC,
            unstable: unstable,
            xattrs: BTreeMap::new(),
        };

        return Self(Arc::new(QRwLock::new(internal)));
    }
}

impl InodeOperations for Symlink {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::Symlink;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::Symlink;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::Symlink;
    }

    fn WouldBlock(&self) -> bool {
        return false;
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

    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let file = FileInternal {
            UniqueId: NewUID(),
            Dirent: dirent.clone(),
            flags: QMutex::new((flags, None)),
            offset: QLock::New(0),
            FileOp: Arc::new(SymlinkFileOperations {}),
        };

        return Ok(File(Arc::new(file)));
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let mut internal = self.write();

        internal.unstable.Size = internal.Target.len() as i64;
        internal.unstable.Usage = internal.unstable.Size;
        return Ok(internal.unstable.clone());
    }

    fn Getxattr(&self, _dir: &Inode, name: &str, _size: usize) -> Result<Vec<u8>> {
        match self.read().xattrs.get(name) {
            None => Err(Error::SysError(SysErr::ENOATTR)),
            Some(s) => Ok(s.clone()),
        }
    }

    fn Setxattr(&self, _dir: &mut Inode, name: &str, value: &[u8], _flags: u32) -> Result<()> {
        self.write()
            .xattrs
            .insert(name.to_string(), value.to_vec());
        return Ok(());
    }

    fn Listxattr(&self, _dir: &Inode, _size: usize) -> Result<Vec<String>> {
        let mut res = Vec::new();
        for (name, _) in &self.read().xattrs {
            res.push(name.clone());
        }

        return Ok(res);
    }

    fn Removexattr(&self, _dir: &Inode, name: &str) -> Result<()> {
        match self.write().xattrs.remove(name) {
            None => return Err(Error::SysError(SysErr::ENOATTR)),
            Some(_) => return Ok(())
        }
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }

    fn SetPermissions(&self, _task: &Task, _dir: &mut Inode, _p: FilePermissions) -> bool {
        return false;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.write().unstable.SetOwner(task, owner);
        return Ok(());
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.write().unstable.SetTimestamps(task, ts);
        return Ok(());
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Ok(self.read().Target.clone());
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::ErrResolveViaReadlink);
    }

    fn AddLink(&self, _task: &Task) {
        self.write().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.write().unstable.Links -= 1;
    }

    fn IsVirtual(&self) -> bool {
        return true;
    }

    fn Sync(&self) -> Result<()> {
        return Ok(());
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

pub struct SymlinkFileOperations {}

impl Waitable for SymlinkFileOperations {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        panic!("SymlinkFileOperations doesn't support Waitable::Readiness");
    }

    fn EventRegister(&self, _task: &Task, _e: &WaitEntry, _mask: EventMask) {
        panic!("SymlinkFileOperations doesn't support Waitable::EventRegister");
    }

    fn EventUnregister(&self, _task: &Task, _e: &WaitEntry) {
        panic!("SymlinkFileOperations doesn't support Waitable::EventUnregister");
    }
}

impl SpliceOperations for SymlinkFileOperations {}

impl FileOperations for SymlinkFileOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::SymlinkFileOperations;
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(
        &self,
        _task: &Task,
        _f: &File,
        _whence: i32,
        _current: i64,
        _offset: i64,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
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
        _task: &Task,
        _f: &File,
        _dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
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

impl SockOperations for SymlinkFileOperations {}
