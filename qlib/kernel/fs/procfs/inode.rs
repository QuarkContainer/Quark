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

use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::mutex::*;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::qlock::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::threadmgr::thread::*;
use super::super::super::uid::*;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::fsutil::file::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::super::mount::*;

#[derive(Clone)]
pub struct TaskOwnedInodeOps {
    pub iops: Arc<Iops>,
    pub creds: Credentials,
}

impl InodeOperations for TaskOwnedInodeOps {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::TaskOwnedInodeOps;
    }

    fn InodeType(&self) -> InodeType {
        return self.iops.InodeType();
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::TaskOwned;
    }

    fn WouldBlock(&self) -> bool {
        return self.iops.WouldBlock();
    }

    fn Lookup(&self, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return self.iops.Lookup(task, dir, name);
    }

    fn Create(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        flags: &FileFlags,
        perm: &FilePermissions,
    ) -> Result<File> {
        return self.iops.Create(task, dir, name, flags, perm);
    }

    fn CreateDirectory(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        return self.iops.CreateDirectory(task, dir, name, perm);
    }

    fn CreateLink(&self, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()> {
        return self.iops.CreateLink(task, dir, oldname, newname);
    }

    fn CreateHardLink(
        &self,
        task: &Task,
        dir: &mut Inode,
        target: &Inode,
        name: &str,
    ) -> Result<()> {
        return self.iops.CreateHardLink(task, dir, target, name);
    }

    fn CreateFifo(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        return self.iops.CreateFifo(task, dir, name, perm);
    }

    fn Remove(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.iops.Remove(task, dir, name);
    }

    fn RemoveDirectory(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.iops.RemoveDirectory(task, dir, name);
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
        return self.iops.Rename(
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
        return self.iops.GetFile(task, dir, dirent, flags);
    }

    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        let mut unstable = self.iops.UnstableAttr(task)?;
        let creds = self.creds.lock();
        unstable.Owner = FileOwner {
            UID: creds.EffectiveKUID,
            GID: creds.EffectiveKGID,
        };

        return Ok(unstable);
    }

    fn Getxattr(&self, dir: &Inode, name: &str, size: usize) -> Result<Vec<u8>> {
        return self.iops.Getxattr(dir, name, size);
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        return self.iops.Setxattr(dir, name, value, flags);
    }

    fn Listxattr(&self, dir: &Inode, size: usize) -> Result<Vec<String>> {
        return self.iops.Listxattr(dir, size);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return self.iops.Check(task, inode, reqPerms);
    }

    fn SetPermissions(&self, task: &Task, dir: &mut Inode, f: FilePermissions) -> bool {
        return self.iops.SetPermissions(task, dir, f);
    }

    fn SetOwner(&self, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        return self.iops.SetOwner(task, dir, owner);
    }

    fn SetTimestamps(&self, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return self.iops.SetTimestamps(task, dir, ts);
    }

    fn Truncate(&self, task: &Task, dir: &mut Inode, size: i64) -> Result<()> {
        return self.iops.Truncate(task, dir, size);
    }

    fn Allocate(&self, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()> {
        return self.iops.Allocate(task, dir, offset, length);
    }

    fn ReadLink(&self, task: &Task, dir: &Inode) -> Result<String> {
        return self.iops.ReadLink(task, dir);
    }

    fn GetLink(&self, task: &Task, dir: &Inode) -> Result<Dirent> {
        return self.iops.GetLink(task, dir);
    }

    fn AddLink(&self, task: &Task) {
        return self.iops.AddLink(task);
    }

    fn DropLink(&self, task: &Task) {
        return self.iops.DropLink(task);
    }

    fn IsVirtual(&self) -> bool {
        return self.iops.IsVirtual();
    }

    fn Sync(&self) -> Result<()> {
        return self.iops.Sync();
    }

    fn StatFS(&self, task: &Task) -> Result<FsInfo> {
        return self.iops.StatFS(task);
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

pub struct StaticFileInodeOpsInternal {
    pub fsType: u64,
    pub unstable: UnstableAttr,

    pub content: Arc<Vec<u8>>,
}

#[derive(Clone)]
pub struct StaticFileInodeOps(pub Arc<QRwLock<StaticFileInodeOpsInternal>>);

impl Deref for StaticFileInodeOps {
    type Target = Arc<QRwLock<StaticFileInodeOpsInternal>>;

    fn deref(&self) -> &Arc<QRwLock<StaticFileInodeOpsInternal>> {
        &self.0
    }
}

impl InodeOperations for StaticFileInodeOps {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::StaticFileInodeOps;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::SpecialFile;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::StaticFile;
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
        return Ok(File(Arc::new(FileInternal {
            UniqueId: NewUID(),
            Dirent: dirent.clone(),
            flags: QMutex::new((flags.clone(), None)),
            offset: QLock::New(0),
            FileOp: StaticFile {
                content: self.read().content.clone(),
            }
            .into(),
        })));
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let u = self.read().unstable;
        return Ok(u);
    }

    fn Getxattr(&self, _dir: &Inode, _name: &str, _size: usize) -> Result<Vec<u8>> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Setxattr(&self, _dir: &mut Inode, _name: &str, _value: &[u8], _flags: u32) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Listxattr(&self, _dir: &Inode, _size: usize) -> Result<Vec<String>> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        if reqPerms.write {
            return Ok(false);
        }

        return ContextCanAccessFile(task, inode, reqPerms);
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

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Ok(());
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Ok(());
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK));
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

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

pub fn NewProcInode(
    iops: Iops,
    msrc: &Arc<QMutex<MountSource>>,
    typ: InodeType,
    thread: Option<Thread>,
) -> Inode {
    let deviceId = PROC_DEVICE.lock().id.DeviceID();
    let inodeId = PROC_DEVICE.lock().NextIno();

    let sattr = StableAttr {
        Type: typ,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: 4096,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    if thread.is_some() {
        let newiops = TaskOwnedInodeOps {
            iops: Arc::new(iops),
            creds: thread.unwrap().lock().creds.clone(),
        };

        return Inode::New(newiops.into(), msrc, &sattr);
    }

    return Inode::New(iops, msrc, &sattr);
}

pub fn NewStaticProcInode(
    task: &Task,
    msrc: &Arc<QMutex<MountSource>>,
    contents: &Arc<Vec<u8>>,
) -> Inode {
    let unstable = WithCurrentTime(
        task,
        &UnstableAttr {
            Owner: ROOT_OWNER,
            Perms: FilePermissions::FromMode(FileMode(0o444)),
            ..Default::default()
        },
    );

    let iops = StaticFileInodeOps(Arc::new(QRwLock::new(StaticFileInodeOpsInternal {
        fsType: FSMagic::PROC_SUPER_MAGIC,
        unstable: unstable,
        content: contents.clone(),
    })));

    return NewProcInode(iops.into(), msrc, InodeType::SpecialFile, None);
}

pub fn NewStaticProcInodeWithString(
    task: &Task,
    msrc: &Arc<QMutex<MountSource>>,
    contents: &str,
) -> Inode {
    return NewStaticProcInode(task, msrc, &Arc::new(contents.as_bytes().to_vec()));
}

pub fn NewStaticProcInodeWithHostFile(
    task: &Task,
    msrc: &Arc<QMutex<MountSource>>,
    filename: &str,
) -> Result<Inode> {
    let bufSize = 1024;
    let mut content: Vec<u8> = Vec::with_capacity(bufSize);
    content.resize(bufSize, 0);
    let filenameSlice = filename.as_bytes();
    let filenameAddr = &filenameSlice[0] as *const _ as u64;
    let ret = HostSpace::ReadContent(
        filenameAddr,
        filename.len(),
        &content[0] as *const _ as u64,
        bufSize,
    );

    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }

    return Ok(NewStaticProcInode(
        task,
        msrc,
        &Arc::new(content[0..ret as usize].to_vec()),
    ));
}
