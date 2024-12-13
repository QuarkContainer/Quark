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
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use crate::GUEST_HOST_SHARED_ALLOCATOR;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::qmsg::*;
use super::super::super::kernel::time::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::Kernel::*;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::super::mount::*;
use super::tmpfs_dir::*;

pub fn NewTmpfsFileInode(
    task: &Task,
    uattr: UnstableAttr,
    msrc: &Arc<QMutex<MountSource>>,
) -> Result<Inode> {
    let mut fstat = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
    let tmpfd = HostSpace::NewTmpfsFile(TmpfsFileType::File, &mut *fstat as *mut _ as u64) as i32;
    if tmpfd < 0 {
        return Err(Error::SysError(-tmpfd));
    }

    let inode = Inode::NewHostInode(task, msrc, tmpfd, &*fstat, true, false, true)?;

    let inodeops = inode.lock().InodeOp.clone();
    let hostiops = match inodeops.as_any().downcast_ref::<HostInodeOp>() {
        None => return Err(Error::SysError(SysErr::EBADF)),
        Some(iops) => iops.clone(),
    };

    let ops = TmpfsFileInodeOp {
        inodeops: hostiops,
        uattr: Arc::new(QMutex::new(uattr)),
    };

    let deviceId = TMPFS_DEVICE.lock().DeviceID();
    let inodeId = TMPFS_DEVICE.lock().NextIno();
    let attr = StableAttr {
        Type: InodeType::RegularFile,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    return Ok(Inode::New(ops.into(), msrc, &attr));
}

#[derive(Clone)]
pub struct TmpfsFileInodeOp {
    pub inodeops: HostInodeOp,
    pub uattr: Arc<QMutex<UnstableAttr>>,
}

impl InodeOperations for TmpfsFileInodeOp {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::TmpfsFileInodeOp;
    }

    fn InodeType(&self) -> InodeType {
        return self.inodeops.InodeType();
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::TmpfsFile;
    }

    fn WouldBlock(&self) -> bool {
        return self.inodeops.WouldBlock();
    }

    fn Lookup(&self, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return self.inodeops.Lookup(task, dir, name);
    }

    fn Create(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        flags: &FileFlags,
        perm: &FilePermissions,
    ) -> Result<File> {
        return self.inodeops.Create(task, dir, name, flags, perm);
    }

    fn CreateDirectory(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        return self.inodeops.CreateDirectory(task, dir, name, perm);
    }

    fn CreateLink(&self, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()> {
        return self.inodeops.CreateLink(task, dir, oldname, newname);
    }

    fn CreateHardLink(
        &self,
        task: &Task,
        dir: &mut Inode,
        target: &Inode,
        name: &str,
    ) -> Result<()> {
        return self.inodeops.CreateHardLink(task, dir, target, name);
    }

    fn CreateFifo(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        return self.inodeops.CreateFifo(task, dir, name, perm);
    }

    //fn RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<QMutex<Dirent>>) -> Result<()> ;
    fn Remove(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.inodeops.Remove(task, dir, name);
    }

    fn RemoveDirectory(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.inodeops.RemoveDirectory(task, dir, name);
    }

    fn Rename(
        &self,
        task: &Task,
        _dir: &mut Inode,
        oldParent: &Inode,
        oldname: &str,
        newParent: &Inode,
        newname: &str,
        replacement: bool,
    ) -> Result<()> {
        return TmpfsRename(task, oldParent, oldname, newParent, newname, replacement);
    }

    fn Bind(
        &self,
        task: &Task,
        dir: &Inode,
        name: &str,
        data: &BoundEndpoint,
        perms: &FilePermissions,
    ) -> Result<Dirent> {
        return self.inodeops.Bind(task, dir, name, data, perms);
    }

    fn BoundEndpoint(&self, task: &Task, inode: &Inode, path: &str) -> Option<BoundEndpoint> {
        return self.inodeops.BoundEndpoint(task, inode, path);
    }

    fn GetFile(&self, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        //let mut flags = flags;
        //flags.Read = true;
        //flags.Write = true;
        return self.inodeops.GetFile(task, dir, dirent, flags);
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let (size, blocks) = self.inodeops.Size()?;

        let mut ret = *self.uattr.lock();
        ret.Usage = blocks * 512;
        ret.Size = size;
        return Ok(ret);
    }

    fn Getxattr(&self, dir: &Inode, name: &str, size: usize) -> Result<Vec<u8>> {
        return self.inodeops.Getxattr(dir, name, size);
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        return self.inodeops.Setxattr(dir, name, value, flags);
    }

    fn Listxattr(&self, dir: &Inode, size: usize) -> Result<Vec<String>> {
        return self.inodeops.Listxattr(dir, size);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return self.inodeops.Check(task, inode, reqPerms);
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, f: FilePermissions) -> bool {
        self.uattr.lock().SetPermissions(task, &f);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.uattr.lock().SetOwner(task, owner);
        return Ok(());
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.uattr.lock().SetTimestamps(task, ts);
        return Ok(());
    }

    fn Truncate(&self, task: &Task, dir: &mut Inode, size: i64) -> Result<()> {
        return self.inodeops.Truncate(task, dir, size);
    }

    fn Allocate(&self, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()> {
        return self.inodeops.Allocate(task, dir, offset, length);
    }

    fn ReadLink(&self, task: &Task, dir: &Inode) -> Result<String> {
        return self.inodeops.ReadLink(task, dir);
    }

    fn GetLink(&self, task: &Task, dir: &Inode) -> Result<Dirent> {
        return self.inodeops.GetLink(task, dir);
    }

    fn AddLink(&self, _task: &Task) {
        self.uattr.lock().Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.uattr.lock().Links -= 1;
    }

    fn IsVirtual(&self) -> bool {
        return true;
    }

    fn Sync(&self) -> Result<()> {
        return self.inodeops.Sync();
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Ok(TMPFS_FSINFO);
    }

    fn Mappable(&self) -> Result<MMappable> {
        return self.inodeops.Mappable();
    }
}
