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
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
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
use super::super::ramfs::symlink::*;
use super::tmpfs_dir::*;

pub fn NewTmpfsSymlink(
    task: &Task,
    target: &str,
    owner: &FileOwner,
    msrc: &Arc<QMutex<MountSource>>,
) -> Inode {
    let s = Symlink::New(task, owner, target);
    let s = TmpfsSymlink(s);

    let deviceId = TMPFS_DEVICE.lock().DeviceID();
    let inodeId = TMPFS_DEVICE.lock().NextIno();
    let attr = StableAttr {
        Type: InodeType::Symlink,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    return Inode::New(s.into(), msrc, &attr);
}

#[derive(Clone)]
pub struct TmpfsSymlink(Symlink);

impl InodeOperations for TmpfsSymlink {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::TmpfsSymlink;
    }

    fn InodeType(&self) -> InodeType {
        return self.0.InodeType();
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::TmpfsSymlink;
    }

    fn WouldBlock(&self) -> bool {
        return self.0.WouldBlock();
    }

    fn Lookup(&self, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return self.0.Lookup(task, dir, name);
    }

    fn Create(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        flags: &FileFlags,
        perm: &FilePermissions,
    ) -> Result<File> {
        return self.0.Create(task, dir, name, flags, perm);
    }

    fn CreateDirectory(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        return self.0.CreateDirectory(task, dir, name, perm);
    }

    fn CreateLink(&self, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()> {
        return self.0.CreateLink(task, dir, oldname, newname);
    }

    fn CreateHardLink(
        &self,
        task: &Task,
        dir: &mut Inode,
        target: &Inode,
        name: &str,
    ) -> Result<()> {
        return self.0.CreateHardLink(task, dir, target, name);
    }

    fn CreateFifo(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        return self.0.CreateFifo(task, dir, name, perm);
    }

    //fn RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<QMutex<Dirent>>) -> Result<()> ;
    fn Remove(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.0.Remove(task, dir, name);
    }

    fn RemoveDirectory(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()> {
        return self.0.RemoveDirectory(task, dir, name);
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
        return self.0.Bind(task, dir, name, data, perms);
    }

    fn BoundEndpoint(&self, task: &Task, inode: &Inode, path: &str) -> Option<BoundEndpoint> {
        return self.0.BoundEndpoint(task, inode, path);
    }

    fn GetFile(&self, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        return self.0.GetFile(task, dir, dirent, flags);
    }

    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        return self.0.UnstableAttr(task);
    }

    fn Getxattr(&self, dir: &Inode, name: &str, size: usize) -> Result<Vec<u8>> {
        return self.0.Getxattr(dir, name, size);
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        return self.0.Setxattr(dir, name, value, flags);
    }

    fn Listxattr(&self, dir: &Inode, size: usize) -> Result<Vec<String>> {
        return self.0.Listxattr(dir, size);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return self.0.Check(task, inode, reqPerms);
    }

    fn SetPermissions(&self, task: &Task, dir: &mut Inode, f: FilePermissions) -> bool {
        return self.0.SetPermissions(task, dir, f);
    }

    fn SetOwner(&self, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        return self.0.SetOwner(task, dir, owner);
    }

    fn SetTimestamps(&self, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return self.0.SetTimestamps(task, dir, ts);
    }

    fn Truncate(&self, task: &Task, dir: &mut Inode, size: i64) -> Result<()> {
        return self.0.Truncate(task, dir, size);
    }

    fn Allocate(&self, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()> {
        return self.0.Allocate(task, dir, offset, length);
    }

    fn ReadLink(&self, task: &Task, dir: &Inode) -> Result<String> {
        return self.0.ReadLink(task, dir);
    }

    fn GetLink(&self, task: &Task, dir: &Inode) -> Result<Dirent> {
        return self.0.GetLink(task, dir);
    }

    fn AddLink(&self, task: &Task) {
        return self.0.AddLink(task);
    }

    fn DropLink(&self, task: &Task) {
        return self.0.DropLink(task);
    }

    fn IsVirtual(&self) -> bool {
        return self.0.IsVirtual();
    }

    fn Sync(&self) -> Result<()> {
        return self.0.Sync();
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Ok(TMPFS_FSINFO);
    }

    fn Mappable(&self) -> Result<MMappable> {
        return self.0.Mappable();
    }
}
