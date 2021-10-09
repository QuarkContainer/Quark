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
use alloc::sync::Arc;
use ::qlib::mutex::*;
use core::any::Any;

use socket::unix::transport::unix::BoundEndpoint;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::auth::*;
use super::super::super::qlib::device::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::pipe::pipe::*;
use super::super::super::kernel::pipe::node::*;
use super::super::super::task::*;
use super::super::attr::*;
use super::super::mount::*;
use super::super::flags::*;
use super::super::file::*;
use super::super::inode::*;
use super::super::dirent::*;
use super::super::host::hostinodeop::*;
use super::tmpfs_dir::*;

pub fn NewTmpfsFifoInode(task: &Task, perms: &FilePermissions, msrc: &Arc<QMutex<MountSource>>) -> Result<Inode> {
    // First create a pipe.
    let pipe = Pipe::New(task, true, DEFAULT_PIPE_SIZE, MemoryDef::PAGE_SIZE as usize);

    let iops = NewPipeInodeOps(task, perms, pipe);
    let fifo = TmpfsFifoInodeOp(iops);

    let deviceId = TMPFS_DEVICE.lock().DeviceID();
    let inodeId = TMPFS_DEVICE.lock().NextIno();
    let attr = StableAttr {
        Type: InodeType::Pipe,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    return Ok(Inode::New(&Arc::new(fifo), msrc, &attr));
}

pub struct TmpfsFifoInodeOp(PipeIops);

impl InodeOperations for TmpfsFifoInodeOp {
    fn as_any(&self) -> &Any {
        return self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::TmpfsFifoInodeOp;
    }

    fn InodeType(&self) -> InodeType {
        return self.0.InodeType();
    }

    fn InodeFileType(&self) -> InodeFileType{
        return InodeFileType::TmpfsFifo;
    }

    fn WouldBlock(&self) -> bool {
        return self.0.WouldBlock();
    }

    fn Lookup(&self, _task: &Task, _dir: &Inode, _name: &str) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn Create(&self, _task: &Task, _dir: &mut Inode, _name: &str, _flags: &FileFlags, _perm: &FilePermissions) -> Result<File> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateLink(&self, _task: &Task, _dir: &mut Inode, _oldname: &str, _newname: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateHardLink(&self, _task: &Task, _dir: &mut Inode, _target: &Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateFifo(&self, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()>{
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn Rename(&self, task: &Task, _dir: &mut Inode, oldParent: &Inode, oldname: &str, newParent: &Inode, newname: &str, replacement: bool) -> Result<()> {
        return TmpfsRename(task, oldParent, oldname, newParent, newname, replacement)
    }

    fn Bind(&self, _task: &Task, _dir: &Inode, _name: &str, _data: &BoundEndpoint, _perms: &FilePermissions) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return None;
    }

    fn GetFile(&self, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        return self.0.GetFile(task, dir, dirent, flags)
    }

    fn UnstableAttr(&self, task: &Task, dir: &Inode) -> Result<UnstableAttr> {
        return self.0.UnstableAttr(task, dir)
    }

    fn Getxattr(&self, dir: &Inode, name: &str) -> Result<String> {
        return self.0.Getxattr(dir, name)
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &str) -> Result<()> {
        return self.0.Setxattr(dir, name, value)
    }

    fn Listxattr(&self, dir: &Inode) -> Result<Vec<String>> {
        return self.0.Listxattr(dir)
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms)
    }

    fn SetPermissions(&self, task: &Task, dir: &mut Inode, f: FilePermissions) -> bool {
        self.0.SetPermissions(task, dir, f);
        return true;
    }

    fn SetOwner(&self, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        return self.0.SetOwner(task, dir, owner);
    }

    fn SetTimestamps(&self, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return self.0.SetTimestamps(task, dir, ts)
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Ok(())
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EPIPE))
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK))
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK))
    }

    fn AddLink(&self, task: &Task) {
        self.0.AddLink(task)
    }

    fn DropLink(&self, task: &Task) {
        self.0.DropLink(task)
    }

    fn IsVirtual(&self) -> bool {
        return false;
    }

    fn Sync(&self) -> Result<()> {
        return Ok(())
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Ok(TMPFS_FSINFO)
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}