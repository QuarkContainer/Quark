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
use crate::qlib::mutex::*;
use core::ops::Deref;
use alloc::string::String;
use alloc::vec::Vec;
use core::any::Any;

use super::super::super::socket::unix::transport::unix::BoundEndpoint;
use super::super::super::fs::tmpfs::tmpfs_dir::*;
use super::super::super::fs::mount::*;
use super::super::super::fs::attr::*;
use super::super::super::fs::inode::*;
use super::super::super::fs::flags::*;
use super::super::super::fs::file::*;
use super::super::super::fs::host::hostinodeop::*;
use super::super::super::fs::dirent::*;
use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::task::*;
use super::super::super::kernel::time::*;
use super::pipe::*;

pub struct PipeIopsInternal {
    // p is the underlying Pipe object representing this fifo.
    pub p: Pipe,

    pub fsType: u64,
    pub unstable: UnstableAttr,
}

pub fn NewPipeInodeOps(task: &Task, perms: &FilePermissions, p: Pipe) -> PipeIops {
    let unstable = WithCurrentTime(task, &UnstableAttr {
        Owner: task.FileOwner(),
        Perms: *perms,
        ..Default::default()
    });

    let internal = PipeIopsInternal {
        p: p,
        fsType: FSMagic::PIPEFS_MAGIC,
        unstable: unstable,
    };

    return PipeIops(Arc::new(QMutex::new(internal)))
}

pub struct PipeIops(Arc<QMutex<PipeIopsInternal>>);

impl Deref for PipeIops {
    type Target = Arc<QMutex<PipeIopsInternal>>;

    fn deref(&self) -> &Arc<QMutex<PipeIopsInternal>> {
        &self.0
    }
}

impl InodeOperations for PipeIops {
    fn as_any(&self) -> &Any {
        return self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::PipeIops;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::Pipe;
    }

    fn InodeFileType(&self) -> InodeFileType{
        return InodeFileType::Pipe;
    }

    fn WouldBlock(&self) -> bool {
        return true;
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

    fn GetFile(&self, task: &Task, _dir: &Inode, _dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let i = self.lock();

        if flags.Read && !flags.Write { // O_RDONLY
            let r = i.p.Open(task, &flags);
            i.p.rWakeup.Broadcast();

            if i.p.isNamed && !flags.NonBlocking && !i.p.HasWriters() {
                let wWakeup = i.p.wWakeup.clone();
                core::mem::drop(i);
                wWakeup.Wait(task)?
            }

            return Ok(r)
        } else if flags.Write && !flags.Read { // O_WRONLY
            let w = i.p.Open(task, &flags);
            i.p.wWakeup.Broadcast();

            if i.p.isNamed && !i.p.HasReaders() {
                // On a nonblocking, write-only open, the open fails with ENXIO if the
                // read side isn't open yet.
                if flags.NonBlocking {
                    return Err(Error::SysError(SysErr::ENXIO))
                }

                let rWakeup = i.p.rWakeup.clone();
                core::mem::drop(i);
                rWakeup.Wait(task)?;
            }

            return Ok(w)
        } else if flags.Write && flags.Read { // O_RDWR
            let rw = i.p.Open(task, &flags);
            i.p.rWakeup.Broadcast();
            i.p.wWakeup.Broadcast();

            return Ok(rw)
        } else {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    }

    fn UnstableAttr(&self, _task: &Task, _dir: &Inode) -> Result<UnstableAttr> {
        let u = self.lock().unstable;
        return Ok(u)
    }

    fn Getxattr(&self, _dir: &Inode, _name: &str) -> Result<String> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP))
    }

    fn Setxattr(&self, _dir: &mut Inode, _name: &str, _value: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP))
    }

    fn Listxattr(&self, _dir: &Inode) -> Result<Vec<String>> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP))
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms)
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, p: FilePermissions) -> bool {
        self.lock().unstable.SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.lock().unstable.SetOwner(task, owner);
        return Ok(())
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.lock().unstable.SetTimestamps(task, ts);
        return Ok(())
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

    fn AddLink(&self, _task: &Task) {
        self.lock().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.lock().unstable.Links -= 1;
    }

    fn IsVirtual(&self) -> bool {
        return false;
    }

    fn Sync(&self) -> Result<()> {
        return Ok(())
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Ok(FsInfo {
            Type: FSMagic::PIPEFS_MAGIC,
            TotalBlocks: 0,
            FreeBlocks: 0,
            TotalFiles: 0,
            FreeFiles: 0,
        })
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}
