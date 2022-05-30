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
use core::ops::Deref;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::uid::NewUID;
use super::super::attr::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::inotify::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::super::mount::*;
use super::dir::*;
use super::terminal::*;

pub fn NewMasterNode(
    task: &Task,
    d: &DirInodeOperations,
    owner: &FileOwner,
    p: &FilePermissions,
) -> Inode {
    let unstable = WithCurrentTime(
        task,
        &UnstableAttr {
            Owner: *owner,
            Perms: *p,
            ..Default::default()
        },
    );

    let iops = MasterInodeOperations(Arc::new(QRwLock::new(MasterInodeOperationsInternal {
        fsType: FSMagic::DEVPTS_SUPER_MAGIC,
        unstable: unstable,
        d: d.clone(),
    })));

    let deviceId = PTS_DEVICE.lock().id.DeviceID();
    let inodeId = PTS_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::CharacterDevice,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: 1024,
        DeviceFileMajor: TTYAUX_MAJOR,
        DeviceFileMinor: PTMX_MINOR,
    };

    let msrc = d.lock().msrc.clone();
    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: Arc::new(iops),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        Watches: Watches::default(),
        MountSource: msrc,
        Overlay: None,
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

pub struct MasterInodeOperationsInternal {
    pub fsType: u64,
    pub unstable: UnstableAttr,
    pub d: DirInodeOperations,
}

#[derive(Clone)]
pub struct MasterInodeOperations(Arc<QRwLock<MasterInodeOperationsInternal>>);

impl Deref for MasterInodeOperations {
    type Target = Arc<QRwLock<MasterInodeOperationsInternal>>;

    fn deref(&self) -> &Arc<QRwLock<MasterInodeOperationsInternal>> {
        &self.0
    }
}

impl InodeOperations for MasterInodeOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::MasterInodeOperations;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::CharacterDevice;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::TTYMaster;
    }

    fn WouldBlock(&self) -> bool {
        return true;
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }

    fn Getxattr(&self, _dir: &Inode, _name: &str) -> Result<String> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Setxattr(&self, _dir: &mut Inode, _name: &str, _value: &str) -> Result<()> {
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

    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        _dirent: &Dirent,
        _flags: FileFlags,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::ENXIO));
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

pub struct MasterFileOperations {
    pub d: DirInodeOperations,
    pub t: Arc<Terminal>,
}

impl Waitable for MasterFileOperations {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        panic!("MasterFileOperations doesn't support Waitable::Readiness");
    }

    fn EventRegister(&self, _task: &Task, _e: &WaitEntry, _mask: EventMask) {
        panic!("MasterFileOperations doesn't support Waitable::EventRegister");
    }

    fn EventUnregister(&self, _task: &Task, _e: &WaitEntry) {
        panic!("MasterFileOperations doesn't support Waitable::EventUnregister");
    }
}

impl SpliceOperations for MasterFileOperations {}

impl FileOperations for MasterFileOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::MasterFileOperations;
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
        return Err(Error::SysError(SysErr::ESPIPE));
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
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let mut buf: [u8; 4096] = [0; 4096];

        let mut size = IoVec::NumBytes(dsts);
        if size > buf.len() {
            size = buf.len();
        }

        let cnt = self
            .t
            .ld
            .lock()
            .OutputQueueRead(task, &mut buf[..size as usize])? as usize;
        let res = task.CopyDataOutToIovs(&buf[0..cnt], dsts, false)?;

        assert!(res == cnt as usize, "MasterFileOperations:ReadAt fail");
        return Ok(res as i64);
    }

    fn WriteAt(
        &self,
        task: &Task,
        _f: &File,
        srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let size = IoVec::NumBytes(srcs);
        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;

        let res = self
            .t
            .ld
            .lock()
            .InputQueueWrite(task, &mut buf.buf[0..len as usize])?;
        return Ok(res);
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
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn Ioctl(&self, task: &Task, _f: &File, _fd: i32, request: u64, val: u64) -> Result<()> {
        let cmd = request;
        match cmd {
            IoCtlCmd::FIONREAD => return self.t.ld.lock().OutputQueueReadSize(task, val),
            IoCtlCmd::TCGETS => return self.t.ld.lock().GetTermios(task, val),
            IoCtlCmd::TCSETS => return self.t.ld.lock().SetTermios(task, val),
            IoCtlCmd::TCSETSW => {
                //This should drain the output queue first.
                return self.t.ld.lock().SetTermios(task, val);
            }
            IoCtlCmd::TIOCGPTN => {
                let n = self.t.n;
                task.CopyOutObj(&n, val)?;
                return Ok(());
            }
            IoCtlCmd::TIOCSPTLCK => {
                //Implement pty locking. For now just pretend we do.
                return Ok(());
            }
            IoCtlCmd::TIOCGWINSZ => {
                //This should drain the output queue first.
                return self.t.ld.lock().GetWindowSize(task, val);
            }
            IoCtlCmd::TIOCSWINSZ => {
                //This should drain the output queue first.
                return self.t.ld.lock().SetWindowSize(task, val);
            }
            _ => return Err(Error::SysError(SysErr::ENOTTY)),
        }
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

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for MasterFileOperations {}
