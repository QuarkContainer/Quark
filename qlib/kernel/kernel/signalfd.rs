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
use alloc::sync::Arc;
use core::any::Any;
use core::ops::Deref;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::fs::anon::*;
use super::super::fs::attr::*;
use super::super::fs::dentry::*;
use super::super::fs::dirent::*;
use super::super::fs::file::*;
use super::super::fs::flags::*;
use super::super::fs::host::hostinodeop::*;
use super::super::task::*;
use super::super::threadmgr::thread::*;
use super::super::SignalDef::*;
use super::waiter::*;

// SFD_NONBLOCK is a signalfd(2) flag.
pub const SFD_NONBLOCK: i32 = 0o00004000;

// SFD_CLOEXEC is a signalfd(2) flag.
pub const SFD_CLOEXEC: i32 = 0o02000000;

#[derive(Debug, Default)]
#[repr(C)]
pub struct SignalfdSiginfo {
    pub signo: u32,
    pub errno: i32,
    pub code: i32,
    pub pid: u32,
    pub uid: u32,
    pub fd: i32,
    pub tid: u32,
    pub band: u32,
    pub overrun: u32,
    pub trapno: u32,
    pub status: i32,
    pub int: i32,
    pub ptr: u64,
    pub utime: u64,
    pub stime: u64,
    pub addr: u64,
    pub addrlsb: u16,
    pub _pad1: [u8; 32],
    pub _pad2: [u8; 16], //[u8; 48]
}

pub struct SignalOperationInternal {
    // target is the original task target.
    //
    // The semantics here are a bit broken. Linux will always use current
    // for all reads, regardless of where the signalfd originated. We can't
    // do exactly that because we need to plumb the context through
    // EventRegister in order to support proper blocking behavior. This
    // will undoubtedly become very complicated quickly.
    pub target: Thread,

    // mask is the signal mask,
    pub mask: QMutex<SignalSet>,
}

pub struct SignalOperation(Arc<SignalOperationInternal>);

impl Deref for SignalOperation {
    type Target = Arc<SignalOperationInternal>;

    fn deref(&self) -> &Arc<SignalOperationInternal> {
        &self.0
    }
}

impl SignalOperation {
    pub fn NewSignalFile(task: &Task, mask: SignalSet) -> File {
        // name matches fs/signalfd.c:signalfd4.
        let inode = NewAnonInode(task);
        let dirent = Dirent::New(&inode, "anon_inode:[signalfd]");

        let intern = SignalOperationInternal {
            target: task.Thread(),
            mask: QMutex::new(mask),
        };

        let fops = Self(Arc::new(intern));
        return File::New(
            &dirent,
            &FileFlags {
                Read: true,
                Write: true,
                ..Default::default()
            },
            fops,
        );
    }

    pub fn Mask(&self) -> SignalSet {
        return *self.mask.lock();
    }

    pub fn SetMask(&self, mask: SignalSet) {
        *self.mask.lock() = mask
    }
}

impl SpliceOperations for SignalOperation {}

impl FileOperations for SignalOperation {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::SignalOperation;
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
        _task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let mut info = match self.target.Sigtimedwait(self.Mask(), 0) {
            Ok(info) => info,
            Err(_) => return Err(Error::SysError(SysErr::EAGAIN)),
        };

        let mut infoNative = SignalfdSiginfo {
            signo: info.Signo as u32,
            errno: info.Errno,
            code: info.Code,
            ..Default::default()
        };

        {
            let sigChild = info.SigChld();
            infoNative.pid = sigChild.pid as u32;
            infoNative.uid = sigChild.uid;
            infoNative.status = sigChild.status;
        }

        {
            let sigTimer = info.SigTimer();
            infoNative.overrun = sigTimer.overrun as u32;
        }

        {
            let sigFault = info.SigFault();
            infoNative.addr = sigFault.addr;
        }

        let addr = &infoNative as *const _ as u64;
        let len = core::mem::size_of_val(&infoNative);

        assert!(dsts.len() == 1);
        let count = dsts[0].CopyFrom(&IoVec::NewFromAddr(addr, len));

        return Ok(count as i64);
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

    fn Append(&self, _task: &Task, _f: &File, _srcs: &[IoVec]) -> Result<(i64, i64)> {
        return Err(Error::SysError(SysErr::EINVAL));
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

impl Waitable for SignalOperation {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        if mask & READABLE_EVENT != 0 && self.target.PendingSignalsNolock().0 & self.Mask().0 != 0 {
            return READABLE_EVENT;
        }

        return 0;
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, _mask: EventMask) {
        self.target.SignalRegister(task, e, self.mask.lock().0)
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        self.target.SignalUnregister(task, e);
    }
}

impl SockOperations for SignalOperation {}
