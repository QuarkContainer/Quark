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

use core::any::Any;

use super::super::super::super::kernel::time::*;
use super::super::super::super::qlib::common::*;
use super::super::super::super::kernel::waiter::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::task::*;

use super::super::super::file::*;
use super::super::super::attr::*;
use super::super::super::dirent::*;
//use super::super::super::flags::*;
use super::super::super::dentry::*;
use super::super::super::host::hostinodeop::*;
use super::*;

pub enum FileOptionsData {
    None,
}

pub type Seek = fn(data: &FileOptionsData, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64>;
pub type ReadDir = fn(data: &FileOptionsData, task: &Task, f: &File, offset: i64, serializer: &mut DentrySerializer) -> Result<i64>;
pub type ReadAt = fn(data: &FileOptionsData, task: &Task, f: &File, dsts: &mut [IoVec], offset: i64, _blocking: bool) -> Result<i64>;
pub type WriteAt = fn(data: &FileOptionsData, task: &Task, f: &File, srcs: &[IoVec], offset: i64, _blocking: bool) -> Result<i64>;
pub type Fsync = fn(data: &FileOptionsData, task: &Task, f: &File, start: i64, end: i64, syncType: SyncType) -> Result<()>;
pub type Flush = fn(data: &FileOptionsData, task: &Task, f: &File) -> Result<()>;

pub type UnstableAttrFn = fn(data: &FileOptionsData, task: &Task, f: &File) -> Result<UnstableAttr>;
pub type Ioctl = fn(data: &FileOptionsData, task: &Task, f: &File, fd: i32, request: u64, val: u64) -> Result<()>;

pub type IterateDir = fn(task: &Task, data: &FileOptionsData, d: &Dirent, dirCtx: &mut DirCtx, offset: i32) -> (i32, Result<i64>);

pub type Mmap = fn(data: &FileOptionsData, _task: &Task, _f: &File, len: u64, hugePage: bool, offset: u64, share: bool, prot: u64) -> Result<u64>;

pub type Connect = fn(data: &FileOptionsData, task: &Task, socketaddr: &[u8], blocking: bool) -> Result<i64>;
pub type Accept = fn(data: &FileOptionsData, task: &Task, addr: &mut [u8], addrlen: &mut u32, flags: i32, _blocking: bool) -> Result<i64>;
pub type Bind = fn(data: &FileOptionsData, task: &Task, sockaddr: &[u8]) -> Result<i64>;
pub type Listen = fn(data: &FileOptionsData, task: &Task, backlog: i32) -> Result<i64>;
pub type Shutdown = fn(data: &FileOptionsData, task: &Task, how: i32) -> Result<i64>;
pub type GetSockOpt = fn(data: &FileOptionsData, task: &Task, level: i32, name: i32, addr: &mut [u8]) -> Result<i64>;
pub type SetSockOpt = fn(data: &FileOptionsData, task: &Task, level: i32, name: i32, opt: &[u8]) -> Result<i64>;
pub type GetSockName = fn(data: &FileOptionsData, task: &Task, socketaddr: &mut [u8]) -> Result<i64>;
pub type GetPeerName = fn(data: &FileOptionsData, task: &Task, socketaddr: &mut [u8]) -> Result<i64>;
pub type RecvMsg = fn(data: &FileOptionsData, _task: &Task, dst: &mut [IoVec], flags: i32, msgHdr: &mut MsgHdr, deadline: Option<Time>) -> Result<i64>;
pub type SendMsg = fn(data: &FileOptionsData, _task: &Task, src: &[IoVec], flags: i32, msgHdr: &mut MsgHdr, deadline: Option<Time>) -> Result<i64>;
pub type SetRecvTimeout = fn(data: &FileOptionsData, nanoseconds: i64);
pub type RecvTimeout = fn(data: &FileOptionsData) -> i64;
pub type SetSendTimeout = fn(data: &FileOptionsData, nanoseconds: i64);
pub type SendTimeout = fn(data: &FileOptionsData) -> i64;
pub type Mappable = fn(data: &FileOptionsData) -> Result<HostInodeOp>;
pub type Readiness = fn(data: &FileOptionsData, mask: EventMask) -> EventMask;
pub type EventRegister = fn(data: &FileOptionsData, e: &WaitEntry, mask: EventMask);
pub type EventUnregister = fn(data: &FileOptionsData, e: &WaitEntry);

pub fn FileGenericSeek_Seek(_data: &FileOptionsData, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
    return SeekWithDirCursor(task, f, whence, current, offset, None)
}

pub fn FileZeroSeek_Seek(_data: &FileOptionsData, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
    return Ok(0)
}

pub fn FileNoSeek_Seek(_data: &FileOptionsData, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
    return Err(Error::SysError(SysErr::EINVAL))
}

pub fn FilePipeSeek_Seek(_data: &FileOptionsData, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
    return Err(Error::SysError(SysErr::ESPIPE))
}

pub fn FileNotDir_Readdir(_data: &FileOptionsData, _task: &Task, _f: &File, _offset: i64, _serializer: &mut DentrySerializer) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTDIR))
}

pub fn FileNotDir_IterateDir(_task: &Task, _data: &FileOptionsData, _d: &Dirent, _dirCtx: &mut DirCtx, _offset: i32) -> (i32, Result<i64>) {
    return (0, Err(Error::SysError(SysErr::ENOTDIR)))
}

pub fn FileNoFsync_Fsync(_data: &FileOptionsData, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
    return Err(Error::SysError(SysErr::EINVAL))
}

pub fn FileNoopFsync_Fsync(_data: &FileOptionsData, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
    return Ok(())
}

pub fn FileNoopFlush_Flush(_data: &FileOptionsData, _task: &Task, _f: &File) -> Result<()> {
    return Ok(())
}

pub fn FileNoIoctl_Ioctl(_data: &FileOptionsData, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTTY))
}

pub fn FileNoSplice_ReadAt(_data: &FileOptionsData, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

pub fn FileNoSplice_WriteAt(_data: &FileOptionsData, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

pub fn FileNoopWrite_WriteAt(_data: &FileOptionsData, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    //return Ok(srcs.NumBytes() as i64)
    panic!("FileNoopWrite_WriteAt: need implement");
}

pub fn FileNoWrite_WriteAt(_data: &FileOptionsData, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    //return Ok(srcs.NumBytes() as i64)
    panic!("FileNoWrite_WriteAt: need implement");
}

pub fn FileNoopRead_ReadAt(_data: &FileOptionsData, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    return Ok(0)
}

fn FileNoRead_ReadAt(_data: &FileOptionsData, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    return Err(Error::SysError(SysErr::EINVAL))
}

fn DirFileOperations_Seek(_data: &FileOptionsData, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
    return SeekWithDirCursor(task, f, whence, current, offset, None)
}

fn DirFileOperations_ReadAt(_data: &FileOptionsData, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

fn DirFileOperations_WriteAt(_data: &FileOptionsData, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS))
}

fn DirFileOperations_Fsync(_data: &FileOptionsData, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
    return Ok(())
}

fn DirFileOperations_Flush(_data: &FileOptionsData, _task: &Task, _f: &File) -> Result<()> {
    return Ok(())
}

fn DirFileOperations_Ioctl(_data: &FileOptionsData, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
    return Err(Error::SysError(SysErr::ENOTTY))
}

fn FileUseInodeUnstableAttr_UnstableAttr(_data: &FileOptionsData, task: &Task, f: &File) -> Result<UnstableAttr> {
    let inode = f.Dirent.Inode();
    return inode.UnstableAttr(task);
}

fn FileNoMMap_Mmap(_data: &FileOptionsData, _task: &Task, _f: &File, _len: u64, _hugePage: bool, _offset: u64, _share: bool, _prot: u64) -> Result<u64> {
    return Err(Error::SysError(SysErr::ENODEV))
}

fn FileNoMMap_Mappable(_data: &FileOptionsData) -> Result<HostInodeOp> {
    return Err(Error::SysError(SysErr::ENODEV))
}

fn AlwaysReady_Readiness(_data: &FileOptionsData, mask: EventMask) -> EventMask {
    return mask
}

fn AlwaysReady_EventRegister(_data: &FileOptionsData, _e: &WaitEntry, _mask: EventMask) {
}

fn AlwaysReady_EventUnregister(_data: &FileOptionsData, _e: &WaitEntry) {
}

fn NoSock_Connect(_data: &FileOptionsData, _task: &Task, _socketaddr: &[u8], _blocking: bool) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_Accept(_data: &FileOptionsData, _task: &Task, _addr: &mut [u8], _addrlen: &mut u32, _flags: i32, _blocking: bool) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_Bind(_data: &FileOptionsData, _task: &Task, _sockaddr: &[u8]) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_Listen(_data: &FileOptionsData, _task: &Task, _backlog: i32) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_Shutdown(_data: &FileOptionsData, _task: &Task, _how: i32) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_GetSockOpt(_data: &FileOptionsData, _task: &Task, _level: i32, _name: i32, _addr: &mut [u8]) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_SetSockOpt(_data: &FileOptionsData, _task: &Task, _level: i32, _name: i32, _opt: &[u8]) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_GetSockName(_data: &FileOptionsData, _task: &Task, _socketaddr: &mut [u8]) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_GetPeerName(_data: &FileOptionsData, _task: &Task, _socketaddr: &mut [u8]) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_RecvMsg(_data: &FileOptionsData, _task: &Task, _dst: &mut [IoVec], _flags: i32, _msgHdr: &mut MsgHdr, _deadline: Option<Time>) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_SendMsg(_data: &FileOptionsData, _task: &Task, _src: &[IoVec], _flags: i32, _msgHdr: &mut MsgHdr, _deadline: Option<Time>) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOTSOCK))
}

fn NoSock_SetRecvTimeout(_data: &FileOptionsData, _nanoseconds: i64) {
    return
}

fn NoSock_RecvTimeout(_data: &FileOptionsData) -> i64 {
    return 0
}

fn NoSock_SetSendTimeout(_data: &FileOptionsData, _nanoseconds: i64) {
    return
}

fn NoSock_SendTimeout(_data: &FileOptionsData) -> i64 {
    return 0
}

pub struct FileOptionsUtil {
    pub data: FileOptionsData,

    pub seek    : Seek,
    pub readDir : ReadDir,
    pub readAt  : ReadAt,
    pub writeAt : WriteAt,
    pub fsync   : Fsync,
    pub flush   : Flush,
    pub unstableAttr: UnstableAttrFn,
    pub ioctl   : Ioctl,
    pub iterateDir: IterateDir,
    pub mmap    : Mmap,
    pub connect : Connect,
    pub accept  : Accept,
    pub bind    : Bind,
    pub listen  : Listen,
    pub shutdown: Shutdown,
    pub getSockOpt: GetSockOpt,
    pub setSockOpt: SetSockOpt,
    pub getSockName: GetSockName,
    pub getPeerName: GetPeerName,
    pub recvMsg: RecvMsg,
    pub sendMsg: SendMsg,
    pub setRecvTimeout: SetRecvTimeout,
    pub recvTimeout: RecvTimeout,
    pub setSendTimeout: SetSendTimeout,
    pub sendTimeout: SendTimeout,
    pub mappable: Mappable,
    pub readiness: Readiness,
    pub eventRegister: EventRegister,
    pub eventUnregister: EventUnregister,
}


impl Default for FileOptionsUtil {
    fn default() -> Self {
        return Self {
            data: FileOptionsData::None,

            seek: FileGenericSeek_Seek,
            readDir: FileNotDir_Readdir,
            iterateDir: FileNotDir_IterateDir,
            readAt: FileNoRead_ReadAt,
            writeAt: FileNoWrite_WriteAt,
            fsync: FileNoFsync_Fsync,
            flush: FileNoopFlush_Flush,
            unstableAttr: FileUseInodeUnstableAttr_UnstableAttr,
            ioctl: FileNoIoctl_Ioctl,
            mmap: FileNoMMap_Mmap,
            mappable: FileNoMMap_Mappable,

            connect: NoSock_Connect,
            accept: NoSock_Accept,
            bind: NoSock_Bind,
            listen: NoSock_Listen,
            shutdown: NoSock_Shutdown,
            getSockOpt: NoSock_GetSockOpt,
            setSockOpt: NoSock_SetSockOpt,
            getSockName: NoSock_GetSockName,
            getPeerName: NoSock_GetPeerName,
            recvMsg: NoSock_RecvMsg,
            sendMsg: NoSock_SendMsg,
            setRecvTimeout: NoSock_SetRecvTimeout,
            recvTimeout: NoSock_RecvTimeout,
            setSendTimeout: NoSock_SetSendTimeout,
            sendTimeout: NoSock_SendTimeout,

            readiness: AlwaysReady_Readiness,
            eventRegister: AlwaysReady_EventRegister,
            eventUnregister: AlwaysReady_EventUnregister,
        }
    }
}

impl FileOptionsUtil {
    pub fn SetFileGenericSeek(&mut self) {
        self.seek = FileGenericSeek_Seek;
    }

    pub fn SetFileZeroSeek(&mut self) {
        self.seek = FileZeroSeek_Seek;
    }

    pub fn SetFileNoSeek(&mut self) {
        self.seek = FileNoSeek_Seek;
    }

    pub fn SetFilePipeSeek(&mut self) {
        self.seek = FilePipeSeek_Seek;
    }

    pub fn SetFileNotDir(&mut self) {
        self.readDir = FileNotDir_Readdir;
        self.iterateDir = FileNotDir_IterateDir;
    }

    pub fn SetFileNoFsync(&mut self) {
        self.fsync = FileNoFsync_Fsync;
    }

    pub fn SetFileNoopFsync(&mut self) {
        self.fsync = FileNoopFsync_Fsync;
    }

    pub fn SetFileNoopFlush(&mut self) {
        self.flush = FileNoopFlush_Flush;
    }

    pub fn SetFileNoIoctl(&mut self) {
        self.ioctl = FileNoIoctl_Ioctl;
    }

    pub fn SetFileNoSplice(&mut self) {
        self.readAt = FileNoSplice_ReadAt;
        self.writeAt = FileNoSplice_WriteAt;
    }

    pub fn SetFileNoopWrite(&mut self) {
        self.writeAt = FileNoopWrite_WriteAt;
    }

    pub fn SetFileNoWrite(&mut self) {
        self.writeAt = FileNoWrite_WriteAt;
    }

    pub fn SetFileNoopRead(&mut self) {
        self.readAt = FileNoopRead_ReadAt;
    }

    pub fn SetFileNoRead(&mut self) {
        self.readAt = FileNoRead_ReadAt;
    }

    pub fn SetDirFileOperations(&mut self) {
        self.seek = DirFileOperations_Seek;
        self.readAt = DirFileOperations_ReadAt;
        self.writeAt = DirFileOperations_WriteAt;
        self.flush = DirFileOperations_Flush;
        self.ioctl = DirFileOperations_Ioctl;
    }

    pub fn SetFileUseInodeUnstableAttr(&mut self) {
        self.unstableAttr = FileUseInodeUnstableAttr_UnstableAttr;
    }

    pub fn SetNoMMap(&mut self) {
        self.mmap = FileNoMMap_Mmap;
        self.mappable = FileNoMMap_Mappable;
    }

    pub fn SetAlwaysReady(&mut self) {
        self.readiness = AlwaysReady_Readiness;
        self.eventRegister = AlwaysReady_EventRegister;
        self.eventUnregister = AlwaysReady_EventUnregister;
    }

    pub fn SetNoSock(&mut self) {
        self.connect = NoSock_Connect;
        self.accept = NoSock_Accept;
        self.bind = NoSock_Bind;
        self.listen = NoSock_Listen;
        self.shutdown = NoSock_Shutdown;
        self.getSockOpt = NoSock_GetSockOpt;
        self.setSockOpt = NoSock_SetSockOpt;
        self.getSockName = NoSock_GetSockName;
        self.getPeerName = NoSock_GetPeerName;
        self.recvMsg = NoSock_RecvMsg;
        self.sendMsg = NoSock_SendMsg;
        self.setRecvTimeout = NoSock_SetRecvTimeout;
        self.recvTimeout = NoSock_RecvTimeout;
        self.setSendTimeout = NoSock_SetSendTimeout;
        self.sendTimeout = NoSock_SendTimeout;
    }
}

impl SpliceOperations for FileOptionsUtil {}

impl FileOperations for FileOptionsUtil {
    fn as_any(&self) -> &Any {
        return self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::FileOptionsUtil
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return (self.seek)(&self.data, task, f, whence, current, offset)
    }

    fn ReadDir(&self, task: &Task, f: &File, offset: i64, serializer: &mut DentrySerializer) -> Result<i64> {
        return (self.readDir)(&self.data, task, f, offset, serializer)
    }

    fn ReadAt(&self, task: &Task, f: &File, dsts: &mut [IoVec], offset: i64, blocking: bool) -> Result<i64> {
        return (self.readAt)(&self.data, task, f, dsts, offset, blocking)
    }

    fn WriteAt(&self, task: &Task, f: &File, srcs: &[IoVec], offset: i64, blocking: bool) -> Result<i64> {
        return (self.writeAt)(&self.data, task, f, srcs, offset, blocking)
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0))
    }

    fn Fsync(&self, task: &Task, f: &File, start: i64, end: i64, syncType: SyncType) -> Result<()> {
        return (self.fsync)(&self.data, task, f, start, end, syncType)
    }

    fn Flush(&self, task: &Task, f: &File) -> Result<()> {
        return (self.flush)(&self.data, task, f)
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        return (self.unstableAttr)(&self.data, task, f)
    }

    fn Ioctl(&self, task: &Task, f: &File, fd: i32, request: u64, val: u64) -> Result<()> {
        return (self.ioctl)(&self.data, task, f, fd, request, val)
    }

    fn IterateDir(&self, task: &Task,d: &Dirent, dirCtx: &mut DirCtx, offset: i32) -> (i32, Result<i64>) {
        return (self.iterateDir)(task, &self.data, d, dirCtx, offset)
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return (self.mappable)(&self.data)
    }
}

impl SockOperations for FileOptionsUtil {}

impl Waitable for FileOptionsUtil {
    fn Readiness(&self, _task: &Task,mask: EventMask) -> EventMask {
        return (self.readiness)(&self.data, mask)
    }

    fn EventRegister(&self, _task: &Task,e: &WaitEntry, mask: EventMask) {
        return (self.eventRegister)(&self.data, e, mask)
    }

    fn EventUnregister(&self, _task: &Task,e: &WaitEntry) {
        return (self.eventUnregister)(&self.data, e)
    }
}