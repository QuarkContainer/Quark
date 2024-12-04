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
use core::slice;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::mem::seq::*;
use super::super::kernel::waiter::*;
use super::super::task::*;

use super::super::fs::anon::*;
use super::super::fs::attr::*;
use super::super::fs::dentry::*;
use super::super::fs::dirent::*;
use super::super::fs::file::*;
use super::super::fs::flags::*;
use super::super::fs::host::hostinodeop::*;

use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::GuestHostSharedAllocator;

// Constants for eventfd2(2).
pub const EFD_SEMAPHORE: i32 = 0x1;
pub const EFD_CLOEXEC: i32 = Flags::O_CLOEXEC;
pub const EFD_NONBLOCK: i32 = Flags::O_NONBLOCK;

pub struct EventOperationsInternal {
    // Queue is used to notify interested parties when the event object
    // becomes readable or writable.
    pub wq: Queue,

    // val is the current value of the event counter.
    pub val: u64,

    // semMode specifies whether the event is in "semaphore" mode.
    pub semMode: bool,
}

pub fn NewEventfd(task: &Task, initVal: u64, semMode: bool) -> File {
    // name matches fs/eventfd.c:eventfd_file_create.
    let inode = NewAnonInode(task);
    let dirent = Dirent::New(&inode, "anon_inode:[eventfd]");

    let internal = EventOperationsInternal {
        wq: Queue::default(),
        val: initVal,
        semMode: semMode,
    };

    let ops = EventOperations(Arc::new_in(QMutex::new(internal), GUEST_HOST_SHARED_ALLOCATOR));

    return File::New(
        &dirent,
        &FileFlags {
            Read: true,
            Write: true,
            ..Default::default()
        },
        ops.into(),
    );
}

#[derive(Clone)]
pub struct EventOperations(Arc<QMutex<EventOperationsInternal>, GuestHostSharedAllocator>);
impl Deref for EventOperations {
    type Target = Arc<QMutex<EventOperationsInternal>, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QMutex<EventOperationsInternal>, GuestHostSharedAllocator> {
        &self.0
    }
}

impl EventOperations {
    pub fn Read(&self, _task: &Task, dst: BlockSeq) -> Result<()> {
        let val: u64;

        {
            let mut e = self.lock();

            if e.val == 0 {
                return Err(Error::SysError(SysErr::EAGAIN));
            }

            // Update the value based on the mode the event is operating in.
            if e.semMode {
                val = 1;
                // Consistent with Linux, this is done even if writing to memory fails.
                e.val -= 1;
            } else {
                val = e.val;
                e.val = 0;
            }
        }

        // Notify writers. We do this even if we were already writable because
        // it is possible that a writer is waiting to write the maximum value
        // to the event.
        let queue = self.lock().wq.clone();
        queue.Notify(WRITEABLE_EVENT);

        let ptr = &val as *const _ as u64 as *const u8;
        let buf = unsafe { slice::from_raw_parts(ptr, 8) };
        dst.CopyOut(buf);

        return Ok(());
    }

    pub fn Write(&self, _task: &Task, src: BlockSeq) -> Result<()> {
        let mut val: u64 = 0;
        let ptr = &mut val as *mut _ as u64 as *mut u8;
        let buf = unsafe { slice::from_raw_parts_mut(ptr, 8) };
        src.CopyIn(buf);

        return self.Signal(val);
    }

    pub fn Signal(&self, val: u64) -> Result<()> {
        if val == core::u64::MAX {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        {
            let mut e = self.lock();

            // We only allow writes that won't cause the value to go over the max
            // uint64 minus 1.
            if val > core::u64::MAX - 1 - e.val {
                return Err(Error::SysError(SysErr::EAGAIN));
            }

            e.val += val;
        }

        let queue = self.lock().wq.clone();
        queue.Notify(READABLE_EVENT);

        return Ok(());
    }
}

impl Waitable for EventOperations {
    // Readiness returns the ready events for the event fd.
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let e = self.lock();

        let mut ready = 0;
        if e.val > 0 {
            ready |= READABLE_EVENT;
        }

        if e.val < core::u64::MAX - 1 {
            ready |= WRITEABLE_EVENT;
        }

        return mask & ready;
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let q = self.lock().wq.clone();
        q.EventRegister(task, e, mask)
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let q = self.lock().wq.clone();
        q.EventUnregister(task, e)
    }
}

impl SpliceOperations for EventOperations {}

impl FileOperations for EventOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::EventOperations;
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
        let size = IoVec::NumBytes(dsts);
        if size < 8 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let buf = DataBuff::New(size);
        self.Read(task, buf.BlockSeq())?;
        task.CopyDataOutToIovs(&buf.buf, dsts, false)?;
        return Ok(8);
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
        if size < 8 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;

        self.Write(task, buf.BlockSeqWithLen(len))?;
        return Ok(8);
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

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<u64> {
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

impl SockOperations for EventOperations {}
