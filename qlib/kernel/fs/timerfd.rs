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
use core::slice;

use super::super::super::common::*;
use super::super::super::linux::time::*;
use super::super::super::linux_def::*;
use super::super::kernel::time::*;
use super::super::kernel::timer::timer::*;
use super::super::kernel::timer::*;
use super::super::kernel::waiter::*;
use super::super::task::*;

use super::anon::*;
use super::attr::*;
use super::dentry::*;
use super::dirent::*;
use super::file::*;
use super::flags::*;
use super::host::hostinodeop::*;

// Constants for eventfd2(2).
pub const EFD_SEMAPHORE: i32 = 0x1;
pub const EFD_CLOEXEC: i32 = Flags::O_CLOEXEC;
pub const EFD_NONBLOCK: i32 = Flags::O_NONBLOCK;

pub struct TimerOperationsInternal {
    // Queue is used to notify interested parties when the event object
    // becomes readable or writable.
    pub queue: Queue,

    // val is the number of timer expirations since the last successful call to
    // Readv, Preadv, or SetTime. val is accessed using atomic memory
    // operations.
    pub val: QMutex<u64>,
}

impl TimerOperationsInternal {
    pub fn New() -> Self {
        return Self {
            queue: Queue::default(),
            val: QMutex::new(0),
        };
    }
}

impl TimerListenerTrait for TimerOperationsInternal {
    // Notify implements TimerListener.Notify.
    fn Notify(&self, exp: u64) {
        {
            *self.val.lock() += exp;
        };

        self.queue.Notify(READABLE_EVENT);
    }

    // Destroy implements ktime.TimerListener.Destroy.
    fn Destroy(&self) {}
}

pub fn NewTimerfd(task: &Task, clockId: i32) -> Result<File> {
    // name matches fs/eventfd.c:eventfd_file_create.
    let inode = NewAnonInode(task);
    let dirent = Dirent::New(&inode, "anon_inode:[timerfd]");

    let internal = Arc::new(TimerOperationsInternal::New());

    let clock = match clockId {
        CLOCK_MONOTONIC => MONOTONIC_CLOCK.clone(),
        CLOCK_REALTIME => REALTIME_CLOCK.clone(),
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    };

    let timer = Timer::New(&clock, TimerListener::TimerOperations(internal.clone()));

    let tops = TimerOperations {
        ops: internal,
        timer: timer,
    };

    // Timerfds reject writes, but the Write flag must be set in order to
    // ensure that our Writev/Pwritev methods actually get called to return
    // the correct errors.
    return Ok(File::New(
        &dirent,
        &FileFlags {
            Read: true,
            Write: true,
            ..Default::default()
        },
        tops,
    ));
}

pub struct TimerOperations {
    pub ops: Arc<TimerOperationsInternal>,
    pub timer: Timer,
}

impl TimerOperations {
    // PauseTimer pauses the associated Timer.
    pub fn PauseTimer(&self) {
        self.timer.Pause();
    }

    // ResumeTimer resumes the associated Timer.
    pub fn ResumeTimer(&self) {
        self.timer.Resume();
    }

    // Clock returns the associated Timer's Clock.
    pub fn Clock(&self) -> Clock {
        return self.timer.Clock();
    }

    // GetTime returns the associated Timer's setting and the time at which it was
    // observed.
    pub fn GetTime(&self) -> (Time, Setting) {
        return self.timer.Get();
    }

    // SetTime atomically changes the associated Timer's setting, resets the number
    // of expirations to 0, and returns the previous setting and the time at which
    // it was observed.
    pub fn SetTime(&self, s: &Setting) -> (Time, Setting) {
        self.timer.SwapAnd(s, || {
            *self.ops.val.lock() = 0;
        })
    }

    pub fn SwapVal(&self, val: u64) -> u64 {
        let mut v = self.ops.val.lock();
        let old = *v;
        *v = val;
        return old;
    }
}

impl Waitable for TimerOperations {
    // Readiness returns the ready events for the event fd.
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let mut ready = 0;
        let val = *self.ops.val.lock();
        if val != 0 {
            ready |= READABLE_EVENT;
        }

        return mask & ready;
    }

    // EventRegister implements waiter.Waitable.EventRegister.
    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let q = self.ops.queue.clone();
        q.EventRegister(task, e, mask)
    }

    // EventUnregister implements waiter.Waitable.EventUnregister.
    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let q = self.ops.queue.clone();
        q.EventUnregister(task, e)
    }
}

impl SpliceOperations for TimerOperations {}

impl FileOperations for TimerOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::TimerOperations;
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
        if IoVec::NumBytes(dsts) < 8 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let val = self.SwapVal(0);

        if val > 0 {
            let ptr = &val as *const _ as u64 as *const u8;
            let buf = unsafe { slice::from_raw_parts(ptr, 8) };
            task.CopyDataOutToIovs(buf, dsts)?;

            return Ok(8);
        }

        return Err(Error::SysError(SysErr::EAGAIN));
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
        return Err(Error::SysError(SysErr::ESPIPE));
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

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for TimerOperations {}
