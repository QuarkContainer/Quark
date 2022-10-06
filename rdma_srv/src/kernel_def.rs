// Copyright (c) 2021 Quark Container Authors
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

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::qlib::common::*;
use super::qlib::control_msg::*;
use super::qlib::kernel::memmgr::pma::*;
use super::qlib::kernel::quring::uring_async::UringAsyncMgr;
use super::qlib::kernel::task::*;
use super::qlib::kernel::Kernel::*;
use super::qlib::loader::*;
use super::qlib::mem::list_allocator::*;
use super::qlib::mutex::*;
use super::qlib::perf_tunning::*;
use super::qlib::qmsg::*;
use super::qlib::task_mgr::*;
use super::qlib::vcpu_mgr::*;
use super::qlib::*;
use crate::qlib::fileinfo::*;

impl<'a> ShareSpace {
    pub fn AQCall(&self, _msg: &HostOutputMsg) {}

    pub fn Schedule(&self, _taskId: u64) {}
}

impl<'a> ShareSpace {
    pub fn LogFlush(&self, _partial: bool) {}
}

impl ShareSpace {
    pub fn Init(&mut self, _vcpuCount: usize, _controlSock: i32) {}

    pub fn TlbShootdown(&self, _vcpuMask: u64) -> i64 {
        return 0;
    }

    pub fn Yield() {}

    pub fn CheckVcpuTimeout(&self) {}
}

impl<T: ?Sized> QMutexIntern<T> {
    pub fn GetID() -> u64 {
        0
    }
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfType {
    Start,
    Other,
    QCall,
    AQCall,
    AQHostCall,
    BusyWait,
    IdleWait,
    BufWrite,
    End,
    User, //work around for kernel clone
    Idle, //work around for kernel clone

    ////////////////////////////////////////
    Blocked,
    Kernel,
}

impl CounterSet {
    pub const PERM_COUNTER_SET_SIZE: usize = 1;
    pub fn GetPerfId(&self) -> usize {
        0
    }

    pub fn PerfType(&self) -> &str {
        return "PerfPrint::Host";
    }
}

pub fn switch(_from: TaskId, _to: TaskId) {}

pub fn OpenAt(_task: &Task, _dirFd: i32, _addr: u64, _flags: u32) -> Result<i32> {
    return Ok(0);
}

pub fn SignalProcess(_signalArgs: &SignalArgs) {}

pub fn StartRootContainer(_para: *const u8) {}
pub fn StartExecProcess(_fd: i32, _process: Process) {}
pub fn StartSubContainerProcess(_elfEntry: u64, _userStackAddr: u64, _kernelStackAddr: u64) {}

pub unsafe fn CopyPageUnsafe(_to: u64, _from: u64) {}

impl CPULocal {
    pub fn CpuId() -> usize {
        return 0;
    }

    pub fn Wakeup(&self) {}
}

impl PageMgr {
    pub fn CopyVsysCallPages(&self, _addr: u64) {}
}

pub fn ClockGetTime(_clockId: i32) -> i64 {
    0
}

pub fn VcpuFreq() -> i64 {
    0
}

pub fn NewSocket(_fd: i32) -> i64 {
    0
}

pub fn UringWake(_idx: usize, _minCompleted: u64) {}

impl HostSpace {
    pub fn Close(_fd: i32) -> i64 {
        0
    }

    pub fn Call(_msg: &mut Msg, _mustAsync: bool) -> u64 {
        0
    }

    pub fn HCall(_msg: &mut Msg, _lock: bool) -> u64 {
        0
    }
}

#[inline]
pub fn child_clone(_userSp: u64) {}

pub fn InitX86FPState(_data: u64, _useXsave: bool) {}

impl OOMHandler for ListAllocator {
    fn handleError(&self, _a: u64, _b: u64) {
        panic!("qvisor OOM: Heap allocator fails to allocate memory block");
    }
}

impl ListAllocator {
    pub fn initialize(&self) {}

    pub fn Check(&self) {}
}

#[inline]
pub fn VcpuId() -> usize {
    return CPULocal::CpuId();
}

impl IOMgr {
    pub fn Init() -> Result<Self> {
        return Err(Error::Common(format!("IOMgr can't init in kernel")));
    }
}

impl UringAsyncMgr {
    pub fn FreeSlot(&self, id: usize) {
        self.freeids.lock().push_back(id as _);
    }

    pub fn Clear(&self) {
        loop {
            let id = match self.freeids.lock().pop_front() {
                None => break,
                Some(id) => id,
            };
            self.freeSlot(id as _);
        }
    }
}

impl HostAllocator {
    pub const fn New() -> Self {
        return Self {
            listHeapAddr: AtomicU64::new(0),
            initialized: AtomicBool::new(true),
        };
    }

    pub fn Init(&self, heapAddr: u64) {
        self.listHeapAddr.store(heapAddr, Ordering::SeqCst)
    }
}

unsafe impl GlobalAlloc for HostAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        return self.Allocator().alloc(layout);
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.Allocator().dealloc(ptr, layout);
    }
}

pub fn Invlpg(_addr: u64) {}

pub fn HyperCall64(_type_: u16, _para1: u64, _para2: u64, _para3: u64, _para4: u64) {}

pub fn IsKernel() -> bool {
    return false;
}