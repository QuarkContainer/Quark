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
//

use core::sync::atomic::Ordering;

use super::qlib::kernel::asm::*;
use super::qlib::kernel::taskMgr::*;
use super::qlib::kernel::threadmgr::task_sched::*;
use super::qlib::kernel::SHARESPACE;

use super::qlib::*;
use super::qlib::loader::*;
use super::qlib::qmsg::*;
use super::qlib::uring::util::*;
use super::qlib::task_mgr::*;
use super::qlib::ShareSpace;
use super::qlib::mutex::*;
use super::qlib::perf_tunning::*;
use super::qlib::vcpu_mgr::*;
use super::qlib::common::*;
use super::qlib::uring::*;
use super::qlib::control_msg::*;
use super::qlib::mem::list_allocator::*;
use super::qlib::kernel::task::*;
use super::Kernel::HostSpace;
use super::syscalls::sys_file::*;
use super::boot::controller::*;

impl IoUring {
    /// Initiate asynchronous I/O.
    #[inline]
    pub fn submit(&self, idx: usize) -> Result<usize> {
        self.submit_and_wait(idx, 0)
    }

    pub fn submit_and_wait(&self, idx: usize, want: usize) -> Result<usize> {
        let len = self.sq_len();

        let mut flags = 0;

        if want > 0 {
            flags |= sys::IORING_ENTER_GETEVENTS;
        }

        if self.params.0.flags & sys::IORING_SETUP_SQPOLL != 0 {
            if self.sq_need_wakeup() {
                if want > 0 {
                    flags |= sys::IORING_ENTER_SQ_WAKEUP;
                } else {
                    HostSpace::UringWake(idx, 0);
                    return Ok(0)
                }
            } else if want == 0 {
                // fast poll
                return Ok(len);
            }
        }

        unsafe { self.enter(idx, len as _, want as _, flags) }
    }

    pub unsafe fn enter(
        &self,
        idx: usize,
        to_submit: u32,
        min_complete: u32,
        flag: u32
    ) -> Result<usize> {
        return io_uring_enter(idx, to_submit, min_complete, flag)
    }

    pub fn sq_len(&self) -> usize {
        unsafe {
            let head = (*self.sq.lock().head).load(Ordering::Acquire);
            let tail = unsync_load(self.sq.lock().tail);

            tail.wrapping_sub(head) as usize
        }
    }

    pub fn sq_need_wakeup(&self) -> bool {
        unsafe {
            (*self.sq.lock().flags).load(Ordering::Acquire) & sys::IORING_SQ_NEED_WAKEUP != 0
        }
    }
}

pub fn io_uring_enter(
    idx: usize,
    to_submit: u32,
    min_complete: u32,
    flags: u32,
    //sig: *const sigset_t,
) -> Result<usize> {
    let ret = HostSpace::IoUringEnter(idx, to_submit, min_complete, flags);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret as usize)
}

impl OOMHandler for ListAllocator {
    fn handleError(&self, size:u64, alignment:u64) {
        HostSpace::KernelOOM(size, alignment);
    }
}

impl ListAllocator {
    pub fn initialize(&self)-> () {
        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn Check(&self) {
        Task::StackOverflowCheck();
    }
}

impl CPULocal {
    pub fn Wakeup(&self) {
        // the uring eventwrite maynot return successfully, likely linux bug
        // todo: fix this.
        //IOURING.EventfdWrite(self.vcpuId, self.eventfd);

        // look like hcall based eventwrite is faster than qcall.
        // todo: root cause this
        //super::Kernel::HostSpace::EventfdWriteAsync(self.eventfd);
        super::Kernel::HostSpace::EventfdWrite(self.eventfd);
    }
}

impl<'a> ShareSpace {
    pub fn AQCall(&self, msg: &HostOutputMsg) {
        loop {
            match self.QOutput.TryPush(msg) {
                Ok(()) => {
                    break;
                }
                Err(_) => (),
            };
        }

        if super::SHARESPACE.HostProcessor() == 0 {
            let vcpuId = super::SHARESPACE.scheduler.WakeOne();
            if vcpuId < 0 &&super::SHARESPACE.NeedHostProcess()  {
                HyperCall64(HYPERCALL_QCALL, 0, 0, 0);
            }
        }
    }

    pub fn Schedule(&self, taskId: u64) {
        self.scheduler.Schedule(TaskId::New(taskId));
    }

    pub fn Yield() {
        HostSpace::VcpuYield();
    }
}

impl<T: ?Sized> QMutexIntern<T> {
    pub fn GetID() -> u64 {
        return Task::TaskAddress();
    }
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfType {
    Start,
    Kernel,
    User,
    Read,
    Write,
    Open,
    KernelHandling,
    Print,
    Idle,
    PageFault,
    QCall,
    SysCall,
    Blocked,
    HostInputProcess,
    End,
}

impl CounterSet {
    pub const PERM_COUNTER_SET_SIZE : usize = 8;

    pub fn GetPerfId(&self) -> usize {
        CPULocal::CpuId() as usize
    }

    pub fn PerfType(&self) -> &str {
        return "PerfPrint::Kernel"
    }
}

#[inline]
pub fn switch(from: TaskId, to: TaskId) {
    Task::Current().PerfGoto(PerfType::Blocked);
    Task::Current().AccountTaskEnter(SchedState::Blocked);

    CPULocal::SetCurrentTask(to.Addr());
    let fromCtx = from.GetTask();
    let toCtx = to.GetTask();

    if !SHARESPACE.config.read().KernelPagetable {
        toCtx.SwitchPageTable();
    }
    toCtx.SetFS();

    fromCtx.Check();
    toCtx.Check();
    debug!("switch {:x}->{:x}", from.data, to.data);

    unsafe {
        context_swap(fromCtx.GetContext(), toCtx.GetContext(), 1, 0);
    }

    Task::Current().PerfGofrom(PerfType::Blocked);
    Task::Current().AccountTaskLeave(SchedState::Blocked);
}

pub fn OpenAt(task: &Task, dirFd: i32, addr: u64, flags: u32) -> Result<i32> {
    return openAt(task, dirFd, addr, flags)
}


pub fn SignalProcess(signalArgs: &SignalArgs) {
    *MSG.lock() = Some(signalArgs.clone());
    CreateTask(SignalHandler, 0 as *const u8, false);
}

pub fn StartRootContainer(para: *const u8) {
    super::StartRootContainer(para)
}

pub fn StartExecProcess(fd: i32, process: Process) {
    super::StartExecProcess(fd, process)
}
pub fn StartSubContainerProcess(elfEntry: u64, userStackAddr: u64, kernelStackAddr: u64) {
    super::StartSubContainerProcess(elfEntry, userStackAddr, kernelStackAddr)
}

extern "C" {
    pub fn CopyPageUnsafe(to: u64, from: u64);
}