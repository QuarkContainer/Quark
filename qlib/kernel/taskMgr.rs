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
use core::arch::asm;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::{GLOBAL_ALLOCATOR, IS_GUEST};

use super::super::super::kernel_def::*;
use super::super::kernel::GlobalRDMASvcCli;
use super::super::linux_def::*;
use super::super::task_mgr::*;
use super::super::vcpu_mgr::*;
use super::quring::uring_mgr::*;
use super::task::*;
use super::Kernel::HostSpace;
use super::Shutdown;
use super::ASYNC_PROCESS;
use super::KERNEL_STACK_ALLOCATOR;
use super::SHARESPACE;
use super::TSC;

static ACTIVE_TASK: AtomicU32 = AtomicU32::new(0);

pub fn IncrActiveTask() -> u32 {
    return ACTIVE_TASK.fetch_add(1, Ordering::SeqCst);
}

pub fn DecrActiveTask() -> u32 {
    return ACTIVE_TASK.fetch_sub(1, Ordering::SeqCst);
}

pub fn AddNewCpu() {
    let mainTaskId = TaskStore::CreateFromThread();
    CPULocal::SetWaitTask(mainTaskId.PrivateTaskAddr(), mainTaskId.SharedTaskAddr());
    CPULocal::SetCurrentTask(mainTaskId.PrivateTaskAddr(), mainTaskId.SharedTaskAddr());
}

pub fn CreateTask(runFnAddr: u64, para: *const u8, kernel: bool) {
    let taskId = { TaskStore::CreateTask(runFnAddr, para, kernel) };
    SHARESPACE.scheduler.NewTask(taskId);
}

extern "C" {
    pub fn context_swap(_fromCxt: u64, _toCtx: u64, _one: u64, _zero: u64);
}

pub const IO_WAIT_CYCLES: i64 = 20_000_000;

pub const WAIT_CYCLES: i64 = 1_000_000; // 1ms

pub fn IOWait() {
    let mut start = TSC.Rdtsc();

    while !Shutdown() {
        if PollAsyncMsg() > 10 {
            start = TSC.Rdtsc();
        }

        let currentTime = TSC.Rdtsc();
        if currentTime - start >= IO_WAIT_CYCLES || Shutdown() {
            // after change the state, check again in case new message coming
            if PollAsyncMsg() > 10 && !Shutdown() {
                start = TSC.Rdtsc();
                continue;
            }

            //debug!("IOWait sleep");
            HostSpace::IOWait();
            //debug!("IOWait wakeup");
            start = TSC.Rdtsc();
        }
    }

    loop {
        HostSpace::IOWait();
    }
}

pub fn WaitFn() -> ! {
    let mut task = TaskId::default();
    loop {
        let next = if task.PrivateTaskAddr() == 0 {
            SHARESPACE.scheduler.GetNext()
        } else {
            let tmp = task;
            task = TaskId::default();
            Some(tmp)
        };
        //info!("WaitFn, next:{:?}\n",next);
        match next {
            None => {
                SHARESPACE.scheduler.IncreaseHaltVcpuCnt();

                //debug!("vcpu sleep");
                task = HostSpace::VcpuWait();
                //debug!("vcpu wakeup");
                SHARESPACE.scheduler.DecreaseHaltVcpuCnt();
            }

            Some(newTask) => {
                let (t, t_wp) = CPULocal::CurrentTask();
                let current = TaskId::New(t, t_wp);
                CPULocal::Myself().SwitchToRunning();
                if !Task::Current().context.savefpsate {
                    Task::Current().SaveFp();
                }
                switch(current, newTask);

                info!("WaitFn() switch(current, newTask)");
                let (pendingFreeStack, pendingFreeStackWrapper) = CPULocal::PendingFreeStack();
                if pendingFreeStack != 0 {
                    //(*PAGE_ALLOCATOR).Free(pendingFreeStack, DEFAULT_STACK_PAGES).unwrap();
                    let task = TaskId::New(pendingFreeStack, pendingFreeStackWrapper).GetPrivateTask();
                    //free X86fpstate
                    task.context.X86fpstate.take();

                    KERNEL_STACK_ALLOCATOR.Free(pendingFreeStack).unwrap();


                    let tw_size  = core::mem::size_of::<TaskWrapper>();
                    let layout = core::alloc::Layout::from_size_align(tw_size, 2).
                                                expect("WaitFn layout for TaskWrapper failed");
                    unsafe {
                        GLOBAL_ALLOCATOR.DeallocShareBuf(pendingFreeStackWrapper as *mut u8, layout.size(), layout.align());
                    };

                    CPULocal::SetPendingFreeStack(0, 0);
                }

                if Shutdown() {
                    //error!("shutdown: {}", super::AllocatorPrint(10));
                    super::Kernel::HostSpace::ExitVM(super::EXIT_CODE.load(QOrdering::SEQ_CST));
                }

                // todo: free heap cache
                //while super::ALLOCATOR.Free() {}
            }
        }
    }
}

#[inline]
pub fn PollAsyncMsg() -> usize {
    if Shutdown() {
        return 0;
    }

    let mut ret = QUringTrigger();
    // ret += GlobalRDMASvcCli().DrainCompletionQueue();
    if SHARESPACE.config.read().EnableRDMA {
        ret += GlobalRDMASvcCli().ProcessRDMASvcMessage();
    }
    if Shutdown() {
        return 0;
    }

    ASYNC_PROCESS.Process();

    //error!("PollAsyncMsg 4 count {}", ret);
    return ret;
}

#[inline]
pub fn ProcessOne() -> bool {
    return QUringProcessOne();
}

pub fn Wait() {

    assert!(IS_GUEST == true,  "pub fn Wait() is called by host");

    CPULocal::Myself().ToSearch(&SHARESPACE);
    let start = TSC.Rdtsc();

    let vcpuId = CPULocal::CpuId() as usize;
    let mut next = SHARESPACE.scheduler.GetNext();
    loop {
        if let Some(newTask) = next {

            let (c_t, c_twrp) = CPULocal::CurrentTask();
            let current = TaskId::New(c_t, c_twrp);
            //let vcpuId = newTask.GetTask().queueId;
            //assert!(CPULocal::CpuId()==vcpuId, "cpu {}, target cpu {}", CPULocal::CpuId(), vcpuId);

            CPULocal::Myself().SwitchToRunning();
            if current.PrivateTaskAddr() != newTask.PrivateTaskAddr() {
                if !Task::Current().context.savefpsate {
                    Task::Current().SaveFp();
                }
                switch(current, newTask);
            }

            // the context is still current, no switch needed.
            break;
        }

        //super::ALLOCATOR.Free();

        let currentTime = TSC.Rdtsc();
        if currentTime - start >= WAIT_CYCLES {
            let (c_t, c_twrp) = CPULocal::CurrentTask();
            let (w_t, w_twrp) = CPULocal::WaitTask();

            let current = TaskId::New(c_t, c_twrp);
            let waitTask = TaskId::New(w_t, w_twrp);
            let oldTask = SHARESPACE.scheduler.queue[vcpuId].ResetWorkingTask();

            match oldTask {
                None => {
                    if !Task::Current().context.savefpsate {
                        Task::Current().SaveFp();
                    }
                    switch(current, waitTask);
                    break;
                }
                Some(t) => next = Some(t),
            }
        } else {
            if PollAsyncMsg() == 0 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    asm!("pause");
                }
            }

            next = SHARESPACE.scheduler.GetNext();
        }
    }
}

pub fn SwitchToNewTask() -> ! {
    CPULocal::Myself().ToSearch(&SHARESPACE);

    let current = Task::Current().GetPrivateTaskId();

    let (wait_t, wait_tp) = CPULocal::WaitTask();
    let waitTask = TaskId::New(wait_t, wait_tp);

    switch(current, waitTask);
    panic!("SwitchToNewTask end impossible");
}

impl Scheduler {
    pub fn Steal(&self, vcpuId: usize) -> Option<TaskId> {
        if self.GlobalReadyTaskCnt() == 0 {
            return None;
        }

        let vcpuCount = self.vcpuCnt;
        match self.queue[0].Steal() {
            None => (),
            Some(t) => return Some(t),
        }

        // skip the current vcpu
        for i in 1..vcpuCount {
            let idx = (i + vcpuId) % vcpuCount;
            match self.queue[idx].Steal() {
                None => (),
                Some(t) => return Some(t),
            }
        }

        return None;
    }

    // steal scheduling
    pub fn GetNext(&self) -> Option<TaskId> {
        let vcpuId = CPULocal::CpuId() as usize;

        match self.queue[vcpuId].Next() {
            None => (),
            Some((t, global)) => {
                //error!("Next ... {:x?}/{}", t, global);
                if global {
                    self.DecReadyTaskCount();
                }
                return Some(t);
            }
        }

        match self.Steal(vcpuId) {
            None => return None,
            Some(t) => {
                self.DecReadyTaskCount();
                //error!("stealing ... {:x?}", t);
                let task = match self.queue[vcpuId].SwapWoringTask(t) {
                    None => {
                        t.GetSharedTask().SetQueueId(vcpuId);
                        t
                    }
                    Some(task) => {
                        self.Schedule(t, true); // reschedule the task
                        task
                    }
                };
                return Some(task);
            }
        }
    }

    pub fn Print(&self) -> String {
        let mut str = alloc::string::String::new();
        let vcpuCount = self.vcpuCnt;
        for i in 0..vcpuCount {
            let hasWorking = self.queue[i].data.lock().workingTaskReady;
            if self.queue[i].Len() > 0 || hasWorking {
                str += &format!("{}:{:x?}", i, &self.queue[i]);
            }
        }

        return str;
    }

    pub fn Schedule(&self, taskId: TaskId, cpuAff: bool) {
        let vcpuId = taskId.GetSharedTask().QueueId();
        //assert!(CPULocal::CpuId()==vcpuId, "cpu {}, target cpu {}", CPULocal::CpuId(), vcpuId);
        self.ScheduleQ(taskId, vcpuId as _, cpuAff);
    }

    pub fn NewTask(&self, taskId: TaskId) -> usize {
        self.ScheduleQ(taskId, 0, false);
        return 0;
    }
}

pub fn Yield() {
    if SHARESPACE.scheduler.GlobalReadyTaskCnt() == 0 {
        return;
    }

    assert!(IS_GUEST == true);
    
    let private_task_id = Task::PrivateTaskID();
    let task = unsafe { &*(private_task_id as *const Task) };
    let task_wp_id = task.taskWrapperId;

    let taskID = TaskId::New(private_task_id, task_wp_id);

    SHARESPACE.scheduler.Schedule(taskID, false);
    Wait();
}

pub fn NewTask(taskId: TaskId) {
    SHARESPACE.scheduler.NewTask(taskId);
}

pub fn ScheduleQ(taskId: TaskId, cpuAff: bool) {
    SHARESPACE
        .scheduler
        .ScheduleQ(taskId, taskId.Queue(), cpuAff);
}
