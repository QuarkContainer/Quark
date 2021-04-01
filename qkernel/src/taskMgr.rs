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

use core::sync::atomic::{Ordering, AtomicU32};

use super::task::*;
use super::SHARESPACE;
use super::qlib::task_mgr::*;
use super::Kernel::HostSpace;
use super::qlib::perf_tunning::*;
use super::qlib::vcpu_mgr::*;
use super::threadmgr::task_sched::*;
use super::KERNEL_STACK_ALLOCATOR;
use super::memmgr::pma::*;
use super::quring::uring_mgr::*;
use super::asm::*;

static ACTIVE_TASK: AtomicU32 = AtomicU32::new(0);

pub fn IncrActiveTask() -> u32 {
    return ACTIVE_TASK.fetch_add(1, Ordering::SeqCst);
}

pub fn DecrActiveTask() -> u32 {
    return ACTIVE_TASK.fetch_sub(1, Ordering::SeqCst);
}

pub fn AddNewCpu() {
    let mainTaskId = TaskStore::CreateFromThread();
    CPULocal::SetWaitTask(mainTaskId.Addr());
    CPULocal::SetCurrentTask(mainTaskId.Addr());
}

pub fn Current() -> TaskId {
    return TaskId::New(CPULocal::CurrentTask());
}

pub fn CreateTask(runFn: TaskFn, para: *const u8) {
    let taskId = { TaskStore::CreateTask(runFn, para) };
    SHARESPACE.scheduler.NewTask(taskId);
}

extern "C" {
    pub fn context_swap(_fromCxt: u64, _toCtx: u64, _one: u64, _zero: u64);
    pub fn context_swap_to(_fromCxt: u64, _toCtx: u64, _one: u64, _zero: u64);
}

#[inline]
fn switch(from: TaskId, to: TaskId) {
    Task::Current().PerfGoto(PerfType::Blocked);
    Task::Current().AccountTaskEnter(SchedState::Blocked);

    CPULocal::SetCurrentTask(to.Addr());
    let fromCtx = from.GetTask();
    let toCtx = to.GetTask();

    toCtx.SwitchPageTable();
    toCtx.SetFS();
    unsafe {
        context_swap(fromCtx.GetContext(), toCtx.GetContext(), 1, 0);
    }

    Task::Current().PerfGofrom(PerfType::Blocked);
    Task::Current().AccountTaskLeave(SchedState::Blocked);
}

fn switch_to(to: TaskId) {
    to.GetTask().AccountTaskLeave(SchedState::Blocked);

    CPULocal::SetCurrentTask(to.Addr());
    let toCtx = to.GetTask();

    toCtx.SwitchPageTable();
    toCtx.SetFS();
    unsafe {
        context_swap_to(0, toCtx.GetContext(), 1, 0);
    }
}

pub const IO_WAIT_CYCLES : i64 = 20_000_000; // 1ms
pub const WAIT_CYCLES : i64 = 2_000_000; // 1ms

pub fn IOWait() {
    let mut start = Rdtsc();

    loop {
        if PollAsyncMsg() > 10 {
            start = Rdtsc();
        }

        let currentTime = Rdtsc();
        if currentTime - start >= IO_WAIT_CYCLES {
            SHARESPACE.kernelIOThreadWaiting.store(true, Ordering::Release);

            // after change the state, check again in case new message coming
            if PollAsyncMsg() > 10 {
                start = Rdtsc();
                SHARESPACE.kernelIOThreadWaiting.store(false, Ordering::Release);
                continue;
            }

            HostSpace::IOWait();
            start = Rdtsc();
            SHARESPACE.kernelIOThreadWaiting.store(false, Ordering::Release);
        }
    }
}

pub fn WaitFn() {
    CPULocal::SetCPUState(VcpuState::Searching);
    loop {
        let next = { SHARESPACE.scheduler.GetNext() };
        match next {
            None => {
                CPULocal::SetCPUState(VcpuState::Waiting);
                SHARESPACE.scheduler.IncreaseHaltVcpuCnt();
                if SHARESPACE.scheduler.GlobalReadyTaskCnt() == 0 {
                    HostSpace::Hlt();
                }

                SHARESPACE.scheduler.DecreaseHaltVcpuCnt();
                CPULocal::SetCPUState(VcpuState::Searching);
            }

            Some(newTask) => {
                //error!("WaitFn newTask1 is {:x?}", &newTask);
                CPULocal::SetCPUState(VcpuState::Running);
                let current = TaskId::New(CPULocal::CurrentTask());
                //error!("WaitFn newTask2 is {:x?}", &newTask);
                switch(current, newTask);

                CPULocal::SetCPUState(VcpuState::Searching);
                let pendingFreeStack = CPULocal::PendingFreeStack();
                if pendingFreeStack != 0 {
                    //(*PAGE_ALLOCATOR).Free(pendingFreeStack, DEFAULT_STACK_PAGES).unwrap();
                    KERNEL_STACK_ALLOCATOR.Free(pendingFreeStack).unwrap();
                    CPULocal::SetPendingFreeStack(0);
                }

                FreePageTables();
            }
        }
    }
}

#[inline]
pub fn PollAsyncMsg() -> usize {
    return HostInputProcess() + QUringTrigger();
}

#[inline]
pub fn ProcessOne() -> bool {
    if QUringProcessOne() {
        return true;
    }

    return HostInputProcessOne(true);
}

pub fn Wait() {
    CPULocal::SetCPUState(VcpuState::Searching);
    let start = Rdtsc();

    loop {
        let next = { SHARESPACE.scheduler.GetNext() };

        if let Some(newTask) = next {
            CPULocal::SetCPUState(VcpuState::Running);
            let current = TaskId::New(CPULocal::CurrentTask());
            if current.data != newTask.data {
                switch(current, newTask);
            }

            break;
        }

        let currentTime = Rdtsc();
        if currentTime - start >= WAIT_CYCLES {
            let current = TaskId::New(CPULocal::CurrentTask());
            let waitTask = TaskId::New(CPULocal::WaitTask());
            switch(current, waitTask);
            break;
        } else {
            if !ProcessOne() {
                unsafe { llvm_asm!("pause" :::: "volatile"); }
            }
        }
    }
}

pub fn SwitchToNewTask() -> ! {
    CPULocal::SetCPUState(VcpuState::Searching);

    let current = Task::TaskId();
    let waitTask = TaskId::New(CPULocal::WaitTask());
    switch(current, waitTask);
    panic!("SwitchToNewTask end impossible");
}

impl Scheduler {
    // steal scheduling
    pub fn GetNext(&self) -> Option<TaskId> {
        let vcpuId = CPULocal::CpuId() as usize;
        let vcpuCount = self.vcpuCnt.load(Ordering::Relaxed);

        for i in vcpuId..vcpuId+vcpuCount {
            match self.GetNextForCpu(vcpuId, i % vcpuCount) {
                None => (),
                Some(t) => {
                    return Some(t)
                }
            }
        }

        return None;
    }

    #[inline]
    pub fn GetNextForCpu(&self, currentCpuId: usize, vcpuId: usize) -> Option<TaskId> {
        // only stealing task from running VCPU
        if vcpuId != 0 && currentCpuId != vcpuId && CPULocal::GetCPUState(vcpuId) != VcpuState::Running {
            return None;
        }

        let count = self.queue[vcpuId].lock().len();
        for _ in 0..count {
            let task = self.queue[vcpuId].Dequeue();
            match task {
                None => return None,
                Some(taskId) => {
                    self.readyTaskCnt.fetch_sub(1, Ordering::SeqCst);

                    if taskId.GetTask().context.ready !=0 || taskId.data == Task::Current().taskId {
                        //the task is in the queue, but the context has not been setup
                        if currentCpuId != vcpuId { //stealing
                            //error!("cpu currentCpuId {} stealing task {:x?} from cpu {}", currentCpuId, taskId, vcpuId);

                            taskId.GetTask().queueId = currentCpuId;
                        }

                        //error!("GetNextForCpu task is {:x?}", taskId);
                        return task
                    }

                    self.Schedule(taskId);
                }
            }
        }

        return None;
    }

    pub fn Schedule(&self, taskId: TaskId) {
        let vcpuId = taskId.GetTask().queueId;
        self.KScheduleQ(taskId, vcpuId);
    }

    pub fn KScheduleQ(&self, task: TaskId, vcpuId: usize) {
        //error!("KScheduleQ task {:x?}, vcpuId {}", task, vcpuId);
        self.ScheduleQ(task, vcpuId as u64);
    }

    pub fn NewTask(&self, taskId: TaskId) -> usize {
        self.ScheduleQ(taskId, 0);
        return 0;
    }
}

pub fn Yield() {
    SHARESPACE.scheduler.Schedule(Task::TaskId());
    Wait();
}

#[inline]
pub fn HostInputProcessOne(try: bool) -> bool {
    let m = if try {
        SHARESPACE.AQHostInputTryPop()
    } else {
        SHARESPACE.AQHostInputPop()
    };

    match m {
        None => (),
        Some(m) => {
            m.Process();
            return true;
        }
    };

    return false
}

#[inline]
pub fn HostInputProcess() -> usize {
    let mut count = 0;
    loop {
        //drain the async message from host
        if HostInputProcessOne(false) {
            count += 1;
        } else {
            return count
        }
    }
}

pub fn NewTask(taskId: TaskId) {
    SHARESPACE.scheduler.NewTask(taskId);
}

pub fn ScheduleQ(taskId: TaskIdQ) {
    SHARESPACE.scheduler.KScheduleQ(taskId.TaskId(), taskId.Queue() as usize);
}
