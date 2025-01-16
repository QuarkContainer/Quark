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

use super::mutex::*;
use alloc::collections::vec_deque::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use cache_padded::CachePadded;
use core::cmp::PartialEq;
use core::panic;
use core::sync::atomic::AtomicIsize;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

use super::vcpu_mgr::*;

use crate::qlib::kernel::arch::tee::is_cc_active;

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct TaskId {
    pub data: u64,
}

impl TaskId {
    #[inline]
    pub const fn New(addr: u64) -> Self {
        return Self { data: addr };
    }

    #[inline]
    pub fn Addr(&self) -> u64 {
        return self.data;
    }

    #[inline]
    pub fn QueueId(&self) -> u64 {
        if !is_cc_active(){
            return self.GetTask().QueueId() as u64;
        } else {
            return self.GetTaskWrapper().QueueId() as u64;
        }
    }

    pub fn SetQueueId(&self, queueId: usize) {
        if !is_cc_active(){
            return self.GetTask().SetQueueId(queueId);
        } else {
            return self.GetTaskWrapper().SetQueueId(queueId)
        }
    }

    #[inline]
    pub fn Ready(&self) -> u64 {
        if !is_cc_active(){
            return self.GetTask().Ready() as u64;
        } else {
            return self.GetTaskWrapper().Ready() as u64;
        }
    }
}

#[derive(Debug, Default)]
pub struct Links {
    pub prev: AtomicU64,
    pub next: AtomicU64,
}

#[derive(Default)]
#[repr(C)]
#[repr(align(128))]
pub struct Scheduler {
    pub queue: Vec<CachePadded<TaskQueue>>,
    pub vcpuCnt: usize,
    pub taskCnt: AtomicUsize,

    // use AtomicIsize instead of usize to handle the race condition
    pub readyTaskCnt: AtomicIsize,
    pub haltVcpuCnt: AtomicUsize,

    pub vcpuWaitMask: AtomicU64,
    pub VcpuArr: Vec<CPULocal>,
}

impl Scheduler {
    pub fn New(vcpuCount: usize) -> Self {
        let mut vcpuArr: Vec<CPULocal> = Vec::with_capacity(vcpuCount);
        let mut queue: Vec<CachePadded<TaskQueue>> = Vec::with_capacity(vcpuCount);
        for _i in 0..vcpuCount {
            vcpuArr.push(CPULocal::default());
            queue.push(CachePadded::new(TaskQueue::default()));
        }

        return Self {
            VcpuArr: vcpuArr,
            queue: queue,
            vcpuCnt: vcpuCount,
            ..Default::default()
        };
    }

    pub fn CleanVcpPageCache(&self) {
        for vcpu in self.VcpuArr.iter() {
            vcpu.AllocatorMut().Clear();
        }
    }

    pub fn DecreaseHaltVcpuCnt(&self) {
        self.haltVcpuCnt.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn IncreaseHaltVcpuCnt(&self) -> usize {
        return self.haltVcpuCnt.fetch_add(1, Ordering::SeqCst);
    }

    pub fn HaltVcpuCnt(&self) -> usize {
        return self.haltVcpuCnt.load(Ordering::SeqCst);
    }

    pub fn IdleVcpuCnt(&self) -> usize {
        return self.vcpuWaitMask.load(Ordering::SeqCst).count_ones() as usize;
    }

    pub fn ReadyForHibernate(&self) -> bool {
        // Hibernate needs to wait all vcpu is halt
        // there are 2 exception: first is io_thread, second is the hibernate vcpu
        let ret = self.IdleVcpuCnt() + 2 == self.vcpuCnt;
        // error!(
        //     "ReadyForHibernate haltvcpu {}/{}, vcpu count is {}/{:b}",
        //     self.HaltVcpuCnt(),
        //     self.IdleVcpuCnt(),
        //     self.vcpuCnt,
        //     self.vcpuWaitMask.load(Ordering::SeqCst)
        // );
        return ret;
    }

    pub fn ReadyTaskCnt(&self, vcpuId: usize) -> u64 {
        //return self.readyTaskCnt.load(Ordering::SeqCst) as u64
        return self.queue[vcpuId].Len();
    }

    pub fn PrintQ(&self, vcpuId: u64) -> String {
        return format!("{:x?}", self.queue[vcpuId as usize]);
    }

    #[inline(always)]
    pub fn GlobalReadyTaskCnt(&self) -> isize {
        self.readyTaskCnt.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn IncReadyTaskCount(&self) -> isize {
        let cnt = self.readyTaskCnt.fetch_add(1, Ordering::SeqCst) + 1;
        return cnt;
    }

    #[inline(always)]
    pub fn DecReadyTaskCount(&self) -> isize {
        let cnt = self.readyTaskCnt.fetch_sub(1, Ordering::SeqCst) - 1;
        return cnt;
    }

    pub fn ScheduleQ(&self, task: TaskId, vcpuId: u64, cpuAff: bool) {
        if self.queue[vcpuId as usize].Enqueue(task, cpuAff) {
            self.IncReadyTaskCount();
        }

        //error!("ScheduleQ task {:x?}, vcpuId {}", task, vcpuId);
        if vcpuId == 0 {
            self.WakeOne();
            return;
        }

        let state = self.VcpuArr[vcpuId as usize].State();
        if state == VcpuState::Waiting {
            //error!("ScheduleQ: vcpu {} is waiting ..., wake it up", vcpuId);
            self.VcpuArr[vcpuId as usize].Wakeup();
        } else if state == VcpuState::Running {
            self.WakeOne();
        }
    }

    pub fn WakeOne(&self) -> i64 {
        loop {
            let mask = self.vcpuWaitMask.load(Ordering::Acquire);

            let vcpuId = mask.trailing_zeros() as usize;
            if vcpuId >= 64 {
                return -1;
            }

            if self.WakeIdleCPU(vcpuId) {
                return vcpuId as i64;
            }
        }
    }

    pub fn WakeAll(&self) {
        for i in 1..self.vcpuCnt {
            self.WakeIdleCPU(i);
        }
    }

    pub fn WakeIdleCPU(&self, vcpuId: usize) -> bool {
        let vcpuMask = 1u64 << vcpuId;
        let prev = self.vcpuWaitMask.fetch_and(!vcpuMask, Ordering::Acquire);

        let wake = (prev & vcpuMask) != 0;
        if wake {
            self.VcpuArr[vcpuId].Wakeup();
        }

        return wake;
    }
}

#[derive(Debug)]
pub struct TaskQueueIntern {
    pub workingTask: TaskId,
    pub workingTaskReady: bool,
    pub queue: VecDeque<TaskId>,
}

impl Default for TaskQueueIntern {
    fn default() -> Self {
        return Self {
            workingTask: TaskId::New(0),
            workingTaskReady: false,
            queue: VecDeque::with_capacity(512),
        };
    }
}

#[derive(Debug)]
pub struct TaskQueue {
    pub queueSize: AtomicUsize,
    pub data: QMutex<TaskQueueIntern>,
}

impl Default for TaskQueue {
    fn default() -> Self {
        return Self::New();
    }
}

impl TaskQueue {
    pub fn New() -> Self {
        return Self {
            queueSize: AtomicUsize::new(0),
            data: QMutex::new(TaskQueueIntern::default()),
        };
    }

    // used by the vcpu owner to get next task
    pub fn Next(&self) -> Option<(TaskId, bool)> {
        let mut data = self.data.lock();
        if data.workingTaskReady {
            data.workingTaskReady = false;
            return Some((data.workingTask, false));
        }

        match data.queue.pop_front() {
            None => return None,
            Some(taskId) => {
                self.queueSize.fetch_sub(1, Ordering::Release);
                data.workingTask = taskId;
                return Some((taskId, true));
            }
        }
    }

    pub fn ResetWorkingTask(&self) -> Option<TaskId> {
        let mut data = self.data.lock();
        if data.workingTaskReady {
            data.workingTaskReady = false;
            return Some(data.workingTask);
        } else {
            data.workingTask = TaskId::New(0);
            return None;
        }
    }

    // return: None: No working task fouond
    // Some(task) => there is workingtask set,
    pub fn SwapWoringTask(&self, task: TaskId) -> Option<TaskId> {
        let mut data = self.data.lock();
        if data.workingTaskReady {
            data.workingTaskReady = false;
            return Some(data.workingTask);
        } else {
            data.workingTask = task;
            return None;
        }
    }

    // try to steal task from other vcpu's queue
    pub fn Steal(&self) -> Option<TaskId> {
        if self.queueSize.load(Ordering::Acquire) == 0 {
            return None;
        }

        match self.data.try_lock() {
            None => return None,
            Some(mut data) => {
                for _ in 0..data.queue.len() {
                    match data.queue.pop_front() {
                        None => panic!("TaskQueue none task"),
                        Some(taskId) => {
                            if taskId.Ready() != 0 {
                                self.queueSize.fetch_sub(1, Ordering::Release);
                                return Some(taskId);
                            }
                            if data.queue.len() == data.queue.capacity() {
                                panic!("queue is full");
                            } else {
                                data.queue.push_back(taskId);
                            }
                        }
                    }
                }
            }
        }

        return None;
    }

    // return whether it is put in global available queue
    pub fn Enqueue(&self, task: TaskId, cpuAff: bool) -> bool {
        //error!("Enqueue {:x?}/{}", task, cpuAff);
        let mut data = self.data.lock();

        if cpuAff && task == data.workingTask {
            data.workingTaskReady = true;
            return false;
        }

        if data.queue.len() == data.queue.capacity()
        {
            panic!("queue is full");
        } else {
            data.queue.push_back(task);
        }

        self.queueSize.fetch_add(1, Ordering::Release);
        return true;
    }

    pub fn ToString(&self) -> String {
        return format!("{:x?} ", self);
    }

    pub fn Len(&self) -> u64 {
        return self.queueSize.load(Ordering::Acquire) as u64;
    }
}
