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

use alloc::collections::vec_deque::VecDeque;
use alloc::vec::Vec;
use super::mutex::*;
use core::ops::Deref;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use core::sync::atomic::AtomicU64;
use alloc::string::String;
use cache_padded::CachePadded;

use super::vcpu_mgr::*;

#[derive(Debug, Copy, Clone, Default)]
pub struct TaskId {
    pub data: u64
}

impl TaskId {
    #[inline]
    pub const fn New(addr: u64) -> Self {
        return Self {
            data: addr
        }
    }

    #[inline]
    pub fn Addr(&self) -> u64 {
        return self.data;
    }
}

// task with cpu queue id
// bit 0~11 queue id, 12~63 task address & !0xfff
#[derive(Debug, Copy, Clone, Default)]
pub struct TaskIdQ {
    data: u64
}

impl TaskIdQ {
    #[inline]
    // the last 12 bits of addr should be zero and queue < 4096
    pub fn New(addr: u64, queue: u64) -> Self {
        assert!((addr & 0xfff) ==0 && queue < 4096, "TaskIdQ::New addr is {:x}", addr);
        return Self {
            data: addr | queue
        }
    }

    #[inline]
    pub fn Addr(&self) -> u64 {
        return self.data & !0xfff;
    }

    #[inline]
    pub fn Queue(&self) -> u64 {
        return self.data & 0xfff;
    }

    #[inline]
    pub fn TaskId(&self) -> TaskId {
        return TaskId::New(self.Addr())
    }
}

#[derive(Default)]
#[repr(C)]
#[repr(align(128))]
pub struct Scheduler {
    pub queue: Vec<CachePadded<TaskQueue>>,
    pub vcpuCnt: usize,
    pub taskCnt: AtomicUsize,
    pub readyTaskCnt: AtomicUsize,
    pub haltVcpuCnt: AtomicUsize,

    pub vcpuWaitMask: AtomicU64,
    pub VcpuArr : Vec<CPULocal>,
}

impl Scheduler {
    pub fn New(vcpuCount: usize) -> Self {
        let mut vcpuArr : Vec<CPULocal> = Vec::with_capacity(vcpuCount);
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
        }
    }

    pub fn DecreaseHaltVcpuCnt(&self) {
        self.haltVcpuCnt.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn IncreaseHaltVcpuCnt(&self) -> usize {
        return self.haltVcpuCnt.fetch_add(1, Ordering::SeqCst);
    }

    pub fn HaltVcpuCnt(&self) -> usize {
        return self.haltVcpuCnt.load(Ordering::Acquire);
    }

    #[inline(always)]
    pub fn GlobalReadyTaskCnt(&self) -> usize {
        self.readyTaskCnt.load(Ordering::Acquire)
    }

    pub fn ReadyTaskCnt(&self, vcpuId: usize) -> u64 {
        //return self.readyTaskCnt.load(Ordering::SeqCst) as u64
        return self.queue[vcpuId].Len();
    }

    pub fn PrintQ(&self, vcpuId: u64) -> String {
        return format!("{:x?}", self.queue[vcpuId as usize].lock());
    }

    #[inline(always)]
    pub fn IncReadyTaskCount(&self) -> usize {
        let cnt = self.readyTaskCnt.fetch_add(1, Ordering::SeqCst) + 1;
        return cnt
    }

    #[inline(always)]
    pub fn DecReadyTaskCount(&self) -> usize {
        let cnt = self.readyTaskCnt.fetch_sub(1, Ordering::SeqCst) - 1;
        return cnt;
    }

    pub fn ScheduleQ(&self, task: TaskId, vcpuId: u64) {
        let _cnt = {
            let mut queue = self.queue[vcpuId as usize].lock();
            queue.push_back(task);
            self.IncReadyTaskCount()
        };

        //error!("ScheduleQ task {:x?}, vcpuId {}", task, vcpuId);
        if vcpuId == 0 {
            self.WakeOne();
            return
        }

        let state = self.VcpuArr[vcpuId as usize].State();
        if state == VcpuState::Waiting {
            //error!("ScheduleQ: vcpu {} is waiting ..., wake it up", vcpuId);
            self.VcpuArr[vcpuId as usize].Wakeup();
        } else if state == VcpuState::Running {
            self.WakeOne();
        }
    }

    pub fn AllTasks(&self) -> Vec<TaskId> {
        let mut ret = Vec::new();
        for i in 0..8 {
            for t in self.queue[i].lock().iter() {
                ret.push(*t)
            }
        }

        return ret;
    }

    pub fn WakeOne(&self) -> i64 {
        loop {
            let mask = self.vcpuWaitMask.load(Ordering::Acquire);

            let vcpuId = mask.trailing_zeros() as usize;
            if vcpuId >= 64 {
                return -1;
            }

            if self.WakeIdleCPU(vcpuId) {
                return vcpuId as i64
            }
        }
    }

    pub fn WakeAll(&self) {
        for i in 1..self.vcpuCnt {
            self.WakeIdleCPU(i);
        }
    }

    pub fn WakeIdleCPU(&self, vcpuId: usize) -> bool {
        let vcpuMask = (1<<vcpuId) as u64;
        let prev = self.vcpuWaitMask.fetch_and(!vcpuMask, Ordering::Acquire);

        let wake = (prev & vcpuMask) != 0;
        if wake {
            self.VcpuArr[vcpuId].Wakeup();
        }

        return wake;

        /*let state = self.VcpuArr[vcpuId].State();
        if state == VcpuState::Waiting {
            self.VcpuArr[vcpuId].Wakeup();
            return true
        }

        return false*/
    }
}

pub struct TaskQueue(pub QMutex<VecDeque<TaskId>>);

impl Deref for TaskQueue {
    type Target = QMutex<VecDeque<TaskId>>;

    fn deref(&self) -> &QMutex<VecDeque<TaskId>> {
        &self.0
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        return Self::New();
    }
}

impl TaskQueue {
    pub fn New() -> Self {
        return TaskQueue(QMutex::new(VecDeque::with_capacity(128)));
    }

    pub fn Dequeue(&self) -> Option<TaskId> {
        return self.lock().pop_front();
    }

    pub fn Enqueue(&self, task: TaskId) {
        self.lock().push_back(task);
    }

    pub fn ToString(&self) -> String {
        return format!("{:x?} ", self.lock());
    }

    pub fn Len(&self) -> u64 {
        return self.lock().len() as u64;
    }
}
