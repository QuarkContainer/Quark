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

use alloc::collections::vec_deque::VecDeque;
use alloc::vec::Vec;
use spin::Mutex;
use core::ops::Deref;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use core::sync::atomic::AtomicU64;

use super::MAX_VCPU_COUNT;
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
        assert!((addr & 0xfff) ==0 && queue < 4096);
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
pub struct Scheduler {
    pub queue: [TaskQueue; MAX_VCPU_COUNT],
    pub vcpuCnt: AtomicUsize,
    pub taskCnt: AtomicUsize,
    pub readyTaskCnt: AtomicUsize,
    pub haltVcpuCnt: AtomicUsize,

    pub vcpuWaitMask: AtomicU64,
    pub VcpuArr : [CPULocal; MAX_VCPU_COUNT],
}

impl Scheduler {
    pub fn DecreaseHaltVcpuCnt(&self) {
        self.haltVcpuCnt.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn IncreaseHaltVcpuCnt(&self) -> usize {
        return self.haltVcpuCnt.fetch_add(1, Ordering::SeqCst);
    }

    pub fn HaltVcpuCnt(&self) -> usize {
        return self.haltVcpuCnt.load(Ordering::Acquire);
    }

    pub fn SetVcpuCnt(&self, vcpuCnt: usize) {
        self.vcpuCnt.store(vcpuCnt, Ordering::SeqCst)
    }

    pub fn GetVcpuCnt(&self) -> usize {
        self.vcpuCnt.load(Ordering::SeqCst)
    }

    pub fn GlobalReadyTaskCnt(&self) -> usize {
        self.readyTaskCnt.load(Ordering::SeqCst)
    }

    pub fn ReadyTaskCnt(&self, vcpuId: usize) -> u64 {
        //return self.readyTaskCnt.load(Ordering::SeqCst) as u64
        return self.queue[vcpuId].Len();
    }

    pub fn ScheduleQ(&self, task: TaskId, vcpuId: u64) {
        self.readyTaskCnt.fetch_add(1, Ordering::SeqCst);
        self.queue[vcpuId as usize].Enqueue(task);

        //error!("ScheduleQ task {:x?}, vcpuId {}", task, vcpuId);
        if vcpuId == 0 {
            self.WakeOne();
            return
        }

        let state = self.VcpuArr[vcpuId as usize].State();
        if state == VcpuState::Waiting {
            //error!("ScheduleQ: vcpu {} is waiting ..., wake it up", vcpuId);
            self.VcpuArr[vcpuId as usize].Wakeup();
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

    pub fn VcpuSetWaiting(&self, vcpuId: usize) {
        self.VcpuArr[vcpuId].SetState(VcpuState::Waiting)
    }

    pub fn VcpuSetRunning(&self, vcpuId: usize) {
        self.VcpuArr[vcpuId].SetState(VcpuState::Running)
    }

    pub fn VcpuSetSearching(&self, vcpuId: usize) {
        self.VcpuArr[vcpuId].SetState(VcpuState::Searching)
    }

    pub fn WakeOne(&self) {
        let mask = self.vcpuWaitMask.load(Ordering::Acquire);
        let vcpuId = mask.trailing_zeros() as usize;
        if vcpuId >= 64 {
            return;
        }

        self.WakeIdleCPU(vcpuId);
    }

    pub fn WakeAll(&self) {
        for i in 1..self.vcpuCnt.load(Ordering::Relaxed) {
            self.WakeIdleCPU(i);
        }
    }

    pub fn WakeIdleCPU(&self, vcpuId: usize) {
        if vcpuId == 0 {
            self.WakeOne();
            return
        }

        let state = self.VcpuArr[vcpuId].State();
        if state == VcpuState::Waiting {
            self.VcpuArr[vcpuId].Wakeup();
        }
    }
}

pub struct TaskQueue(pub Mutex<VecDeque<TaskId>>);

impl Deref for TaskQueue {
    type Target = Mutex<VecDeque<TaskId>>;

    fn deref(&self) -> &Mutex<VecDeque<TaskId>> {
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
        return TaskQueue(Mutex::new(VecDeque::with_capacity(128)));
    }

    pub fn Dequeue(&self) -> Option<TaskId> {
        return self.lock().pop_front();
    }

    pub fn Enqueue(&self, task: TaskId) {
        self.lock().push_back(task);
    }

    pub fn Print(&self) {
        //info!("start TaskQueue::Print...");

        for id in self.lock().iter() {
            info!("Get task {:x}", (*id).Addr());
        }

        //info!("end TaskQueue::Print...");
    }

    pub fn Len(&self) -> u64 {
        return self.lock().len() as u64;
    }
}
