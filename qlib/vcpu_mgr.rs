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

use core::sync::atomic::Ordering;
use core::sync::atomic::AtomicU64;

use super::ShareSpace;
use super::mem::list_allocator::*;

#[derive(Clone, Debug, PartialEq, Copy)]
#[repr(u64)]
pub enum VcpuState {
    Searching,
    Waiting,
    Running,
}

#[derive(Debug, Default)]
#[repr(C)]
#[repr(align(128))]
pub struct CPULocal {
    pub kernelStack: AtomicU64,             //offset 0
    pub userStack: AtomicU64,               //offset 8
    pub vcpuId: usize,                      //offset 16
    pub waitTask: AtomicU64,                //offset 24
    pub currentTask: AtomicU64,             //offset 32
    pub pendingFreeStack: AtomicU64,        //offset 40
    pub state: AtomicU64,                   //offset 48

    pub switchCount: AtomicU64,
    pub uringMsgCount: AtomicU64,
    pub data: u64, // for eventfd data writing and reading
    pub eventfd: i32,
    pub epollfd: i32,
    pub allocator: VcpuAllocator,
}

impl CPULocal {
    pub fn State(&self) -> VcpuState {
        let state = self.state.load(Ordering::Acquire);
        return unsafe { core::mem::transmute(state) };
    }

    pub fn ToSearch(&self, sharespace: &ShareSpace) -> u64 {
        assert!(self.state.load(Ordering::SeqCst)!=VcpuState::Searching as u64, "state is {}", self.state.load(Ordering::SeqCst));
        self.state.store(VcpuState::Searching as u64, Ordering::SeqCst);
        return sharespace.IncrVcpuSearching();
    }

    pub fn ToWaiting(&self, sharespace: &ShareSpace) -> u64 {
        assert!(self.state.load(Ordering::SeqCst)==VcpuState::Searching as u64, "state is {}", self.state.load(Ordering::SeqCst));
        self.state.store(VcpuState::Waiting as u64, Ordering::SeqCst);
        let searchingCnt = sharespace.DecrVcpuSearching();
        return searchingCnt;
    }

    pub fn ToRunning(&self, sharespace: &ShareSpace) -> u64 {
        assert!(self.state.load(Ordering::SeqCst)==VcpuState::Searching as u64, "state is {}", self.state.load(Ordering::SeqCst));
        self.state.store(VcpuState::Running as u64, Ordering::SeqCst);
        let searchingCnt = sharespace.DecrVcpuSearching();
        return searchingCnt;
    }

    pub fn IncrUringMsgCnt(&self, cnt: u64) -> u64 {
        return self.uringMsgCount.fetch_add(cnt, Ordering::Relaxed);
    }
}