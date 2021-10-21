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

#[derive(Clone, Debug, PartialEq, Copy)]
#[repr(u64)]
pub enum VcpuState {
    Waiting,
    Searching, // wait in kernel to look for new jobs
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
    pub data: u64, // for eventfd data writing and reading
    pub eventfd: i32,
}

impl CPULocal {
    pub fn SwapState(&self, state: VcpuState) -> VcpuState {
        let old = self.state.swap(state as u64, Ordering::SeqCst);
        return unsafe { core::mem::transmute(old) };
    }

    pub fn SetState(&self, state: VcpuState) {
        self.state.store(state as u64, Ordering::Release);
    }

    pub fn State(&self) -> VcpuState {
        let state = self.state.load(Ordering::Acquire);
        return unsafe { core::mem::transmute(state) };
    }

    pub fn SetWaiting(&self) {
        self.SetState(VcpuState::Waiting)
    }

    pub fn SetRunning(&self) {
        self.SetState(VcpuState::Running)
    }

    pub fn SetSearching(&self) {
        self.SetState(VcpuState::Searching)
    }
}