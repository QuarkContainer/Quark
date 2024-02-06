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

use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use core::sync::atomic::{AtomicI64, AtomicU8};
use core::cell::UnsafeCell;

use super::mem::list_allocator::*;
use super::ShareSpace;

#[derive(Clone, Debug, PartialEq, Copy)]
#[repr(u64)]
pub enum VcpuState {
    Searching,
    Waiting,
    Running,
}

#[derive(Clone, Debug, PartialEq, Copy)]
#[repr(u8)]
pub enum VcpuMode {
    Kernel,
    User,
}

#[derive(Debug, Default)]
#[repr(C)]
#[repr(align(128))]
pub struct CPULocal {
    pub kernelStack: AtomicU64,      //offset 0
    pub userStack: AtomicU64,        //offset 8
    pub vcpuId: usize,               //offset 16
    pub waitTask: AtomicU64,         //offset 24
    pub currentTask: AtomicU64,      //offset 32
    pub pendingFreeStack: AtomicU64, //offset 40
    pub state: AtomicU64,            //offset 48

    pub switchCount: AtomicU64,
    pub uringMsgCount: AtomicU64,
    pub tlbEpoch: AtomicU64,
    pub data: u64, // for eventfd data writing and reading
    pub eventfd: i32,
    pub epollfd: i32,
    pub allocator: UnsafeCell<VcpuAllocator>,
    pub pageAllocator: UnsafeCell<PageAllocator>,

    // it is the time to enter guest ring3. If it is in ring0, the vale will be zero
    pub enterAppTimestamp: AtomicI64,
    pub interruptMask: AtomicU64,
    pub mode: AtomicU8,
}


unsafe impl Send for CPULocal {}
unsafe impl Sync for CPULocal {}


impl CPULocal {
    pub fn State(&self) -> VcpuState {
        let state = self.state.load(Ordering::SeqCst);
        return unsafe { core::mem::transmute(state) };
    }

    pub fn AllocatorMut(&self) -> &mut VcpuAllocator {
        //return unsafe { &mut *(&self.allocator as *const _ as u64 as *mut VcpuAllocator) };
        return unsafe { &mut *self.allocator.get() }
    }

    pub fn ToSearch(&self, sharespace: &ShareSpace) -> u64 {
        assert!(
            self.state.load(Ordering::Acquire) != VcpuState::Searching as u64,
            "state is {}",
            self.state.load(Ordering::Acquire)
        );
        self.state
            .store(VcpuState::Searching as u64, Ordering::Release);
        return sharespace.IncrVcpuSearching();
    }

    pub fn ToWaiting(&self, sharespace: &ShareSpace) -> u64 {
        assert!(
            self.state.load(Ordering::Acquire) == VcpuState::Searching as u64,
            "state is {}",
            self.state.load(Ordering::Acquire)
        );
        self.state
            .store(VcpuState::Waiting as u64, Ordering::Release);
        let searchingCnt = sharespace.DecrVcpuSearching();
        return searchingCnt;
    }

    pub fn ToRunning(&self, sharespace: &ShareSpace) -> u64 {
        assert!(
            self.state.load(Ordering::Acquire) == VcpuState::Searching as u64,
            "state is {}",
            self.state.load(Ordering::Acquire)
        );
        self.state
            .store(VcpuState::Running as u64, Ordering::Release);
        let searchingCnt = sharespace.DecrVcpuSearching();
        return searchingCnt;
    }

    pub fn IncrUringMsgCnt(&self, cnt: u64) -> u64 {
        return self.uringMsgCount.fetch_add(cnt, Ordering::Relaxed);
    }

    pub fn ResetEnterAppTimestamp(&self) -> i64 {
        return self.enterAppTimestamp.swap(0, Ordering::SeqCst);
    }

    pub fn SetEnterAppTimestamp(&self, val: i64) {
        self.enterAppTimestamp.store(val, Ordering::SeqCst)
    }

    pub fn EnterAppTimestamp(&self) -> i64 {
        return self.enterAppTimestamp.load(Ordering::SeqCst);
    }

    pub fn SetMode(&self, mode: VcpuMode) {
        return self.mode.store(mode as u8, Ordering::SeqCst);
    }

    pub fn GetMode(&self) -> VcpuMode {
        let mode = self.mode.load(Ordering::SeqCst);
        return unsafe { core::mem::transmute(mode) };
    }

    pub fn ResetInterruptMask(&self) -> u64 {
        return self.interruptMask.swap(0, Ordering::SeqCst);
    }

    pub fn SetInterruptMask(&self, mask: u64) {
        self.interruptMask.fetch_or(mask, Ordering::SeqCst);
    }

    pub const TLB_SHOOTDOWN_MASK: u64 = 1 << 0;
    pub const THREAD_TIMEOUT: u64 = 1 << 1;

    pub fn InterruptTlbShootdown(&self) {
        self.SetInterruptMask(Self::TLB_SHOOTDOWN_MASK);
    }

    pub fn InterruptThreadTimeout(&self) {
        self.SetInterruptMask(Self::THREAD_TIMEOUT);
    }

    pub fn InterruptByTlbShootdown(mask: u64) -> bool {
        return mask & Self::TLB_SHOOTDOWN_MASK != 0;
    }

    pub fn InterruptByThreadTimeout(mask: u64) -> bool {
        return mask & Self::THREAD_TIMEOUT != 0;
    }
}
