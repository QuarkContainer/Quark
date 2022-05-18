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

use libc::*;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use super::qlib::SysCallID;
use super::qlib::MAX_VCPU_COUNT;
use super::vmspace::syscall::*;
use super::*;

pub struct SyncMgr {
    pub sharespaceReady: AtomicU32,
    pub vcpuWait: [u32; MAX_VCPU_COUNT],
    pub waitMask: u32,
    // if there is vcpu thread wait at host, vcpuWait = 1
}

impl SyncMgr {
    pub const fn New() -> Self {
        return Self {
            sharespaceReady: AtomicU32::new(0),
            vcpuWait: [0; MAX_VCPU_COUNT],
            waitMask: 0,
        };
    }

    // timeout: ms
    pub fn Futex(addr: u64, op: i32, val: i32, _timeout: i64, uaddr2: u64, val3: i32) -> i32 {
        let nr = SysCallID::sys_futex as usize;
        unsafe {
            return syscall6(
                nr,
                addr as usize,
                op as usize,
                val as usize,
                0,
                uaddr2 as usize,
                val3 as usize,
            ) as i32;
        }
    }

    pub fn FutexWaitTimeout(
        addr: u64,
        op: i32,
        val: i32,
        timeout: i64,
        uaddr2: u64,
        val3: i32,
    ) -> i32 {
        let ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: timeout * 1000_000,
        };

        let nr = SysCallID::sys_futex as usize;
        unsafe {
            return syscall6(
                nr,
                addr as usize,
                op as usize,
                val as usize,
                &ts as *const _ as usize,
                uaddr2 as usize,
                val3 as usize,
            ) as i32;
        }
    }

    pub fn WaitVcpu(vcpuId: usize) -> i32 {
        //error!("WaitVcpu cpu is {}", vcpuId);
        //defer!(error!("WaitVcpu leave cpu is {}", vcpuId));

        let addr = {
            let mut syncMgr = super::SYNC_MGR.lock();
            let addr = &syncMgr.vcpuWait[vcpuId] as *const _ as u64;
            syncMgr.vcpuWait[vcpuId] = 1;
            syncMgr.waitMask |= 1 << vcpuId;
            addr
        };

        //let ret = Self::FutexWaitTimeout(addr, FUTEX_WAIT, 1, 10, 0, 0);
        let ret = Self::Futex(addr, FUTEX_WAIT, 1, 0, 0, 0);

        let mut syncMgr = super::SYNC_MGR.lock();
        syncMgr.vcpuWait[vcpuId] = 0;
        syncMgr.waitMask &= !(1 << vcpuId);
        return ret;
    }

    // if vcpuId > 0, wakeup this vcpuId
    // if vcpuId = 0, wakeup anyone vcpu
    pub fn WakeVcpu(vcpuId: usize) -> i32 {
        let mut syncMgr = super::SYNC_MGR.lock(); //need to lock, in case there are 2 vcpu thread enter waiting state when the io_thread are waking up

        let vcpuId = if vcpuId > 0 {
            vcpuId as usize
        } else {
            if syncMgr.waitMask == 0 {
                // async host message trigger
                return 0;
            }

            // async host message trigger, find first waiting vcpu
            let lastone = (syncMgr.waitMask & !1).trailing_zeros();
            lastone as usize
        };

        if vcpuId >= 32 {
            return 0;
        }

        if syncMgr.waitMask & (1 << vcpuId) != 0 {
            // if target vcpu is waiting
            //there is waiting thread, need to call futex wake
            let addr = &syncMgr.vcpuWait[vcpuId] as *const _ as u64;
            syncMgr.vcpuWait[vcpuId] = 0;
            let count = Self::Futex(addr, FUTEX_WAKE, INT_MAX, 0, 0, 0);
            return count;
        } else {
            //no waiting thread
            return 0;
        }
    }

    pub fn SharespaceReady(&mut self) -> &mut AtomicU32 {
        return &mut self.sharespaceReady;
    }

    pub fn WaitShareSpaceReady() -> i32 {
        let addr = super::SYNC_MGR.lock().SharespaceReady().get_mut() as *const _ as u64;

        let ret = Self::Futex(addr, FUTEX_WAIT, 0, 0, 0, 0);
        return ret;
    }

    pub fn WakeShareSpaceReady() -> i32 {
        let mut syncMgr = super::SYNC_MGR.lock();

        syncMgr.sharespaceReady.store(1, Ordering::Relaxed);
        let addr = syncMgr.sharespaceReady.get_mut() as *const _ as u64;
        let ret = Self::Futex(addr, FUTEX_WAKE, INT_MAX, 0, 0, 0);

        return ret;
    }
}
