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

use crate::arch::vCPU;
use core::sync::atomic::Ordering;
use libc::*;
use std::os::unix::io::AsRawFd;

use super::qlib::common::*;
use super::qlib::pagetable::*;
use super::runc::runtime::vm::*;
use super::*;

#[repr(C)]
pub struct SignalMaskStruct {
    length: u32,
    mask1: u32,
    mask2: u32,
    _pad: u32,
}

pub struct HostPageAllocator {
    pub allocator: AlignedAllocator,
}

impl HostPageAllocator {
    pub fn New() -> Self {
        return Self {
            allocator: AlignedAllocator::New(0x1000, 0x10000),
        };
    }
}

impl Allocator for HostPageAllocator {
    fn AllocPage(&self, _incrRef: bool) -> Result<u64> {
        let ret = self.allocator.Allocate()?;
        ZeroPage(ret);
        return Ok(ret);
    }
}

impl RefMgr for HostPageAllocator {
    fn Ref(&self, _addr: u64) -> Result<u64> {
        //panic!("HostPageAllocator doesn't support Ref");
        return Ok(1);
    }

    fn Deref(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support Deref");
    }

    fn GetRef(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support GetRef");
    }
}

impl KVMVcpu {
    pub fn VcpuWait(&self) -> i64 {
        let sharespace = &SHARE_SPACE;
        loop {
            if !super::runc::runtime::vm::IsRunning() {
                return -1;
            }

            {
                sharespace.IncrHostProcessor();
                Self::GuestMsgProcess(sharespace);

                defer!({
                    // last processor in host
                    if sharespace.DecrHostProcessor() == 0 {
                        Self::GuestMsgProcess(sharespace);
                    }
                });
            }

            let ret = sharespace.scheduler.WaitVcpu(sharespace, self.id, true);
            match ret {
                Ok(taskId) => return taskId as i64,
                Err(Error::Exit) => return -1,
                Err(e) => panic!("HYPERCALL_HLT wait fail with error {:?}", e),
            }
        }
    }
}
