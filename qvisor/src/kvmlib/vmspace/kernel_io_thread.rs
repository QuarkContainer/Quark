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

use super::super::qlib::ShareSpace;
use super::super::qlib::common::*;
use super::super::*;

pub struct KIOThread {
    pub eventfd: i32,
}

impl KIOThread {
    pub fn New() -> Self {
        return Self{
            eventfd: 0
        }
    }

    pub fn Init(&self, eventfd: i32) {
        unsafe {
            *(&self.eventfd as * const _ as u64 as * mut i32) = eventfd;
        }
    }

    pub fn Wait(&self, sharespace: &ShareSpace) -> Result<()> {
        loop {
            let mut data : u64 = 0;
            let ret = unsafe {
                libc::read(self.eventfd, &mut data as * mut _ as *mut libc::c_void, 8)
            };

            if ret < 0 {
                panic!("KIOThread::Wakeup fail... eventfd is {}, errno is {}",
                        self.eventfd, errno::errno().0);
            }

            if !super::super::runc::runtime::vm::IsRunning() {
                return Err(Error::Exit)
            }

            if sharespace.ReadyAsyncMsgCnt() > 0 || URING_MGR.lock().ring.HasCompleteEntry() {
                return Ok(())
            }
        }
    }

    pub fn Wakeup(&self, sharespace: &ShareSpace) {
        if sharespace.kernelIOThreadWaiting.load(Ordering::Acquire) {
            let val : u64 = 1;
            let ret = unsafe {
                libc::write(self.eventfd, &val as * const _ as *const libc::c_void, 8)
            };
            if ret < 0 {
                panic!("KIOThread::Wakeup fail...");
            }
        }
    }
}


