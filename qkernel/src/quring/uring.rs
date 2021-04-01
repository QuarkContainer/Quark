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

use core::sync::atomic;

use super::super::qlib::uring::*;
use super::super::qlib::uring::util::*;
use super::super::qlib::uring::porting::*;
use super::super::qlib::common::*;
use super::super::Kernel::HostSpace;

impl IoUring {
    /// Initiate asynchronous I/O.
    #[inline]
    pub fn submit(&self) -> Result<usize> {
        self.submit_and_wait(0)
    }

    pub fn submit_and_wait(&self, want: usize) -> Result<usize> {
        let len = self.sq_len();

        let mut flags = 0;

        if want > 0 {
            flags |= sys::IORING_ENTER_GETEVENTS;
        }

        if self.params.0.flags & sys::IORING_SETUP_SQPOLL != 0 {
            if self.sq_need_wakeup() {
                if want > 0 {
                    flags |= sys::IORING_ENTER_SQ_WAKEUP;
                } else {
                    super::super::Kernel::HostSpace::UringWake();
                    return Ok(0)
                }
            } else if want == 0 {
                // fast poll
                return Ok(len);
            }
        }

        unsafe { self.enter(len as _, want as _, flags) }
    }

    pub fn sq_len(&self) -> usize {
        unsafe {
            let head = (*self.sq.head).load(atomic::Ordering::Acquire);
            let tail = unsync_load(self.sq.tail);

            tail.wrapping_sub(head) as usize
        }
    }

    pub fn sq_need_wakeup(&self) -> bool {
        unsafe {
            (*self.sq.flags).load(atomic::Ordering::Acquire) & sys::IORING_SQ_NEED_WAKEUP != 0
        }
    }

    pub unsafe fn enter(
        &self,
        to_submit: u32,
        min_complete: u32,
        flag: u32
    ) -> Result<usize> {
        return io_uring_enter(self.fd.as_raw_fd(), to_submit, min_complete, flag)
    }
}

pub fn io_uring_enter(
    fd: i32,
    to_submit: u32,
    min_complete: u32,
    flags: u32,
    //sig: *const sigset_t,
) -> Result<usize> {
    let ret = HostSpace::IoUringEnter(fd, to_submit, min_complete, flags);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret as usize)
}