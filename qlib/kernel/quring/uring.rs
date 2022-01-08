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

use core::sync::atomic;

use super::super::super::uring::*;
use super::super::super::uring::util::*;
use super::super::super::common::*;
use super::super::Kernel::HostSpace;

impl IoUring {
    /// Initiate asynchronous I/O.
    #[inline]
    pub fn submit(&self, idx: usize) -> Result<usize> {
        self.submit_and_wait(idx, 0)
    }

    pub fn submit_and_wait(&self, idx: usize, want: usize) -> Result<usize> {
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
                    super::super::Kernel::HostSpace::UringWake(idx, 0);
                    return Ok(0)
                }
            } else if want == 0 {
                // fast poll
                return Ok(len);
            }
        }

        unsafe { self.enter(idx, len as _, want as _, flags) }
    }

    pub fn sq_len(&self) -> usize {
        unsafe {
            let head = (*self.sq.lock().head).load(atomic::Ordering::Acquire);
            let tail = unsync_load(self.sq.lock().tail);

            tail.wrapping_sub(head) as usize
        }
    }

    pub fn sq_need_wakeup(&self) -> bool {
        unsafe {
            (*self.sq.lock().flags).load(atomic::Ordering::Acquire) & sys::IORING_SQ_NEED_WAKEUP != 0
        }
    }

    pub unsafe fn enter(
        &self,
        idx: usize,
        to_submit: u32,
        min_complete: u32,
        flag: u32
    ) -> Result<usize> {
        return io_uring_enter(idx, to_submit, min_complete, flag)
    }
}

pub fn io_uring_enter(
    idx: usize,
    to_submit: u32,
    min_complete: u32,
    flags: u32,
    //sig: *const sigset_t,
) -> Result<usize> {
    let ret = HostSpace::IoUringEnter(idx, to_submit, min_complete, flags);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret as usize)
}