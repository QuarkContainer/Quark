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

impl IoUring {
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

