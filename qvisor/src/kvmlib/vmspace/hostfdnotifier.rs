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

use spin::RwLock;
use libc::*;
use core::ops::Deref;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::ShareSpace;
use super::VMSpace;

pub struct FdNotifierInternal {
    //eventfd which guest notify host for message
    pub eventfd: i32,
}

pub struct HostFdNotifier(RwLock<FdNotifierInternal>);

impl Deref for HostFdNotifier {
    type Target = RwLock<FdNotifierInternal>;

    fn deref(&self) -> &RwLock<FdNotifierInternal> {
        &self.0
    }
}

impl HostFdNotifier {
    pub fn New() -> Self {
        let eventfd = unsafe {
            //eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)
            eventfd(0, EFD_CLOEXEC)
        };

        if eventfd == -1 {
            panic!("FdNotifier::New create eventfd fail, error is {}", errno::errno().0);
        }

        let internal = FdNotifierInternal {
            eventfd: eventfd,
        };

        return Self(RwLock::new(internal))
    }

    pub fn Eventfd(&self) -> i32 {
        return self.read().eventfd;
    }

    pub fn Notify(&self)  {
        let data: u64 = 1;
        let ret = unsafe {
            write(self.read().eventfd, &data as *const _ as *const c_void, 8)
        };

        if ret == -1 {
            let errno = errno::errno().0;
            error!("hostfdnotifier Trigger fail to write data to the eventfd, errno is {}", errno);
        }
    }

    pub const MAX_EVENTS: usize = 128;
    pub fn WaitAndNotify(&self, shareSpace: &'static ShareSpace, timeout: i32) -> Result<i32> {
        let waitTime = if timeout == -1 { // blocked
            //shareSpace.WaitInHost();
            if shareSpace.ReadyAsyncMsgCnt() > 0 {
                0
            } else {
                -1
            }
            //10
        } else {
            timeout
        };

        let eventfd = self.read().eventfd;
        let mask : EventMask = EVENT_IN;
        let mut e = pollfd {
            fd: eventfd,
            events: mask as i16,
            revents: 0,
        };

        loop {
            if waitTime >= 0 {
                let ret = unsafe {
                    poll(&mut e, 1, waitTime)
                };

                let ret = VMSpace::GetRet(ret as i64) as i32;
                // Interrupted by signal, try again.
                if ret == -SysErr::EINTR {
                    continue;
                }

                // error happen
                if ret < 0 {
                    panic!("WaitAndNotify waitfail with error {}", ret);
                }

                // timeout
                if ret == 0 {
                    return Ok(0)
                }
            }

            // eventfd is triggered
            let mut data : u64 = 0;
            let ret = unsafe {
                libc::read(eventfd, &mut data as * mut _ as *mut libc::c_void, 8)
            };

            if ret < 0 {
                panic!("WaitAndNotify... eventfd is {}, errno is {}",
                       eventfd, errno::errno().0);
            }
            return Ok(1)
        }
    }

}