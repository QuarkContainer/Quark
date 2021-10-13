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

pub struct HostFdNotifier {
    //eventfd which guest notify host for message
    pub eventfd: i32,
}

impl HostFdNotifier {
    pub fn New() -> Self {
        let eventfd = unsafe {
             eventfd(0, EFD_CLOEXEC)
        };

        if eventfd == -1 {
            panic!("FdNotifier::New create eventfd fail, error is {}", errno::errno().0);
        }

        return Self {
            eventfd: eventfd,
        };
    }

    pub fn Eventfd(&self) -> i32 {
        return self.eventfd;
    }

    pub fn Notify(&self)  {
        let data: u64 = 1;
        let ret = unsafe {
            write(self.eventfd, &data as *const _ as *const c_void, 8)
        };

        if ret == -1 {
            let errno = errno::errno().0;
            error!("hostfdnotifier Trigger fail to write data to the eventfd, errno is {}", errno);
        }
    }

    pub fn Wait(&self) {
        let eventfd = self.eventfd;
        let mut data : u64 = 0;
        let ret = unsafe {
            libc::read(eventfd, &mut data as * mut _ as *mut libc::c_void, 8)
        };

        if ret < 0 {
            panic!("WaitAndNotify... eventfd is {}, errno is {}",
                   eventfd, errno::errno().0);
        }
    }

}