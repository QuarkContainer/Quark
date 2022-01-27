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

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
//use super::super::qlib::qmsg::input::*;
//use super::super::SHARE_SPACE;
use super::super::IO_MGR;

#[repr(C)]
#[repr(packed)]
#[derive(Default, Copy, Clone, Debug)]
pub struct EpollEvent {
    pub Event: u32,
    pub U64: u64
}


pub struct HostFdNotifier {
    //main epoll fd
    pub epollfd: i32,
}

impl HostFdNotifier {
    pub fn New() -> Self {
        let epfd = unsafe {
            epoll_create1(0)
        };

        if epfd == -1 {
            panic!("FdNotifier::New create epollfd fail, error is {}", errno::errno().0);
        }

        return Self {
            epollfd: epfd,
        };
    }

    pub fn Epollfd(&self) -> i32 {
        return self.epollfd;
    }

    pub fn EpollCtlAdd(&self, fd: i32, mask: EventMask) -> Result<()> {
        return self.WaitFd(fd, LibcConst::EPOLL_CTL_ADD as _, mask);
    }

    pub fn EpollCtlMod(&self, fd: i32, mask: EventMask) -> Result<()> {
        return self.WaitFd(fd, LibcConst::EPOLL_CTL_MOD as _, mask);
    }

    pub fn EpollCtlDel(&self, fd: i32) -> Result<()> {
        return self.WaitFd(fd, LibcConst::EPOLL_CTL_DEL as _, 0);
    }

    pub fn WaitFd(&self, fd: i32, op: u32, mask: EventMask) -> Result<()> {
        let n = self;

        let epollfd = n.epollfd;

        let mut ev = epoll_event {
            events: mask as u32 | EPOLLET as u32,
            u64: fd as u64
        };

        let ret = unsafe {
            epoll_ctl(epollfd, op as i32, fd, &mut ev as *mut epoll_event)
        };

        if ret == -1 {
            return Err(Error::SysError(errno::errno().0))
        }

        return Ok(())
    }

    pub const MAX_EVENTS: usize = 8;
    pub fn HostEpollWait(&self) -> i64 {
        let mut events = [EpollEvent::default(); Self::MAX_EVENTS];
        let addr = &mut events[0] as * mut _ as u64;

        let epollfd = self.epollfd;
        let nfds = unsafe {
            epoll_wait(epollfd, addr as _, Self::MAX_EVENTS as i32, 0)
        };

        if nfds > 0 {
            for e in &events[0..nfds as usize] {
                let fd = e.U64 as i32;
                let event = e.Event as EventMask;
                Self::FdNotify(fd, event);
            }
        }

        if nfds < 0 {
            return nfds as i64
        }

        return 0 as i64
    }

    pub fn FdNotify(fd: i32, mask: EventMask) {
        IO_MGR.Notify(fd, mask);
    }
}