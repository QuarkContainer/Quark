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

use std::io;
use libc::*;

use super::qlib::common::*;

impl Error {
    pub fn FromIOErr(err: io::Error) -> Error {
        return Error::SysError(err.raw_os_error().unwrap())
    }
}

pub fn GetRet(ret: i32) -> Result<i32> {
    if ret == -1 {
        return Err(Error::SysError(errno::errno().0))
    }

    return Ok(ret)
}

pub fn SysRet(ret: i64) -> i64 {
    if ret == -1 {
        return -errno::errno().0 as i64
    }

    return ret
}

pub fn GetNoRet(ret: i32) -> Result<()> {
    if ret == -1 {
        return Err(Error::SysError(errno::errno().0))
    }

    return Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Cond {
    eventfd: i32
}

impl Cond {
    pub fn New() -> Result<Self> {
        let res = unsafe {
            eventfd(0, 0)
        };

        let eventfd = GetRet(res)?;
        return Ok(Self{
            eventfd: eventfd
        })
    }

    pub fn Wait(&self) -> Result<()> {
        let mut data: u64 = 0;

        unsafe {
            read(self.eventfd, &mut data as *mut _ as *mut libc::c_void, 8);
            close(self.eventfd);
        }

        return Ok(())
    }

    pub fn Notify(&self) -> Result<()> {
        let data: u64 = 1;
        unsafe {
            write(self.eventfd, &data as *const _ as *const libc::c_void, 8);
            close(self.eventfd);
        }

        Ok(())
    }
}

pub struct Events {
    pub events: [epoll_event; 128],
    pub count: usize,
}

impl Events {
    pub fn New() -> Self {
        return Self {
            events: [epoll_event { events: 0, u64: 0 }; 128],
            count: 0,
        }
    }

    pub fn SetCount(&mut self, cnt: usize) {
        self.count = cnt;
    }

    pub fn Slice(&self) -> &[epoll_event] {
        return &self.events[0..self.count]
    }

    pub fn SliceMut(&mut self) -> &mut [epoll_event] {
        return &mut self.events
    }
}

pub struct Epoll {
    pub epollfd: i32,
}

impl Epoll {
    pub fn New() -> Result<Self>  {
        let epollfd = unsafe {
            epoll_create1(0)
        };

        if epollfd == -1 {
            info!("USrvSocket epoll_create fail");
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        return Ok(Self{
            epollfd: epollfd,
        })
    }

    pub fn Unblock(fd: i32) -> Result<()> {
        let flags = unsafe {
            fcntl(fd, F_GETFL, 0)
        };

        if flags < 0 {
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        let ret = unsafe {
            fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        };

        if ret < 0 {
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        return Ok(())
    }

    pub fn Addfd(&self, fd: i32, events: u32) -> Result<()> {
        Self::Unblock(fd)?;

        let mut event = epoll_event {
            events: events,
            u64: fd as u64,
        };

        let ret = unsafe {
            epoll_ctl(self.epollfd, EPOLL_CTL_ADD, fd, &mut event)
        };

        if ret < 0 {
            error!("USrvSocket epoll_ctl add fd fail with err {}", errno::errno().0 as i32);
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        return Ok(())
    }

    pub fn Poll(&self, events: &mut Events) -> Result<()> {
        let slice = events.SliceMut();
        let nfds = unsafe {
            epoll_wait(self.epollfd, &mut slice[0], slice.len() as i32, -1)
        };

        if nfds == -1 {
            return Err(Error::Common(format!("UCallController wait fail with err {}", errno::errno().0)));
        }

        events.SetCount(nfds as usize);
        return Ok(())
    }
}