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

pub mod fdinfo;
pub mod file_range_mgr;
pub mod socket_info;
//pub mod rdma;

use core::sync::atomic::AtomicI32;
use libc::*;
use spin::Mutex;

use crate::qlib::fileinfo::*;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::SysCallID;
use super::super::*;
use super::syscall::*;

const START_FD: i32 = 0; //stdin:0, stdout:1, stderr:2
const MAX_FD: i32 = 65535; //skip stdin, stdout, stderr

//map between guest/process fd to host fd

impl IOMgr {
    pub fn Print(&self) {
        info!("fdTbl is {:?}", self.fdTbl);
    }

    pub fn GetEventFd(&self) -> i32 {
        return self.eventfd;
    }

    pub fn Init() -> Result<Self> {
        let eventfd = unsafe { eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK) };

        if eventfd == -1 {
            let errno = errno::errno().0;
            error!("EpollMgr create pipe fail");
            return Err(Error::SysError(errno));
        }

        //info!("EpollMgr eventfd = {}", eventfd);

        let res = Self {
            eventfd: eventfd,
            epollfd: AtomicI32::new(0),
            fdTbl: Mutex::new(FdTbl::New()),
        };

        res.DrainPipe()?;

        return Ok(res);
    }

    //this needs to be called after Notify
    pub fn DrainPipe(&self) -> Result<()> {
        let mut data: u64 = 0;

        let nr = SysCallID::sys_read as usize;
        let ret = unsafe {
            syscall3(
                nr,
                self.eventfd as usize,
                &mut data as *mut _ as usize,
                8 as usize,
            ) as i32
        };

        if ret > 0 || -ret == EAGAIN || -ret == EWOULDBLOCK {
            return Ok(());
        } else {
            return Err(Error::SysError(ret));
        }
    }

    //return guest fd
    pub fn AddFile(&self, fd: i32) -> i32 {
        self.fdTbl
            .lock()
            .AddFile(fd)
            .expect("hostfdMap: guest fd alloc fail");
        return fd;
    }

    pub fn AddSocket(&self, fd: i32) -> i32 {
        self.fdTbl
            .lock()
            .AddSocket(fd)
            .expect("hostfdMap: guest fd alloc fail");
        return fd;
    }

    //ret: true: exist, false: not exist
    pub fn RemoveFd(&self, fd: i32) -> Option<FdInfo> {
        let fdInfo = self.fdTbl.lock().Remove(fd);
        return fdInfo;
    }

    pub fn GetFdByHost(&self, fd: i32) -> Option<i32> {
        if self.fdTbl.lock().Contains(fd) {
            return Some(fd);
        }

        return None;
    }

    pub fn AddWait(&self, fd: i32, mask: EventMask) {
        let fdInfo = self.GetByHost(fd);
        match fdInfo {
            None => (),
            Some(fdInfo) => {
                fdInfo.lock().AddWait(mask).unwrap();
            }
        }
    }

    pub fn RemoveWait(&self, fd: i32, mask: EventMask) {
        let fdInfo = self.GetByHost(fd);
        match fdInfo {
            None => (),
            Some(fdInfo) => {
                fdInfo.lock().RemoveWait(mask).unwrap();
            }
        }
    }
}

impl FdTbl {
    pub fn New() -> Self {
        let mut res = Self::default();

        res.Insert(0, FdInfo::NewFile(0));
        res.Insert(1, FdInfo::NewFile(1));
        res.Insert(2, FdInfo::NewFile(2));

        return res;
    }

    pub fn AddFile(&mut self, osfd: i32) -> Result<FdInfo> {
        let fdInfo = FdInfo::NewFile(osfd);

        self.Insert(osfd, fdInfo.clone());
        return Ok(fdInfo);
    }

    pub fn AddSocket(&mut self, osfd: i32) -> Result<FdInfo> {
        let fdInfo = FdInfo::NewSocket(osfd);

        self.Insert(osfd, fdInfo.clone());
        return Ok(fdInfo);
    }
}
