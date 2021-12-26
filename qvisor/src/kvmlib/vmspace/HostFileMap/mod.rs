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
pub mod rdma_socket;

use std::collections::BTreeMap;
use libc::*;

use self::fdinfo::*;
use super::super::qlib::common::*;
use super::super::qlib::SysCallID;
use super::super::qlib::linux_def::*;
use super::syscall::*;
use super::super::*;

const START_FD: i32 = 0; //stdin:0, stdout:1, stderr:2
const MAX_FD: i32 = 65535; //skip stdin, stdout, stderr

//map between guest/process fd to host fd
pub struct IOMgr {
    //guest hostfd to fdInfo
    pub fdTbl: FdTbl,
    pub eventfd: i32,
}

unsafe impl Send for IOMgr {}

impl IOMgr {
    pub fn Print(&self) {
        info!("fdTbl is {:?}", self.fdTbl);
    }

    pub fn GetEventFd(&self) -> i32 {
        return self.eventfd;
    }

    pub fn Init() -> Result<Self> {
        let eventfd = unsafe {
            eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)
        };

        if eventfd == -1 {
            let errno = errno::errno().0;
            error!("EpollMgr create pipe fail");
            return Err(Error::SysError(errno))
        }

        info!("EpollMgr eventfd = {}", eventfd);

        let mut res = Self {
            eventfd: eventfd,
            fdTbl: FdTbl::New(),
        };

        res.DrainPipe()?;

        return Ok(res);
    }

    //this needs to be called after Notify
    pub fn DrainPipe(&mut self) -> Result<()> {
        let mut data: u64 = 0;

        let nr = SysCallID::sys_read as usize;
        let ret = unsafe {
            syscall3(nr, self.eventfd as usize, &mut data as *mut _ as usize, 8 as usize) as i32
        };

        if ret > 0 || -ret == EAGAIN || -ret == EWOULDBLOCK {
            return Ok(())
        } else {
            return Err(Error::SysError(ret))
        }
    }

    //return guest fd
    pub fn AddFd(&mut self, osfd: i32, epollable: bool) -> i32 {
        let fdInfo = self.fdTbl.Alloc(osfd, epollable).expect("hostfdMap: guest fd alloc fail");
        return fdInfo.lock().osfd;
    }

    pub fn SetUnblock(osfd: i32) {
        unsafe {
            /*let mut flags = fcntl(osfd, F_GETFL, 0);
            if flags == -1 {
                panic!("SetUnblock: can't F_GETFL for fd");
            }

            flags |= Flags::O_NONBLOCK as i32;*/
            let flags = Flags::O_NONBLOCK as i32;

            let ret = fcntl(osfd, F_SETFL, flags);
            if ret == -1 {
                panic!("SetUnblock: can't F_SETFL for fd");
            }
        }
    }

    //ret: true: exist, false: not exist
    pub fn RemoveFd(&mut self, hostfd: i32) -> Option<FdInfo> {
        let fdInfo = self.fdTbl.Remove(hostfd);
        return fdInfo;
    }

    pub fn GetFdByHost(&self, hostfd: i32) -> Option<i32> {
        match self.fdTbl.Get(hostfd) {
            None => {
                //self.Print();
                None
            }
            Some(fdInfo) => Some(fdInfo.lock().osfd),
        }
    }

    pub fn GetByHost(&self, hostfd: i32) -> Option<FdInfo> {
        match self.fdTbl.Get(hostfd) {
            None => {
                None
            }
            Some(fdInfo) => Some(fdInfo.clone()),
        }
    }
}

//guest fdset for one process
#[derive(Debug, Clone)]
pub struct FdTbl {
    //pub gaps: GapMgr,
    //map between guest fd to host fd
    //pub map: BTreeMap<i32, osfd>,
    pub map: BTreeMap<i32, FdInfo>,
}

impl FdTbl {
    pub fn New() -> Self {
        let mut res = Self {
            map: BTreeMap::new(),
        };

        res.map.insert(0, FdInfo::New(0, true));
        res.map.insert(1, FdInfo::New(1, true));
        res.map.insert(2, FdInfo::New(2, true));

        return res
    }

    pub fn Alloc(&mut self, osfd: i32, epollable: bool) -> Result<FdInfo> {
        let fdInfo = FdInfo::New(osfd, epollable);

        self.map.insert(osfd, fdInfo.clone());
        return Ok(fdInfo)
    }

    pub fn Take(&mut self, osfd: i32, epollable: bool) -> Result<FdInfo> {
        let fdInfo = FdInfo::New(osfd, epollable);

        self.map.insert(osfd as i32, fdInfo.clone());
        return Ok(fdInfo)
    }

    pub fn Get(&self, fd: i32) -> Option<FdInfo> {
        match self.map.get(&fd) {
            None => None,
            Some(fdInfo) => Some(fdInfo.clone()),
        }
    }

    pub fn Remove(&mut self, fd: i32) -> Option<FdInfo> {
        //self.gaps.Free(fd as u64, 1);
        self.map.remove(&fd)
    }
}
