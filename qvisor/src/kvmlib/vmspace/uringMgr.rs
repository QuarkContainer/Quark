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

use alloc::vec::Vec;

use super::super::qlib::common::*;
use super::super::qlib::uring::sys::sys::*;
use super::super::qlib::uring::*;

use super::super::*;
use super::host_uring::*;

//#[derive(Debug)]
pub struct UringMgr {
    pub fd: i32,
    pub eventfd: i32,
    pub fds: Vec<i32>,
    pub ring: IoUring,
}

pub const FDS_SIZE : usize = 4096;

impl UringMgr {
    pub fn New(size: usize) -> Self {
        let mut fds = Vec::with_capacity(FDS_SIZE);
        for _i in 0..FDS_SIZE {
            fds.push(-1);
        }

        //let ring = Builder::default().setup_sqpoll(50).setup_sqpoll_cpu(0).build(size as u32).expect("InitUring fail");
        let ring = Builder::default().setup_sqpoll(50).build(size as u32).expect("InitUring fail");

        let ret = Self {
            fd: ring.fd.0,
            eventfd: 0,
            fds: fds,
            ring: ring,
        };

        ret.Register(IORING_REGISTER_FILES, &ret.fds[0] as * const _ as u64, ret.fds.len() as u32).expect("InitUring register files fail");
        return ret;
    }

    pub fn Setup(&mut self, submission: u64, completion: u64) -> Result<i32> {
        self.ring.CopyTo(submission, completion);
        return Ok(0)
    }

    pub fn SetupEventfd(&mut self, eventfd: i32) {
        self.eventfd = eventfd;

        self.Register(IORING_REGISTER_EVENTFD, &self.eventfd as * const _ as u64, 1).expect("InitUring register eventfd fail");
    }

    pub fn Enter(&mut self, toSumbit: u32, minComplete:u32, flags: u32) -> Result<i32> {
        let ret = IOUringEnter(self.fd, toSumbit, minComplete, flags);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32))
        }

        return Ok(ret as i32)
    }

    pub fn Wake(&self) -> Result<()> {
        let ret = IOUringEnter(self.fd, 1, 0, IORING_ENTER_SQ_WAKEUP);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32))
        }

        return Ok(());
    }

    pub fn Register(&self, opcode: u32, arg: u64, nrArgs: u32) -> Result<()> {
        let ret = IOUringRegister(self.fd, opcode, arg, nrArgs);
        if ret < 0 {
            error!("IOUringRegister get fail {}", ret);
            return Err(Error::SysError(-ret as i32))
        }

        return Ok(())
    }

    pub fn UnRegisterFile(&mut self) -> Result<()> {
        return self.Register(IORING_UNREGISTER_FILES, 0, 0)
    }

    pub fn Addfd(&mut self, fd: i32) -> Result<()> {
        self.fds[fd as usize] = fd;

        let fu = sys::io_uring_files_update {
            offset : fd as u32,
            resv: 0,
            fds: self.fds[fd as usize..].as_ptr() as _,
        };

        return self.Register(IORING_REGISTER_FILES_UPDATE, &fu as * const _ as u64, 1);
    }

    pub fn Removefd(&mut self, fd: i32) -> Result<()> {
        self.fds[fd as usize] = -1;
        let fu = sys::io_uring_files_update {
            offset : fd as u32,
            resv: 0,
            fds: self.fds[fd as usize..].as_ptr() as _,
        };

        return self.Register(IORING_REGISTER_FILES_UPDATE, &fu as * const _ as u64, 1);
    }
}


