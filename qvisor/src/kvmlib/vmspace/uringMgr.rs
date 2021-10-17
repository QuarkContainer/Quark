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

use alloc::slice;

use super::super::qlib::common::*;
use super::super::qlib::uring::sys::sys::*;
use super::super::qlib::uring::*;
use super::super::qlib::linux_def::*;
use super::super::VMS;

use super::super::*;
use super::host_uring::*;

//#[derive(Debug)]
pub struct UringMgr {
    pub fd: i32,
    pub eventfd: i32,
    pub fds: &'static mut [i32],
    pub ring: IoUring,
}

pub const FDS_SIZE : usize = 8192;

impl UringMgr {
    pub fn New(size: usize) -> Self {
        let pages = (FDS_SIZE * 4 + 4095) / 4096;
        let fdsAddr = VMS.lock().allocator.as_mut().unwrap().AllocPages(pages as u64).expect("UringMgr allocpages fail");
        let fds = unsafe {
            slice::from_raw_parts_mut(fdsAddr as *mut i32, FDS_SIZE)
        };

        for i in 0..FDS_SIZE {
            fds[i] = -1;
        }

        //let ring = Builder::default().setup_sqpoll(50).setup_sqpoll_cpu(0).build(size as u32).expect("InitUring fail");
        let ring = Builder::default().setup_sqpoll(10).dontfork().build(size as u32).expect("InitUring fail");

        let ret = Self {
            fd: ring.fd.0,
            eventfd: 0,
            fds: fds,
            ring: ring,
        };

        ret.Register(IORING_REGISTER_FILES, &ret.fds[0] as * const _ as u64, ret.fds.len() as u32).expect("InitUring register files fail");
        return ret;
    }

    pub fn Probe(&self) {
        error!("UringMgr probe is ....");
        let mut probe = Probe::new();
        self.RegisterProbe(probe.as_mut_ptr() as *const _ as u64, Probe::COUNT as u32).unwrap();
        error!("opcode::Write::CODE support is {}", probe.is_supported(opcode::Write::CODE as u8));
        error!("opcode::WriteFixed::CODE support is {}", probe.is_supported(opcode::WriteFixed::CODE as u8));
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

    pub fn RegisterBuff(&self, addr: u64, size: usize) -> Result<()> {
        let ioVec = IoVec::NewFromAddr(addr, size);
        return self.Register(IORING_REGISTER_BUFFERS, &ioVec as * const _ as u64, 1);
    }

    pub fn RegisterProbe(&self, probe: u64, size: u32) -> Result<()> {
        return self.Register(IORING_REGISTER_PROBE, probe, size);
    }

    pub fn UnRegisterFile(&mut self) -> Result<()> {
        return self.Register(IORING_UNREGISTER_FILES, 0, 0)
    }

    pub fn GetFds(&self) -> u64 {
        return self.fds[0..].as_ptr() as _;
    }

    pub fn GetUringFd(&self) -> i32 {
        return self.fd;
    }

    pub fn Addfd(&mut self, fd: i32) -> Result<()> {
        if fd as usize >= self.fds.len() {
            error!("Addfd out of bound fd {}", fd);
            panic!("Addfd out of bound fd {}", fd)
        }
        self.fds[fd as usize] = fd;

        let fu = sys::io_uring_files_update {
            offset : fd as u32,
            resv: 0,
            fds: self.fds[fd as usize..].as_ptr() as _,
        };

        return self.Register(IORING_REGISTER_FILES_UPDATE, &fu as * const _ as u64, 1);
    }

    pub fn Removefd(&mut self, fd: i32) -> Result<()> {
        if fd as usize >= self.fds.len() {
            error!("Removefd out of bound fd {}", fd);
            panic!("Removefd out of bound fd {}", fd)
        }

        self.fds[fd as usize] = -1;
        let fu = sys::io_uring_files_update {
            offset : fd as u32,
            resv: 0,
            fds: self.fds[fd as usize..].as_ptr() as _,
        };

        return self.Register(IORING_REGISTER_FILES_UPDATE, &fu as * const _ as u64, 1);
    }
}


