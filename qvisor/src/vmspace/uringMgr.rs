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

use super::super::print::*;
use super::super::qlib::common::*;
use super::super::qlib::uring::sys::sys::*;
use super::super::qlib::uring::*;

use super::super::*;

//#[derive(Debug)]
pub struct UringMgr {
    pub fds: Vec<i32>,
    pub uringSize: usize,
}

impl Drop for UringMgr {
    fn drop(&mut self) {
        self.Close();
    }
}

pub const FDS_SIZE: usize = 1024 * 16;

impl UringMgr {
    pub fn New(size: usize) -> Self {
        let fdsSize = if QUARK_CONFIG.lock().UringFixedFile {
            FDS_SIZE
        } else {
            0
        };

        let mut fds = Vec::with_capacity(fdsSize);
        for _i in 0..fdsSize {
            fds.push(-1);
        }

        let ret = Self {
            fds: fds,
            uringSize: size,
        };

        return ret;
    }

    pub fn Close(&mut self) {
        let logfd = LOG.Logfd();
        for fd in &self.fds {
            if *fd >= 0 && *fd != logfd {
                unsafe {
                    libc::close(*fd);
                }
            }
        }
    }

    pub fn Register(&self, _opcode: u32, _arg: u64, _nrArgs: u32) -> Result<()> {
        // // it is inited
        // if self.uringfd != -1 {
        //     self.RegisterOne(self.uringfd, opcode, arg, nrArgs)?;
        // }

        return Ok(());
    }

    // pub fn RegisterOne(&self, fd: i32, opcode: u32, arg: u64, nrArgs: u32) -> Result<()> {
    //     let ret = IOUringRegister(fd, opcode, arg, nrArgs);
    //     if ret < 0 {
    //         error!("IOUringRegister get fail {}", ret);
    //         return Err(Error::SysError(-ret as i32));
    //     }

    //     return Ok(());
    // }

    pub fn UnRegisterFile(&mut self) -> Result<()> {
        return self.Register(IORING_UNREGISTER_FILES, 0, 0);
    }

    pub fn Addfd(&mut self, fd: i32) -> Result<()> {
        if !QUARK_CONFIG.lock().UringFixedFile {
            return Ok(());
        }

        if fd as usize >= self.fds.len() {
            error!("Addfd out of bound fd {}", fd);
            panic!("Addfd out of bound fd {}", fd)
        }
        self.fds[fd as usize] = fd;

        let fu = sys::io_uring_files_update {
            offset: fd as u32,
            resv: 0,
            fds: self.fds[fd as usize..].as_ptr() as _,
        };

        return self.Register(IORING_REGISTER_FILES_UPDATE, &fu as *const _ as u64, 1);
    }

    pub fn Removefd(&mut self, fd: i32) -> Result<()> {
        if !QUARK_CONFIG.lock().UringFixedFile {
            return Ok(());
        }

        if fd as usize >= self.fds.len() {
            error!("Removefd out of bound fd {}", fd);
            panic!("Removefd out of bound fd {}", fd)
        }

        self.fds[fd as usize] = -1;
        let fu = sys::io_uring_files_update {
            offset: fd as u32,
            resv: 0,
            fds: self.fds[fd as usize..].as_ptr() as _,
        };

        return self.Register(IORING_REGISTER_FILES_UPDATE, &fu as *const _ as u64, 1);
    }
}
