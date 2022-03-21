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

use libc;
use std::io;
use std::os::unix::io::{AsRawFd, RawFd};
use alloc::string::ToString;

use super::super::qlib::common::*;
use super::super::qlib::cstring::*;

const DEFAULT_PTMX: &'static str = "/dev/ptmx";

pub fn NewPty() -> Result<(Master, Slave)> {
    let master = NewMaster()?;
    master.grantpt()?;
    master.unlockpt()?;
    let slave = master.NewSlave()?;

    return Ok((master, slave))
}

pub fn NewMaster() -> Result<Master> {
    let cstr = CString::New(DEFAULT_PTMX);
    return Master::new(cstr.Ptr() as * const libc::c_char)
}

pub trait Descriptor: AsRawFd {
    /// The constructor function `open` opens the path
    /// and returns the fd.
    fn open(path: *const libc::c_char,
            flag: i32,
            mode: Option<i32>)
            -> Result<RawFd> {
        unsafe {
            match libc::open(path, flag, mode.unwrap_or_default()) {
                -1 => Err(Error::SysError(-errno::errno().0)),
                fd => {
                    Ok(fd)
                },
            }
        }
    }

    /// The function `close` leaves the fd.
    fn close(&self) -> Result<()> {
        unsafe {
            match libc::close(self.as_raw_fd()) {
                -1 => Err(Error::SysError(-errno::errno().0)),
                _ => Ok(()),
            }
        }
    }

    /// The destructor function `drop` call the method `close`
    /// and panic if a error is occurred.
    fn drop(&self) {
        if self.close().is_err() {
            unimplemented!();
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Master {
    pub pty: RawFd,
}


impl Master {
    pub fn new(path: *const ::libc::c_char) -> Result<Self> {
        info!("before open");
        let res = match Self::open(path, libc::O_RDWR, None) {
            Err(e) => Err(e),
            Ok(fd) => Ok(Master { pty: fd }),
        };

        info!("after open");
        return res;
    }

    pub fn dup(&self) -> Result<i32> {
        unsafe {
            match libc::dup(self.as_raw_fd()) {
                -1 => Err(Error::SysError(-errno::errno().0)),
                d => Ok(d),
            }
        }
    }

    /// Change UID and GID of slave pty associated with master pty whose
    /// fd is provided, to the real UID and real GID of the calling thread.
    pub fn grantpt(&self) -> Result<i32> {
        unsafe {
            match libc::grantpt(self.as_raw_fd()) {
                -1 => Err(Error::SysError(-errno::errno().0)),
                c => Ok(c),
            }
        }
    }

    /// Unlock the slave pty associated with the master to which fd refers.
    pub fn unlockpt(&self) -> Result<i32> {
        unsafe {
            match libc::unlockpt(self.as_raw_fd()) {
                -1 => Err(Error::SysError(-errno::errno().0)),
                c => Ok(c),
            }
        }
    }

    /// Returns a pointer to a static buffer, which will be overwritten on
    /// subsequent calls.
    pub fn ptsname(&self) -> Result<*const libc::c_char> {
        unsafe {
            match libc::ptsname(self.as_raw_fd()) {
                c if c.is_null() => Err(Error::Common("ptsname fail with NULL".to_string())),
                c => Ok(c),
            }
        }
    }

    pub fn NewSlave(&self) -> Result<Slave> {
        let ptsname = self.ptsname()?;
        let res = Slave::new(ptsname);
        return res;
    }
}

impl Descriptor for Master {}

impl AsRawFd for Master {
    /// The accessor function `as_raw_fd` returns the fd.
    fn as_raw_fd(&self) -> RawFd {
        self.pty
    }
}

impl io::Read for Master {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        unsafe {
            match libc::read(self.as_raw_fd(),
                             buf.as_mut_ptr() as *mut libc::c_void,
                             buf.len()) {
                -1 => Ok(0),
                len => Ok(len as usize),
            }
        }
    }
}

impl io::Write for Master {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe {
            match libc::write(self.as_raw_fd(),
                              buf.as_ptr() as *const libc::c_void,
                              buf.len()) {
                -1 => Err(io::Error::last_os_error()),
                ret => Ok(ret as usize),
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Slave {
    pub pty: RawFd,
}

impl Slave {
    /// The constructor function `new` returns the Slave interface.
    pub fn new(path: *const ::libc::c_char) -> Result<Self> {
        info!("Slave::new");
        match Self::open(path, libc::O_RDWR, None) {
            Err(e) => Err(e),
            Ok(fd) => Ok(Slave { pty: fd }),
        }
    }

    pub fn dup2(&self, std: i32) -> Result<i32> {
        unsafe {
            match libc::dup2(self.as_raw_fd(), std) {
                -1 => Err(Error::SysError(-errno::errno().0)),
                d => Ok(d),
            }
        }
    }

    pub fn dup(&self) -> Result<i32> {
        unsafe {
            match libc::dup(self.as_raw_fd()) {
                -1 => Err(Error::SysError(-errno::errno().0)),
                d => Ok(d),
            }
        }
    }
}

impl Descriptor for Slave {}

impl AsRawFd for Slave {
    /// The accessor function `as_raw_fd` returns the fd.
    fn as_raw_fd(&self) -> RawFd {
        self.pty
    }
}
/*
impl Drop for Slave {
    fn drop(&mut self) {
        Descriptor::drop(self);
    }
}*/