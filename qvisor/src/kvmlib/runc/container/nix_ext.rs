// Copyright (c) 2021 Quark Container Authors
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

// Functions in libc that haven't made it into nix yet
use libc;
use std::os::unix::io::RawFd;
use nix::fcntl::{open, OFlag};
use nix::sys::stat::Mode;
use nix::unistd::{close, write};

use super::super::super::qlib::common::*;
use super::super::super::qlib::cstring::*;

#[inline]
pub fn lsetxattr(
    path: &CString,
    name: &CString,
    value: &CString,
    len: usize,
    flags: i32,
) -> Result<()> {
    let res = unsafe {
        libc::lsetxattr(
            path.Ptr() as *const libc::c_char,
            name.Ptr() as *const libc::c_char,
            value.Ptr() as *const libc::c_void,
            len,
            flags,
        )
    };

    return Error::MapRes(res as i32);
}

#[inline]
pub fn fchdir(fd: RawFd) -> Result<()> {
    let res = unsafe { libc::fchdir(fd) };
    return Error::MapRes(res as i32);
}

#[inline]
pub fn setgroups(gids: &[libc::gid_t]) -> Result<()> {
    let res = unsafe { libc::setgroups(gids.len(), gids.as_ptr()) };
    return Error::MapRes(res as i32);
}

#[inline]
pub fn setrlimit(
    resource: libc::c_int,
    soft: libc::c_ulonglong,
    hard: libc::c_ulonglong,
) -> Result<()> {
    let rlim = &libc::rlimit {
        rlim_cur: soft,
        rlim_max: hard,
    };
    let res = unsafe { libc::setrlimit(resource as u32, rlim) };
    return Error::MapRes(res as i32);
}

#[inline]
pub fn clearenv() -> Result<()> {
    let res = unsafe { libc::clearenv() };
    return Error::MapRes(res as i32);
}

#[cfg(target_env = "gnu")]
#[inline]
pub fn putenv(string: &CString) -> Result<()> {
    // NOTE: gnue takes ownership of the string so we pass it
    //       with into_raw.
    //       This prevents the string to be de-allocated.
    //       According to
    //       https://www.gnu.org/software/libc/manual/html_node/Environment-Access.html
    //       the variable will be accessable from the exec'd program
    //       throughout its lifetime, as such this is not going to be re-claimed
    //       and will show up as leak in valgrind and friends.
    let ptr = string.Ptr();
    let res = unsafe { libc::putenv(ptr as *mut libc::c_char) };
    return Error::MapRes(res as i32);
}

#[cfg(not(target_env = "gnu"))]
pub fn putenv(string: &CString) -> Result<()> {
    let res = unsafe { libc::putenv(string.as_ptr() as *mut libc::c_char) };
    return Error::MapRes(res as i32);
}

const EXEC_PATH: &'static str = "/proc/self/attr/exec";

pub fn setexeccon(label: &str) -> Result<()> {
    let fd = open(EXEC_PATH, OFlag::O_RDWR, Mode::empty()).map_err(|e| Error::IOError(format!("setexeccon error is {:?}", e)))?;
    defer!(close(fd).unwrap());
    write(fd, label.as_bytes()).map_err(|e| Error::IOError(format!("setexeccon error is {:?}", e)))?;
    Ok(())
}

const XATTR_NAME: &'static str = "security.selinux";

pub fn setfilecon(file: &str, label: &str) -> Result<()> {
    let path = CString::New(file);
    let name = CString::New(XATTR_NAME);
    let value = CString::New(label);
    lsetxattr(&path, &name, &value, label.len(), 0)?;
    Ok(())
}
