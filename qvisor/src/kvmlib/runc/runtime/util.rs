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

use nix::fcntl::*;
use nix::unistd::*;
use nix::sys::stat::Mode;
use libc;
use caps::*;

use super::super::super::qlib::common::*;
use super::super::super::util::*;
use super::super::oci::*;

pub fn WriteIDMapping(path: &str, maps: &[LinuxIDMapping])  -> Result<()> {
    let mut data = String::new();
    for m in maps {
        let val = format!("{} {} {}\n", m.container_id, m.host_id, m.size);
        data = data + &val;
    }

    if !data.is_empty() {
        let fd = Open(path, OFlag::O_WRONLY, Mode::empty())?;
        defer!(close(fd).unwrap());
        Write(fd, data.as_bytes())?;
    }
    Ok(())
}

pub fn Open(path: &str, flag: OFlag, mode: Mode) -> Result<i32> {
    return open(path, flag, mode)
        .map_err(|e| Error::IOError(format!("io error is {:?}", e)));
}

pub fn Write(fd: i32, data: &[u8]) -> Result<usize> {
    return write(fd, data)
        .map_err(|e| Error::IOError(format!("io error is {:?}", e)));
}

#[inline]
pub fn SetRLimit(
    resource: u32,
    soft: u64,
    hard: u64,
) -> Result<()> {
    let rlim = &libc::rlimit {
        rlim_cur: soft,
        rlim_max: hard,
    };
    let res = unsafe { libc::setrlimit(resource, rlim) };
    return GetNoRet(res)
}

pub fn Unshare(flags: i32) -> Result<()> {
    let ret = unsafe {
        libc::unshare(flags)
    };

    return GetNoRet(ret)
}

pub fn Close(fd: i32) -> Result<()> {
    let ret = unsafe {
        libc::close(fd)
    };

    return GetNoRet(ret)
}

pub fn SetNamespace(fd: i32, nstype: i32) -> Result<()> {
    let ret = unsafe {
        libc::setns(fd, nstype)
    };

    return GetNoRet(ret)
}

pub fn SetID(uid: u32, gid: u32) -> Result<()> {
    let uid = Uid::from_raw(uid);
    let gid = Gid::from_raw(gid);

    // set uid/gid
    if let Err(e) = prctl::set_keep_capabilities(true) {
        return Err(Error::Common(format!("set keep capabilities returned {}", e)));
    };

    {
        setresgid(gid, gid, gid).map_err(|e| Error::IOError(format!("io error is {:?}", e)))?;
    }
    {
        setresuid(uid, uid, uid).map_err(|e| Error::IOError(format!("io error is {:?}", e)))?;
    }
    // if we change from zero, we lose effective caps
    if uid != Uid::from_raw(0) {
        ResetEffective()?;
    }
    if let Err(e) = prctl::set_keep_capabilities(false) {
        return Err(Error::Common(format!("set keep capabilities returned {}", e)));
    };
    Ok(())
}

pub fn ResetEffective() -> Result<()> {
    return set(None, CapSet::Effective, ::caps::all()).map_err(|e| Error::IOError(format!("io error is {:?}", e)));
}
