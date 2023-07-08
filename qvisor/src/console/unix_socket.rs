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

use alloc::string::ToString;
use libc::*;
use std::mem;
use std::os::unix::io::{AsRawFd, RawFd};

use super::super::qlib::common::*;
use super::super::qlib::cstring::*;

#[repr(C)]
union HeaderAlignedBuf {
    // CMSG_SPACE(mem::size_of::<c_int>()) = 24 (linux x86_64),
    buf: [libc::c_char; 256],
    align: libc::cmsghdr,
}

pub struct UnixSocket(RawFd);

impl AsRawFd for UnixSocket {
    /// The accessor function `as_raw_fd` returns the fd.
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

impl Drop for UnixSocket {
    fn drop(&mut self) {
        unsafe {
            close(self.0);
        }
    }
}

impl UnixSocket {
    pub fn NewServer(path: &str) -> Result<Self> {
        let mut server = sockaddr_un {
            sun_family: AF_UNIX as u16,
            sun_path: [0; 108],
        };

        let cstr = CString::New(path);
        let slice = cstr.Slice();

        #[cfg(target_arch="x86_64")]
        for i in 0..slice.len() {
            server.sun_path[i] = slice[i] as i8;
        }
        #[cfg(target_arch="aarch64")]
        server.sun_path.copy_from_slice(slice);

        let sock = unsafe { socket(AF_UNIX, SOCK_STREAM, 0) };

        if sock < 0 {
            info!("USrvSocket create socket fail");
            return Err(Error::SysError(-errno::errno().0 as i32));
        }

        let srvSock = Self(sock);

        let ret = unsafe {
            bind(
                sock,
                &server as *const _ as *const sockaddr,
                110, /*sizeof(sockaddr_un)*/
            )
        };

        if ret < 0 {
            info!("USrvSocket bind socket fail");
            return Err(Error::SysError(-errno::errno().0 as i32));
        }

        let ret = unsafe { listen(sock, 1) };

        if ret < 0 {
            info!("USrvSocket listen socket fail");
            return Err(Error::SysError(-errno::errno().0 as i32));
        }

        return Ok(srvSock);
    }

    pub fn NewClient(path: &str) -> Result<Self> {
        let mut server = sockaddr_un {
            sun_family: AF_UNIX as u16,
            sun_path: [0; 108],
        };

        let cstr = CString::New(path);
        let slice = cstr.Slice();

        #[cfg(target_arch="x86_64")]
        for i in 0..slice.len() {
            server.sun_path[i] = slice[i] as i8;
        }
        #[cfg(target_arch="aarch64")]
        server.sun_path.copy_from_slice(slice);

        let sock = unsafe { socket(AF_UNIX, SOCK_STREAM, 0) };

        if sock < 0 {
            info!("UCliSocket create socket fail");
            return Err(Error::SysError(-errno::errno().0 as i32));
        }

        let cliSocket = Self(sock);

        let ret = unsafe {
            connect(
                sock,
                &server as *const _ as *const sockaddr,
                110, /*sizeof(sockaddr_un)*/
            )
        };

        if ret < 0 {
            info!("UCliSocket connect socket fail, path is {}", path);
            return Err(Error::SysError(-errno::errno().0 as i32));
        }

        return Ok(cliSocket);
    }

    pub fn SendFd(&self, fd: RawFd) -> Result<()> {
        let mut dummy: c_int = 0;
        let msg_len = unsafe { libc::CMSG_SPACE(mem::size_of::<c_int>() as u32) as _ };
        let mut u = HeaderAlignedBuf { buf: [0; 256] };
        let mut iov = libc::iovec {
            iov_base: &mut dummy as *mut c_int as *mut c_void,
            iov_len: mem::size_of_val(&dummy),
        };

        let msg: msghdr = libc::msghdr {
            msg_name: std::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: &mut iov,
            msg_iovlen: 1,
            msg_control: unsafe { u.buf.as_mut_ptr() as *mut c_void },
            msg_controllen: msg_len,
            msg_flags: 0,
        };

        unsafe {
            let hdr = libc::cmsghdr {
                cmsg_level: libc::SOL_SOCKET,
                cmsg_type: libc::SCM_RIGHTS,
                cmsg_len: libc::CMSG_LEN(mem::size_of::<c_int>() as u32) as _,
            };
            // https://github.com/rust-lang/rust-clippy/issues/2881
            #[allow(clippy::cast_ptr_alignment)]
            std::ptr::write_unaligned(libc::CMSG_FIRSTHDR(&msg), hdr);

            // https://github.com/rust-lang/rust-clippy/issues/2881
            #[allow(clippy::cast_ptr_alignment)]
            std::ptr::write_unaligned(
                libc::CMSG_DATA(u.buf.as_mut_ptr() as *const _) as *mut c_int,
                fd,
            );
        }

        let rv = unsafe { libc::sendmsg(self.0, &msg, 0) };
        if rv < 0 {
            return Err(Error::SysError(-errno::errno().0));
        }

        Ok(())
    }

    pub fn RecvFd(&self) -> Result<RawFd> {
        let mut dummy: c_int = -1;
        let msg_len = unsafe { libc::CMSG_SPACE(mem::size_of::<c_int>() as u32) as _ };
        let mut u = HeaderAlignedBuf { buf: [0; 256] };
        let mut iov = libc::iovec {
            iov_base: &mut dummy as *mut c_int as *mut c_void,
            iov_len: mem::size_of_val(&dummy),
        };
        let mut msg: msghdr = libc::msghdr {
            msg_name: std::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: &mut iov,
            msg_iovlen: 1,
            msg_control: unsafe { u.buf.as_mut_ptr() as *mut c_void },
            msg_controllen: msg_len,
            msg_flags: 0,
        };

        unsafe {
            let rv = libc::recvmsg(self.0, &mut msg, 0);
            match rv {
                0 => Err(Error::Common("UnExpected Eof".to_string())),
                rv if rv < 0 => return Err(Error::SysError(-errno::errno().0)),
                rv if rv == mem::size_of::<c_int>() as isize => {
                    let hdr: *mut libc::cmsghdr =
                        if msg.msg_controllen >= mem::size_of::<libc::cmsghdr>() as _ {
                            msg.msg_control as *mut libc::cmsghdr
                        } else {
                            return Err(Error::Common("bad control msg (header)".to_string()));
                        };

                    if (*hdr).cmsg_level != libc::SOL_SOCKET || (*hdr).cmsg_type != libc::SCM_RIGHTS
                    {
                        return Err(Error::Common("bad control msg (level)".to_string()));
                    }

                    if msg.msg_controllen
                        != libc::CMSG_SPACE(mem::size_of::<c_int>() as u32) as usize
                    {
                        return Err(Error::Common("bad control msg (len)".to_string()));
                    }

                    #[allow(clippy::cast_ptr_alignment)]
                    let fd = std::ptr::read_unaligned(libc::CMSG_DATA(hdr) as *mut c_int);
                    if libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) < 0 {
                        return Err(Error::SysError(-errno::errno().0));
                    }
                    return Ok(fd);
                }
                _ => {
                    return Err(Error::Common("bad control msg (ret code)".to_string()));
                }
            }
        }
    }
}
