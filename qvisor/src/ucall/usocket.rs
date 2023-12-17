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
use alloc::string::ToString;
use libc::*;
use nix::sys::socket::ControlMessageOwned;
use nix::sys::socket::{recvmsg, sendmsg, ControlMessage, MsgFlags};
use nix::sys::uio::IoVec;
use std::{thread, time};

use super::super::qlib::common::*;
use super::super::qlib::control_msg::*;
use super::super::qlib::cstring::*;
use super::super::qlib::linux_def::*;
use super::super::URING_MGR;
use super::ucall::*;

#[derive(Debug)]
pub struct USocket {
    pub socket: i32,
}

impl USocket {
    pub fn Drop(&self) {
        if self.socket == -1 {
            return;
        }

        URING_MGR.lock().Removefd(self.socket).unwrap();

        unsafe {
            close(self.socket);
        }
    }

    // this is designed for the QVisor signal sending to QKernel.
    // As there is no real unix connection setup, the SendResponse won't work
    pub fn DummyUSocket() -> Self {
        return Self { socket: -1 };
    }

    pub fn CreateServerSocket(path: &str) -> Result<i32> {
        let mut server = sockaddr_un {
            sun_family: AF_UNIX as u16,
            sun_path: [0; 108],
        };

        let cstr = CString::New(path);
        let slice = cstr.Slice();
        server.sun_path[0..slice.len()].copy_from_slice(slice);

        let sock = unsafe { socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0) };

        if sock < 0 {
            info!("USrvSocket create socket fail");
            return Err(Error::SysError(errno::errno().0 as i32));
        }

        let ret = unsafe {
            bind(
                sock,
                &server as *const _ as *const sockaddr,
                110, /*sizeof(sockaddr_un)*/
            )
        };

        if ret < 0 {
            info!("USrvSocket bind socket fail");
            return Err(Error::SysError(errno::errno().0 as i32));
        }

        let ret = unsafe { listen(sock, 5) };

        if ret < 0 {
            info!("USrvSocket listen socket fail");
            return Err(Error::SysError(-errno::errno().0 as i32));
        }

        return Ok(sock);
    }

    pub fn InitClient(path: &str) -> Result<Self> {
        let mut server = sockaddr_un {
            sun_family: AF_UNIX as u16,
            sun_path: [0; 108],
        };

        let cstr = CString::New(path);
        let slice = cstr.Slice();
        server.sun_path[0..slice.len()].copy_from_slice(slice);

        let sock = unsafe { socket(AF_UNIX, SOCK_STREAM, 0) };

        if sock < 0 {
            info!("UCliSocket create socket fail");
            return Err(Error::SysError(errno::errno().0 as i32));
        }

        let cliSocket = Self { socket: sock };

        let waitCnt = 2;
        for i in 0..waitCnt {
            let ret = unsafe {
                connect(
                    sock,
                    &server as *const _ as *const sockaddr,
                    110, /*sizeof(sockaddr_un)*/
                )
            };

            if ret < 0 {
                info!(
                    "UCliSocket connect socket fail, path is {}, error is {}",
                    path,
                    errno::errno().0
                );
                if errno::errno().0 != SysErr::ECONNREFUSED || i == waitCnt - 1 {
                    return Err(Error::SysError(errno::errno().0 as i32));
                }
            } else {
                break;
            }

            let hundred_millis = time::Duration::from_millis(100);
            thread::sleep(hundred_millis);
        }

        return Ok(cliSocket);
    }

    const MAX_FILES: usize = 16 * 4;

    pub fn ReadWithFds(&self, buf: &mut [u8]) -> Result<(usize, Vec<i32>)> {
        let iovec = [IoVec::from_mut_slice(buf)];
        let mut space: Vec<u8> = vec![0; Self::MAX_FILES];

        loop {
            match recvmsg(self.socket, &iovec, Some(&mut space), MsgFlags::empty()) {
                Ok(msg) => {
                    let cnt = msg.bytes;

                    let mut iter = msg.cmsgs();
                    match iter.next() {
                        Some(ControlMessageOwned::ScmRights(fds)) => {
                            return Ok((cnt, fds.to_vec()))
                        }
                        None => return Ok((cnt, Vec::new())),
                        _ => break,
                    }
                }
                Err(errno) => {
                    if errno as i32 == EINTR {
                        continue;
                    }
                    return Err(Error::IOError(format!(
                        "ReadWithFds io::error is {:?}",
                        errno
                    )));
                }
            };
        }

        return Err(Error::IOError("ReadWithFds can't get fds".to_string()));
    }

    //read buf.len() data
    pub fn ReadAll(&self, buf: &mut [u8]) -> Result<()> {
        let mut len = buf.len();
        while len > 0 {
            let cnt = unsafe { read(self.socket, &mut buf[0] as *mut _ as *mut c_void, len) };

            if cnt < 0 {
                info!("UCliSocket read socket fail");
                return Err(Error::SysError(errno::errno().0 as i32));
            }

            len -= cnt as usize;
        }

        return Ok(());
    }

    pub fn WriteWithFds(&self, buf: &[u8], fds: &[i32]) -> Result<usize> {
        let iov = [IoVec::from_slice(&buf)];
        let cmsg = [ControlMessage::ScmRights(&fds)];
        let size = sendmsg(self.socket, &iov, &cmsg, MsgFlags::empty(), None)
            .map_err(|e| Error::IOError(format!("WriteWithFds io::error is {:?}", e)))?;

        return Ok(size);
    }

    //write buf.len() data
    pub fn WriteAll(&self, buf: &[u8]) -> Result<()> {
        let mut len = buf.len();
        while len > 0 {
            let cnt = unsafe { write(self.socket, &buf[0] as *const _ as *const c_void, len) };

            if cnt < 0 {
                info!("UCliSocket read socket fail");
                return Err(Error::SysError(-errno::errno().0 as i32));
            }

            len -= cnt as usize;
        }

        return Ok(());
    }

    pub fn WriteLen(&self, len: usize, fds: &[i32]) -> Result<()> {
        let len = len as u32;
        let ptr = &len as *const _ as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, 4) };

        let size = self.WriteWithFds(slice, fds)?;

        if size == slice.len() {
            return Ok(());
        }

        return self.WriteAll(&slice[size..]);
    }

    pub fn ReadLen(&self) -> Result<(usize, Vec<i32>)> {
        let mut len: u32 = 0;
        let ptr = &mut len as *mut _ as *mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, 4) };
        let (size, fds) = self.ReadWithFds(slice)?;

        if size < slice.len() {
            let slice = &mut slice[size..];
            self.ReadAll(slice)?;
        }

        return Ok((len as usize, fds));
    }

    pub fn GetReq(&self) -> Result<(UCallReq, Vec<i32>)> {
        let (len, fds) = self.ReadLen()?;
        let mut buf: [u8; UCALL_BUF_LEN] = [0; UCALL_BUF_LEN];

        assert!(len < UCALL_BUF_LEN, "UCallClient::Call req is too long");
        self.ReadAll(&mut buf[0..len])?;
        let req: UCallReq = serde_json::from_slice(&buf[0..len])
            .map_err(|e| Error::Common(format!("UCallSrv deser error is {:?}", e)))?;

        return Ok((req, fds));
    }

    pub fn SendResp(&self, resp: &UCallResp) -> Result<()> {
        if self.socket == -1 {
            return Ok(());
        }

        let req = serde_json::to_vec(resp)
            .map_err(|e| Error::Common(format!("UCallSrv ser error is {:?}", e)))?;
        self.WriteLen(req.len(), &[])?;
        self.WriteAll(&req)?;
        return Ok(());
    }
}
