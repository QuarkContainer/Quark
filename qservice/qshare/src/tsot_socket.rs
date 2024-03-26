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

// TsotSocket is a tcp client socket initialized from admin process such as mulitenant gateway

use std::io::IoSlice;
use std::os::fd::{FromRawFd, IntoRawFd, AsRawFd, RawFd};
use std::os::unix::net::UnixStream as StdStream;
use std::os::unix::net::SocketAncillary;

// use tokio::net::TcpStream;
use tokio::net::UnixStream;

use crate::common::*;
use crate::tsot_msg::*;

pub struct TsotClient {
    pub stream: UnixStream,
    pub uid: [u8; 16]
}

impl TsotClient {
    pub async fn Init() -> Result<Self> {
        let stream = UnixStream::connect(TSOT_HOST_SOCKET_PATH).await?;

        // let gatewayRegister = GatewayRegisterReq {
        //     gatewayUid: uuid::Uuid::new_v4().into_bytes()
        // };

        return Ok(Self{
            stream: stream,
            uid: uuid::Uuid::new_v4().into_bytes(),
        })
    }

    pub fn SendMsg(&self, msg: TsotMessage) -> Result<()> {
        let socket = msg.socket;
        let msgAddr = &msg.msg as * const _ as u64 as * const u8;
        let writeBuf = unsafe {
            std::slice::from_raw_parts(msgAddr, BUFF_SIZE)
        };

        let bufs = &[
            IoSlice::new(writeBuf)
        ][..];

        let raw_fd: RawFd = self.stream.as_raw_fd();
        let stdStream : StdStream = unsafe { StdStream::from_raw_fd(raw_fd) };

        let mut ancillary_buffer = [0; 128];
        let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
        let res = if socket >= 0 {
            let fds = [socket];
            ancillary.add_fds(&fds[..]);
            stdStream.send_vectored_with_ancillary(bufs, &mut ancillary)
        } else {
            stdStream.send_vectored_with_ancillary(bufs, &mut ancillary)
        };

        // take ownership of stdstream to avoid fd close
        let _ = stdStream.into_raw_fd();

        let size = match res {
            Err(e) => {
                return Err(e.into());
            }
            Ok(s) => s
        };

        assert!(size == BUFF_SIZE);

        return Ok(())
    }

    // pub fn Connect(namespace: &str, ipAddr: [u8; 4], port: u16) -> Result<TcpStream> {

    // }
}


impl Drop for TsotMessage {
    fn drop(&mut self) {
        unsafe {
            if self.socket >= 0 {
                libc::close(self.socket);
            }
        }
    }
}