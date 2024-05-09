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

use nix::sys::socket::ControlMessageOwned;
use nix::sys::socket::{recvmsg, MsgFlags};
use nix::sys::uio::IoVec;
use std::collections::BTreeMap;
use std::io::IoSlice;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use std::os::unix::net::SocketAncillary;
use std::os::unix::net::UnixStream as StdStream;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

// use tokio::net::TcpStream;
use tokio::net::{TcpSocket, TcpStream, UnixStream};
use tokio::sync::oneshot;

use crate::common::*;
use crate::tsot_msg::*;
use tokio::sync::{mpsc, Notify};

#[derive(Debug)]
pub struct ConnectResp {
    pub error: String,
}

#[derive(Debug)]
pub struct TsotClient {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub inputTx: mpsc::Sender<TsotMessage>,
    pub inputRx: Mutex<Option<mpsc::Receiver<TsotMessage>>>,

    pub stream: UnixStream,
    pub uid: [u8; 16],

    pub nextReqId: AtomicU32,

    pub requests: Mutex<BTreeMap<u32, oneshot::Sender<ConnectResp>>>,
}

impl TsotClient {
    pub async fn New() -> Result<Self> {
        let stream = UnixStream::connect(TSOT_HOST_SOCKET_PATH).await?;
        let (tx, rx) = mpsc::channel::<TsotMessage>(30);

        let client = Self {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            inputTx: tx,
            inputRx: Mutex::new(Some(rx)),

            stream: stream,
            uid: uuid::Uuid::new_v4().into_bytes(),

            nextReqId: AtomicU32::new(1),
            requests: Mutex::new(BTreeMap::new()),
        };

        let gatewayRegister = GatewayRegisterReq {
            gatewayUid: client.uid.clone(),
        };

        client
            .SendMsg(TsotMessage::from(TsotMsg::GatewayRegisterReq(
                gatewayRegister,
            )))
            .unwrap();

        client.WaitforRegisterResp().await.unwrap();

        return Ok(client);
    }

    pub async fn WaitforRegisterResp(&self) -> Result<()> {
        let mut readBuf: [u8; BUFF_SIZE] = [0; BUFF_SIZE];
        let readbufAddr = &readBuf[0] as *const _ as u64;
        let mut offset = 0;
        while offset != BUFF_SIZE {
            self.stream.readable().await?;
            let cnt = self.stream.try_read(&mut readBuf[offset..])?;
            offset += cnt;
        }

        let msg = unsafe { *(readbufAddr as *const TsotMsg) };

        match msg {
            TsotMsg::GatewayRegisterResp(resp) => {
                if resp.errorCode != 0 {
                    return Err(Error::CommonError(format!(
                        "TsotClient init fail with error {}",
                        resp.errorCode
                    )));
                }
            }
            m => {
                return Err(Error::CommonError(format!(
                    "TsotClient get unexpect message {:?}",
                    m
                )))
            }
        }

        return Ok(());
    }

    pub fn Close(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }

    pub async fn Process(&self) {
        match self.ProcessMsgs().await {
            Ok(()) => (),
            Err(Error::SocketClose) => {
                self.stop.store(false, Ordering::SeqCst);
            }
            Err(e) => {
                error!("TsotClient process fail with error {:?}", e);
                return;
            }
        }
    }

    pub async fn ProcessMsgs(&self) -> Result<()> {
        let mut rx = self.inputRx.lock().unwrap().take().unwrap();

        let mut msg: Option<TsotMessage> = None;

        loop {
            match msg.take() {
                None => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            break;
                        }
                        _res = self.stream.readable() => {
                            self.ProcessRead()?;
                        }
                        m = rx.recv() => {
                            match m {
                                None => (),
                                Some(m) => {
                                    msg = Some(m);
                                }
                            }
                        }
                    }
                }
                Some(m) => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            break;
                        }
                        _ = self.stream.readable() => {
                            self.ProcessRead()?;

                            // return the msg
                            msg = Some(m);
                        }
                        _ = self.stream.writable() => {
                            self.SendMsg(m)?;
                        }
                    }
                }
            }
        }

        return Ok(());
    }

    const MAX_FILES: usize = 16 * 4;
    pub fn ReadWithFds(&self, socket: i32, buf: &mut [u8]) -> Result<(usize, Vec<i32>)> {
        let iovec = [IoVec::from_mut_slice(buf)];
        let mut space: Vec<u8> = vec![0; Self::MAX_FILES];

        match recvmsg(socket, &iovec, Some(&mut space), MsgFlags::empty()) {
            Ok(msg) => {
                let cnt = msg.bytes;

                let mut iter = msg.cmsgs();
                match iter.next() {
                    Some(ControlMessageOwned::ScmRights(fds)) => return Ok((cnt, fds.to_vec())),
                    None => return Ok((cnt, Vec::new())),
                    _ => return Ok((cnt, Vec::new())),
                }
            }
            Err(errno) => return Err(Error::SysError(errno as i32)),
        };
    }

    pub fn ProcessRead(&self) -> Result<()> {
        let mut readBuf: [u8; BUFF_SIZE * 2] = [0; BUFF_SIZE * 2];
        let readbufAddr = &readBuf[0] as *const _ as u64;
        let raw_fd: RawFd = self.stream.as_raw_fd();

        defer!(
            // reset the tokio::unixstream state
            let mut buf =[0; 0];
            self.stream.try_read(&mut buf).ok();
        );
        let (size, fds) = match self.ReadWithFds(raw_fd, &mut readBuf) {
            Ok((size, fds)) => (size, fds),
            Err(Error::SysError(11)) => {
                // EAGAIN
                return Ok(());
            }
            Err(e) => {
                return Err(e);
            }
        };

        if size == 0 {
            return Err(Error::SocketClose);
        }

        let msg = unsafe { *(readbufAddr as *const TsotMsg) };

        if fds.len() == 0 {
            self.ProcessMsg(msg, None)?;
        } else {
            self.ProcessMsg(msg, Some(fds[0]))?;
        }

        return Ok(());
    }

    pub fn ProcessMsg(&self, msg: TsotMsg, _socket: Option<RawFd>) -> Result<()> {
        match msg {
            TsotMsg::GatewayConnectResp(m) => {
                let reqId = m.reqId;
                let tx = self.requests.lock().unwrap().remove(&reqId);
                match tx {
                    None => {
                        return Err(Error::CommonError(format!(
                            "TsotClient ProcessMsg get unknown reqid {}",
                            reqId
                        )))
                    }
                    Some(tx) => {
                        let err = if m.errorCode == 0 {
                            String::new()
                        } else {
                            format!("TsotClient connect fail with error {}", m.errorCode)
                        };
                        tx.send(ConnectResp { error: err }).unwrap();
                    }
                }
            }
            m => {
                error!("ProcessMsg get unimplement msg {:?}", &m);
                unimplemented!()
            }
        }

        return Ok(());
    }

    pub fn SendMsg(&self, msg: TsotMessage) -> Result<()> {
        let socket = msg.socket;
        let msgAddr = &msg.msg as *const _ as u64 as *const u8;
        let writeBuf = unsafe { std::slice::from_raw_parts(msgAddr, BUFF_SIZE) };

        let bufs = &[IoSlice::new(writeBuf)][..];

        let raw_fd: RawFd = self.stream.as_raw_fd();
        let stdStream: StdStream = unsafe { StdStream::from_raw_fd(raw_fd) };

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
            Ok(s) => s,
        };

        assert!(size == BUFF_SIZE);

        return Ok(());
    }

    pub fn EnqMsg(&self, msg: TsotMessage) -> Result<()> {
        match self.inputTx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_e) => {
                return Err(Error::MpscSendFull(format!(
                    "TsotClient Enqueue message fulll"
                )));
            }
        }
    }

    pub async fn Connect(
        &self,
        tenant: &str,
        namespace: &str,
        ipAddr: [u8; 4],
        port: u16,
    ) -> Result<TcpStream> {
        let tcpSocket = TcpSocket::new_v4()?;
        let sock = tcpSocket.as_raw_fd();

        let podNamespace = format!("{}/{}", tenant, namespace);

        assert!(podNamespace.len() < 64);

        let reqId = self.nextReqId.fetch_add(1, Ordering::SeqCst);
        let mut req = GatewayConnectReq {
            reqId: reqId,
            podNamespace: [0; 64],
            dstIp: IpAddress::New(&ipAddr).0,
            dstPort: port,
            srcPort: 123,
        };

        for i in 0..podNamespace.len() {
            req.podNamespace[i] = podNamespace.as_bytes()[i];
        }

        let msg = TsotMessage {
            socket: sock,
            msg: TsotMsg::GatewayConnectReq(req),
        };

        let (tx, rx) = oneshot::channel();

        self.requests.lock().unwrap().insert(reqId, tx);

        self.EnqMsg(msg)?;

        match rx.await {
            Ok(v) => {
                if v.error.len() == 0 {
                    // take ownership
                    let sock = tcpSocket.into_raw_fd();
                    let stdstream = unsafe { std::net::TcpStream::from_raw_fd(sock) };
                    let stream = tokio::net::TcpStream::from_std(stdstream)?;
                    return Ok(stream);
                } else {
                    return Err(Error::CommonError(v.error));
                }
            }
            Err(e) => {
                return Err(Error::CommonError(format!("{:?}", e)));
            }
        }
    }
}
