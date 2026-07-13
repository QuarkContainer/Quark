// Copyright (c) 2023 Quark Container Authors / 2018 The gVisor Authors.
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

use std::net::Ipv4Addr;
use std::net::SocketAddrV4;
use std::os::fd::AsRawFd;
use std::os::fd::IntoRawFd;
use std::os::unix::io::FromRawFd;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::net::TcpSocket;
use tokio::net::TcpStream;
use tokio::sync::Notify;

use qshare::common::*;
use qshare::tsot_msg::ErrCode;

use super::cidr_util::{IsEgressIp, IsPodIp};
use super::peer_mgr::PEER_MGR;
use super::pod_broker::PodBroker;
use super::pod_broker_mgr::POD_BRORKER_MGRS;

fn leak_connected_fd(stream: TcpStream) {
    std::mem::forget(stream);
}


#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsotErrCode {
    Ok = 0,
    Reject = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct TsotConnReq {
    pub podNamespace: [u8; 64],
    pub dstIp: u32,
    pub dstPort: u16,
    pub srcIp: u32,
    pub srcPort: u16,
}

impl TsotConnReq {
    pub fn GetNamespace(&self) -> Result<String> {
        for i in 0..self.podNamespace.len() {
            if self.podNamespace[i] == 0 {
                if i == 0 {
                    return Ok("Default".to_owned());
                }
                let str = std::str::from_utf8(&self.podNamespace[0..i])?;
                return Ok(str.to_owned());
            }
        }

        let str = std::str::from_utf8(&self.podNamespace)?;
        return Ok(str.to_owned());
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TsotConnResp {
    pub errcode: u32,
}

/// the service to waiting for peer tcp connection
pub struct ConnectionSvc {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub port: u16,
}

impl ConnectionSvc {
    pub fn New(port: u16) -> Self {
        return Self {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            port: port,
        };
    }

    pub fn Close(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }

    pub async fn Process(&self) -> Result<()> {
        //let addr = format!("0.0.0.0:{}", self.port);
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr).await?;

        loop {
            // tokio::select! {
            //     _ = self.closeNotify.notified() => {
            //         self.stop.store(false, Ordering::SeqCst);
            //         return Ok(())
            //     }
            //     res = listener.accept() => {
            //         error!("ConnectionSvc::accept {:?}", &res);
            //         match res {
            //             Err(_) => continue,
            //             Ok((stream, _peerAddr)) => {
            //                 tokio::spawn(async move {
            //                     let conn = TcpSvcConnection::New(stream);
            //                     match conn.Process().await {
            //                         Err(e) => error!("ConnectionSvc::ProcessConnection fail with error {:?}", e),
            //                         Ok(_) => (),
            //                     }
            //                 });
            //             }
            //         }
            //     }
            // }

            let res = listener.accept().await;

            error!("ConnectionSvc 1 get connection ...");
            match res {
                Err(_) => continue,
                Ok((stream, _peerAddr)) => {
                    tokio::spawn(async move {
                        let conn = TcpSvcConnection::New(stream);
                        match conn.Process().await {
                            Err(e) => {
                                error!("ConnectionSvc::ProcessConnection fail with error {:?}", e)
                            }
                            Ok(_) => (),
                        }
                    });
                }
            }
        }
    }
}

pub struct TcpSvcConnection {
    pub stream: TcpStream,
}

impl TcpSvcConnection {
    pub fn New(stream: TcpStream) -> Self {
        return Self { stream: stream };
    }

    pub async fn ProcessConnectionInner(&self) -> Result<()> {
        error!("TcpSvcConnection ProcessConnectionInner 1");
        let connReq = self.ReadConnReq().await?;

        error!("TcpSvcConnection ProcessConnectionInner 2 {:?}", &connReq);
        let namespace = connReq.GetNamespace()?;
        let socket = self.stream.as_raw_fd();

        POD_BRORKER_MGRS.HandlePeerConnect(
            &namespace,
            connReq.dstIp,
            connReq.dstPort,
            connReq.srcIp,
            connReq.srcPort,
            socket,
        )?;
        return Ok(());
    }

    pub async fn Process(self) -> Result<()> {
        match self.ProcessConnectionInner().await {
            Err(_e) => {
                let resp = TsotConnResp {
                    errcode: TsotErrCode::Reject as _,
                };

                self.WriteConnResp(resp).await?;
                return Ok(());
            }
            Ok(()) => {
                let resp = TsotConnResp {
                    errcode: TsotErrCode::Ok as _,
                };

                self.WriteConnResp(resp).await?;
                // PeerConnectNotify duplicates this fd to the server guest via SCM_RIGHTS.
                // Drop our relay copy so TCP teardown (FIN) can propagate to the client.
                drop(self.stream);
                return Ok(());
            }
        }
    }

    pub async fn ReadConnReq(&self) -> Result<TsotConnReq> {
        const REQ_SIZE: usize = std::mem::size_of::<TsotConnReq>();
        let mut readBuf = [0; REQ_SIZE];
        let mut offset = 0;

        while offset < REQ_SIZE {
            self.stream.readable().await?;
            let cnt = self.stream.try_read(&mut readBuf[offset..])?;
            offset += cnt;
        }

        let msg = unsafe { *(&readBuf[0] as *const _ as u64 as *const TsotConnReq) };

        return Ok(msg);
    }

    pub async fn WriteConnResp(&self, resp: TsotConnResp) -> Result<()> {
        const RESP_SIZE: usize = std::mem::size_of::<TsotConnResp>();
        let addr = &resp as *const _ as u64 as *const u8;
        let buf = unsafe { std::slice::from_raw_parts(addr, RESP_SIZE) };
        let mut offset = 0;

        while offset < RESP_SIZE {
            self.stream.writable().await?;
            let cnt = self.stream.try_write(&buf[offset..])?;
            offset += cnt;
        }

        return Ok(());
    }
}

pub struct TcpClientConnection {
    pub podBroker: PodBroker,
    pub isPodConnection: bool,

    pub socket: i32,

    pub reqId: u32,
    pub podNamespace: String,
    pub dstIp: u32,
    pub dstPort: u16,
    pub srcIp: u32,
    pub srcPort: u16,
}

impl TcpClientConnection {
    pub async fn PodConnectProcess(self) {
        let podBroker = self.podBroker.clone();
        let res = if IsEgressIp(self.dstIp) {
            self.ConnectDirect().await
        } else {
            self.ProcessConnection()
                .await
                .map(|stream| leak_connected_fd(stream))
        };
        match res {
            Ok(()) => {
                podBroker
                    .HandlePodConnectResp(self.reqId, ErrCode::None as i32)
                    .unwrap();
            }
            Err(e) => {
                error!(
                    "PodConnectProcess reqId {} dst {:x}:{} failed {:?}",
                    self.reqId, self.dstIp, self.dstPort, e
                );
                podBroker
                    .HandlePodConnectResp(self.reqId, ErrCode::ECONNREFUSED as i32)
                    .unwrap();
            }
        }
    }

    pub async fn GatewayConnectProcess(self) {
        let podBroker = self.podBroker.clone();
        let res = if IsEgressIp(self.dstIp) {
            self.ConnectDirect().await
        } else {
            self.ProcessConnection()
                .await
                .map(|stream| leak_connected_fd(stream))
        };
        match res {
            Ok(()) => {
                podBroker
                    .HandleGatewayConnectResp(self.reqId, ErrCode::None as i32)
                    .unwrap();
            }
            Err(_e) => {
                podBroker
                    .HandleGatewayConnectResp(self.reqId, ErrCode::ECONNREFUSED as i32)
                    .unwrap();
            }
        }
    }

    pub async fn ProcessConnection(&self) -> Result<TcpStream> {
        let stream = self.Connect().await?;
        let mut req = TsotConnReq {
            podNamespace: [0; 64],
            dstIp: self.dstIp,
            dstPort: self.dstPort,
            srcIp: self.srcIp,
            srcPort: self.srcPort,
        };

        for i in 0..self.podNamespace.as_bytes().len() {
            req.podNamespace[i] = self.podNamespace.as_bytes()[i];
        }

        self.WriteConnReq(&stream, req).await?;

        let resp = self.ReadConnResp(&stream).await?;

        if resp.errcode != TsotErrCode::Ok as u32 {
            return Err(Error::CommonError(format!(
                "TcpClientConnection connect fail with error {:?}",
                resp.errcode
            )));
        }

        return Ok(stream);
    }

    /// Connect the guest fd directly to an external (non-pod) destination.
    ///
    /// The fd was created by tokio (`TcpSocket::new_v4`) and is already non-blocking.
    /// We reconstruct a `TcpSocket` from it so tokio's async connect drives the
    /// EINPROGRESS → ready cycle properly without any blocking-mode gymnastics.
    pub async fn ConnectDirect(&self) -> Result<()> {
        let socket = unsafe { TcpSocket::from_raw_fd(self.socket) };

        if self.srcIp != 0 {
            let src_addr = SocketAddrV4::new(Ipv4Addr::from(self.srcIp), 0);
            socket.bind(src_addr.into()).map_err(|e| {
                error!("ConnectDirect bind pod src {:x} failed {:?}", self.srcIp, e);
                Error::from(e)
            })?;
        }

        let dst_addr = SocketAddrV4::new(Ipv4Addr::from(self.dstIp), self.dstPort);
        let stream = socket.connect(dst_addr.into()).await.map_err(|e| {
            error!(
                "ConnectDirect to {:x}:{} failed {:?}",
                self.dstIp, self.dstPort, e
            );
            Error::from(e)
        })?;

        leak_connected_fd(stream);
        Ok(())
    }

    pub async fn Connect(&self) -> Result<TcpStream> {
        if !IsPodIp(self.dstIp) {
            return Err(Error::NotExist(format!(
                "Connect relay requires pod dst ip {:x}",
                self.dstIp
            )));
        }
        let peer = PEER_MGR.LookforPeer(self.dstIp)?;
        let relay_addr = SocketAddrV4::new(Ipv4Addr::from(peer.hostIp), peer.port);

        let socket = unsafe { TcpSocket::from_raw_fd(self.socket) };
        let stream = socket.connect(relay_addr.into()).await.map_err(|e| {
            error!(
                "Connect relay to {:x}:{} for pod dst {:x}:{} failed {:?}",
                peer.hostIp, peer.port, self.dstIp, self.dstPort, e
            );
            Error::from(e)
        })?;
        return Ok(stream);
    }

    pub async fn ReadConnResp(&self, stream: &TcpStream) -> Result<TsotConnResp> {
        const RESP_SIZE: usize = std::mem::size_of::<TsotConnResp>();
        let mut readBuf = [0; RESP_SIZE];
        let mut offset = 0;

        while offset < RESP_SIZE {
            stream.readable().await?;
            let cnt = stream.try_read(&mut readBuf[offset..])?;
            offset += cnt;
        }

        let msg = unsafe { *(&readBuf[0] as *const _ as u64 as *const TsotConnResp) };

        return Ok(msg);
    }

    pub async fn WriteConnReq(&self, stream: &TcpStream, resp: TsotConnReq) -> Result<()> {
        const REQ_SIZE: usize = std::mem::size_of::<TsotConnReq>();
        let addr = &resp as *const _ as u64 as *const u8;
        let buf = unsafe { std::slice::from_raw_parts(addr, REQ_SIZE) };
        let mut offset = 0;

        while offset < REQ_SIZE {
            stream.writable().await?;
            let cnt = stream.try_write(&buf[offset..])?;
            offset += cnt;
        }

        return Ok(());
    }
}
