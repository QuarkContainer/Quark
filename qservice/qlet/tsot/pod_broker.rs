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

use core::ops::Deref;
use nix::sys::socket::ControlMessageOwned;
use nix::sys::socket::{recvmsg, MsgFlags};
use nix::sys::uio::IoVec;
use std::collections::HashMap;
use std::io::IoSlice;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use std::os::unix::net::SocketAncillary;
use std::os::unix::net::UnixStream as StdStream;
use std::sync::atomic::*;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::net::TcpSocket;
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tokio::sync::Notify;

use qshare::common::*;
use qshare::tsot_msg::*;

use crate::pod_mgr::pod_sandbox::PodSandbox;
use crate::pod_mgr::NAMESPACE_MGR;

use super::conn_svc::TcpClientConnection;
use super::dns_proxy::DnsProxyReq;
use super::dns_proxy::DNS_PROXY;
use super::pod_broker_mgr::POD_BRORKER_MGRS;

#[derive(Debug, Default)]
pub struct PodIdentity {
    pub podUid: String,
    pub namespace: String,
    pub podIp: u32,
}

#[derive(Debug)]
pub struct PodBrokerInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub stream: UnixStream,
    pub buff: [u8; BUFF_SIZE],
    pub offset: AtomicUsize,

    pub outputTx: mpsc::Sender<TsotMessage>,
    pub outputRx: Mutex<Option<mpsc::Receiver<TsotMessage>>>,

    pub podSandbox: Mutex<Option<PodSandbox>>,

    // port to backlogs
    pub listeningPorts: Mutex<HashMap<u16, u32>>,

    // reqId to ConnectReq
    pub connecting: Mutex<HashMap<u32, ConnectReq>>,
}

#[derive(Debug, Clone)]
pub struct PodBroker(Arc<PodBrokerInner>);

impl Deref for PodBroker {
    type Target = Arc<PodBrokerInner>;

    fn deref(&self) -> &Arc<PodBrokerInner> {
        &self.0
    }
}

impl PodBroker {
    pub fn New(stream: UnixStream) -> Self {
        let (tx, rx) = mpsc::channel::<TsotMessage>(30);

        let inner = PodBrokerInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            stream: stream,
            buff: [0; BUFF_SIZE],
            offset: AtomicUsize::new(0),

            outputTx: tx,
            outputRx: Mutex::new(Some(rx)),

            podSandbox: Mutex::new(None),

            listeningPorts: Mutex::new(HashMap::new()),
            connecting: Mutex::new(HashMap::new()),
        };

        return Self(Arc::new(inner));
    }

    pub fn Drop(&self) -> Result<()> {
        match self.podSandbox.lock().unwrap().take() {
            Some(podSandbox) => {
                let inner = podSandbox.lock().unwrap();
                POD_BRORKER_MGRS.RemoveBroker(&inner.namespace, inner.ip.0, &inner.name)?;
            }
            None => (),
        }

        return Ok(());
    }

    pub async fn Init(&self, gatewayRegister: bool) -> Result<()> {
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
            TsotMsg::PodRegisterReq(register) => {
                if gatewayRegister {
                    let resp = PodRegisterResp {
                        containerIp: 0,
                        errorCode: ErrCode::ECONNREFUSED as _,
                    };

                    self.SendMsg(TsotMsg::PodRegisterResp(resp).into())?;
                    return Err(Error::CommonError(
                        "Gateway register can't use PodRegisterReq".to_owned(),
                    ));
                }
                let podUid = uuid::Uuid::from_bytes(register.podUid).to_string();
                let podSandbox = NAMESPACE_MGR.GetPodSandbox(&podUid)?;

                *self.podSandbox.lock().unwrap() = Some(podSandbox.clone());

                let inner = podSandbox.lock().unwrap();
                POD_BRORKER_MGRS.AddPodBroker(
                    &inner.namespace,
                    inner.ip.0,
                    &inner.name,
                    self.clone(),
                )?;

                let resp = PodRegisterResp {
                    containerIp: inner.ip.0,
                    errorCode: ErrCode::None as _,
                };

                self.SendMsg(TsotMsg::PodRegisterResp(resp).into())?;
            }
            TsotMsg::GatewayRegisterReq(register) => {
                if !gatewayRegister {
                    let resp = GatewayRegisterResp {
                        errorCode: ErrCode::ECONNREFUSED as _,
                    };

                    self.SendMsg(TsotMsg::GatewayRegisterResp(resp).into())?;
                    return Err(Error::CommonError(
                        "Pod register can't use GatewayRegisterReq".to_owned(),
                    ));
                }

                let gatewayUid = uuid::Uuid::from_bytes(register.gatewayUid).to_string();
                let podSandbox = PodSandbox::New(
                    &gatewayUid,
                    "system",
                    "gateway",
                    IpAddress::New(&[127, 1, 2, 3]),
                );

                *self.podSandbox.lock().unwrap() = Some(podSandbox.clone());

                let inner = podSandbox.lock().unwrap();
                POD_BRORKER_MGRS.AddPodBroker(
                    &inner.namespace,
                    inner.ip.0,
                    &inner.name,
                    self.clone(),
                )?;

                let resp = GatewayRegisterResp {
                    errorCode: ErrCode::None as _,
                };

                self.SendMsg(TsotMsg::GatewayRegisterResp(resp).into())?;
            }
            m => {
                return Err(Error::CommonError(format!(
                    "ProcessRegisteMsg get unexpect message {:?}",
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

    pub async fn Process(&self, gatewayRegister: bool) {
        let res = self.Init(gatewayRegister).await;
        match res {
            Ok(()) => (),
            Err(e) => {
                error!("podBroker init fail with error {:?}", e);
                return;
            }
        }

        match self.ProcessMsgs().await {
            Ok(()) => (),
            Err(Error::SocketClose) => {
                self.stop.store(false, Ordering::SeqCst);
            }
            Err(e) => {
                error!("podBroker process fail with error {:?}", e);
                return;
            }
        }

        match self.Drop() {
            Ok(()) => (),
            Err(e) => {
                error!("podBroker process Drop with error {:?}", e);
                return;
            }
        }
    }

    pub async fn ProcessMsgs(&self) -> Result<()> {
        let mut rx = self.outputRx.lock().unwrap().take().unwrap();

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

    pub fn EnqMsg(&self, msg: TsotMessage) -> Result<()> {
        match self.outputTx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_e) => {
                return Err(Error::MpscSendFull(format!(
                    "PodBroker Enqueue message fulll"
                )));
            }
        }
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
        let mut readBuf: [u8; BUFF_SIZE * 1] = [0; BUFF_SIZE * 1];
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

    pub fn ProcessPodRegisterReq(&self, req: PodRegisterReq) -> Result<()> {
        let podUid = uuid::Uuid::from_bytes(req.podUid).to_string();
        let resp = match NAMESPACE_MGR.GetPodSandbox(&podUid) {
            Err(_e) => PodRegisterResp {
                containerIp: 0,
                errorCode: ErrCode::PodUidDonotExisit as _,
            },
            Ok(podSandbox) => PodRegisterResp {
                containerIp: podSandbox.lock().unwrap().ip.0,
                errorCode: ErrCode::PodUidDonotExisit as _,
            },
        };

        self.SendMsg(TsotMsg::PodRegisterResp(resp).into())?;
        return Ok(());
    }

    pub fn ProcessCreateSocketReq(&self, _req: CreateSocketReq) -> Result<()> {
        let stream = TcpSocket::new_v4()?;
        let fd = stream.into_raw_fd();
        let resp = CreateSocketResp {};

        let message = TsotMessage {
            socket: fd,
            msg: TsotMsg::CreateSocketResp(resp),
        };

        return self.EnqMsg(message);
    }

    pub fn ProcessListenReq(&self, req: ListenReq) -> Result<()> {
        match self
            .listeningPorts
            .lock()
            .unwrap()
            .insert(req.port, req.backlog)
        {
            None => return Ok(()),
            Some(port) => {
                error!("ProcessListenReq existing port {}", port);
                return Ok(());
            }
        }
    }

    pub fn ProcessAccept(&self, req: AcceptReq) -> Result<()> {
        match self.listeningPorts.lock().unwrap().get_mut(&req.port) {
            Some(count) => {
                *count += 1;
            }
            None => {
                error!("ProcessAccept not existing port {}", req.port);
            }
        }

        return Ok(());
    }

    pub fn ProcessStopListenReq(&self, req: StopListenReq) -> Result<()> {
        match self.listeningPorts.lock().unwrap().remove(&req.port) {
            Some(_) => return Ok(()),
            None => {
                error!("ProcessStopListenReq not existing port {}", req.port);
                return Ok(());
            }
        }
    }

    pub fn ProcessConnectReq(&self, req: PodConnectReq, socket: i32) -> Result<()> {
        let sandbox = self.podSandbox.lock().unwrap();
        let sandbox = sandbox.as_ref().unwrap();
        let sandbox = sandbox.lock().unwrap();
        let connection = TcpClientConnection {
            podBroker: self.clone(),
            isPodConnection: true,
            socket: socket,
            reqId: req.reqId,
            podNamespace: sandbox.namespace.clone(),
            dstIp: req.dstIp,
            dstPort: req.dstPort,
            srcIp: sandbox.ip.0,
            srcPort: req.srcPort,
        };

        match self
            .connecting
            .lock()
            .unwrap()
            .insert(req.reqId, ConnectReq::PodConnectReq(req.clone()))
        {
            Some(_) => {
                panic!("ProcessConnectReq existing reqId {:?}", req);
            }
            None => (),
        }

        tokio::spawn(async move {
            connection.PodConnectProcess().await;
        });

        return Ok(());
    }

    pub fn ProcessGatewayConnectReq(&self, req: GatewayConnectReq, socket: i32) -> Result<()> {
        let sandbox = self.podSandbox.lock().unwrap();

        let mut namespacelen = 0;
        for i in 0..req.podNamespace.len() {
            if req.podNamespace[i] == 0 {
                namespacelen = i;
                break;
            }
        }

        let namespace = String::from_utf8(req.podNamespace[..namespacelen].to_vec())?;
        let sandbox = sandbox.as_ref().unwrap();
        let sandbox = sandbox.lock().unwrap();
        let connection = TcpClientConnection {
            podBroker: self.clone(),
            isPodConnection: false,
            socket: socket,
            reqId: req.reqId,
            podNamespace: namespace,
            dstIp: req.dstIp,
            dstPort: req.dstPort,
            srcIp: sandbox.ip.0,
            srcPort: req.srcPort,
        };

        match self
            .connecting
            .lock()
            .unwrap()
            .insert(req.reqId, ConnectReq::GatewayConnectReq(req.clone()))
        {
            Some(_) => {
                panic!("ProcessConnectReq existing reqId {:?}", req);
            }
            None => (),
        }

        tokio::spawn(async move {
            connection.GatewayConnectProcess().await;
        });

        return Ok(());
    }

    pub fn ProcessDnsReq(&self, req: DnsReq) -> Result<()> {
        let dnsProxyReq = DnsProxyReq {
            reqId: req.reqId,
            podBroker: self.clone(),
            domains: req.GetDomains(),
        };
        DNS_PROXY.EnqMsg(dnsProxyReq);
        return Ok(());
    }

    pub fn ProcessMsg(&self, msg: TsotMsg, socket: Option<RawFd>) -> Result<()> {
        match msg {
            TsotMsg::PodRegisterReq(m) => {
                self.ProcessPodRegisterReq(m)?;
            }
            TsotMsg::CreateSocketReq(m) => {
                self.ProcessCreateSocketReq(m)?;
            }
            TsotMsg::ListenReq(m) => self.ProcessListenReq(m)?,
            TsotMsg::AcceptReq(m) => {
                self.ProcessAccept(m)?;
            }
            TsotMsg::StopListenReq(m) => {
                self.ProcessStopListenReq(m)?;
            }
            TsotMsg::PodConnectReq(m) => {
                if socket.is_none() {
                    return Err(Error::CommonError(format!("ConnectReq has no socket")));
                }
                self.ProcessConnectReq(m, socket.unwrap())?;
            }
            TsotMsg::GatewayConnectReq(m) => {
                if socket.is_none() {
                    return Err(Error::CommonError(format!("ConnectReq has no socket")));
                }
                self.ProcessGatewayConnectReq(m, socket.unwrap())?;
            }
            TsotMsg::DnsReq(m) => {
                self.ProcessDnsReq(m)?;
            }

            m => {
                error!("ProcessMsg get unimplement msg {:?}", &m);
                unimplemented!()
            }
        }

        return Ok(());
    }

    pub fn HandlePodRegisterResp(&self, containerIp: u32, errorCode: u32) -> Result<()> {
        let msg = PodRegisterResp {
            containerIp: containerIp,
            errorCode: errorCode,
        };

        return self.EnqMsg(TsotMsg::PodRegisterResp(msg).into());
    }

    pub fn HandlePodHibernate(&self) -> Result<()> {
        let msg = Hibernate { _type: 1 };

        let message = TsotMessage {
            socket: 0,
            msg: TsotMsg::Hibernate(msg),
        };

        return self.EnqMsg(message);
    }

    pub fn HandlePodWalkup(&self) -> Result<()> {
        let msg = Wakeup { _type: 1 };

        let message = TsotMessage {
            socket: 0,
            msg: TsotMsg::Wakeup(msg),
        };

        return self.EnqMsg(message);
    }

    pub fn HandleNewPeerConnection(
        &self,
        peerIp: u32,
        peerPort: u16,
        dstPort: u16,
        socket: i32,
    ) -> Result<()> {
        match self.listeningPorts.lock().unwrap().get_mut(&dstPort) {
            Some(count) => {
                if *count == 0 {
                    return Err(Error::NotExist(format!(
                        "target container doesn't listening the port {dstPort}"
                    )));
                }

                *count -= 1;
            }
            None => {
                return Err(Error::NotExist(format!(
                    "target container doesn't listening the port {dstPort}"
                )));
            }
        }

        let msg = PeerConnectNotify {
            peerIp: peerIp,
            peerPort: peerPort,
            localPort: dstPort,
        };

        let message = TsotMessage {
            socket: socket,
            msg: TsotMsg::PeerConnectNotify(msg),
        };

        return self.EnqMsg(message);
    }

    pub fn HandlePodConnectResp(&self, reqId: u32, errorCode: i32) -> Result<()> {
        match self.connecting.lock().unwrap().remove(&reqId) {
            None => {
                error!("HandleConnectResp get non exist reqId {}", reqId);
                return Ok(());
            }
            Some(_) => (),
        }

        let msg = PodConnectResp {
            reqId: reqId,
            errorCode: errorCode,
        };

        return self.EnqMsg(TsotMsg::PodConnectResp(msg).into());
    }

    pub fn HandleGatewayConnectResp(&self, reqId: u32, errorCode: i32) -> Result<()> {
        match self.connecting.lock().unwrap().remove(&reqId) {
            None => {
                error!("HandleConnectResp get non exist reqId {}", reqId);
                return Ok(());
            }
            Some(_) => (),
        }

        let msg = GatewayConnectResp {
            reqId: reqId,
            errorCode: errorCode,
        };

        return self.EnqMsg(TsotMsg::GatewayConnectResp(msg).into());
    }
}
