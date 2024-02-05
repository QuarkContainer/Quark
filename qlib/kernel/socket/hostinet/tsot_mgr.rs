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

use core::sync::atomic::AtomicU32;
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use spin::Mutex;
use alloc::collections::VecDeque;
use core::ops::Deref;
use core::fmt;

use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::common::*;
use crate::qlib::kernel::kernel::waiter::Queue;
use crate::qlib::socket_buf::{AcceptSocket, SocketBuff, SocketBuffIntern};
use crate::qlib::tsot_msg::*;
use crate::qlib::linux_def::*;
use crate::qlib::kernel::tcpip::tcpip::{SockAddr, SockAddrInet};

use super::loopbacksocket::LoopbackSocket;
use super::tsotsocket::TsotSocketOperations;

#[derive(Debug, Default)]
pub struct TsotBindings {
    // bind port --> (bind count, reusePort)
    pub bindings: Mutex<BTreeMap<u16, (usize, bool)>>,

    // key: port
    pub listeningSockets: Mutex<BTreeMap<u16, TsotListenSocket>>,
}

impl TsotBindings {
    pub fn Bind(&self, port: u16, reusePort: bool) -> Result<()> {
        let mut bindings = self.bindings.lock();
        match bindings.get_mut(&port) {
            None => {
                bindings.insert(port, (1, reusePort));
            }
            Some((count, currReusePort)) => {
                if *currReusePort && reusePort {
                    *count += 1;
                } else {
                    return Err(Error::SysError(SysErr::EADDRINUSE));
                }
            }
        }

        return Ok(())
    }

    pub fn RemoveBind(&self, port: u16) {
        let mut bindings = self.bindings.lock();
        match bindings.get_mut(&port) { 
            None => {
                error!("TsotSocketMgr::RemoveBind the port {} doesn't exist", port);
                return;
            }
            Some((count, _currReusePort)) => {
                *count -= 1;
                if *count > 0 {
                    return;
                }
            }
        }

        bindings.remove(&port);
    }

    pub fn Listen(&self, fd: i32, port: u16, backlog: u32, queue: &Queue) -> Result<(TsotAcceptQueue, u32)> {
        let acceptQueue;
        let finalBacklog;

        let mut listeningSockets = self.listeningSockets.lock();
        match listeningSockets.get_mut(&port) {
            None => {
                let mut queues = BTreeMap::new();
                queues.insert(fd, (queue.clone(), backlog));
                acceptQueue = TsotAcceptQueue::New();
                let listenSocket = TsotListenSocket {
                    port: port,
                    queues: queues,
                    acceptQueue: acceptQueue.clone(),
                    backlog: backlog,
                };
                    
                listeningSockets.insert(port, listenSocket);
                finalBacklog = backlog;
            }
            Some(listenSocket) => {
                acceptQueue = listenSocket.acceptQueue.clone();
                match listenSocket.queues.get_mut(&fd) {
                    None => {
                        listenSocket.backlog += backlog;
                        listenSocket.queues.insert(fd, (queue.clone(), backlog));
                    }
                    Some((_queue, originalBacklog)) => {
                        listenSocket.backlog += backlog - *originalBacklog;
                        *originalBacklog = backlog;
                    }
                }
                listenSocket.backlog += backlog;
                finalBacklog = listenSocket.backlog;
            }
        }
        
        return Ok((acceptQueue, finalBacklog));
    }

    /// return: need to stop listening the port
    pub fn StopListen(&self, fd: i32, port: u16) -> Result<bool> {
        let mut listeningSockets = self.listeningSockets.lock();
        match listeningSockets.get_mut(&port) {
            None => {
                return Ok(false)
            }
            Some(listenSocket) => {
                if !listenSocket.Remove(fd)? {
                    return Ok(false)
                }
            }
        }

        match self.listeningSockets.lock().remove(&port) {
            None => {
                error!("TsotSocketMgr::StopListen port doesn't exist");
                return Ok(false)
            }
            Some(_) => (),
        }
        return Ok(true)
    }

    pub fn NewConnection(&self, fd: i32, addr: QIPv4Addr, port: u16, sockBuf: AcceptSocket, queue: Queue) -> Result<()> {
        match self.listeningSockets.lock().get(&port) {
            None => return Err(Error::SysError(SysErr::EADDRINUSE)),
            Some(sock) => {
                sock.NewConnection(fd, addr, port, sockBuf, queue)?;
            }
        }

        return Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct QIPv4Addr (pub u32);

impl From<u32> for QIPv4Addr {
    fn from(addr: u32) -> Self {
        return Self(addr)
    }
}

impl From<&[u8; 4]> for QIPv4Addr {
    fn from(addr: &[u8; 4]) -> Self {
        return Self::New(addr)
    }
}

impl QIPv4Addr {
    pub fn New(addr: &[u8; 4]) -> Self {
        let mut bytes = [0; 4];
        for i in 0..4 {
            bytes[i] = addr[3-i];
        }
        let ipAddr = unsafe {
            *(&bytes[0] as * const _ as u64 as * const u32)
        };

        return Self(ipAddr)
    }

    pub fn Loopback() -> Self {
        return Self::from(&[127, 0, 0, 1]);
    }

    fn AsBytes<'a>(&self) -> &'a [u8] {
        let ptr = &self.0 as * const _ as u64 as * const u8;
        return unsafe {
            core::slice::from_raw_parts(ptr, 4)
        }
    }

    pub fn ToBytes(&self) -> [u8; 4] {
        let mut addr = [0; 4];
        let bytes = self.AsBytes();
        for i in 0..4 {
            addr[i] = bytes[3-i];
        }

        return addr;
    }

    pub fn IsLoopback(&self) -> bool {
        // like [127, 0, 0, 1] i.e. 127.0.0.1
        return self.AsBytes()[3] == 127;
    }

    pub fn IsAny(&self) -> bool {
        return self.0 == 0;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QIPv4Endpoint {
    pub ip: QIPv4Addr,
    pub port: u16,
}

impl QIPv4Endpoint {
    pub fn New(ip: QIPv4Addr, port: u16) -> Self {
        return Self {
            ip: ip,
            port: port,
        }
    }

    pub fn ToSockAddr(&self) -> SockAddr {
        let addr = SockAddrInet {
            Family: AFType::AF_INET as u16,
            Port: self.port,
            Addr: self.ip.ToBytes(),
            Zero: [0; 8],
        };

        return SockAddr::Inet(addr)
    }
}

// #[derive(Debug)]
pub struct TsotSocketMgr {
    pub currReqId: AtomicU64,

    // key: reqId
    pub connectingSockets: Mutex<BTreeMap<u16, TsotSocketOperations>>,

    pub localIpAddr: AtomicU32,

    pub bindAddrs: Mutex<BTreeMap<QIPv4Addr, TsotBindings>>,
}

impl Default for TsotSocketMgr {
    fn default() -> Self {
        return Self {
            currReqId: AtomicU64::new(0),
            connectingSockets: Mutex::new(BTreeMap::new()),
            localIpAddr: AtomicU32::new(0),
            bindAddrs: Mutex::new(BTreeMap::new()),
        }
    }
}

impl TsotSocketMgr {
    pub fn SetLocalIpAddr(&self, addr: u32) {
        self.localIpAddr.store(addr, Ordering::SeqCst);
    }

    pub fn LocalIpAddr(&self) -> QIPv4Addr {
        return self.localIpAddr.load(Ordering::Relaxed).into();
    }
    
    pub fn NextReqId(&self) -> u16 {
        return self.currReqId.fetch_add(1, Ordering::SeqCst) as u16;
    }

    pub fn ValidAddr(&self, ip: QIPv4Addr) -> Result<()> {
        if ip.IsLoopback() || ip == self.LocalIpAddr() {
            return Ok(())
        }

        return Err(Error::SysError(SysErr::EINVAL));
    }

    pub fn Bind(&self, ip: QIPv4Addr, port: u16, reusePort: bool) -> Result<()> {
        self.ValidAddr(ip)?;
        let mut bindingAddrs = self.bindAddrs.lock();
        
        if !bindingAddrs.contains_key(&ip) {
            bindingAddrs.insert(ip, TsotBindings::default());
        }
        
        match bindingAddrs.get_mut(&ip) {
            Some(addr) => {
                addr.Bind(port, reusePort)?;
            }
            None => {
                unreachable!()
            }
        }
        
        return Ok(())
    }

    pub fn RemoveBind(&self, ip: QIPv4Addr, port: u16) -> Result<()> {
        self.ValidAddr(ip)?;
        let mut bindingAddrs = self.bindAddrs.lock();
        
        if !bindingAddrs.contains_key(&ip) {
            bindingAddrs.insert(ip, TsotBindings::default());
        }
        
        match bindingAddrs.get_mut(&ip) {
            Some(addr) => {
                addr.RemoveBind(port);
            }
            None => {
                unreachable!()
            }
        }

        return Ok(())
    }

    // todo: handle multiple listen to change backlog
    pub fn Listen(&self, ip: QIPv4Addr, fd: i32, port: u16, backlog: u32, queue: &Queue) -> Result<TsotAcceptQueue> {
        self.ValidAddr(ip)?;
        let mut bindingAddrs = self.bindAddrs.lock();
        
        if !bindingAddrs.contains_key(&ip) {
            bindingAddrs.insert(ip, TsotBindings::default());
        }
        
        let (acceptQueue, finalBacklog) = match bindingAddrs.get_mut(&ip) {
            Some(addr) => {
                addr.Listen(fd, port, backlog, queue)?
            }
            None => {
                unreachable!()
            }
        };

        drop(bindingAddrs);

        if !ip.IsLoopback() {
            let msg = TsotMsg::ListenReq(ListenReq {
                port: port,
                backlog: finalBacklog,
            }).into();
    
            self.SendMsg(&msg)?;
        }
        
        return Ok(acceptQueue);
    }

    pub fn StopListen(&self, fd: i32, ip: QIPv4Addr, port: u16) -> Result<()> {
        self.ValidAddr(ip)?;
        let mut bindingAddrs = self.bindAddrs.lock();

        match bindingAddrs.get_mut(&ip) {
            Some(addr) => {
                if !addr.StopListen(fd, port)? {
                    return Ok(())
                }
            }
            None => {
                return Ok(())
            }
        };

        drop(bindingAddrs);

        if !ip.IsLoopback() {
            let msg = TsotMsg::StopListenReq(StopListenReq {
                port: port
            }).into();
    
            self.SendMsg(&msg)?;
        }

        return Ok(())

    }

    pub fn Accept(&self, port: u16) -> Result<()> {
        let msg = TsotMsg::AcceptReq(AcceptReq {
            port: port
        }).into();

        self.SendMsg(&msg)?;
        return Ok(())
    }

    pub fn Connect(&self, dstIp: QIPv4Addr, dstPort: u16, srcPort: u16, socket: i32, ops: &TsotSocketOperations) -> Result<()> {
        let reqId = self.NextReqId();
        let connectReq = ConnectReq {
            reqId: reqId,
            dstIp: dstIp.0,
            dstPort: dstPort,
            srcPort: srcPort,
        };

        let msg = TsotMessage {
            socket: socket,
            msg: TsotMsg::ConnectReq(connectReq)
        };

        self.SendMsg(&msg)?;

        self.connectingSockets.lock().insert(reqId, ops.clone());

        return Ok(())
    }


    pub fn NewPeerConnection(&self, fd: i32, peerAddr: QIPv4Addr, port: u16, sockBuf: SocketBuff) -> Result<()> {
        let sockBuf = AcceptSocket::SocketBuff(sockBuf);
        
        let listens = self.bindAddrs.lock();
        match listens.get(&peerAddr) { 
            None => {
                return Err(Error::SysError(SysErr::EADDRINUSE))
            }
            Some(addr) => {
                addr.NewConnection(fd, peerAddr, port, sockBuf, Queue::default())?;
                return Ok(())
            }
        }
    }

    pub fn NewLoopbackConnection(
        &self, 
        fd: i32, 
        peerAddr: QIPv4Addr, 
        port: u16, 
        socket: LoopbackSocket,
        serverQueue: Queue
    ) -> Result<()> {
        let sockBuf = AcceptSocket::LoopbackSocket(socket);
        
        let listens = self.bindAddrs.lock();
        match listens.get(&peerAddr) { 
            None => {
                return Err(Error::SysError(SysErr::EADDRINUSE))
            }
            Some(addr) => {
                addr.NewConnection(fd, peerAddr, port, sockBuf, serverQueue)?;
                return Ok(())
            }
        }
    }

    pub fn Process(&self) -> Result<()> {
        loop {
            let msg = match self.RecvMsg() {
                Err(_) => return Ok(()),
                Ok(m) => m,
            };

            let fd = msg.socket;

            match msg.msg {
                TsotMsg::PeerConnectNotify(m) => {
                    let sockBuf = SocketBuff(Arc::new(SocketBuffIntern::default()));
                    self.NewPeerConnection(
                        fd, 
                        m.peerIp.into(), 
                        m.peerPort, 
                        sockBuf
                    )?;
                }
                TsotMsg::ConnectResp(m) => {
                    let connectingSocket = match self.connectingSockets.lock().get(&m.reqId) {
                        None => {
                            error!("TsotSocketMgr::ConnectResp no connecting request {:?}", &m);
                            continue
                        }
                        Some(socket) => socket.clone()
                    };

                    connectingSocket.SetConnErrno(m.errorCode as _);
                    connectingSocket.queue.Notify(EVENT_READ)
                }
                _ => ()
            };
        }
    }
}

#[derive(Debug, Clone)]
pub struct TsotListenSocket {
    pub port: u16,
    // hostfd to (Queue, backlog), in case of reusePort
    pub queues: BTreeMap<i32, (Queue, u32)>,
    pub acceptQueue: TsotAcceptQueue,
    pub backlog: u32,
}

impl TsotListenSocket {
    pub fn NewConnection(
        &self, 
        fd: i32, 
        addr: QIPv4Addr, 
        port: u16, 
        sockBuf: AcceptSocket, 
        queue: Queue
    ) -> Result<()> {
        let trigger = self.acceptQueue.EnqSocket(fd, addr.0, port, sockBuf, queue);
        if trigger {
            for (_, (queue, _)) in &self.queues {
                queue.Notify(EVENT_READ);
            }
        }
        
        return Ok(())
    }

    // return: need to remove the whole listenSocket
    pub fn Remove(&mut self, fd: i32) -> Result<bool> {
        match self.queues.remove(&fd) {
            None => return Ok(false),
            Some((_queue, backlog)) => {
                self.backlog -= backlog;
            }
        };

        return Ok(self.backlog == 0);
    }
}

// #[derive(Debug, Clone)]
// pub struct TsotConnectingSocket {
//     pub reqId: u16,
//     pub tsotSocket: TsotSocketOperations,
// }


#[derive(Default, Debug)]
pub struct TsotAcceptItem {
    pub fd: i32,
    pub addr: u32,
    pub port: u16,
    pub sockBuf: AcceptSocket,
    pub queue: Queue
}

#[derive(Clone, Debug)]
pub struct TsotAcceptQueue(Arc<Mutex<TsotAcceptQueueIntern>>);

impl Deref for TsotAcceptQueue {
    type Target = Arc<Mutex<TsotAcceptQueueIntern>>;

    fn deref(&self) -> &Arc<Mutex<TsotAcceptQueueIntern>> {
        &self.0
    }
}

impl TsotAcceptQueue {
    pub fn New() -> Self {
        let inner = TsotAcceptQueueIntern {
            aiQueue: VecDeque::new(),
            error: 0,
        };

        return Self(Arc::new(Mutex::new(inner)));
    }

    // return whether trigger queue notify
    pub fn EnqSocket(
        &self,
        fd: i32,
        addr: u32,
        port: u16,
        sockBuf: AcceptSocket,
        queue: Queue
    ) -> bool {
        let mut inner = self.lock();
        let item = TsotAcceptItem {
            fd: fd,
            addr: addr,
            port: port,
            sockBuf: sockBuf,
            queue
        };

        inner.aiQueue.push_back(item);
        let trigger = inner.aiQueue.len() == 1;
        return trigger;
    }
}

pub struct TsotAcceptQueueIntern {
    pub aiQueue: VecDeque<TsotAcceptItem>,
    pub error: i32,
}

impl fmt::Debug for TsotAcceptQueueIntern {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TsotAcceptQueueIntern aiQueue {:x?}", self.aiQueue)
    }
}

impl Drop for TsotAcceptQueueIntern {
    fn drop(&mut self) {
        for ai in &mut self.aiQueue {
            HostSpace::Close(ai.fd);
        }
    }
}

impl TsotAcceptQueueIntern {
    pub fn SetErr(&mut self, error: i32) {
        self.error = error
    }

    pub fn Err(&self) -> i32 {
        return self.error;
    }

    pub fn DeqSocket(&mut self) -> Result<TsotAcceptItem> {
        match self.aiQueue.pop_front() {
            None => {
                if self.error != 0 {
                    return Err(Error::SysError(self.error));
                }
                return Err(Error::SysError(SysErr::EAGAIN));
            }
            Some(item) => return Ok(item),
        }
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.aiQueue.len() > 0 {
            event |= READABLE_EVENT;
        }

        if self.error != 0 {
            event |= EVENT_ERR;
        }

        return event;
    }
}
