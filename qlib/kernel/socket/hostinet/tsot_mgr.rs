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

use core::sync::atomic::{AtomicU64, Ordering};

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use spin::Mutex;

use crate::qlib::{common::*, AFType, TcpSockAddr};
use crate::qlib::kernel::kernel::waiter::Queue;
use crate::qlib::socket_buf::*;
use crate::qlib::tsot_msg::*;
use crate::qlib::kernel::tcpip::tcpip::SockAddrInet;

use super::tsotsocket::TsotSocketOperations;

// #[derive(Debug)]
pub struct TsotSocketMgr {
    pub currReqId: AtomicU64,

    // key: port
    pub listeningSockets: Mutex<BTreeMap<u16, TsotListenSocket>>,
    
    // key: reqId
    pub connectingSocket: Mutex<BTreeMap<u16, TsotSocketOperations>>,
}

impl TsotSocketMgr {
    pub fn NextReqId(&self) -> u16 {
        return self.currReqId.fetch_add(1, Ordering::SeqCst) as u16;
    }

    // todo: handle multiple listen to change backlog
    pub fn Listen(&self, port: u16, backlog: u32, queue: &Queue, acceptQueue: &AcceptQueue) -> Result<()> {
        let msg = TsotMsg::ListenReq(ListenReq {
            port: port,
            backlog: backlog,
        }).into();

        self.SendMsg(&msg)?;

        let listenSocket = TsotListenSocket {
            port: port,
            queue: queue.clone(),
            acceptQueue: acceptQueue.clone(),
        };

        self.listeningSockets.lock().insert(port, listenSocket);

        return Ok(())
    }

    pub fn StopListen(&self, port: u16) -> Result<()> {
        match self.listeningSockets.lock().remove(&port) {
            None => {
                error!("TsotSocketMgr::StopListen port doesn't exist");
                return Ok(())
            }
            Some(_) => (),
        }

        let msg = TsotMsg::StopListenReq(StopListenReq {
            port: port
        }).into();

        self.SendMsg(&msg)?;
        return Ok(())
    }

    pub fn Accept(&self, port: u16) -> Result<()> {
        let msg = TsotMsg::AcceptReq(AcceptReq {
            port: port
        }).into();

        self.SendMsg(&msg)?;
        return Ok(())
    }

    pub fn Connect(&self, dstIp: u32, dstPort: u16, srcPort: u16, socket: i32, ops: &TsotSocketOperations) -> Result<()> {
        let reqId = self.NextReqId();
        let connectReq = ConnectReq {
            reqId: reqId,
            dstIp: dstIp,
            dstPort: dstPort,
            srcPort: srcPort,
        };

        let msg = TsotMessage {
            socket: socket,
            msg: TsotMsg::ConnectReq(connectReq)
        };

        self.SendMsg(&msg)?;

        self.connectingSocket.lock().insert(reqId, ops.clone());

        return Ok(())
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
                    let addr = SockAddrInet {
                        Family: AFType::AF_INET as u16,
                        Port: m.peerPort,
                        Addr: m.PeerAddrBytes(),
                        Zero: [0; 8],
                    };

                    let listeningSocket = match self.listeningSockets.lock().get(&m.localPort) {
                        None => {
                            error!("TsotSocketMgr::PeerConnectNotify no listening port {}", &m.localPort);
                            continue
                        }
                        Some(socket) => socket.clone()
                    };

                    let sockBuf = SocketBuff(Arc::new(SocketBuffIntern::default()));
        
                    listeningSocket.acceptQueue.EnqSocket(fd, TcpSockAddr::NewFromInet(addr), 6,  sockBuf.into(), listeningSocket.queue.clone());
                }
                TsotMsg::ConnectResp(_m) => {

                }
                _ => ()
            };
        }
    }
}

#[derive(Debug, Clone)]
pub struct TsotListenSocket {
    pub port: u16,
    pub queue: Queue,
    pub acceptQueue: AcceptQueue,
}

// #[derive(Debug, Clone)]
// pub struct TsotConnectingSocket {
//     pub reqId: u16,
//     pub tsotSocket: TsotSocketOperations,
// }