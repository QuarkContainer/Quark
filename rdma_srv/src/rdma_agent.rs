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

use alloc::sync::Arc;
use core::sync::atomic::Ordering;
use spin::Mutex;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::CString;
use std::ops::{Deref, DerefMut};
use std::os::unix::io::{AsRawFd, RawFd};
use std::{env, mem, ptr, thread, time};

use super::id_mgr::IdMgr;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::socket_buf::SocketBuff;
use super::qlib::unix_socket::UnixSocket;
use super::rdma::*;
use super::rdma_channel::*;
use super::rdma_conn::*;
use super::rdma_ctrlconn::*;
use super::rdma_srv::*;

pub struct SockInfo {
    pub sockfd: u32,
    pub status: SockStatus,
    // pub acceptQueue: AcceptQueue,
}

pub struct RDMAAgentIntern {
    pub id: u32,

    // client id passed when initialize RDMASvcCli, can use container id for container.
    pub clientId: String,

    // the unix socket fd between rdma client and RDMASrv
    pub sockfd: i32,

    // the memfd share memory with rdma client
    pub client_memfd: i32,

    // the eventfd which send notification to client
    pub client_eventfd: i32,

    // the memory region shared with client
    pub shareMemRegion: MemRegion,

    pub shareRegion: Mutex<&'static mut ClientShareRegion>,

    pub ioBufIdMgr: Mutex<IdMgr>,

    pub keys: Vec<[u32; 2]>,
    // TODO: indexes allocated for io buffer.

    //sockfd -> sockInfo
    // pub sockInfos: Mutex<HashMap<u32, SockInfo>>,
}

#[derive(Clone)]
pub struct RDMAAgent(Arc<RDMAAgentIntern>);

impl Deref for RDMAAgent {
    type Target = Arc<RDMAAgentIntern>;

    fn deref(&self) -> &Arc<RDMAAgentIntern> {
        &self.0
    }
}

impl RDMAAgent {
    pub fn New(id: u32, clientId: String, connSock: i32, clientEventfd: i32) -> Self {
        let path = "/home/qingming/rdma_srv";
        println!("RDMAAgent::New 1");
        let memfdname = CString::new("RDMASrvMemFd").expect("CString::new failed for RDMASrvMemFd");
        let memfd = unsafe { libc::memfd_create(memfdname.as_ptr(), libc::MFD_ALLOW_SEALING) };
        let size = mem::size_of::<ClientShareRegion>();
        let _ret = unsafe { libc::ftruncate(memfd, size as i64) };
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                memfd,
                0,
            )
        };

        println!("addr: 0x{:x}", addr as u64);

        // Send srv fd to client.
        // let srv_sock = UnixSocket::NewServer(path).unwrap();
        // let conn_sock = UnixSocket::Accept(srv_sock.as_raw_fd()).unwrap();
        // conn_sock.SendFd(fd).unwrap();

        //start from 2M registration.
        let mr = RDMA
            .CreateMemoryRegion(addr as u64, 2 * 1024 * 1024)
            .unwrap();
        let shareRegion = unsafe {
            let addr = addr as *mut ClientShareRegion;
            &mut (*addr)
        };

        shareRegion.sq.Init();
        shareRegion.cq.Init();

        Self(Arc::new(RDMAAgentIntern {
            id: id,
            clientId: clientId,
            sockfd: connSock,
            client_memfd: memfd,
            client_eventfd: clientEventfd,
            shareMemRegion: MemRegion { addr: 0, len: 0 },
            shareRegion: Mutex::new(shareRegion),
            ioBufIdMgr: Mutex::new(IdMgr::Init(0, 20)),
            keys: vec![[mr.LKey(), mr.RKey()]],
            // sockInfos: Mutex::new(HashMap::new()),
        }))
    }

    pub fn NewDummyAgent() -> Self {
        Self(Arc::new(RDMAAgentIntern {
            id: 0,
            clientId: String::new(),
            sockfd: 0,
            client_memfd: 0,
            client_eventfd: 0,
            shareMemRegion: MemRegion { addr: 0, len: 0 },
            shareRegion: unsafe {
                let addr = 0 as *mut ClientShareRegion;
                Mutex::new(&mut (*addr))
            },
            ioBufIdMgr: Mutex::new(IdMgr::Init(0, 0)),
            keys: vec![[0, 0]],
            // sockInfos: Mutex::new(HashMap::new()),
        }))
    }

    pub fn CreateServerRDMAChannel(
        &self,
        connectRequest: &ConnectRequest,
        rdmaConn: RDMAConn,
    ) -> RDMAChannel {
        let channelId = RDMA_SRV.channelIdMgr.lock().AllocId().unwrap();
        let ioBufIndex = self.ioBufIdMgr.lock().AllocId().unwrap() as usize;
        let shareRegion = self.shareRegion.lock();
        let sockBuf = Arc::new(SocketBuff::InitWithShareMemory(
            MemoryDef::DEFAULT_BUF_PAGE_COUNT,
            &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
        ));
        let rdmaChannel = RDMAChannel::CreateRDMAChannel(
            channelId,
            self.keys[ioBufIndex / 16][0],
            self.keys[ioBufIndex / 16][1],
            sockBuf,
            rdmaConn,
            connectRequest,
            ioBufIndex as u32,
            self,
        );
        rdmaChannel
    }

    pub fn CreateClientRDMAChannel(
        &self,
        connectReq: &RDMAConnectReq,
        rdmaConn: RDMAConn,
        shareRegion: &ClientShareRegion,
    ) -> RDMAChannel {
        println!("RDMAAgent::CreateClientRDMAChannel 1");
        let channelId = RDMA_SRV.channelIdMgr.lock().AllocId().unwrap();
        println!("RDMAAgent::CreateClientRDMAChannel 2");
        let ioBufIndex = self.ioBufIdMgr.lock().AllocId().unwrap() as usize;
        println!("RDMAAgent::CreateClientRDMAChannel 3");
        // let shareRegion = self.shareRegion.lock();
        println!("RDMAAgent::CreateClientRDMAChannel 4");
        let sockBuf = Arc::new(SocketBuff::InitWithShareMemory(
            MemoryDef::DEFAULT_BUF_PAGE_COUNT,
            &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
        ));
        println!("RDMAAgent::CreateClientRDMAChannel 5");
        let rdmaChannel = RDMAChannel::CreateClientChannel(
            channelId,
            // connectReq.sockfd,
            self.keys[ioBufIndex / 16][0],
            self.keys[ioBufIndex / 16][1],
            sockBuf,
            rdmaConn,
            &connectReq,
            ioBufIndex as u32,
            self,
        );
        rdmaChannel
    }

    pub fn HandleClientRequest(&self) {
        loop {
            let shareRegion = self.shareRegion.lock();
            match shareRegion.sq.Pop() {
                Some(rdmaRequest) => self.HandleClientRequestInternal(rdmaRequest, &shareRegion),
                None => {
                    // println!("No more request for agent: {}", self.id);
                    break;
                }
            }
        }
    }

    pub fn SendResponse(&self, response: RDMAResp) {
        let mut shareRegion = self.shareRegion.lock();
        // println!("RDMAAgent::SendResponse: {:?} before", response);
        shareRegion.cq.Push(response);
        // println!("RDMAAgent::SendResponse: {:?}", response);
        if shareRegion.clientBitmap.load(Ordering::SeqCst) == 1 {
            let data = 16u64;
            let ret = unsafe {
                libc::write(
                    self.client_eventfd,
                    &data as *const _ as *const libc::c_void,
                    mem::size_of_val(&data) as usize,
                )
            };
            // println!("RDMAAgent::SendResponse, ret: {}", ret);
            if ret < 0 {
                println!("error: {}", std::io::Error::last_os_error());
            }
        }
    }

    fn HandleClientRequestInternal(&self, rdmaReq: RDMAReq, shareRegion: &ClientShareRegion) {
        match rdmaReq.msg {
            RDMAReqMsg::RDMAListen(msg) => {
                RDMA_SRV.srvEndPoints.lock().insert(
                    Endpoint {
                        ipAddr: msg.ipAddr,
                        port: msg.port,
                    },
                    SrvEndpoint {
                        agentId: self.id,
                        sockfd: msg.sockfd,
                        endpoint: Endpoint {
                            ipAddr: msg.ipAddr,
                            port: msg.port,
                        },
                        status: SrvEndPointStatus::Listening,
                    },
                );
                println!(
                    "insert listening: ipAddr: {}, port: {}",
                    msg.ipAddr, msg.port
                );
                // self.sockInfos.lock().insert(
                //     msg.sockfd,
                //     SockInfo {
                //         sockfd: msg.sockfd,
                //         status: SockStatus::LISTENING,
                //     },
                // );
            }
            RDMAReqMsg::RDMAConnect(msg) => {
                println!("connect: ipAddr: {}, port: {}", msg.dstIpAddr, msg.dstPort);
                //TODOCtrlPlane: need get nodeIp from dstIpAddr
                match RDMA_CTLINFO.podIpInfo.lock().get(&msg.dstIpAddr) {
                    Some(nodeIpAddr) => {
                        println!("RDMAReqMsg::RDMAConnect 1");
                        let conns = RDMA_SRV.conns.lock();
                        let rdmaConn = conns.get(nodeIpAddr).unwrap();
                        println!("RDMAReqMsg::RDMAConnect 2");
                        let rdmaChannel =
                            self.CreateClientRDMAChannel(&msg, rdmaConn.clone(), shareRegion);
                        println!("RDMAReqMsg::RDMAConnect 3");
                        // self.sockInfos.lock().insert(
                        //     msg.sockfd,
                        //     SockInfo {
                        //         sockfd: msg.sockfd,
                        //         status: SockStatus::CONNECTING,
                        //     },
                        // );
                        println!("RDMAReqMsg::RDMAConnect 4");
                        RDMA_SRV
                            .channels
                            .lock()
                            .insert(rdmaChannel.localId, rdmaChannel.clone());

                        println!("RDMAReqMsg::RDMAConnect 5");
                        let connectReqeust = rdmaChannel.CreateConnectRequest(msg.sockfd);
                        println!("ConnectRequest before send: {:?}", connectReqeust);
                        rdmaConn
                            .ctrlChan
                            .lock()
                            .SendControlMsg(ControlMsgBody::ConnectRequest(connectReqeust))
                            .expect("fail to send msg");
                    }
                    None => {
                        println!("TODO: return error as no ip mapping is found");
                    }
                }
            }
            RDMAReqMsg::RDMAWrite(msg) => {
                println!("RDMAAgent::RDMAReqMsg::RDMAWrite, msg: {:?}", msg);
                match RDMA_SRV.channels.lock().get(&msg.channelId) {
                    Some(rdmaChannel) => {
                        rdmaChannel.RDMASend();
                    }
                    None => {
                        panic!("RDMAChannel with id {} does not exist!", msg.channelId);
                    }
                }
            }
            RDMAReqMsg::RDMARead(msg) => {
                println!("RDMAAgent::RDMAReqMsg::RDMARead, msg: {:?}", msg);
                match RDMA_SRV.channels.lock().get(&msg.channelId) {
                    Some(rdmaChannel) => {
                        rdmaChannel.SendConsumedData();
                    }
                    None => {
                        panic!("RDMAChannel with id {} does not exist!", msg.channelId);
                    }
                }
            }
            RDMAReqMsg::RDMAShutdown(msg) => {
                println!("RDMAAgent::RDMAReqMsg::RDMAShutdown, msg: {:?}", msg);
                match RDMA_SRV.channels.lock().get(&msg.channelId) {
                    Some(rdmaChannel) => {
                        rdmaChannel.Shutdown(msg.howto);
                    }
                    None => {
                        panic!("RDMAChannel with id {} does not exist!", msg.channelId);
                    }
                }
            }
            RDMAReqMsg::RDMACloseChannel(msg) => {
                println!("RDMAAgent::RDMAReqMsg::RDMACloseChannel, msg: {:?}", msg);
                let rdmaChannel = RDMA_SRV
                    .channels
                    .lock()
                    .get(&msg.channelId)
                    .unwrap()
                    .clone();

                rdmaChannel.Close();
            }
        }
    }
}
