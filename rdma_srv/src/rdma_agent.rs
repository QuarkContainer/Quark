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
use std::net::{IpAddr, Ipv4Addr};
use std::ops::{Deref, DerefMut};
use std::os::unix::io::{AsRawFd, RawFd};
use std::str::FromStr;
use std::{env, mem, ptr, thread, time};

use super::id_mgr::IdMgr;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::socket_buf::{SocketBuff, SocketBuffIntern};
use super::qlib::unix_socket::UnixSocket;
use super::rdma::*;
use super::rdma_channel::*;
use super::rdma_conn::*;
use super::rdma_ctrlconn::*;
use super::rdma_srv::*;
// use super::qlib::kernel::TSC;

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
    pub memoryRegions: Mutex<Vec<MemoryRegion>>,
    //sockfd -> sockInfo
    // pub sockInfos: Mutex<HashMap<u32, SockInfo>>,
    pub udpMR: MemoryRegion,
    pub udpRecvBufferAllocator: Mutex<UDPBufferAllocator>,
    pub podId: [u8; 64],
    pub vpcId: Mutex<u32>,
    pub ipAddr: Mutex<u32>, // currently only support single IPv4 address for each pod.
}

impl Drop for RDMAAgentIntern {
    fn drop(&mut self) {        
        unsafe {
            self.memoryRegions.lock().clear();
            if self.sockfd != 0 {
                libc::close(self.sockfd);
            }

            if self.shareMemRegion.addr != 0 {
                libc::munmap(
                    self.shareMemRegion.addr as *mut _,
                    self.shareMemRegion.len as usize,
                );
            }
        }
    }
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
    pub fn New(
        id: u32,
        clientId: String,
        connSock: i32,
        clientEventfd: i32,
        podId: [u8; 64],
    ) -> Self {
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

        let shareRegion = unsafe {
            let addr = addr as *mut ClientShareRegion;
            &mut (*addr)
        };

        //start from 2M registration.
        let tcpMR = RDMA
            .CreateMemoryRegion(&shareRegion.iobufs as *const _ as u64, 2 * 64 * 1024 * 1024)
            .unwrap();

        let udpBufSent = &shareRegion.udpBufSent as *const _ as u64;
        let udpMR = RDMA
            .CreateMemoryRegion(
                udpBufSent,
                mem::size_of::<UDPPacket>() * UDP_RECV_PACKET_COUNT,
            )
            .unwrap();

        // debug!("RDMAAgent::New shareRegion: size: {}, addr: {:x}, clientBit: {:x}, cq: {:x}, sq: {:x} ioMeta: {:x}, buff: {:x}",
        //  size,
        //  shareRegion as *const _ as u64,
        //  &(shareRegion.clientBitmap) as *const _ as u64,
        //  &(shareRegion.cq) as *const _ as u64,
        //  &(shareRegion.sq) as *const _ as u64,
        //  &(shareRegion.ioMetas) as *const _ as u64,
        //  &(shareRegion.iobufs) as *const _ as u64);
        let udpRecvBufferAllocator = UDPBufferAllocator::New(
            &shareRegion.udpBufRecv as *const _ as u64,
            UDP_RECV_PACKET_COUNT as u32,
        );

        shareRegion.sq.Init();
        shareRegion.cq.Init();

        Self(Arc::new(RDMAAgentIntern {
            id: id,
            clientId: clientId,
            sockfd: connSock,
            client_memfd: memfd,
            client_eventfd: clientEventfd,
            shareMemRegion: MemRegion {
                addr: addr as u64,
                len: size as u64,
            },
            shareRegion: Mutex::new(shareRegion),
            ioBufIdMgr: Mutex::new(IdMgr::Init(0, 1024)),
            keys: vec![[tcpMR.LKey(), tcpMR.RKey()]],
            memoryRegions: Mutex::new(vec![tcpMR]),
            udpMR,
            udpRecvBufferAllocator: Mutex::new(udpRecvBufferAllocator),
            podId,
            vpcId: Mutex::new(0),
            ipAddr: Mutex::new(0),
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
            memoryRegions: Mutex::new(vec![]),
            udpMR: MemoryRegion::default(),
            udpRecvBufferAllocator: Mutex::new(UDPBufferAllocator::default()),
            podId: [0; 64],
            vpcId: Mutex::new(0),
            ipAddr: Mutex::new(0),
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
        let sockBuf = SocketBuff(Arc::new(SocketBuffIntern::InitWithShareMemory(
            MemoryDef::DEFAULT_BUF_PAGE_COUNT,
            &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
            true,
        )));

        let rdmaChannel = RDMAChannel::CreateRDMAChannel(
            channelId,
            self.keys[ioBufIndex / 1024][0],
            self.keys[ioBufIndex / 1024][1],
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
        // shareRegion: &ClientShareRegion,
    ) -> RDMAChannel {
        let channelId = RDMA_SRV.channelIdMgr.lock().AllocId().unwrap();
        let ioBufIndex = self.ioBufIdMgr.lock().AllocId().unwrap() as usize;
        let shareRegion = self.shareRegion.lock();
        let sockBuf = SocketBuff(Arc::new(SocketBuffIntern::InitWithShareMemory(
            MemoryDef::DEFAULT_BUF_PAGE_COUNT,
            &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _ as u64,
            &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
            &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
            true,
        )));

        let rdmaChannel = RDMAChannel::CreateClientChannel(
            channelId,
            self.keys[ioBufIndex / 1024][0],
            self.keys[ioBufIndex / 1024][1],
            sockBuf,
            rdmaConn,
            &connectReq,
            ioBufIndex as u32,
            self,
        );
        rdmaChannel
    }

    pub fn HandleClientRequest(&self) -> usize {
        let mut count = 0;
        loop {
            let request = self.shareRegion.lock().sq.Pop();
            count += 1;
            match request {
                Some(rdmaRequest) => self.HandleClientRequestInternal(rdmaRequest),
                None => {
                    count -= 1;
                    // println!("No more request for agent: {}", self.id);
                    break;
                }
            }
        }
        count
    }

    pub fn SendResponse(&self, response: RDMAResp) {
        let mut shareRegion = self.shareRegion.lock();
        // if matches!(response.msg, RDMARespMsg::RDMANotify(_)) {
        //     let mut readBufHeadTailAddr = &shareRegion.ioMetas as *const _ as u64 - 24;
        //     let mut i = 1;
        //     loop {
        //         // unsafe {
        //         //     *(readBufHeadTailAddr as *mut u32) = i;
        //         //     *((readBufHeadTailAddr + 4) as *mut u32) = i + 1;
        //         //     *((readBufHeadTailAddr + 8) as *mut u32) = i + 2;
        //         //     *((readBufHeadTailAddr + 12) as *mut u32) = i + 3;
        //         //     *((readBufHeadTailAddr + 16) as *mut u64) = i as u64 + 4;
        //         // }

        //         i += 5;

        //         debug!(
        //             "RDMARespMsg::RDMANotify, readBufHeadTailAddr: {:x}, readHead: {}, readTail: {}, writehead: {}, writeTail: {}, consumedData: {}",
        //             readBufHeadTailAddr,
        //             unsafe { *(readBufHeadTailAddr as *const u32) },
        //             unsafe { *((readBufHeadTailAddr + 4) as *const u32) },
        //             unsafe { *((readBufHeadTailAddr + 8) as *const u32) },
        //             unsafe { *((readBufHeadTailAddr + 12) as *const u32) },
        //             unsafe { *((readBufHeadTailAddr + 16) as *const u64) }
        //         );
        //         readBufHeadTailAddr += 24;
        //         if readBufHeadTailAddr > (&shareRegion.iobufs as *const _ as u64) {
        //             break;
        //         }
        //     }
        //     let mut i = 0;
        //     readBufHeadTailAddr = &shareRegion.iobufs as *const _ as u64;
        //     loop {
        //         debug!(
        //             "RDMARespMsg::RDMANotify, buf: {:x}, val: {}",
        //             readBufHeadTailAddr,
        //             unsafe { *((readBufHeadTailAddr + i) as *const u8) },
        //         );
        //         i += 1;
        //         if i > 16 {
        //             break;
        //         }
        //     }
        // }

        shareRegion.cq.Push(response);
        if shareRegion.clientBitmap.load(Ordering::Acquire) == 1 {
            let data = 16u64;
            let ret = unsafe {
                libc::write(
                    self.client_eventfd,
                    &data as *const _ as *const libc::c_void,
                    mem::size_of_val(&data) as usize,
                )
            };
            if ret < 0 {
                println!("error: {}", std::io::Error::last_os_error());
            }
        }
        // profiling
        // if matches!(response.msg, RDMARespMsg::RDMAConnect(_)) {
        //     // println!("qq1: After push RDMAConnect to CQ (SendResponse)");
        //     RDMA_SRV.timestamps.lock().push(TSC.Rdtsc()); //4
        //     let len = RDMA_SRV.timestamps.lock().len();
        //     let mut i = 1;
        //     let mut v1 = RDMA_SRV.timestamps.lock()[0];
        //     let mut v2 = RDMA_SRV.timestamps.lock()[1];
        //     println!("qq, Connect time is: {}", RDMA_SRV.timestamps.lock()[len - 1] - v1);
        //     loop {
        //         println!("{}", v2 - v1);
        //         i += 1;
        //         if i == len {
        //             break;
        //         }
        //         v1 = v2;
        //         v2 = RDMA_SRV.timestamps.lock()[i];            
        //     }
        //     RDMA_SRV.timestamps.lock().clear();
        // }
    }

    fn HandleClientRequestInternal(&self, rdmaReq: RDMAReq) {
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
                    "RDMAAgent::RDMAListen, ipAddr: {}, port: {}",
                    msg.ipAddr, msg.port
                );
            }
            RDMAReqMsg::RDMAListenUsingPodId(msg) => {
                let mut podId: [u8; 64] = [0; 64];
                if RDMA_CTLINFO.isK8s {
                    podId = msg.podId;
                } else {
                    let podIdStr = "server".to_string();
                    // if podIdStr.len() != podId.len() {
                    //     panic!(
                    //         "podId len: {} is not equal to podIdStr len: {}",
                    //         podId.len(),
                    //         podIdStr.len()
                    //     );
                    // }

                    podIdStr
                        .bytes()
                        .zip(podId.iter_mut())
                        .for_each(|(b, ptr)| *ptr = b);
                }
                RDMA_SRV.srvPodIdEndpoints.lock().insert(
                    EndpointUsingPodId {
                        podId,
                        port: msg.port,
                    },
                    SrvEndpointUsingPodId {
                        agentId: self.id,
                        sockfd: msg.sockfd,
                        podId: msg.podId,
                        port: msg.port,
                        status: SrvEndPointStatus::Listening,
                    },
                );
                // error!("RDMAReqMsg::RDMAListenUsingPodId, podId: {:?}, port: {}", podId, msg.port);
            }
            RDMAReqMsg::RDMAConnect(msg) => {
                let mut msgCloned = msg.clone();

                match RDMA_CTLINFO.IsService(msgCloned.dstIpAddr, &msgCloned.dstPort) {
                    None => {}
                    Some(ipWithPort) => {
                        println!("RDMAConnect: The traffic is connecting to a service. Change the connection to {:?}", ipWithPort);
                        msgCloned.dstIpAddr = ipWithPort.ip;
                        msgCloned.dstPort = ipWithPort.port.port;
                    }
                }
                match RDMA_CTLINFO.get_node_ip_by_pod_ip(&msgCloned.dstIpAddr) {
                    Some(nodeIpAddr) => {
                        let conns = RDMA_SRV.conns.lock();
                        let rdmaConn = conns.get(&nodeIpAddr).unwrap();
                        let rdmaChannel = self.CreateClientRDMAChannel(&msgCloned, rdmaConn.clone());

                        RDMA_SRV
                            .channels
                            .lock()
                            .insert(rdmaChannel.localId, rdmaChannel.clone());

                        let connectReqeust = rdmaChannel.CreateConnectRequest(msgCloned.sockfd, 1);
                        rdmaConn
                            .ctrlChan
                            .lock()
                            .SendControlMsg(ControlMsgBody::ConnectRequest(connectReqeust));
                        // .expect("fail to send msg");
                    }
                    None => {
                        println!("TODO: return error as no ip to node mapping is found, dstIpAddr: {}, dstPort: {}", msgCloned.dstIpAddr, msgCloned.dstPort);
                    }
                }
            }
            RDMAReqMsg::RDMAConnectUsingPodId(msg) => {
                // println!("qq1: RDMAReqMsg::RDMAConnectUsingPodId Enter, msg: {:?}", msg);
                // RDMA_SRV.timestamps.lock().push(TSC.Rdtsc()); //0
                let vpcId;
                let ipAddr;
                let mut dstIpAddr = msg.dstIpAddr;
                if RDMA_CTLINFO.isK8s {
                    vpcId = *self.vpcId.lock();
                    ipAddr = self.ipAddr.lock().to_be();
                } else {
                    let mut podId = "client".to_string();
                    if dstIpAddr == 2130706433 || dstIpAddr == 0 {
                        podId = "server".to_string();
                    }
                    
                    let vpcIpAddr = RDMA_CTLINFO
                        .podIdToVpcIpAddr
                        .lock()
                        .get(&podId)
                        .unwrap()
                        .clone();
                    vpcId = vpcIpAddr.vpcId;
                    ipAddr = vpcIpAddr.ipAddr.to_be();
                }
                if vpcId == 0 {
                    panic!("TODO: VpcIdIpAddr hasn't been obtained from control plan yet, should handle this in negative case");
                }

                let srcVpcIpAddr = VpcIpAddr { vpcId, ipAddr };
                
                if dstIpAddr == 2130706433 || dstIpAddr == 0 { //handle dstIpAddr = 127.0.0.1 or 0
                    dstIpAddr = ipAddr;
                }
                let mut dstPort = msg.dstPort;
                if RDMA_CTLINFO.IsEgress(dstIpAddr) {
                    self.SendControlMsgInternal(
                        msg.sockfd,
                        RDMA_CTLINFO.localIp_get(),
                        // ipAddr.to_be(),
                        srcVpcIpAddr,
                        msg.srcPort,
                        dstIpAddr,
                        dstPort,
                    );
                } else {
                    // error!("RDMAConnectUsingPodId: Connect to ip {} port {}", dstIpAddr, dstPort);
                    match RDMA_CTLINFO.IsService(dstIpAddr, &dstPort) {
                        None => {}
                        Some(ipWithPort) => {
                            println!("RDMAConnectUsingPodId: The traffic is connecting to a service. Change the connection to {:?}", ipWithPort);
                            dstIpAddr = ipWithPort.ip;
                            dstPort = ipWithPort.port.port;
                        }
                    }
                    match RDMA_CTLINFO.get_node_ip_by_pod_ip(&dstIpAddr) {
                        Some(nodeIpAddr) => {
                            // println!("qq1: RDMAReqMsg::RDMAConnectUsingPodId before SendControlMsgInternal");
                            // RDMA_SRV.timestamps.lock().push(TSC.Rdtsc()); // 1
                            self.SendControlMsgInternal(
                                msg.sockfd,
                                nodeIpAddr,
                                srcVpcIpAddr,
                                msg.srcPort,
                                dstIpAddr,
                                dstPort,
                            );
                        }
                        None => {
                            if !RDMA_CTLINFO.isK8s {
                                if dstIpAddr == 0 {
                                    self.SendControlMsgInternal(
                                        msg.sockfd,
                                        u32::from(Ipv4Addr::from_str("172.16.1.43").unwrap()).to_be(),
                                        srcVpcIpAddr,
                                        msg.srcPort,
                                        dstIpAddr,
                                        dstPort,
                                    );
                                }
                            }
                            else {
                                println!("TODO: return error as no ip to node mapping is found, dstIpAddr: {}, dstPort: {}", dstIpAddr, dstPort);
                            }
                        }
                    }
                }
            }
            RDMAReqMsg::RDMAWrite(msg) => match RDMA_SRV.channels.lock().get(&msg.channelId) {
                Some(rdmaChannel) => {
                    rdmaChannel.RDMASend();
                }
                None => {
                    panic!("RDMAChannel with id {} does not exist!", msg.channelId);
                }
            },
            RDMAReqMsg::RDMARead(msg) => match RDMA_SRV.channels.lock().get(&msg.channelId) {
                Some(rdmaChannel) => {
                    // debug!("RDMAReqMsg::RDMARead, channel id: {}", rdmaChannel.localId);
                    rdmaChannel.SendConsumedData();
                }
                None => {
                    panic!("RDMAChannel with id {} does not exist!", msg.channelId);
                }
            },
            RDMAReqMsg::RDMAShutdown(msg) => match RDMA_SRV.channels.lock().get(&msg.channelId) {
                Some(rdmaChannel) => {
                    rdmaChannel.Shutdown();
                }
                None => {
                    panic!("RDMAChannel with id {} does not exist!", msg.channelId);
                }
            },
            RDMAReqMsg::RDMAClose(msg) => {
                let rdmaChannel = RDMA_SRV
                    .channels
                    .lock()
                    .get(&msg.channelId)
                    .unwrap()
                    .clone();
                rdmaChannel.Close();
            }
            RDMAReqMsg::RDMAPendingShutdown(msg) => {
                match RDMA_SRV.channels.lock().get(&msg.channelId) {
                    Some(rdmaChannel) => {
                        rdmaChannel.PendingShutdown();
                    }
                    None => {
                        panic!("RDMAChannel with id {} does not exist!", msg.channelId);
                    }
                }
            }
            RDMAReqMsg::RDMASendUDPPacket(msg) => {
                // error!("RDMAReqMsg::RDMASendUDPPacket, msg: {:?}", msg);
                let udpPacket = &mut self.shareRegion.lock().udpBufSent[msg.udpBuffIdx as usize];
                let vpcId;
                let ipAddr;
                if RDMA_CTLINFO.isK8s {
                    vpcId = *self.vpcId.lock();
                    ipAddr = self.ipAddr.lock().to_be();
                } else {
                    let podId = "client".to_string();
                    let mut vpcIpAddr = RDMA_CTLINFO
                        .podIdToVpcIpAddr
                        .lock()
                        .get(&podId)
                        .unwrap()
                        .clone();
                    if vpcIpAddr.ipAddr == udpPacket.dstIpAddr {
                        vpcIpAddr = RDMA_CTLINFO
                            .podIdToVpcIpAddr
                            .lock()
                            .get(&"server".to_string())
                            .unwrap()
                            .clone();
                    }
                    vpcId = vpcIpAddr.vpcId;
                    ipAddr = vpcIpAddr.ipAddr.to_be();
                }

                if vpcId == 0 {
                    panic!("TODO: VpcIdIpAddr hasn't been obtained from control plan yet, should handle this in negative case");
                }

                udpPacket.srcIpAddr = ipAddr;
                udpPacket.vpcId = vpcId;

                match RDMA_CTLINFO.get_node_ip_by_pod_ip(&udpPacket.dstIpAddr) {
                    Some(nodeIpAddr) => {
                        let conns = RDMA_SRV.conns.lock();
                        let rdmaConn = conns.get(&nodeIpAddr).unwrap().clone();
                        let wrId = (self.id as u64) << 32 | msg.udpBuffIdx as u64;
                        let laddr = udpPacket as *const _ as u64;
                        // debug!("RDMAReqMsg::RDMASendUDPPacket, 1, agentId: {:x}, udpBuffIdx: {:x}, wrId: {:x}, laddr: {:x}", self.id, msg.udpBuffIdx, wrId, laddr);
                        rdmaConn
                            .RDMAUDQPSend(
                                laddr,
                                (udpPacket.length + UDP_BUFF_OFFSET as u16) as u32,
                                wrId,
                                self.udpMR.LKey(),
                            )
                            .expect("RDMAUDQPSend failed...");
                    }
                    None => {
                        println!("TODO: return error as no ip to node mapping is found");
                    }
                }
            }
            RDMAReqMsg::RDMAReturnUDPBuff(msg) => {
                RDMA_SRV
                    .udpBufferAllocator
                    .lock()
                    .ReturnBuffer(msg.udpBuffIdx);
            }
        }
    }

    fn SendControlMsgInternal(
        &self,
        sockfd: u32,
        nodeIpAddr: u32,
        srcVpcIpAddr: VpcIpAddr,
        srcPort: u16,
        dstIpAddr: u32,
        dstPort: u16,
    ) {
        let conns = RDMA_SRV.conns.lock();
        let rdmaConn = conns.get(&nodeIpAddr).unwrap();

        let rdmaChannel = self.CreateClientRDMAChannel(
            &RDMAConnectReq {
                sockfd: sockfd,
                dstIpAddr: dstIpAddr,
                dstPort: dstPort,
                srcIpAddr: srcVpcIpAddr.ipAddr,
                // vpcId: srcVpcIpAddr.vpcId
                srcPort: srcPort,
            },
            rdmaConn.clone(),
        );

        RDMA_SRV
            .channels
            .lock()
            .insert(rdmaChannel.localId, rdmaChannel.clone());

        let connectReqeust = rdmaChannel.CreateConnectRequest(sockfd, srcVpcIpAddr.vpcId);
        // println!("qq1: SendControlMsgInternal, before rdmaConn::SendControlMsg");
        // RDMA_SRV.timestamps.lock().push(TSC.Rdtsc()); //2
        rdmaConn
            .ctrlChan
            .lock()
            .SendControlMsg(ControlMsgBody::ConnectRequest(connectReqeust));
        // .expect("fail to send msg");
    }

    pub fn HandleUDPPacketRecv(&self, udpPacket: &UDPPacket) {
        let (addr, udpBuffIdx) = self.udpRecvBufferAllocator.lock().GetFreeBuffer();
        if addr != 0 {
            let srcPtr = udpPacket as *const _ as *const u8;
            let dstPtr = addr as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    srcPtr,
                    dstPtr,
                    udpPacket.length as usize + UDP_BUFF_OFFSET,
                );
            }

            self.SendResponse(RDMAResp {
                user_data: 0,
                msg: RDMARespMsg::RDMARecvUDPPacket(RDMARecvUDPPacket { udpBuffIdx }),
            });
        } else {
            error!("No buffer to hold the udp packet, drop!");
        }
    }
}
