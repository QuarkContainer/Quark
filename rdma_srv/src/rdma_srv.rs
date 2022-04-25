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

use spin::Mutex;
use std::collections::HashMap;

use super::id_mgr::IdMgr;
use super::qlib::rdma_share::*;
use super::rdma::*;
use super::rdma_agent::*;
use super::rdma_channel::*;
use super::rdma_conn::*;
use super::rdma_ctrlconn::*;
use lazy_static::lazy_static;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::{env, mem, ptr, thread, time};
use std::ffi::CString;

lazy_static! {
    pub static ref RDMA_SRV: RDMASrv = RDMASrv::New();
    pub static ref RDMA_CTLINFO: CtrlInfo = CtrlInfo::default();
    //pub static ref RDMA_SRV_SHARED_REGION: ShareRegion = ShareRegion::default();
}

#[derive(Clone)]
pub enum SrvEndPointStatus {
    Binded,
    Listening,
}

pub struct SrvEndpoint {
    //pub srvEndpointId: u32, // to be returned as bind
    pub agentId: u32,
    pub sockfd: u32,
    pub endpoint: Endpoint,
    pub status: SrvEndPointStatus, //TODO: double check whether it's needed or not
                                   //pub acceptQueue: [RDMAChannel; 5], // hold rdma channel which can be assigned.
}

pub struct RDMAControlChannelRegion {
    // data buf for sockbuf, it will be mapped in the rdma MR
    pub iobufs: [IOBuf; IO_BUF_COUNT],

    // metadata region for the sockbuf
    pub ioMetas: [IOMetas; IO_BUF_COUNT],
}

pub struct RDMASrv {
    // epoll fd
    pub epollFd: i32,

    // unix socket srv fd
    pub unixSockfd: i32,

    // tcp socket srv fd
    pub tcpSockfd: i32,

    // eventfd which used by rdma client to trigger RDMA srv
    pub eventfd: i32,

    // srv memory region memfd shared with RDMAclient
    pub srvMemfd: i32,

    // srv memory region shared with all RDMAClient
    pub srvMemRegion: MemRegion,

    // rdma connects: remote node ipaddr --> RDMAConn
    pub conns: Mutex<HashMap<u32, RDMAConn>>,

    // todo: tbd: need it?
    // rdma connects: virtual subnet ipaddr --> RDMAConn
    // pub vipMapping: HashMap<u32, RDMAConn>,

    // rdma channels: channelId --> RDMAChannel
    pub channels: Mutex<HashMap<u32, RDMAChannel>>,

    // rdma control channels: qpNum -> RDMAChannel
    pub controlChannels: Mutex<HashMap<u32, RDMAControlChannel>>,
    pub controlChannels2: Mutex<HashMap<u32, RDMAChannel>>,

    // agents: agentId -> RDMAAgent
    pub agents: Mutex<HashMap<u32, RDMAAgent>>,

    // the bitmap to expedite ready container search
    pub shareRegion: &'static ShareRegion,

    // keep track of server endpoint on current node
    pub srvEndPoints: Mutex<HashMap<Endpoint, SrvEndpoint>>,
    pub currNode: Node,

    pub channelIdMgr: Mutex<IdMgr>,

    pub agentIdMgr: Mutex<IdMgr>,
    //TODO: indexes allocated for io buffer.
    pub controlRegion: &'static RDMAControlChannelRegion,
    pub controlChannelRegionAddress: MemRegion,
    pub controlBufIdMgr: Mutex<IdMgr>,
    pub keys: Vec<[u32; 2]>,
}

impl Drop for RDMASrv {
    fn drop(&mut self) {
        //TODO: This is not called because it's global static
        println!("drop RDMASrv");
        unsafe {
            libc::munmap(
                self.controlChannelRegionAddress.addr as *mut libc::c_void,
                self.controlChannelRegionAddress.len as usize,
            );
            libc::munmap(
                self.srvMemRegion.addr as *mut libc::c_void,
                self.srvMemRegion.len as usize,
            );
        }
    }
}

impl RDMASrv {
    pub fn New() -> Self {
        println!("RDMASrv::New");

        let controlSize = mem::size_of::<RDMAControlChannelRegion>();
        let contrlAddr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                controlSize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };

        if contrlAddr == libc::MAP_FAILED {
            println!("failed to mmap control region");
        }

        // println!(
        //     "contrlAddr : 0x{:x}, controlSize is: {}",
        //     contrlAddr as u64, controlSize
        // );

        let memfdname = CString::new("RDMASrvMemFd").expect("CString::new failed for RDMASrvMemFd");
        let memfd = unsafe {
            libc::memfd_create(
                memfdname.as_ptr(),
                libc::MFD_ALLOW_SEALING,
            )
        };
        // println!("memfd::{}", memfd);
        if memfd == -1 {
            panic!(
                "fail to create memfd, error is: {}",
                std::io::Error::last_os_error()
            );
        }
        let size = mem::size_of::<ShareRegion>();
        let _ret = unsafe { libc::ftruncate(memfd, size as i64) };
        let shareRegionAddr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                memfd,
                0,
            )
        };

        if shareRegionAddr as i64 == -1 {
            panic!(
                "fail to mmap share region, error is: {}",
                std::io::Error::last_os_error()
            );
        }

        println!(
            "shareRegionAddr : 0x{:x}, size is: {}",
            shareRegionAddr as u64, size
        );

        //RDMA.Init("", 1);

        //start from 2M registration.
        let mr = RDMA
            .CreateMemoryRegion(contrlAddr as u64, 2 * 1024 * 1024)
            .unwrap();
        return Self {
            epollFd: 0,
            unixSockfd: 0,
            tcpSockfd: 0,
            eventfd: unsafe { libc::eventfd(0, 0) },
            srvMemfd: memfd,
            srvMemRegion: MemRegion {
                addr: shareRegionAddr as u64,
                len: size as u64,
            },
            conns: Mutex::new(HashMap::new()),
            channels: Mutex::new(HashMap::new()),
            agents: Mutex::new(HashMap::new()),
            shareRegion: unsafe {
                let addr = shareRegionAddr as *mut ShareRegion;
                &mut (*addr)
            },
            srvEndPoints: Mutex::new(HashMap::new()),
            currNode: Node::default(),
            channelIdMgr: Mutex::new(IdMgr::Init(1, 1000)),
            agentIdMgr: Mutex::new(IdMgr::Init(0, 1000)),
            controlRegion: unsafe {
                let addr = contrlAddr as *mut RDMAControlChannelRegion;
                &mut (*addr)
            },
            controlChannelRegionAddress: MemRegion {
                addr: contrlAddr as u64,
                len: controlSize as u64,
            },
            controlBufIdMgr: Mutex::new(IdMgr::Init(0, 16)),
            keys: vec![[mr.LKey(), mr.RKey()]],
            controlChannels: Mutex::new(HashMap::new()),
            controlChannels2: Mutex::new(HashMap::new()),
        };
    }

    pub fn getRDMAChannel(&self, channelId: u32) -> Option<RDMAChannel> {
        match self.channels.lock().get(&channelId) {
            None => None,
            Some(rdmaChannel) => Some(rdmaChannel.clone()),
        }
    }

    pub fn ProcessRDMAWriteImmFinish(&self, channelId: u32, qpNum: u32) {
        if channelId != 0 {
            match self.channels.lock().get(&channelId) {
                None => {
                    panic!(
                        "ProcessRDMAWriteImmFinish get unexpected channelId: {}",
                        channelId
                    );
                }
                Some(channel) => {
                    channel.ProcessRDMAWriteImmFinish();
                }
            }
        } else {
            match self.controlChannels.lock().get(&qpNum) {
                None => {
                    panic!("ProcessRDMAWriteImmFinish get unexpected qpNum: {}", qpNum);
                }
                Some(channel) => {
                    channel.ProcessRDMAWriteImmFinish();
                }
            }
        }
    }

    pub fn ProcessRDMARecvWriteImm(&self, channelId: u32, qpNum: u32, recvCount: u32) {
        if channelId != 0 {
            match self.channels.lock().get(&channelId) {
                None => {
                    panic!(
                        "ProcessRDMARecvWriteImm get unexpected channelId: {}",
                        channelId
                    );
                }
                Some(channel) => {
                    channel.ProcessRDMARecvWriteImm(qpNum, recvCount as u64);
                }
            }
        } else {
            match RDMA_SRV.controlChannels.lock().get(&qpNum) {
                //where lock end??
                None => {
                    panic!("ProcessRDMAWriteImmFinish get unexpected qpNum: {}", qpNum);
                }
                Some(channel) => {
                    channel.ProcessRDMARecvWriteImm(qpNum, recvCount as u64);
                }
            }
        }
    }

    pub fn HandleClientRequest(&self) {
        let agentIds = self.shareRegion.getAgentIds();
        // println!("agentIds: {:?}", agentIds);
        let rdmaAgents = self.agents.lock();
        for agentId in agentIds.iter() {
            match rdmaAgents.get(agentId) {
                Some(rdmaAgent) => {
                    rdmaAgent.HandleClientRequest();
                }
                None => {
                    println!("RDMA agent with id {} doesn't exist", agentId);
                }
            }
        }
    }

    // pub fn CreateRDMAChannel(&self, agentId: u32) {
    //     let channelId = self.channelIdMgr.lock().AllocId();

    // }
}

// scenarios:
// a. init
// b. input:
//      1. srv socket accept -> init client connection
//      2. srv tcp socket accept -> init peer connection
//      3. client submit queue
//      4. rdma work complete trigger
//      5. connection mgr callback
// c. de-construction
//      1. connection mgr disconnect
//      2. tcp connection close
//      3. rdma connection disconnect (keepalive?)
// request/response type
