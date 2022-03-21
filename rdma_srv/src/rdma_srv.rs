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

use super::qlib::rdma_share::*;
use super::rdma_agent::*;
use super::rdma_channel::*;
use super::rdma_conn::*;
use super::rdma_ctrlconn::*;
use std::mem::MaybeUninit;

use lazy_static::lazy_static;

lazy_static! {
    pub static ref RDMA_SRV: Mutex<RDMASrv> = Mutex::new(RDMASrv::New());
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
    pub endpoint: Endpoint,
    pub status: SrvEndPointStatus, //TODO: double check whether it's needed or not
                                   //pub acceptQueue: [RDMAChannel; 5], // hold rdma channel which can be assigned.
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
    pub conns: HashMap<u32, RDMAConn>,

    // todo: tbd: need it?
    // rdma connects: virtual subnet ipaddr --> RDMAConn
    // pub vipMapping: HashMap<u32, RDMAConn>,

    // rdma channels: channelId --> RDMAChannel
    pub channels: HashMap<u32, RDMAChannel>,

    // agents: agentId -> RDMAAgent
    pub agents: HashMap<u32, RDMAAgent>,

    // the bitmap to expedite ready container search
    pub shareRegion: &'static ShareRegion,

    // keep track of server endpoint on current node
    pub srvEndPoints: HashMap<Endpoint, SrvEndpoint>,

    pub currNode: Node,
    //TODO: indexes allocated for io buffer.
}

impl RDMASrv {
    pub fn New() -> Self {
        return Self {
            epollFd: 0,
            unixSockfd: 0,
            tcpSockfd: 0,
            eventfd: 0,
            srvMemfd: 0,
            srvMemRegion: MemRegion { addr: 0, len: 0 },
            conns: HashMap::new(),
            channels: HashMap::new(),
            agents: HashMap::new(),
            //shareRegion: &RDMA_SRV_SHARED_REGION,
            shareRegion: unsafe {
                let eventAddr = 0 as *mut ShareRegion; // as &mut qlib::Event;
                &mut (*eventAddr)
            },
            srvEndPoints: HashMap::new(),
            currNode: Node::default(),
        };
    }
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
