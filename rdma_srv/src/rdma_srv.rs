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

use std::collections::HashMap;
use spin::Mutex;

use super::qlib::rdma_share::*;
use super::rdma_conn::*;
use super::rdma_channel::*;
use super::rdma_agent::*;
use super::rdma_ctrlconn::*;

use lazy_static::lazy_static;

lazy_static! {
    //pub static ref RDMA_SRV: Mutex<RDMASrv> = RDMASrv::default();
    //pub static ref RDMA_CTLINFO: CtrlInfo = CtrlInfo::default();
}

pub struct RDMASrv {
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
}

impl RDMASrv {
    /*pub fn New() -> Self {

    }*/
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