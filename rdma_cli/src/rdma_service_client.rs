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
use std::collections::HashSet;
use spin::Mutex;

use super::qlib::rdma_share::*;
use super::rdma_conn::*;
use super::rdma_channel::*;
use super::rdma_agent::*;
use super::rdma_ctrlconn::*;

pub enum SockType {
    TBD,
    SERVER,
    CLIENT
}

pub struct SockFdInfo {
    pub fd: u32,
    pub readBuf: Mutex<ByteStream>,
    pub writeBuf: Mutex<ByteStream>,
    pub srcIpAddr: u32,
    pub srcPort: u16,
    pub dstIpAdr: u32,
    pub dstPort: u16,
    pub status: SockStatus,
    pub duplexMode: DuplexMode, 
    pub sockType: SockType,
    pub channelId: u32,
    pub acceptQueue: [RDMAChannel; 5] //object with remote information.
}

pub struct RDMASvcCli {
    // agent id
    pub agentId: u32,

    // the unix socket fd between rdma client and RDMASrv
    pub sockfd: i32,

    // the memfd share memory with rdma client
    pub memfd: i32,

    // the eventfd which send notification to client
    pub eventfd: i32,

    // the memory region shared with client
    pub shareMemRegion: MemRegion,

    pub clientShareRegion: &'static mut ClientShareRegion,

    // srv memory region shared with all RDMAClient
    pub srvMemRegion: MemRegion,

    // the bitmap to expedite ready container search
    pub srvShareRegion: &'static ShareRegion,

    // // sockfd -> rdmaChannelId
    // pub rdmaChannels: HashMap<u32, u32>,

    // sockfd -> sockFdInfo
    pub sockFdInfos: HashMap<u32, SockFdInfo>,
    
    // ipaddr -> set of used ports
    pub usedPorts: HashMap<u32, HashSet<u16>>,
}