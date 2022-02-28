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
use alloc::sync::Arc;
use alloc::sync::Weak;
use core::sync::atomic::AtomicU64;

use super::rdma_conn::*;
use super::rdma_agent::*;

// RDMA Channel
use super::qlib::bytestream::*;

pub struct RDMAChannelIntern {
    pub localId: u32,
    pub remoteId: u32,
    pub readBuf: Mutex<ByteStream>,
    pub writeBuf: Mutex<ByteStream>,
    pub consumeReadData: &'static AtomicU64,

    // rdma connect to remote node
    pub conn: RDMAConn,

    // rdma agent connected to rdma client
    pub agent: RDMAAgent,

    pub vpcId: u32,
    pub srcIPAddr: u32,
    pub dstIPAddr: u32,
    pub srcPort: u16,
    pub dstPort: u16
}

pub struct RDMAChannel(Arc<RDMAChannelIntern>);
pub struct RDMAChannelWeak(Weak<RDMAChannelIntern>);
