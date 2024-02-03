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

use core::mem::size_of;

pub static TSOT_SOCKET_PATH: &'static str = "/var/run/quark/tsot-socket";

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum ErrCode {
    None = 0,
    PodUidDonotExisit,
    ConnectFail,
}

pub struct TsotMessage {
    pub socket: i32,
    pub msg: TsotMsg,
}

impl Default for TsotMessage {
    fn default() -> Self {
        return Self {
            socket: 0,
            msg: TsotMsg::None
        }
    }
}

impl From<TsotMsg> for TsotMessage {
    fn from(msg: TsotMsg) -> Self {
        return Self {
            socket: -1,
            msg: msg
        }
    }
}


#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TsotMsg {
    None,
    PodRegisterReq(PodRegisterReq),
    ListenReq(ListenReq),
    AcceptReq(AcceptReq),
    StopListenReq(StopListenReq),
    ConnectReq(ConnectReq),

    //////////////////////////////////////////////////////
    // from nodeagent to pod
    PodRegisterResp(PodRegisterResp),
    PeerConnectNotify(PeerConnectNotify),
    ConnectResp(ConnectResp),
}


impl TsotMsg {
    pub fn AsBytes(&self) -> &'static[u8] {
        let addr = self as * const _ as u64 as * const u8;
        return unsafe {
            core::slice::from_raw_parts(addr, size_of::<Self>())
        }
    }

    pub fn FromBytes<'a>(bytes: &'a [u8]) -> &'a Self {
        assert!(bytes.len() == size_of::<Self>());
        let addr = &bytes[0] as * const _ as u64;
        return unsafe {
            &*(addr as * const Self)
        }
    }
}

// from pod to node agent
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PodRegisterReq {
    pub podUid: [u8; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ListenReq {
    pub port: u16,
    pub backlog: u32,
}

// notify nodeagent one connected socket is consumed by user code
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AcceptReq {
    pub port: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StopListenReq {
    pub port: u16,
}

// send with the socket fd
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConnectReq {
    pub reqId: u16,
    pub dstIp: u32,
    pub dstPort: u16,
    pub srcPort: u16,
}

//////////////////////////////////////////////////////
// from nodeagent to pod

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PodRegisterResp {
    // the pod's container IP addr
    pub containerIp: u32,
    pub errorCode: u32
}

// another pod connected to current pod
// send with new socket fd
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PeerConnectNotify {
    pub peerIp: u32,
    pub peerPort: u16,
    pub localPort: u16,
}

impl PeerConnectNotify {
    pub fn PeerAddrBytes(&self) -> [u8; 4] {
        let addr = &self.peerIp as * const _ as u64 as * const u8;
        let bytes = unsafe {
            core::slice::from_raw_parts(addr, 4)
        };

        let mut ret : [u8; 4] = [0; 4];
        
        ret.copy_from_slice(bytes);
        return ret;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConnectResp {
    pub reqId: u16,
    pub errorCode: u32
}




