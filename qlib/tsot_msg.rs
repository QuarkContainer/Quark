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

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec::Vec;
use core::mem::size_of;

pub static TSOT_SOCKET_PATH: &'static str = "/var/run/quark/tsot-socket";
pub static TSOT_HOST_SOCKET_PATH: &'static str = "/var/run/quark_host/tsot-socket";

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum ErrCode {
    None = 0,
    PodUidDonotExisit,
    ECONNREFUSED = 111, //
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TsotMessage {
    pub socket: i32,
    pub msg: TsotMsg,
}

impl Default for TsotMessage {
    fn default() -> Self {
        return Self {
            socket: 0,
            msg: TsotMsg::None,
        };
    }
}

impl From<TsotMsg> for TsotMessage {
    fn from(msg: TsotMsg) -> Self {
        return Self {
            socket: -1,
            msg: msg,
        };
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TsotMsg {
    None,
    PodRegisterReq(PodRegisterReq),

    // admin gateway connection, it could connect to any namespace/pod as a client
    GatewayRegisterReq(GatewayRegisterReq),

    CreateSocketReq(CreateSocketReq),

    ListenReq(ListenReq),
    AcceptReq(AcceptReq),
    StopListenReq(StopListenReq),
    PodConnectReq(PodConnectReq),
    GatewayConnectReq(GatewayConnectReq),
    DnsReq(DnsReq),

    //////////////////////////////////////////////////////
    // from nodeagent to pod
    PodRegisterResp(PodRegisterResp),
    GatewayRegisterResp(GatewayRegisterResp),
    CreateSocketResp(CreateSocketResp),
    PeerConnectNotify(PeerConnectNotify),
    PodConnectResp(PodConnectResp),
    GatewayConnectResp(GatewayConnectResp),
    DnsResp(DnsResp),
    Hibernate(Hibernate),
    Wakeup(Wakeup),
}

pub const BUFF_SIZE: usize = core::mem::size_of::<TsotMsg>();

impl TsotMsg {
    pub fn AsBytes(&self) -> &'static [u8] {
        let addr = self as *const _ as u64 as *const u8;
        return unsafe { core::slice::from_raw_parts(addr, size_of::<Self>()) };
    }

    pub fn FromBytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() == size_of::<Self>());
        let addr = &bytes[0] as *const _ as u64;
        let ret = unsafe { *(addr as *const Self) };
        return ret;
    }
}

// from pod to node agent
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PodRegisterReq {
    pub podUid: [u8; 16],
}

// from pod to node agent
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GatewayRegisterReq {
    pub gatewayUid: [u8; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CreateSocketReq {}

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

#[derive(Debug, Clone, Copy)]
pub enum ConnectReq {
    PodConnectReq(PodConnectReq),
    GatewayConnectReq(GatewayConnectReq),
}

// send with the socket fd
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PodConnectReq {
    pub reqId: u32,
    pub dstIp: u32,
    pub dstPort: u16,
    pub srcPort: u16,
}

// send with the socket fd
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GatewayConnectReq {
    pub reqId: u32,
    pub podNamespace: [u8; 64],
    pub dstIp: u32,
    pub dstPort: u16,
    pub srcPort: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DnsReq {
    pub reqId: u16,
    pub nameslen: u16,
    pub names: [u8; 256],
}

impl DnsReq {
    pub fn GetDomains(&self) -> Vec<String> {
        let namesStr =
            unsafe { String::from_utf8_unchecked(self.names[0..self.nameslen as usize].to_vec()) };

        let names: Vec<&str> = namesStr.split(":").collect();
        let mut domains = Vec::new();
        for name in names {
            domains.push(name.to_owned());
        }
        return domains;
    }
}

//////////////////////////////////////////////////////
// from nodeagent to pod

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PodRegisterResp {
    // the pod's container IP addr
    pub containerIp: u32,
    pub errorCode: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GatewayRegisterResp {
    pub errorCode: u32,
}

// send with new socket fd
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CreateSocketResp {}

// send with new socket fd
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Hibernate {
    pub _type: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Wakeup {
    pub _type: u32,
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
        let addr = &self.peerIp as *const _ as u64 as *const u8;
        let bytes = unsafe { core::slice::from_raw_parts(addr, 4) };

        let mut ret: [u8; 4] = [0; 4];

        ret.copy_from_slice(bytes);
        return ret;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PodConnectResp {
    pub reqId: u32,
    pub errorCode: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GatewayConnectResp {
    pub reqId: u32,
    pub errorCode: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DnsResp {
    pub reqId: u16,
    // maxinum 4 request and response
    pub ips: [u32; 4],
    pub count: usize,
}
