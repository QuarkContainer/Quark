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

impl Drop for TsotMessage {
    fn drop(&mut self) {
        unsafe {
            if self.socket >= 0 {
                libc::close(self.socket);
            }
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
    PodRegisterReq(PodRegisterReq),
    ListenReq(ListenReq),
    Accept(Accept),
    StopListenReq(StopListenReq),
    ConnectReq(ConnectReq),

    //////////////////////////////////////////////////////
    // from nodeagent to pod
    PodRegisterResp(PodRegisterResp),
    PeerConnectNotify(PeerConnectNotify),
    ConnectResp(ConnectResp),
}

// from pod to node agent
#[repr(C)]
#[derive(Debug, Clone, Copy)]
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
pub struct Accept {
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
    pub localPort: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConnectResp {
    pub reqId: u16,
    pub errorCode: u32
}




