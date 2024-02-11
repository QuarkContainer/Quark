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

use std::str::FromStr;
use std::sync::atomic::AtomicU16;
use std::sync::atomic::Ordering;

use crate::qlib::common::*;
use crate::qlib::tsot_msg::*;
use crate::vmspace::kernel::GlobalIOMgr;
use crate::vmspace::kernel::IOURING;
use crate::vmspace::kernel::SHARESPACE;
use crate::VMS;

use super::USocket;

lazy_static! {
    pub static ref TSOT_AGENT: TsotAgent = TsotAgent::New().unwrap();
}

pub struct TsotAgent {
    pub client: USocket,
    pub currentReqId: AtomicU16,
}

impl TsotAgent {
    pub fn New() -> Result<Self> {
        let client = USocket::InitClient(TSOT_SOCKET_PATH)?;

        let socket = client.socket;

        let agent = Self {
            client: client,
            currentReqId: AtomicU16::new(0),
        };

        agent.Register()?;

        IOURING.TsotPollInit(socket);
        
        return Ok(agent);
    }

    pub fn NextReqId(&self) -> u16 {
        return self.currentReqId.fetch_add(1, Ordering::SeqCst);
    }

    pub fn Register(&self) -> Result<()> {
        let str = VMS.lock().podUid.clone();
        let podUid = uuid::Uuid::from_str(&str)?;
        let mut regsiterMsg = PodRegisterReq::default();
        regsiterMsg.podUid.copy_from_slice(podUid.as_bytes());

        let msg = TsotMsg::PodRegisterReq(regsiterMsg).into();
        self.SendMsg(&msg)?;
        let resp = self.RecvMsg()?;

        match resp.msg {
            TsotMsg::PodRegisterResp(m) => {
                if m.errorCode != ErrCode::None as u32 {
                    panic!("TsotAgent::Register fail with error {:?}", m.errorCode);
                }

                SHARESPACE.tsotSocketMgr.SetLocalIpAddr(m.containerIp);

                //info!("TsotAgent::Register success with ip {:?}", QIPv4Addr::from(m.containerIp).ToBytes());
            }
            _ => {
                panic!("TsotAgent::Register get unexpect message {:?}", resp.msg);
            }
        }

        unsafe {
            let socket = self.client.socket;
            let flags = libc::fcntl(socket, libc::F_GETFL, 0);
            let ret = libc::fcntl(socket, libc::F_SETFL, flags | libc::O_NONBLOCK);
            assert!(ret == 0, "UnblockFd fail");
        }

        return Ok(())
    }

    pub fn SendMsg(&self, m: &TsotMessage) -> Result<()> {
        let bytes = m.msg.AsBytes();
        if m.socket > 0 {
            let size = self.client.WriteWithFds(bytes, &[m.socket])?;
            assert!(size == bytes.len());
        } else {
            let ret = self.client.WriteAll(bytes);
            return ret;
        }

        return Ok(())
    }

    pub fn RecvMsg(&self) -> Result<TsotMessage> {
        const MSG_SIZE : usize = std::mem::size_of::<TsotMsg>();
        let mut bytes = [0u8; MSG_SIZE];

        let (size, fds) = self.client.ReadWithFds(&mut bytes)?;
        assert!(size == MSG_SIZE);
        let msg = TsotMsg::FromBytes(&bytes);

        if fds.len() == 0 {
            return Ok(TsotMessage {
                socket: -1,
                msg: msg,
            })
        } else {
            let fd = fds[0];
            let _hostfd = GlobalIOMgr().AddSocket(fd);
            
            return Ok(TsotMessage {
                socket: fd,
                msg: msg,
            })
        }

    }
}