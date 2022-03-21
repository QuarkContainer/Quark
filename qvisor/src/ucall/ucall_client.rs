// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

use serde_json;

use super::super::qlib::common::*;
use super::super::qlib::control_msg::*;
use super::usocket::*;
use super::ucall::*;

pub struct UCallClient {
    pub sock: USocket,
}

impl UCallClient {
    pub fn Init(path: &str) -> Result<Self> {
        let sock = USocket::InitClient(path)?;
        return Ok(Self {
            sock: sock,
        })
    }

    pub fn Call(&self, req: &UCallReq) -> Result<UCallResp> {
        let reqArr = serde_json::to_vec(req).map_err(|e|Error::Common(format!("UCallClient ser error is {:?}", e)))?;
        let fds = req.GetFds();
        match fds {
            None => self.sock.WriteLen(reqArr.len(), &[])?,
            Some(fds) => self.sock.WriteLen(reqArr.len(), fds)?,
        }

        self.sock.WriteAll(&reqArr)?;

        let (len, _fds) = self.sock.ReadLen()?;
        let mut buf : [u8; UCALL_BUF_LEN] = [0; UCALL_BUF_LEN];

        assert!(len < UCALL_BUF_LEN, "UCallClient::Call resp is too long");
        self.sock.ReadAll(&mut buf[0..len])?;
        let resp : UCallResp = serde_json::from_slice(&buf[0..len]).map_err(|e|Error::Common(format!("UCallClient deser error is {:?}", e)))?;
        match resp {
            UCallResp::UCallRespErr(s) => return Err(Error::Common(s)),
            _ => (),
        }
        return Ok(resp);
    }

    pub fn StreamCall(&self, req: &UCallReq) -> Result<()> {
        let reqArr = serde_json::to_vec(req).map_err(|e|Error::Common(format!("UCallClient ser error is {:?}", e)))?;
        let fds = req.GetFds();
        match fds {
            None => self.sock.WriteLen(reqArr.len(), &[])?,
            Some(fds) => self.sock.WriteLen(reqArr.len(), fds)?,
        }

        self.sock.WriteAll(&reqArr)?;
        return Ok(())
    }

    pub fn StreamGetRet(&self) -> Result<UCallResp> {
        let (len, _fds) = self.sock.ReadLen()?;
        let mut buf : [u8; UCALL_BUF_LEN] = [0; UCALL_BUF_LEN];

        assert!(len < UCALL_BUF_LEN, "UCallClient::Call resp is too long");
        self.sock.ReadAll(&mut buf[0..len])?;
        let resp : UCallResp = serde_json::from_slice(&buf[0..len]).map_err(|e|Error::Common(format!("UCallClient deser error is {:?}", e)))?;
        match resp {
            UCallResp::UCallRespErr(s) => return Err(Error::Common(s)),
            _ => (),
        }
        return Ok(resp);
    }
}