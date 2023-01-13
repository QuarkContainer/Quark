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

use alloc::vec::Vec;
use alloc::string::String;

use crate::qlib::bytestream::*;
use crate::qlib::common::*;
use crate::qlib::linux_def::*;

#[repr(u8)]
pub enum UserMsgType {
    UserFuncCall = 1,
    UserFuncResp
}

// serialize format
// len: u32
// type: 1 byte
// msg: UserFuncCall or UserFuncResp
pub enum UserMsg {
    UserFuncCall(UserFuncCall),
    UserFuncResp(UserFuncResp),
}

impl MessageIO for UserMsg {
    // data size, used for write check
    fn Size(&self) -> usize {
        match self {
            UserMsg::UserFuncCall(msg) => {
                // msg type + msgboday
                return 1 + msg.Size()
            }
            UserMsg::UserFuncResp(msg) => {
                // msg type + msgboday
                return 1 + msg.Size()
            }
        }
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let msgType = buf.ReadObj::<u8>()?;

        match msgType {
            x if x == UserMsgType::UserFuncCall as u8 => {
                let msg = UserFuncCall::Read(buf)?;
                return Ok(Self::UserFuncCall(msg))
            }
            x if x == UserMsgType::UserFuncResp as u8 => {
                let msg = UserFuncResp::Read(buf)?;
                return Ok(Self::UserFuncResp(msg))
            }
            _ => return Err(Error::SysError(SysErr::EINVAL))
        }
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        match self {
            Self::UserFuncCall(msg) => {
                buf.WriteObj(&(UserMsgType::UserFuncCall as u8))?;
                msg.Write(buf)?;
            }
            Self::UserFuncResp(msg) => {
                buf.WriteObj(&(UserMsgType::UserFuncResp as u8))?;
                msg.Write(buf)?;
            }
        }

        return Ok(())
    }
}
 
pub struct UserFuncCall {
    pub userdata: u64,
    
    // <len: u16, bytes: [u8]>
    pub funcName: String,
    
    // <len: u16, bytes: [u8]>
    pub payload: Vec<u8>, 
}

impl MessageIO for UserFuncCall {
    // data size, used for write check
    fn Size(&self) -> usize {
        let mut size = 8;
        size += 2 + self.funcName.len();
        size += 2 + self.payload.len();
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let userData = buf.ReadObj::<u64>()?;

        let namelen = buf.ReadObj::<u16>()?;
        let name= buf.ReadString(namelen as usize)?;

        let buflen = buf.ReadObj::<u16>()?;
        let payload = buf.ReadVec(buflen as usize)?;

        let obj = Self {
            userdata: userData,
            funcName: name,
            payload: payload,
        };

        return Ok(obj)
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        buf.WriteObj(&self.userdata)?;
        buf.WriteObj(&(self.funcName.len() as u16))?;
        buf.WriteSlice(self.funcName.as_bytes())?;
        buf.WriteObj(&(self.payload.len() as u16))?;
        buf.WriteSlice(&self.payload)?;

        return Ok(())
    }
}


pub struct UserFuncResp {
    pub userdata: u64,
    pub sessionId: u64,
    pub payload: Vec<u8>,
}

impl MessageIO for UserFuncResp {
    // data size, used for write check
    fn Size(&self) -> usize {
        let mut size = 8;
        size += 8;

        // 2 is the len of payload
        size += 2 + self.payload.len();
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let userData = buf.ReadObj::<u64>()?;
        let sessionId = buf.ReadObj::<u64>()?;
        let buflen = buf.ReadObj::<u16>()?;
        let payload = buf.ReadVec(buflen as usize)?;

        let obj = Self {
            userdata: userData,
            sessionId: sessionId,
            payload: payload,
        };

        return Ok(obj)
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        buf.WriteObj(&self.userdata)?;
        buf.WriteObj(&self.sessionId)?;
        buf.WriteObj(&(self.payload.len() as u16))?;
        buf.WriteSlice(&self.payload)?;

        return Ok(());
    }
}
