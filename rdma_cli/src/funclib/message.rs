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
use std::fmt;

use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::bytestream::*;

pub trait MsgIO : Sized {
    // data size, used for write check
    fn Size(&self) -> usize;

    // read obj
    fn Read(buf: &mut SocketBufIovs) -> Result<Self>;
    
    // write obj
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()>;

    fn Serialize(&self, buf: &mut [u8]) -> Result<usize> {
        if self.Size() > buf.len() {
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let mut iovs = SocketBufIovs::NewFromBuf(buf);
        self.Write(&mut iovs)?;
        return Ok(self.Size());
    }

    fn Deserialize(buf: &[u8]) -> Result<Self> {
        let mut iovs = SocketBufIovs::NewFromBuf(buf);
        return Self::Read(&mut iovs);
    }
}

pub struct QMsg {
    pub messageId: u64,
    pub payload: MsgPayload,
}

impl QMsg {
    pub fn NewErrRespMsg(messageId: u64, errcode: i32) -> Self {
        return Self {
            messageId: messageId,
            payload: MsgPayload::NewErrorFuncResp(errcode),
        }
    }
    
    pub fn NewMsg(messageId: u64, payload: MsgPayload) -> Self {
        return Self {
            messageId: messageId,
            payload: payload,
        }
    }
}

impl fmt::Debug for QMsg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QMsg")
         .field("messageId", &self.messageId)
         .finish()
    }
}

impl MsgIO for QMsg {
    fn Size(&self) -> usize {
        let mut size = 0;
        size += 8; // message Id: 8 bytes
        size += self.payload.Size();
        return size;
    }

    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let messageId = buf.ReadObj::<u64>()?;
        let payload = MsgPayload::Read(buf)?;
        
        let msg = Self {
            messageId: messageId,
            payload: payload,
        };
        return Ok(msg)
    }

    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        buf.WriteObj(&self.messageId)?;
        self.payload.Write(buf)?;
        return Ok(())
    }
}

#[repr(u8)]
pub enum PayloadType {
    UserFuncCall = 1,
    AgentFuncCall,
    FuncResp,
    Credential,
}

// https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#server_error_responses

// Internal Server Error
// The server has encountered a situation it does not know how to handle.
pub const HTTP_INTERN_ERR: i32 = 500;

// Not Implemented
// The request method is not supported by the server and cannot be handled. 
// The only methods that servers are required to support (and therefore that must not return this code) 
// are GET and HEAD.
pub const HTTP_NOT_IMPL: i32 = 501;

// serialize format
// len: u32
// type: 1 byte
// msg: UserFuncCall or UserFuncResp
pub enum MsgPayload {
    UserFuncCall(UserFuncCall),
    AgentFuncCall(AgentFuncCall),
    FuncResp(FuncResp),
    Credential(Credential),
}

impl MsgPayload {
    pub fn NewErrorFuncResp(errcode: i32) -> Self {
        let resp = FuncResp::NewErr(errcode);
        return Self::FuncResp(resp);
    }

    pub fn NewUserFuncCall(funcName: String, payload: Vec<u8>) -> Self {
        let call = UserFuncCall {
            funcName: funcName,
            payload: payload,
        };

        return Self::UserFuncCall(call);
    }

    pub fn NewAgentFuncCall(appId: u64, funcName: String, payload: Vec<u8>) -> Self {
        let call = AgentFuncCall {
            appId: appId,
            funcName: funcName,
            payload: payload,
        };

        return Self::AgentFuncCall(call);
    }
}

impl MsgIO for MsgPayload {
    // data size, used for write check
    fn Size(&self) -> usize {
        match self {
            MsgPayload::UserFuncCall(msg) => {
                // msg type + msgboday
                return 1 + msg.Size()
            }
            MsgPayload::AgentFuncCall(msg) => {
                // msg type + msgboday
                return 1 + msg.Size()
            }
            MsgPayload::FuncResp(msg) => {
                // msg type + msgboday
                return 1 + msg.Size()
            }
            MsgPayload::Credential(msg) => {
                // msg type + msgboday
                return 1 + msg.Size()
            }
        }
    }

    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let msgType = buf.ReadObj::<u8>()?;

        match msgType {
            x if x == PayloadType::UserFuncCall as u8 => {
                let msg = UserFuncCall::Read(buf)?;
                return Ok(Self::UserFuncCall(msg))
            }
            x if x == PayloadType::AgentFuncCall as u8 => {
                let msg = AgentFuncCall::Read(buf)?;
                return Ok(Self::AgentFuncCall(msg))
            }
            x if x == PayloadType::FuncResp as u8 => {
                let msg = FuncResp::Read(buf)?;
                return Ok(Self::FuncResp(msg))
            }
            x if x == PayloadType::Credential as u8 => {
                let msg = Credential::Read(buf)?;
                return Ok(Self::Credential(msg))
            }
            _ => return Err(Error::SysError(SysErr::EINVAL))
        }
    }
    
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        match self {
            Self::UserFuncCall(msg) => {
                buf.WriteObj(&(PayloadType::UserFuncCall as u8))?;
                msg.Write(buf)?;
            }
            Self::AgentFuncCall(msg) => {
                buf.WriteObj(&(PayloadType::AgentFuncCall as u8))?;
                msg.Write(buf)?;
            }
            Self::FuncResp(msg) => {
                buf.WriteObj(&(PayloadType::FuncResp as u8))?;
                msg.Write(buf)?;
            }
            Self::Credential(msg) => {
                buf.WriteObj(&(PayloadType::Credential as u8))?;
                msg.Write(buf)?;
            }
        }

        return Ok(())
    }
}
 

pub struct Credential {
    pub appId: u64,
}

impl MsgIO for Credential {
    // data size, used for write check
    fn Size(&self) -> usize {
        let size = 8;
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let appId = buf.ReadObj::<u64>()?;

        let obj = Self {
            appId: appId,
        };

        return Ok(obj)
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        buf.WriteObj(&self.appId)?;

        return Ok(())
    }
}


pub struct UserFuncCall {
    // <len: u16, bytes: [u8]>
    pub funcName: String,
    
    // <len: u16, bytes: [u8]>
    pub payload: Vec<u8>, 
}

impl MsgIO for UserFuncCall {
    // data size, used for write check
    fn Size(&self) -> usize {
        let mut size = 8;
        size += 2 + self.funcName.len();
        size += 2 + self.payload.len();
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let namelen = buf.ReadObj::<u16>()?;
        let name= buf.ReadString(namelen as usize)?;

        let buflen = buf.ReadObj::<u16>()?;
        let payload = buf.ReadVec(buflen as usize)?;

        let obj = Self {
            funcName: name,
            payload: payload,
        };

        return Ok(obj)
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        buf.WriteObj(&(self.funcName.len() as u16))?;
        buf.WriteSlice(self.funcName.as_bytes())?;
        buf.WriteObj(&(self.payload.len() as u16))?;
        buf.WriteSlice(&self.payload)?;

        return Ok(())
    }
}

pub struct AgentFuncCall {
    pub appId: u64,

    // <len: u16, bytes: [u8]>
    pub funcName: String,
    
    // <len: u16, bytes: [u8]>
    pub payload: Vec<u8>, 
}

impl MsgIO for AgentFuncCall {
    // data size, used for write check
    fn Size(&self) -> usize {
        let mut size = 0;

        // appId id
        size += 8;

        // 2 is the len of funcName
        size += 2 + self.funcName.len();
        
        // 2 is the len of payload
        size += 2 + self.payload.len();
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let appId = buf.ReadObj::<u64>()?;
        
        let funcNameLen = buf.ReadObj::<u16>()?;
        let funcName = buf.ReadString(funcNameLen as usize)?;
        
        let payloadLen = buf.ReadObj::<u16>()?;
        let payload = buf.ReadVec(payloadLen as usize)?;

        let obj = Self {
            appId: appId,
            funcName: funcName,
            payload: payload,
        };

        return Ok(obj)
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        buf.WriteObj(&self.appId)?;

        buf.WriteObj(&(self.funcName.len() as u16))?;
        buf.WriteSlice(&self.funcName.as_bytes())?;

        buf.WriteObj(&(self.payload.len() as u16))?;
        buf.WriteSlice(&self.payload)?;

        return Ok(());
    }
}

pub struct FuncResp {
    pub errcode: i32,
    pub payload: Vec<u8>,
}

impl FuncResp {
    pub fn NewErr(errcode: i32) -> Self {
        return Self {
            errcode: errcode,
            payload: Vec::new(),
        }
    }
}

impl fmt::Debug for FuncResp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FuncResp")
         .field("errcode", &self.errcode)
         .finish()
    }
}

impl MsgIO for FuncResp {
    // data size, used for write check
    fn Size(&self) -> usize {
        let mut size = 0;

        // errcode id
        size += 4;

        // 2 is the len of payload
        size += 2 + self.payload.len();
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut SocketBufIovs) -> Result<Self> {
        let errcode = buf.ReadObj::<i32>()?;
        let buflen = buf.ReadObj::<u16>()?;
        let payload = buf.ReadVec(buflen as usize)?;

        let obj = Self {
            errcode: errcode,
            payload: payload,
        };

        return Ok(obj)
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut SocketBufIovs) -> Result<()> {
        buf.WriteObj(&self.errcode)?;
        buf.WriteObj(&(self.payload.len() as u16))?;
        buf.WriteSlice(&self.payload)?;

        return Ok(());
    }
}
