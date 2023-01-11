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
/*
impl RingBufIO for UserMsg {
    // data size, used for write check
    fn Size(&self) -> usize {
        match self {
            UserMsg::UserFuncCall(msg) => {
                return 4 + 1 + msg.Size()
            }
            UserMsg::UserFuncResp(msg) => {
                return 4 + 1 + msg.Size()
            }
        }
    }

    // read obj, return <Obj, whether trigger>
    fn Read(buf: &mut ByteStreamIntern) -> Result<(Self, bool)> {
        let (userData , trigger) = buf.readObj::<u64>()?;

        let (namelen, _) = buf.readObj::<u16>()?;
        let (name, _)= buf.readString(namelen as usize)?;

        let (buflen , _) = buf.readObj::<u16>()?;
        let (payload, _) = buf.readVec(buflen as usize)?;

        let obj = Self {
            userdata: userData,
            funcName: name,
            payload: payload,
        };

        return Ok((obj, trigger))
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, buf: &mut ByteStreamIntern) -> Result<bool> {
        let size = self.Size();
        let bufSize = buf.AvailableSpace();
        if size < bufSize {
            return Err(Error::NoEnoughSpace);
        }

        buf.writeObj(&self.userdata)?;
        buf.writeObj(&(self.funcName.len() as u16))?;
        buf.writeSlice(self.funcName.as_bytes())?;
        buf.writeObj(&(self.payload.len() as u16))?;
        return buf.writeSlice(&self.payload);
    }
}
 */
pub struct UserFuncCall {
    userdata: u64,
    // <len: u16, bytes: [u8]>
    funcName: String,
    
    // <len: u16, bytes: [u8]>
    payload: Vec<u8>, 
}

impl RingBufIO for UserFuncCall {
    // data size, used for write check
    fn Size(&self) -> usize {
        let mut size = 8;
        size += 2 + self.funcName.len();
        size += 2 + self.payload.len();
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(bs: &mut ByteStreamIntern) -> Result<(Self, bool)> {
        bs.PrepareDataIovs();
        let buf = &mut bs.dataIovs;

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

        let trigger = bs.Consume(obj.Size());

        return Ok((obj, trigger))
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, bs: &mut ByteStreamIntern) -> Result<bool> {
        bs.PrepareSpaceIovs();
        let buf = &mut bs.spaceiovs;

        let size = self.Size();
        let bufSize = buf.Count();
        if size < bufSize {
            return Err(Error::NoEnoughSpace);
        }

        buf.WriteObj(&self.userdata)?;
        buf.WriteObj(&(self.funcName.len() as u16))?;
        buf.WriteSlice(self.funcName.as_bytes())?;
        buf.WriteObj(&(self.payload.len() as u16))?;
        buf.WriteSlice(&self.payload)?;

        let trigger = bs.Produce(size);
        return Ok(trigger)
    }
}


pub struct UserFuncResp {
    userdata: u64,
    payload: Vec<u8>,
}

impl RingBufIO for UserFuncResp {
    // data size, used for write check
    fn Size(&self) -> usize {
        let mut size = 8;
        size += 2 + self.payload.len();
        return size
    }

    // read obj, return <Obj, whether trigger>
    fn Read(bs: &mut ByteStreamIntern) -> Result<(Self, bool)> {
        bs.PrepareDataIovs();
        let buf = &mut bs.dataIovs;

        let userData = buf.ReadObj::<u64>()?;
        let buflen = buf.ReadObj::<u16>()?;
        let payload = buf.ReadVec(buflen as usize)?;

        let obj = Self {
            userdata: userData,
            payload: payload,
        };

        let trigger = bs.Consume(obj.Size());

        return Ok((obj, trigger))
    }
    
    // write obj, return <whether trigger>
    fn Write(&self, bs: &mut ByteStreamIntern) -> Result<bool> {
        bs.PrepareSpaceIovs();
        let buf = &mut bs.spaceiovs;


        let size = self.Size();
        let bufSize = buf.Count();
        if size < bufSize {
            return Err(Error::NoEnoughSpace);
        }

        buf.WriteObj(&self.userdata)?;
        buf.WriteObj(&(self.payload.len() as u16))?;
        buf.WriteSlice(&self.payload)?;

        return Ok(bs.Produce(size));
    }
}
