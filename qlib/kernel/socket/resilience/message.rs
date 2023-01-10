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


pub struct UserMsg <'a> {
    buf: &'a [u8] 
}

impl <'a> UserMsg <'a> {
    pub fn Content(&self) -> UserMsgContent {
        let msgType = self.buf[0];

        if msgType == UserMsgType::UserFuncCall as u8 {
            return UserMsgContent::UserFuncCall(UserFuncCall::New(&self.buf[1..]))
        } else {
            return UserMsgContent::UserFuncResp(UserFuncResp::New(&self.buf[1..]))
        }
    }
}

#[repr(u8)]
pub enum UserMsgType {
    UserFuncCall = 1,
    UserFuncResp
}

pub enum UserMsgContent <'a> {
    UserFuncCall(UserFuncCall<'a>),
    UserFuncResp(UserFuncResp<'a>),
}

pub struct UserFuncCall <'a> {
    // userdata: u64,
    // payload: &[u8],
    buf: &'a [u8] 
}

impl <'a> UserFuncCall <'a> {
    pub fn New(buf: &'a [u8]) -> Self {
        return Self {
            buf: buf,
        }
    }

    pub fn UserData(&self) -> u64 {
        let data = unsafe {
            *(&self.buf[0] as * const _ as u64 as * const u64)
        };
        return data;
    }

    pub fn Payload(&self) -> &'a [u8] {
        return &self.buf[8..];
    }
}

pub struct UserFuncResp <'a> {
    // msgId: u64,
    // payload: &[u8],
    buf: &'a [u8] 
}

impl <'a> UserFuncResp <'a> {
    pub fn New(buf: &'a [u8]) -> Self {
        return Self {
            buf: buf,
        }
    }

    pub fn MsgId(&self) -> u64 {
        let data = unsafe {
            *(&self.buf[0] as * const _ as u64 as * const u64)
        };
        return data;
    }

    pub fn Payload(&self) -> &'a [u8] {
        return &self.buf[8..];
    }
}