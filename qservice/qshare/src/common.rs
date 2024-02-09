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

use serde_json::Error as SerdeJsonError;
use tonic::Status as TonicStatus;
use std::num::ParseIntError;
use std::string::FromUtf8Error;
use std::str::Utf8Error;
use prost::EncodeError;
use prost::DecodeError;
use etcd_client::Error as EtcdError;

#[derive(Debug)]
pub enum Error {
    None,
    SysError(i32),
    SocketClose,
    NotExist(String),
    Exist(String),
    MpscSendFull(String),
    CommonError(String),
    StdIOErr(std::io::Error),
    SerdeJsonError(SerdeJsonError),
    ReqWestErr(reqwest::Error),
    TonicStatus(TonicStatus),
    StdErr(Box<dyn std::error::Error>),
    TonicTransportErr(tonic::transport::Error),
    ParseIntError(ParseIntError),
    RegexError(regex::Error),
    Utf8Error(Utf8Error),
    FromUtf8Error(FromUtf8Error),
    AcquireError(tokio::sync::AcquireError),
    TokioChannFull,
    TokioChannClose,
    IpNetworkError(ipnetwork::IpNetworkError),
    EncodeError(EncodeError),
    DecodeError(DecodeError),
    Timeout,
    MinRevsionErr(MinRevsionErr),
    NewKeyExistsErr(NewKeyExistsErr),
    DeleteRevNotMatchErr(DeleteRevNotMatchErr),
    UpdateRevNotMatchErr(UpdateRevNotMatchErr),
    EtcdError(EtcdError),
    ContextCancel,

}

unsafe impl core::marker::Send for Error {}

impl From<EtcdError> for Error {
    fn from(item: EtcdError) -> Self {
        return Self::EtcdError(item)
    }
}

impl From<EncodeError> for Error {
    fn from(item: EncodeError) -> Self {
        return Self::EncodeError(item)
    }
}

impl From<DecodeError> for Error {
    fn from(item: DecodeError) -> Self {
        return Self::DecodeError(item)
    }
}

impl From<ipnetwork::IpNetworkError> for Error {
    fn from(item: ipnetwork::IpNetworkError) -> Self {
        return Self::IpNetworkError(item)
    }
}

impl From<tokio::sync::AcquireError> for Error {
    fn from(item: tokio::sync::AcquireError) -> Self {
        return Self::AcquireError(item)
    }
}

impl From<Utf8Error> for Error {
    fn from(item: Utf8Error) -> Self {
        return Self::Utf8Error(item)
    }
}

impl From<FromUtf8Error> for Error {
    fn from(item: FromUtf8Error) -> Self {
        return Self::FromUtf8Error(item)
    }
}

impl From<regex::Error> for Error {
    fn from(item: regex::Error) -> Self {
        return Self::RegexError(item)
    }
}

impl From<std::io::Error> for Error {
    fn from(item: std::io::Error) -> Self {
        return Self::StdIOErr(item)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

impl From<SerdeJsonError> for Error {
    fn from(item: SerdeJsonError) -> Self {
        return Self::SerdeJsonError(item)
    }
}

impl From<reqwest::Error> for Error {
    fn from(item: reqwest::Error) -> Self {
        return Self::ReqWestErr(item)
    }
}

impl From<TonicStatus> for Error {
    fn from(item: TonicStatus) -> Self {
        return Self::TonicStatus(item)
    }
}

impl From<Box<dyn std::error::Error>> for Error {
    fn from(item: Box<dyn std::error::Error>) -> Self {
        return Self::StdErr(item)
    }
}

impl From<tonic::transport::Error> for Error {
    fn from(item: tonic::transport::Error) -> Self {
        return Self::TonicTransportErr(item)
    }
}

impl From<ParseIntError> for Error {
    fn from(item: ParseIntError) -> Self {
        return Self::ParseIntError(item)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IpAddress(pub u32);

impl IpAddress {
    pub fn AsBytes(&self) -> [u8; 4] {
        let mut bytes = [0u8; 4];
        bytes[0] = (self.0 >> 24) as u8;
        bytes[1] = (self.0 >> 16) as u8;
        bytes[2] = (self.0 >> 8) as u8;
        bytes[3] = (self.0 >> 0) as u8;
        return bytes;
    }
}

#[derive(Debug)]
pub struct MinRevsionErr {
    pub minRevision: i64,
    pub actualRevision: i64,
}

#[derive(Debug)]
pub struct NewKeyExistsErr {
    pub key: String,
    pub rv: i64,
}

#[derive(Debug)]
pub struct DeleteRevNotMatchErr {
    pub expectRv: i64,
    pub actualRv: i64,
}

#[derive(Debug)]
pub struct UpdateRevNotMatchErr {
    pub expectRv: i64,
    pub actualRv: i64,
}


impl Error {
    pub fn NewMinRevsionErr(minRevision: i64, actualRevision: i64) -> Self {
        return Self::MinRevsionErr(MinRevsionErr { 
            minRevision: minRevision, 
            actualRevision: actualRevision 
        })
    }

    pub fn NewNewKeyExistsErr(key: String, rv: i64) -> Self {
        return Self::NewKeyExistsErr(NewKeyExistsErr{
            key: key,
            rv: rv,
        });
    }

    pub fn NewDeleteRevNotMatchErr(expectRv: i64, actualRv: i64) -> Self {
        return Self::DeleteRevNotMatchErr(DeleteRevNotMatchErr {
            expectRv: expectRv,
            actualRv: actualRv
        })
    }

    pub fn NewUpdateRevNotMatchErr(expectRv: i64, actualRv: i64) -> Self {
        return Self::UpdateRevNotMatchErr(UpdateRevNotMatchErr {
            expectRv: expectRv,
            actualRv: actualRv
        })
    }
}