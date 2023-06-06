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

use etcd_client::Error as EtcdError;
use prost::DecodeError;
use prost::EncodeError;
use tokio::task::JoinError;
use tonic::Status as TonicStatus;
use serde_json::Error as SerdeJsonError;
use std::string::FromUtf8Error;
use std::str::Utf8Error;
use std::num::ParseIntError;

use crate::types::WatchEvent;

pub type Result<T> = core::result::Result<T, Error>;

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

#[derive(Debug)]
pub enum Error {
    NotExist,
    ContextCancel,
    EtcdError(EtcdError),
    CommonError(String),
    MinRevsionErr(MinRevsionErr),
    DecodeError(DecodeError),
    EncodeError(EncodeError),
    NewKeyExistsErr(NewKeyExistsErr),
    DeleteRevNotMatchErr(DeleteRevNotMatchErr),
    UpdateRevNotMatchErr(UpdateRevNotMatchErr),
    TonicStatus(TonicStatus),
    StdErr(Box<dyn std::error::Error>),
    TonicTransportErr(tonic::transport::Error),
    SerdeJsonError(SerdeJsonError),
    MpscSendErrWatchEvent(tokio::sync::mpsc::error::SendError<WatchEvent>),
    Utf8Error(Utf8Error),
    FromUtf8Error(FromUtf8Error),
    JoinError(JoinError),
    ReqWestErr(reqwest::Error),
    StdIOErr(std::io::Error),
    AcquireError(tokio::sync::AcquireError),
    ParseIntError(ParseIntError),
    RegexError(regex::Error),
    RocksdbError(rocksdb::Error),
    TokioChannFull,
    TokioChannClose,

    MpscSendFail,

    EPERM(String), // Operation not permitted
    ENOENT(String), // no such entity
    EINVAL(String), // Invalid argument
    ECONNREFUSED(String),

    Timeout,
    IpNetworkError(ipnetwork::IpNetworkError),
    // ErrRequeue may be returned by a PopProcessFunc to safely requeue
    // the current item. The value of Err will be returned from Pop.
    ErrRequeue
}

unsafe impl core::marker::Send for Error {}

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

impl From<Error> for String {
    fn from(item: Error) -> Self {
        return format!("{:?}", item)
    }
}


impl From<rocksdb::Error> for Error {
    fn from(item: rocksdb::Error) -> Self {
        return Self::RocksdbError(item)
    }
}

impl From<ipnetwork::IpNetworkError> for Error {
    fn from(item: ipnetwork::IpNetworkError) -> Self {
        return Self::IpNetworkError(item)
    }
}

impl From<regex::Error> for Error {
    fn from(item: regex::Error) -> Self {
        return Self::RegexError(item)
    }
}

impl From<ParseIntError> for Error {
    fn from(item: ParseIntError) -> Self {
        return Self::ParseIntError(item)
    }
}

impl From<tokio::sync::AcquireError> for Error {
    fn from(item: tokio::sync::AcquireError) -> Self {
        return Self::AcquireError(item)
    }
}

impl From<reqwest::Error> for Error {
    fn from(item: reqwest::Error) -> Self {
        return Self::ReqWestErr(item)
    }
}

impl From<std::io::Error> for Error {
    fn from(item: std::io::Error) -> Self {
        return Self::StdIOErr(item)
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

impl From<JoinError> for Error {
    fn from(item: JoinError) -> Self {
        return Self::JoinError(item)
    }
}

impl From<tokio::sync::mpsc::error::SendError<WatchEvent>> for Error {
    fn from(item: tokio::sync::mpsc::error::SendError<WatchEvent>) -> Self {
        return Self::MpscSendErrWatchEvent(item)
    }
}

impl From<SerdeJsonError> for Error {
    fn from(item: SerdeJsonError) -> Self {
        return Self::SerdeJsonError(item)
    }
}

impl From<EtcdError> for Error {
    fn from(item: EtcdError) -> Self {
        return Self::EtcdError(item)
    }
}

impl From<DecodeError> for Error {
    fn from(item: DecodeError) -> Self {
        return Self::DecodeError(item)
    }
}

impl From<EncodeError> for Error {
    fn from(item: EncodeError) -> Self {
        return Self::EncodeError(item)
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

impl From<Error> for TonicStatus {
    fn from(item: Error) -> Self {
        match item {
            Error::TonicStatus(status) => return status,
            _ => panic!("TonicStatus covert fail")
        }
    }
}