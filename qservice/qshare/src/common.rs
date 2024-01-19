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

#[derive(Debug)]
pub enum Error {
    None,
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
}

unsafe impl core::marker::Send for Error {}

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
