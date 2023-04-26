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

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(deprecated)]

#[macro_use]
extern crate log;
extern crate simple_logging;

use lazy_static::lazy_static;

pub mod cri;
pub mod runtime;
pub mod pod;
pub mod container;
pub mod node;
pub mod nm_svc;
pub mod pod_sandbox;

use qobjs::common::Result as QResult;
use runtime::image_mgr::ImageMgr;

use qobjs::pb_gen::v1alpha2;
use crate::cri::client::*;
use crate::runtime::runtime::RuntimeMgr;
use crate::runtime::network::*;

lazy_static! {
    pub static ref RUNTIME_MGR: RuntimeMgr = {
        tokio::runtime::Runtime::new().unwrap().block_on( async {
            RuntimeMgr::New(10).await.unwrap()
        })
    };

    pub static ref IMAGE_MGR: ImageMgr = {
        tokio::runtime::Runtime::new().unwrap().block_on( async {
            ImageMgr::New(v1alpha2::AuthConfig::default()).await.unwrap()
        })
    };

    pub static ref NETWORK_PROVIDER: LocalNetworkAddressProvider = {
        LocalNetworkAddressProvider::Init()
    };
}

#[tokio::main]
async fn main() -> QResult<()> {
    use log::LevelFilter;

    simple_logging::log_to_file("/var/log/quark/service_diretory.log", LevelFilter::Info).unwrap();
    

    let client = CriClient::Init().await?;
    println!("version is {:#?}", client.ListImages(None).await?);

    return Ok(())
}

