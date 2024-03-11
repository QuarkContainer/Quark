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
// limitations under

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(deprecated)]

#[macro_use]
extern crate log;
extern crate simple_logging;

pub mod http_gateway;
pub mod namespace_mgr;
pub mod func_mgr;

use namespace_mgr::{NamespaceMgr, NamespaceStore};
use once_cell::sync::OnceCell;

use qshare::common::*;
use http_gateway::*;

pub static NAMESPACE_MGR: OnceCell<NamespaceMgr> = OnceCell::new();
pub static NAMESPACE_STORE: OnceCell<NamespaceStore> = OnceCell::new();

#[tokio::main]
async fn main() -> Result<()> {
    log4rs::init_file("/etc/quark/gateway_logging_config.yaml", Default::default()).unwrap();
    
    NAMESPACE_MGR.set(NamespaceMgr::New(vec!["http://127.0.0.1:8890".to_owned()]).await?).unwrap();
    NAMESPACE_STORE.set(NamespaceStore::New(&vec!["http://127.0.0.1:2379".to_owned()]).await?).unwrap();
    
    error!("gateway ...");
    let gateway = HttpGateway{};
    gateway.HttpServe().await?;
    // StateService().await.unwrap();

    return Ok(())
}