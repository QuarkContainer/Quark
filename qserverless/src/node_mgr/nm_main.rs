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
#[allow(non_camel_case_types)]

extern crate reqwest;
#[macro_use]
extern crate log;
extern crate simple_logging;

use once_cell::sync::OnceCell;
use lazy_static::lazy_static;

use nm_store::NodeMgrCache;
use qobjs::common::Result as QResult;

pub mod nm_svc;
pub mod na_client;
pub mod nodemgr;
pub mod types;
pub mod nm_store;
pub mod etcd;

use crate::nm_svc::*;
use crate::etcd::svc_dir::*;

pub static NM_CACHE : OnceCell<NodeMgrCache> = OnceCell::new();
pub const VERSION: &str = "0.1";

lazy_static! {
    pub static ref SVC_DIR: SvcDir = SvcDir::default();
    pub static ref CACHE_OBJ_TYPES: Vec<&'static str> = vec!["pod", "podset", "package"];
}

#[tokio::main]
async fn main() -> QResult<()> {
    log4rs::init_file("logging_config.yaml", Default::default()).unwrap();
    NM_CACHE.set(NodeMgrCache::New().await.unwrap()).unwrap();
    SVC_DIR.write().await.Init("localhost:2379").await?;
    GrpcService().await.unwrap();

    Ok(())
}


