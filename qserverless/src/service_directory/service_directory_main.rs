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

//#[macro_use]
extern crate lazy_static;
extern crate serde_derive;
#[macro_use]
extern crate log;
extern crate simple_logging;
#[macro_use]
extern crate scopeguard;

pub mod etcd_store;
pub mod watch;
pub mod etcd_client;
pub mod cache;
pub mod svc_dir;
pub mod grpc_srv;

use qobjs::selection_predicate::ListOption;
use qobjs::common::Result as QResult;
use crate::grpc_srv::*;

pub const KEY_PREFIX : &str = "Quark";

#[tokio::main]
async fn main() -> QResult<()> {
    use log::LevelFilter;

    simple_logging::log_to_file("/var/log/quark/service_diretory.log", LevelFilter::Info).unwrap();
    gRpcServer().await?;
    Ok(())
}

