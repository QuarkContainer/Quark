// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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

#[macro_use]
extern crate scopeguard;

#[macro_use]
extern crate log;

use func_context::FuncCallMgr;
use func_node::FuncNodeMgr;
use func_pod::FuncPodMgr;
use lazy_static::lazy_static;

pub mod func_svc;
pub mod func_conn;
pub mod func_context;
pub mod task_queue;
pub mod scheduler;
pub mod func_pod;
pub mod package;
pub mod func_node;
pub mod message;
pub mod grpc_svc;

use package::PackageMgr;
use qobjs::common::*;

lazy_static! {
    pub static ref PACKAGE_MGR: PackageMgr = {
        PackageMgr::New()
    };

    pub static ref FUNC_POD_MGR: FuncPodMgr = {
        FuncPodMgr::New()
    };

    pub static ref FUNC_NODE_MGR: FuncNodeMgr = {
        FuncNodeMgr::New()
    };

    pub static ref FUNC_CALL_MGR: FuncCallMgr = {
        FuncCallMgr::New()
    };
}

#[tokio::main]
async fn main() -> Result<()> {
    log4rs::init_file("logging_config.yaml", Default::default()).unwrap();
    grpc_svc::GrpcService().await.unwrap();
    Ok(())
}
