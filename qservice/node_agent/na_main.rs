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
#![feature(ip_bits)]
#![feature(unix_socket_ancillary_data)]

#[macro_use]
extern crate log;

#[macro_use]
extern crate scopeguard;

extern crate simple_logging;

mod tsot;
mod pod_mgr;

use crate::pod_mgr::cadvisor::client as CadvisorClient;
use crate::pod_mgr::podMgr::PodMgrSvc;
use crate::tsot::tsot_svc::TsotCniSvc;

use qshare::common::*;

lazy_static::lazy_static! {
    pub static ref CADVISOR_CLI: CadvisorClient::Client = {
        CadvisorClient::Client::Init()
    };
}

#[tokio::main]
async fn main() -> Result<()> {
    defer!(error!("node_agent finish"));
    log4rs::init_file("/etc/quark/na_logging_config.yaml", Default::default()).unwrap();

    let podMgrFuture = PodMgrSvc();
    let tostSvcFuture = TsotCniSvc();
    tokio::select! {
        _ = podMgrFuture => (),
        _ = tostSvcFuture => ()
    }
    
    return Ok(())
}