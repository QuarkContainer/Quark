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

use tonic::transport::Server;
use qobjs::common::Result as QResult;
use qobjs::pb_gen::node_mgr_pb;

pub mod nm_svc;
pub mod node_agent;
pub mod nodemgr;
pub mod types;

use crate::nm_svc::*;


#[tokio::main]
async fn main() -> QResult<()> {
    use log::LevelFilter;
    simple_logging::log_to_file("/var/log/quark/service_diretory.log", LevelFilter::Info).unwrap();
    
    /*
    //cadvisor::client::Client::Test().await?;
    let client = cadvisor::client::Client::Init();
    //println!("machine is {:#?}", client.MachineInfo().await?);
    //println!("versioninfo is {:#?}", client.VersionInfo().await?);
    println!("versioninfo is {:#?}", client.GetInfo().await?);
*/
    let inner = NodeMgrSvc::New();
    let svc = node_mgr_pb::node_agent_service_server::NodeAgentServiceServer::new(inner);
    Server::builder().add_service(svc).serve("127.0.0.1:8888".parse().unwrap()).await?;

    Ok(())
}


