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
#![feature(map_first_last)]

#[macro_use]
extern crate log;
extern crate simple_logging;

use std::collections::BTreeMap;
use std::sync::Arc;

use blobstore::blob_client::BlobSvcClientMgr;
use func_agent::func_agent::{FuncAgent, FuncAgentGrpcService};
use func_agent::funcsvc_client::FuncSvcClientMgr;
use funcdir_mgr::FuncDirMgr;
use lazy_static::lazy_static;
use once_cell::sync::OnceCell;

pub mod cri;
pub mod runtime;
pub mod pod;
pub mod container;
pub mod node;
pub mod nm_svc;
pub mod pod_sandbox;
//pub mod message;
pub mod node_status;
pub mod cadvisor;
pub mod store;
pub mod nodeagent_server;
pub mod func_agent;
pub mod blobstore;
pub mod funcdir_mgr;

use qobjs::common::Result as QResult;
use qobjs::config::{NodeAgentConfig, SystemConfig, SYSTEM_CONFIGS};
use qobjs::types::*;
//use qobjs::config::NodeConfiguration;
//use qobjs::nm::NodeAgentMessage;
use runtime::image_mgr::ImageMgr;

use qobjs::crictl;
use store::nodeagent_store::NodeAgentStore;
use tokio::sync::Notify;
use crate::nodeagent_server::NodeAgentServerMgr;
use crate::runtime::runtime::RuntimeMgr;
use crate::runtime::network::*;
use crate::cadvisor::client as CadvisorClient;
use crate::cadvisor::provider::CadvisorInfoProvider;
use crate::blobstore::blob_svc::BlobServiceGrpcService;

pub static RUNTIME_MGR: OnceCell<RuntimeMgr> = OnceCell::new();
pub static IMAGE_MGR: OnceCell<ImageMgr> = OnceCell::new();
pub static CADVISOR_PROVIDER: OnceCell<CadvisorInfoProvider> = OnceCell::new();
pub static NODEAGENT_STORE: OnceCell<NodeAgentStore> = OnceCell::new();
pub static FUNC_SVC_CLIENT: OnceCell<FuncSvcClientMgr> = OnceCell::new();
pub static NODEAGENT_CONFIG1: OnceCell<NodeAgentConfig> = OnceCell::new();
pub static FUNCDIR_MGR: OnceCell<FuncDirMgr> = OnceCell::new();

lazy_static! {
    pub static ref NETWORK_PROVIDER: LocalNetworkAddressProvider = {
        LocalNetworkAddressProvider::Init()
    };

    pub static ref CADVISOR_CLI: CadvisorClient::Client = {
        CadvisorClient::Client::Init()
    };

    pub static ref BLOB_SVC_CLIENT_MGR : BlobSvcClientMgr = {
        BlobSvcClientMgr::default()
    };

    pub static ref NODEAGENT_CONFIG: NodeAgentConfig = {
        let systemConfigs: BTreeMap<String, SystemConfig> = serde_json::from_str(SYSTEM_CONFIGS).unwrap();

        let configName = ConfigName();

        let systemConfig = match systemConfigs.get(&configName) {
            None => panic!("there is no system config named {}", configName),
            Some(c) => c,
        };
    
        systemConfig.nodeAgentConfig.clone()
    };

    pub static ref NODE_READY_NOTIFY : Arc<Notify> = Arc::new(Notify::new());
}

pub fn ConfigName() -> String {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        args[1].clone()
    } else {
        "product".to_string()
    }
}

#[tokio::main]
async fn main() -> QResult<()> {
    log4rs::init_file("na_logging_config.yaml", Default::default()).unwrap();

    info!("NodeAgent start with config name {:?}", &ConfigName());
    let blobSvcAddr = &NODEAGENT_CONFIG.BlobSvcAddr();
     
    let funcAgentSvc = FuncAgentSvc();
    let nodeAgentSvc = NodeAgentSvc();
    let blobSvc = BlobServiceGrpcService(&blobSvcAddr);
    tokio::select! {
        _ = funcAgentSvc => (),
        _ = nodeAgentSvc => (),
        _ = blobSvc => ()
    }

    return Ok(())
}

pub async fn FuncAgentSvc() -> QResult<()> {
    NODE_READY_NOTIFY.notified().await;
    let blobSvcAddr = &NODEAGENT_CONFIG.BlobSvcAddr();
    let funcSvcAddr = &NODEAGENT_CONFIG.FuncSvcAddr();
    let funcAgent = FuncAgent::New(&NODEAGENT_CONFIG.NodeName(), blobSvcAddr);
    FUNC_SVC_CLIENT.set(FuncSvcClientMgr::New(funcSvcAddr, &funcAgent)).unwrap();
    FuncAgentGrpcService(&funcAgent).await?;
    return Ok(());
}

pub async fn NodeAgentSvc() -> QResult<()> {
    CADVISOR_PROVIDER.set(CadvisorInfoProvider::New().await.unwrap()).unwrap();
    RUNTIME_MGR.set(RuntimeMgr::New(10).await.unwrap()).unwrap();
    IMAGE_MGR.set(ImageMgr::New(crictl::AuthConfig::default()).await.unwrap()).unwrap();
    
    //let client = crate::cri::client::CriClient::Init().await?;
    
    let nodeAgentStore= NodeAgentStore::New()?;
    NODEAGENT_STORE.set(nodeAgentStore).unwrap();
    
   
    let config = qobjs::config::NodeConfiguration::Default()?;
    
    let funcDirRoot = format!("{}/{}", &config.RootPath, "func");
    FUNCDIR_MGR.set(FuncDirMgr::New(&funcDirRoot, OBJECTDB_ADDR).await?).unwrap();
    
    let na = crate::node::Run(&NODEAGENT_CONFIG.NodeName(), config).await?;
    
    
    let nodeMgrAddrs = NODEAGENT_CONFIG.nodeMgrAddrs();
    let nodeAgentSrvMgr = NodeAgentServerMgr::New(nodeMgrAddrs);
    
    nodeAgentSrvMgr.Process(&na).await.unwrap();
    return Ok(())
}
