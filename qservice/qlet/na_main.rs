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

use std::sync::Arc;

use crate::pod_mgr::cadvisor::client as CadvisorClient;
use crate::pod_mgr::podMgr::PodMgrSvc;
use crate::tsot::tsot_svc::TsotSvc;

use pod_mgr::node_mgr::NodeMgr;
use qshare::common::*;
use qshare::metastore::informer_factory::InformerFactory;
use qshare::metastore::selection_predicate::ListOption;
use qshare::qlet_config::QletConfig;
use tokio::sync::Notify;

lazy_static::lazy_static! {
    pub static ref CADVISOR_CLI: CadvisorClient::Client = {
        CadvisorClient::Client::Init()
    };

    #[derive(Debug)]
    pub static ref QLET_CONFIG: QletConfig = {
        let args : Vec<String> = std::env::args().collect();
        if args.len() != 2 {
            QletConfig {
                nodeName: "node1".to_string(),
                etcdAddresses: vec![
                    "127.0.0.1:2380".to_string(),
                ],
                nodeIp: "127.0.0.1".to_string(),
                podMgrPort: 8888,
                tsotCniPort: 1234,
                tsotSvcPort: 1235,
                stateSvcPort: 1236,
                cidr: "10.1.1.0/8".to_string(),
                stateSvcAddr: "127.0.0.1:8890".to_string(),
                singleNodeModel: true
            }
        } else {
            let configFilePath = &args[1];
            let config = QletConfig::Load(configFilePath).expect(&format!("can't load config from {}", configFilePath));
            config
        }
    };

    pub static ref NODE_MGR: Arc<NodeMgr> = Arc::new(NodeMgr::default());
}

#[tokio::main]
async fn main() -> Result<()> {
    defer!(error!("qlet finish"));
    log4rs::init_file("/etc/quark/na_logging_config.yaml", Default::default()).unwrap();

    error!("config is {:#?}", &QLET_CONFIG.clone());

    if !QLET_CONFIG.singleNodeModel {
        let stateSvcAddr = &format!("http://{}", QLET_CONFIG.stateSvcAddr);
        let factory = InformerFactory::New(stateSvcAddr, "").await.unwrap();
        factory.AddInformer("node_info", &ListOption::default()).await.unwrap();
        let informer = factory.GetInformer("node_info").await.unwrap();
        let _id1 = informer.AddEventHandler(NODE_MGR.clone()).await.unwrap();
        tokio::spawn(async move {
            let notify = Arc::new(Notify::new());

            // todo: handle statesvc crash
            informer.Process(notify).await.ok();
        });
    }

    let podMgrFuture = PodMgrSvc();
    let tostSvcFuture = TsotSvc();
    tokio::select! {
        res = podMgrFuture => {
            error!("podMgrFuture res is {:?}", res);
        }
        res = tostSvcFuture =>  {
            error!("tostSvcFuturer res is {:?}", res);
        }
    }
    
    return Ok(())
}