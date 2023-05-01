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

use std::collections::BTreeMap;

use lazy_static::lazy_static;
use once_cell::sync::OnceCell;

pub mod cri;
pub mod runtime;
pub mod pod;
pub mod container;
pub mod node;
pub mod nm_svc;
pub mod pod_sandbox;
pub mod message;
pub mod node_status;
pub mod cadvisor;

use qobjs::common::Result as QResult;
//use qobjs::config::NodeConfiguration;
//use qobjs::pb_gen::node_mgr_pb::FornaxCoreMessage;
use runtime::image_mgr::ImageMgr;

use qobjs::pb_gen::v1alpha2;
use crate::runtime::runtime::RuntimeMgr;
use crate::runtime::network::*;
use crate::cadvisor::client as CadvisorClient;
use crate::cadvisor::provider::CadvisorInfoProvider;

pub static RUNTIME_MGR: OnceCell<RuntimeMgr> = OnceCell::new();
pub static IMAGE_MGR: OnceCell<ImageMgr> = OnceCell::new();
pub static CADVISOR_PROVIDER: OnceCell<CadvisorInfoProvider> = OnceCell::new();

lazy_static! {
    pub static ref NETWORK_PROVIDER: LocalNetworkAddressProvider = {
        LocalNetworkAddressProvider::Init()
    };

    pub static ref CADVISOR_CLI: CadvisorClient::Client = {
        CadvisorClient::Client::Init()
    };
}

#[tokio::main]
async fn main() -> QResult<()> {
    use qobjs::pb_gen::node_mgr_pb::{self as NmMsg};
    use log::LevelFilter;
    use nm_svc::*;
    use qobjs::pb_gen::node_mgr_pb::*;
    use k8s_openapi::api::core::v1 as k8s;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    //use serde::{Deserialize, Serialize};

    let podstr = r#"
    {
        "apiVersion":"v1",
        "kind":"Pod",
        "metadata":{
            "name":"nginx",
            "namespace": "default"
        },
        "spec":{
            "hostNetwork": true,
            "containers":[
                {
                    "name":"nginx",
                    "image":"nginx:alpine",
                    "ports":[
                        {
                            "containerPort": 80,
                            "hostIP": "192.168.0.22",
                            "hostPort": 88
                        }
                    ]
                }
            ]
        }
    }"#;

    simple_logging::log_to_file("/var/log/quark/service_diretory.log", LevelFilter::Info).unwrap();
    RUNTIME_MGR.set(RuntimeMgr::New(10).await.unwrap()).unwrap();
    IMAGE_MGR.set(ImageMgr::New(v1alpha2::AuthConfig::default()).await.unwrap()).unwrap();
    CADVISOR_PROVIDER.set(CadvisorInfoProvider::New().await.unwrap()).unwrap();
    
    //let client = CriClient::Init().await?;
    //println!("version is {:#?}", client.ListImages(None).await?);

    let config = qobjs::config::NodeConfiguration::Default()?;

    let (mut rx, na) = crate::node::Run(config).await?;

    println!("main 1");
        
    let _msg = rx.recv().await;
    
    let nc = NmMsg::NodeConfiguration {
        cluster_domain: "".to_string(),
        node: serde_json::to_string(&k8s::Node {
            metadata: ObjectMeta {
                annotations: Some(BTreeMap::new()),
                ..Default::default()
            },
            spec: Some(k8s::NodeSpec {
                pod_cidr: Some("123.1.2.0/24".to_string()),
                pod_cidrs: Some(vec!["123.1.2.0/24".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        })?,
    };
    na.Send(NodeAgentMsg::NodeMgrMsg(FornaxCoreMessage{
        message_type: NmMsg::MessageType::NodeConfiguration as i32,
        message_body: Some(NmMsg::fornax_core_message::MessageBody::NodeConfiguration(nc)),
        ..Default::default()
    }))?;

    let pod: k8s::Pod = serde_json::from_str(podstr)?;
    error!("pod is {:#?}", pod);
    tokio::time::sleep(core::time::Duration::from_secs(2)).await;

    let pa = NmMsg::PodCreate {
        pod_identifier: "podId1".to_string(),
        pod: podstr.to_string(),
        config_map: serde_json::to_string(&k8s::ConfigMap::default())?,
    };

    na.Send(NodeAgentMsg::NodeMgrMsg(FornaxCoreMessage{
        message_type: NmMsg::MessageType::PodCreate as i32,
        message_body: Some(NmMsg::fornax_core_message::MessageBody::PodCreate(pa)),
        ..Default::default()
    }))?;

    let client = crate::cri::client::CriClient::Init().await?;
    println!("pods is {:#?}", client.ListPodSandbox(None).await?);

    println!("main 2");
    loop {
        let msg = rx.recv().await;
        error!("final msg is {:#?}", msg);
    }

    //return Ok(())
}

