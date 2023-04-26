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

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64};
use qobjs::config::NodeConfiguration;
use uuid::Uuid;
use tokio::sync::{mpsc, Notify};
use tokio::sync::Mutex as TMutex;
use std::sync::atomic::Ordering;

use chrono::prelude::*;
use k8s_openapi::api::core::v1 as k8s;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time};

//use qobjs::runtime_types::QuarkNode;
use qobjs::runtime_types::*;

use qobjs::common::*;

use crate::nm_svc::NodeAgentMsg;
use crate::pod::*;
use crate::NETWORK_PROVIDER;

pub struct NodeAgent {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,
    pub node: QuarkNode,

    pub msgChannTx: mpsc::Sender<NodeAgentMsg>,
    pub pods: TMutex<BTreeMap<String, PodAgent>>,
}

impl NodeAgent {
    pub async fn Send(&self, msg: NodeAgentMsg) -> Result<()> {
        match self.msgChannTx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_) => {
                return Err(Error::CommonError(format!("PodAgent send message fail")));
            }
        }
    }
}


impl NodeAgent {
    pub async fn Stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_one();
    }

    //pub fn BuildFornaxPod(&self, state: PodState, configMap: &k8s::ConfigMap, isDaemon: bool) -> Result<QuarkPod> {   
    //}

}

pub fn InitK8sNode() -> Result<k8s::Node> {
    let hostname = hostname::get()?.to_str().unwrap().to_string();
    let mut node = k8s::Node {
        metadata: ObjectMeta  {
            name: Some(hostname),
            namespace: Some(DefaultFornaxCoreNodeNameSpace.to_string()),
            uid: Some(Uuid::new_v4().to_string()),
            resource_version: Some("0".to_owned()),
            generation: Some(0),
            creation_timestamp: Some(Time(Utc::now())),
            ..Default::default()
        },
        spec:  Some(k8s::NodeSpec::default()),
        status: Some(k8s::NodeStatus {
            capacity: Some(BTreeMap::new()),
            allocatable: Some(BTreeMap::new()),
            phase: Some(NodePending.to_string()),
            conditions: Some(Vec::new()),
            addresses: Some(Vec::new()),
            daemon_endpoints: Some(k8s::NodeDaemonEndpoints::default()),
            node_info: Some(k8s::NodeSystemInfo::default()),
            images: Some(Vec::new()),
            volumes_in_use: Some(Vec::new()),
            volumes_attached: Some(Vec::new()),
            ..Default::default()
        })
    };

    node.status.as_mut().unwrap().conditions.as_mut().unwrap().push(k8s::NodeCondition{
        type_   : NodeReady.to_string(),
        status: ConditionFalse.to_string(),
        reason: Some("Node Initialiazing".to_string()),
        message: Some("Node Initialiazing".to_string()),
        last_heartbeat_time: Some(Time(Utc::now())),
        ..Default::default()
    });

    node.status.as_mut().unwrap().conditions.as_mut().unwrap().push(k8s::NodeCondition{
        type_   : NodeNetworkUnavailable.to_string(),
        status: ConditionTrue.to_string(),
        reason: Some("Node Initialiazing".to_string()),
        message: Some("Node Initialiazing".to_string()),
        last_heartbeat_time: Some(Time(Utc::now())),
        ..Default::default()
    });

    node.status.as_mut().unwrap().addresses = Some(NETWORK_PROVIDER.GetNetAddress());

    return Ok(node)
}

pub struct QuarkNode {
    pub nodeConfig: NodeConfiguration,
    pub node: k8s::Node,
    pub revision: AtomicI64,
    pub pods: TMutex<BTreeMap<String, QuarkPod>>, 
}

impl QuarkNode {
    pub fn NewQuarkNode(nodeConfig: &NodeConfiguration) -> Result<QuarkNode> {
        let k8sNode = InitK8sNode()?;
        
        let quarkNode: QuarkNode = QuarkNode {
            nodeConfig: nodeConfig.clone(),
            node: k8sNode,
            revision: AtomicI64::new(0),
            pods: TMutex::new(BTreeMap::new()),
        };
    
        return Ok(quarkNode);
    }
}

