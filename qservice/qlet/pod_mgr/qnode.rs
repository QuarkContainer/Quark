// Copyright (c) 2023 Quark Container Authors / 2018 The gVisor Authors.
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
use std::ops::Deref;
use std::sync::Mutex;
use std::sync::atomic::AtomicI64;
use std::time::SystemTime;
use qshare::node::Node;
use qshare::node::NodeCondition;
use qshare::node::NodeDef;
use qshare::node::NodeStatus;
use qshare::node::ObjectMeta;
use qshare::node::PodDef;
use uuid::Uuid;

use qshare::config::*;
use qshare::consts::*;
use qshare::common::*;
use super::runtime::network::*;
use super::qpod::*;

lazy_static::lazy_static! {
    pub static ref NETWORK_PROVIDER: LocalNetworkAddressProvider = {
        LocalNetworkAddressProvider::Init()
    };
}

pub fn InitK8sNode(hostname: &str) -> Result<Node> {
    let mut node = Node {
        metadata: ObjectMeta  {
            name: hostname.to_owned(),
            annotations: BTreeMap::new(),
            namespace: DefaultNodeMgrNodeNameSpace.to_string(),
            uid: Uuid::new_v4().to_string(),
            resource_version: "0".to_owned(),
            labels: BTreeMap::new(),
        },
        spec:  NodeDef::default(),
        status: NodeStatus {
            capacity: BTreeMap::new(),
            allocatable: BTreeMap::new(),
            phase: NodePending.to_string(),
            conditions: Vec::new(),
            addresses: Vec::new(),
            //node_info: Some(k8s::NodeSystemInfo::default()),
            images: Vec::new(),
            // volumes_in_use: Some(Vec::new()),
            // volumes_attached: Some(Vec::new()),
            ..Default::default()
        }
    };

    node.status.conditions.push(NodeCondition{
        type_   : NodeReady.to_string(),
        status: ConditionFalse.to_string(),
        reason: Some("Node Initialiazing".to_string()),
        message: Some("Node Initialiazing".to_string()),
        last_heartbeat_time: Some(SystemTime::now()),
        ..Default::default()
    });

    node.status.conditions.push(NodeCondition{
        type_   : NodeNetworkUnavailable.to_string(),
        status: ConditionTrue.to_string(),
        reason: Some("Node Initialiazing".to_string()),
        message: Some("Node Initialiazing".to_string()),
        last_heartbeat_time: Some(SystemTime::now()),
        ..Default::default()
    });

    node.status.addresses = NETWORK_PROVIDER.GetNetAddress();

    return Ok(node)
}


#[derive(Debug)]
pub struct QuarkNodeInner {
    pub nodeConfig: NodeConfiguration,
    pub node: Mutex<Node>,
    pub revision: AtomicI64,
    pub pods: Mutex<BTreeMap<String, QuarkPod>>, 
}

#[derive(Clone, Debug)]
pub struct QuarkNode(pub Arc<QuarkNodeInner>);

impl Deref for QuarkNode {
    type Target = Arc<QuarkNodeInner>;

    fn deref(&self) -> &Arc<QuarkNodeInner> {
        &self.0
    }
}

impl QuarkNode {
    pub fn NewQuarkNode(nodename: &str, nodeConfig: &NodeConfiguration) -> Result<QuarkNode> {
        let k8sNode = InitK8sNode(nodename)?;
        
        let inner = QuarkNodeInner {
            nodeConfig: nodeConfig.clone(),
            node: Mutex::new(k8sNode),
            revision: AtomicI64::new(0),
            pods: Mutex::new(BTreeMap::new()),
        };
    
        return Ok(QuarkNode(Arc::new(inner)));
    }

    pub fn ActivePods(&self) -> Vec<PodDef> {
        let map = self.pods.lock().unwrap();
        let mut pods = Vec::new();
        for p in map.values() {
            let pod = (*p.Pod().read().unwrap()).clone();
            pods.push(pod);
        }

        return pods;
    }

    pub fn NodeName(&self) -> String {
        return self.node.lock().unwrap().NodeId();
    }

}


pub struct ContainerWorldSummary {
    pub runningPods: Vec<QuarkPod>,
    pub terminatedPods: Vec<QuarkPod>,
}

pub fn NodeSpecPodCidrChanged(old: &Node, new: &Node) -> bool {
    match ValidateNodeSpec(new) {
        Err(e) => {
            error!("api node spec is not valid, errors {:?}", e);
            return false;
        }
        Ok(()) => ()
    }

    let oldspec = &old.spec;
    let newspec = &new.spec;

    if oldspec.pod_cidr.len() == 0 
        || oldspec.pod_cidr != newspec.pod_cidr { 
        return true;
    }

    return false;
}

pub fn ValidateNodeSpec(node: &Node) -> Result<()> {
    use ipnetwork::IpNetwork;

    let spec = &node.spec;
    if spec.pod_cidr.len() == 0 {
        return Err(Error::CommonError(format!("api node spec pod cidr is nil")));
    } else {
        let _network = spec.pod_cidr.parse::<IpNetwork>()?;
    }

    return Ok(())
}
