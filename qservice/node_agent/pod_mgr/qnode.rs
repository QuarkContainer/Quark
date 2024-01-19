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
use chrono::Utc;
use uuid::Uuid;

use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time};

use qshare::config::*;
use qshare::consts::*;
use qshare::k8s;
use qshare::common::*;
use qshare::k8s_util::K8SUtil;

use super::runtime::network::*;
use super::qpod::*;

lazy_static::lazy_static! {
    pub static ref NETWORK_PROVIDER: LocalNetworkAddressProvider = {
        LocalNetworkAddressProvider::Init()
    };
}

pub fn InitK8sNode(hostname: &str) -> Result<k8s::Node> {
    let mut node = k8s::Node {
        metadata: ObjectMeta  {
            name: Some(hostname.to_owned()),
            annotations: Some(BTreeMap::new()),
            namespace: Some(DefaultNodeMgrNodeNameSpace.to_string()),
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


#[derive(Debug)]
pub struct QuarkNodeInner {
    pub nodeConfig: NodeConfiguration,
    pub node: Mutex<k8s::Node>,
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

    pub fn ActivePods(&self) -> Vec<k8s::Pod> {
        let map = self.pods.lock().unwrap();
        let mut pods = Vec::new();
        for p in map.values() {
            let pod = (*p.Pod().read().unwrap()).clone();
            pods.push(pod);
        }

        return pods;
    }

    pub fn NodeName(&self) -> String {
        return K8SUtil::Id(&self.node.lock().unwrap().metadata);
    }

}


pub struct ContainerWorldSummary {
    pub runningPods: Vec<QuarkPod>,
    pub terminatedPods: Vec<QuarkPod>,
}

pub fn NodeSpecPodCidrChanged(old: &k8s::Node, new: &k8s::Node) -> bool {
    match ValidateNodeSpec(new) {
        Err(e) => {
            error!("api node spec is not valid, errors {:?}", e);
            return false;
        }
        Ok(()) => ()
    }

    let oldspec = old.spec.as_ref().unwrap();
    let newspec = new.spec.as_ref().unwrap();

    if oldspec.pod_cidr.as_ref().unwrap().len() == 0 
    || oldspec.pod_cidr != newspec.pod_cidr 
    || oldspec.pod_cidrs.as_ref().unwrap().len() != newspec.pod_cidrs.as_ref().unwrap().len() {
        return true;
    }

    let mut oldcidrs = oldspec.pod_cidrs.as_ref().unwrap().to_vec();
    let mut newcidrs = newspec.pod_cidrs.as_ref().unwrap().to_vec();

    oldcidrs.sort();
    newcidrs.sort();
    
    for i in 0..oldcidrs.len() {
        if oldcidrs[i] != newcidrs[i] {
            return true;
        }
    }

    return false;
}

pub fn ValidateNodeSpec(node: &k8s::Node) -> Result<()> {
    use ipnetwork::IpNetwork;

    let spec = node.spec.as_ref().unwrap();
    if spec.pod_cidr.is_none() {
        return Err(Error::CommonError(format!("api node spec pod cidr is nil")));
    } else {
        let _network = spec.pod_cidr.as_ref().unwrap().parse::<IpNetwork>()?;
    }

    for cidr in spec.pod_cidrs.as_ref().unwrap() {
        let _network = cidr.parse::<IpNetwork>()?;
    }

    if spec.pod_cidr.is_some() && &spec.pod_cidrs.as_ref().unwrap()[0] != spec.pod_cidr.as_ref().unwrap() {
        return Err(Error::CommonError(format!("node spec podcidrs[0] {} does not match podcidr {}",
            spec.pod_cidrs.as_ref().unwrap()[0], spec.pod_cidr.as_ref().unwrap())));
    }

    return Ok(())
}
