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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use core::ops::Deref;
use std::time::SystemTime;
use async_trait::async_trait;

use qobjs::audit::func_audit::FuncAudit;
use qobjs::k8s;
use qobjs::ObjectMeta;
use qobjs::nm;
use qobjs::runtime_types::DefaultDomainName;
use qobjs::{types::*, selector::Labels};
use qobjs::common::*;
use qobjs::cacher::*;
use qobjs::selection_predicate::*;
use tokio::sync::Notify;

use crate::FUNCAUDIT;
use crate::na_client::NodeAgentClient;

pub fn MetaToDataObject(kind: &str, meta: &ObjectMeta) -> Result<DataObjectInner> {
    let mut labels = BTreeMap::new();
    if let Some(map) = &meta.labels {
        for (k, v) in map {
            labels.insert(k.to_string(), v.to_string());
        }
    }
    
    let mut annotations = BTreeMap::new();
    if let Some(map) = &meta.annotations {
        for (k, v) in map {
            annotations.insert(k.to_string(), v.to_string());
        }
    }

    let resource_version = meta.resource_version.as_deref().unwrap_or("0").to_string();

    let revision = resource_version.parse::<i64>()?;

    let inner = DataObjectInner {
        kind: kind.to_string(),
        namespace: meta.namespace.as_deref().unwrap_or("").to_string(),
        name: meta.name.as_deref().unwrap_or("").to_string(),
        lables: Labels::NewFromMap(labels),
        annotations: Labels::NewFromMap(annotations),
        reversion: revision,
        data: String::new(),
    };

    return Ok(inner);
}

pub fn NodeToDataObject(node: &k8s::Node) -> Result<DataObject> {
    let mut inner = MetaToDataObject(QUARK_NODE, &node.metadata)?;
    inner.data = serde_json::to_string(node)?;
    let dataObj = DataObject(Arc::new(inner));
    return Ok(dataObj)
}

pub fn PodToDataObject(pod: &k8s::Pod) -> Result<DataObject> {
    let mut inner = MetaToDataObject(QUARK_POD, &pod.metadata)?;
    inner.data = serde_json::to_string(pod)?;
    let dataObj = DataObject(Arc::new(inner));
    return Ok(dataObj)
}
#[derive(Debug)]
pub struct NodeMgrCacheInner {
    pub nodes: Option<Cacher>,
    pub pods: Option<Cacher>,
    pub nodePods: BTreeMap<String, BTreeSet<String>>,
    pub nodeAgents: BTreeMap<String, NodeAgentClient>,
    
    // for the node which temp disconnect, when timeout will be removed
    // asume there won't be 2 node has same probation start time
    pub probationNodes: BTreeMap<SystemTime, String>,
    // from nodekey to probation start time
    pub probationNodesMap: BTreeMap<String, SystemTime>,
}

impl NodeMgrCacheInner {
     pub fn RefreshNode(&mut self, node: &k8s::Node, pods: &[k8s::Pod]) -> Result<()> {
        let mut podObjs = Vec::new();
        let nodeObj = NodeToDataObject(node)?;
        let mut set = BTreeSet::new();
        for pod in pods {
            let obj = PodToDataObject(pod)?;
            set.insert(obj.Key());
            podObjs.push(obj);
        }

        let inner = self;
        inner.pods.as_ref().unwrap().Refresh(&podObjs)?;
        inner.nodePods.insert(nodeObj.Key(), set);

        if inner.nodes.as_ref().unwrap().Contains(&nodeObj.Key()) {
            inner.nodes.as_ref().unwrap().ProcessEvent(&WatchEvent {
                type_: EventType::Modified,
                obj: nodeObj,
            })?;
        } else {
            inner.nodes.as_ref().unwrap().ProcessEvent(&WatchEvent {
                type_: EventType::Added,
                obj: nodeObj,
            })?;
        }

        return Ok(());
    }
}

#[derive(Debug, Clone)]
pub struct NodeMgrCache(Arc<RwLock<NodeMgrCacheInner>>);

impl Deref for NodeMgrCache {
    type Target = Arc<RwLock<NodeMgrCacheInner>>;

    fn deref(&self) -> &Arc<RwLock<NodeMgrCacheInner>> {
        &self.0
    }
}

#[async_trait]
impl BackendStore for NodeMgrCache {
    async fn Create(&self, _obj: &DataObject) -> Result<DataObject> {
        unimplemented!();
    }
    
    async fn Update(&self, _expectedRev: i64, _obj: &DataObject) -> Result<DataObject> {
        unimplemented!();
    }

    async fn Delete(&self, _key: &str, _expectedRev: i64) -> Result<i64> {
        unimplemented!();
    }

    async fn Get(&self, _key: &str, _minRevision: i64) -> Result<Option<DataObject>> {
        unimplemented!();
    }

    async fn List(&self, _prefix: &str, _opts: &ListOption) -> Result<DataObjList> {
        unimplemented!();
    }

    fn Register(&self, _cacher: Cacher, _rev: i64, _prefix: String, ready: Arc<Notify>, notify: Arc<Notify>) -> Result<()> {
        
        notify.notify_one();
        ready.notify_one();
        return Ok(())
    }
}

impl NodeMgrCache {
    pub async fn New() -> Result<Self> {
        let inner = NodeMgrCacheInner {
            nodes: None,
            pods: None,
            nodePods: BTreeMap::new(),
            nodeAgents: BTreeMap::new(),
            probationNodes: BTreeMap::new(),
            probationNodesMap: BTreeMap::new(),
        };

        let cache = Self(Arc::new(RwLock::new(inner)));
        let nodesCache = Cacher::New(Arc::new(cache.clone()), QUARK_NODE, 0).await?;
        let podsCache = Cacher::New(Arc::new(cache.clone()), QUARK_POD, 0).await?;

        // init the listRevision as there won't be initial list operation for the nodecache and podcache
        nodesCache.write().unwrap().listRevision = 1;
        podsCache.write().unwrap().listRevision = 1;

        cache.write().unwrap().nodes = Some(nodesCache);
        cache.write().unwrap().pods = Some(podsCache);
        return Ok(cache);
    }

    // workaround scheduler, schedule pod to node one by one
    pub fn SchedulePod(&self, _pod: &k8s::Pod) -> Option<String> {
        let pods = self.GetCacher(QUARK_POD).unwrap();
        let podCount = pods.Count();
        let nodes : Vec<String> = self.read().unwrap().nodeAgents.keys().cloned().collect();
        if nodes.len() == 0 {
            return None;
        }

        return Some(nodes[podCount % nodes.len()].clone());
    }

    pub fn GetCacher(&self, objType: &str) -> Option<Cacher> {
        if objType == QUARK_POD {
            return self.read().unwrap().pods.clone();
        }

        if objType == QUARK_NODE {
            return self.read().unwrap().nodes.clone();
        }

        return None;
    }

    pub fn RegisterNodeAgentClient(&self, naClient: &NodeAgentClient, node: &k8s::Node, pods: &[k8s::Pod]) -> Result<()> {
        let nodeObj = NodeToDataObject(&node)?;
        let nodeKey = nodeObj.Key();

        let mut inner = self.write().unwrap();
        if let Some(time) = inner.probationNodesMap.remove(&nodeKey) {
            inner.probationNodes.remove(&time);
        }

        inner.nodeAgents.insert(nodeKey, naClient.clone());

        inner.RefreshNode(&node, &pods)?;

        return Ok(())
    }


    pub fn ProcessPodEvent(&self, nodeKey: &str, event: &nm::PodEvent) -> Result<()> {
        let k8sPod: k8s::Pod = serde_json::from_str(&event.pod)?;
        let podObj = PodToDataObject(&k8sPod)?;
        let mut inner = self.write().unwrap();

        if event.event_type == nm::EventType::Add as i32 {
            assert!(!inner.nodePods.get(nodeKey).as_deref().unwrap().contains(&podObj.Key()));
            inner.nodePods.get_mut(nodeKey).as_mut().unwrap().insert(podObj.Key());
            inner.pods.as_ref().unwrap().ProcessEvent(&WatchEvent {
                type_: EventType::Added,
                obj: podObj,
            })?;
        } else if event.event_type == nm::EventType::Delete as i32 {
            assert!(inner.nodePods.get(nodeKey).as_deref().unwrap().contains(&podObj.Key()));
            inner.nodePods.get_mut(nodeKey).as_mut().unwrap().remove(&podObj.Key());
            inner.pods.as_ref().unwrap().ProcessEvent(&WatchEvent {
                type_: EventType::Deleted,
                obj: podObj,
            })?;
        } else { // if event.event_type == nm::EventType::Update as i32
            assert!(inner.nodePods.get(nodeKey).as_deref().unwrap().contains(&podObj.Key()));
            inner.pods.as_ref().unwrap().ProcessEvent(&WatchEvent {
                type_: EventType::Modified,
                obj: podObj,
            })?;
        }
        
        return Ok(());
    }

    pub fn ProcessNodeUpdate(&self, nodeKey: &str, update: &nm::NodeUpdate) -> Result<()> {
        assert!(self.read().unwrap().nodes.as_ref().unwrap().Contains(nodeKey), "nodekey is {}", nodeKey);
        
        let node: k8s::Node = serde_json::from_str(&update.node)?;
        let nodeObj: DataObject = NodeToDataObject(&node)?;
        self.read().unwrap().nodes.as_ref().unwrap().ProcessEvent(&WatchEvent {
            type_: EventType::Modified,
            obj: nodeObj,
        })
    }

    pub fn DeleteNode(&self, nodeKey: &str) -> Result<()> {
        let mut inner = self.write().unwrap();
        if let Some(set) = inner.nodePods.remove(nodeKey) {
            for podKey in set {
                let pod = match inner.pods.as_ref().unwrap().GetObject(&podKey) {
                    None => return Err(Error::CommonError(format!("NodeMgrCache::DeleteNode get inconsistent pods set for node {}", nodeKey))),
                    Some(obj) => obj,
                };

                inner.pods.as_ref().unwrap().ProcessEvent(&WatchEvent {
                    type_: EventType::Deleted,
                    obj: pod,
                })?;
            }
        }

        return Ok(());
    }

    pub async fn CreatePod(&self, node: &str, pod: &k8s::Pod, configmap: &k8s::ConfigMap) -> Result<()> {
        let nodeAgentClient = self.read().unwrap().nodeAgents.get(node).cloned();
        match nodeAgentClient {
            None => {
                return Err(Error::CommonError(format!("There is no node named {}", node)));
            }
            Some(client) => {
                return client.CreatePod(pod, configmap).await;
            }
        }
    }

    pub async fn TerminatePod(&self, podId: &str) -> Result<()> {
        let pod = self.read().unwrap().pods.as_ref().unwrap().GetObject(podId).clone();
        match pod {
            None => {
                return Err(Error::CommonError(format!("There is no pod named {}", podId)));
            }
            Some(pod) => {
                let pod : k8s::Pod = serde_json::from_str(&pod.data)?;
                let node = match pod.metadata.annotations.as_ref().unwrap().get(AnnotationNodeMgrNode) {
                    None => return Err(Error::CommonError(format!("pod is not assigned {}", podId))),
                    Some(node) => node.to_string(),
                };

                let nodeAgentClient = self.read().unwrap().nodeAgents.get(&node).cloned();
                match nodeAgentClient {
                    None => {
                        return Err(Error::CommonError(format!("There is no node named {}", node)));
                    }
                    Some(client) => {
                        return client.TerminatePod(podId).await;
                    }
                }
            }
        }
    }

    pub async fn ReadFuncLog(&self, namespace: &str, funcId: &str) -> Result<String> {
        let nodeId = FUNCAUDIT.get().unwrap().GetNode(namespace, funcId).await?;
        let nodeId = format!("{}/{}", DefaultDomainName, nodeId);
        let nodeAgentClient = self.read().unwrap().nodeAgents.get(&nodeId).cloned();
        match nodeAgentClient {
            None => {
                return Err(Error::CommonError(format!("Can't find node with name {}", nodeId)));
            }
            Some(client) => {
                return client.ReadFuncLog(namespace, funcId).await;
            }
        }
    }
}