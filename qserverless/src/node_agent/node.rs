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
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, AtomicI64};
use std::time::{SystemTime, Duration};
use qobjs::config::NodeConfiguration;
use uuid::Uuid;
use tokio::sync::{mpsc, Notify};
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use core::ops::Deref;
use tokio::time;

use chrono::{prelude::*};
use qobjs::k8s;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time};
use qobjs::k8s_util::K8SUtil;

//use qobjs::runtime_types::QuarkNode;
use qobjs::runtime_types::*;
use qobjs::common::*;
use qobjs::nm::{self as NmMsg};
use qobjs::types::*;

use crate::nm_svc::{NodeAgentMsg, PodCreate};
use crate::node_status::{SetNodeStatus, IsNodeStatusReady};
use crate::{pod::*, NODEAGENT_STORE, RUNTIME_MGR};
use crate::NETWORK_PROVIDER;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeAgentState {
    Initializing,
    Initialized,
    Registering,
    Registered,
    Ready,
}

pub struct NodeAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub state: Mutex<NodeAgentState>,

    pub node: QuarkNode,
    pub agentChann: mpsc::Sender<NodeAgentMsg>,
    pub agentRx: Mutex<Option<mpsc::Receiver<NodeAgentMsg>>>,
    
    pub pods: Mutex<BTreeMap<String, PodAgent>>,
}

#[derive(Clone)]
pub struct NodeAgent(pub Arc<NodeAgentInner>);

impl Deref for NodeAgent {
    type Target = Arc<NodeAgentInner>;

    fn deref(&self) -> &Arc<NodeAgentInner> {
        &self.0
    }
}

impl NodeAgent {
    pub fn Send(&self, msg: NodeAgentMsg) -> Result<()> {
        match self.agentChann.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_) => {
                return Err(Error::CommonError(format!("PodAgent send message fail")));
            }
        }
    }
}

impl NodeAgent {
    pub fn New(node: &QuarkNode) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<NodeAgentMsg>(30);
        
        let inner = NodeAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            state: Mutex::new(NodeAgentState::Initializing),
            node: node.clone(),
            agentChann: tx,
            agentRx: Mutex::new(Some(rx)),
            pods: Mutex::new(BTreeMap::new()),
        };

        let agent = NodeAgent(Arc::new(inner));
        return Ok(agent);
    }

    // clean up all qserverless created pods, used when nodeagent restarted
    // work around when the nodeagent persistent store is not ready
    pub async fn CleanPods() -> Result<()> {
        let pods = RUNTIME_MGR.get().unwrap().GetPods().await?;
        for p in pods {
            let pod = p.sandbox.unwrap();

            info!("removing pod {} annotations {:#?}", &pod.id, &pod.annotations);
            // skip call sandbox which not created by qserverless
            if !pod.annotations.contains_key(AnnotationNodeMgrNode) {
                continue;
            }

            let containers = RUNTIME_MGR.get().unwrap().GetPodContainers(&pod.id).await?;
            let mut containerIds = Vec::new();
            for container in containers {
                containerIds.push(container.id);
            }
            RUNTIME_MGR.get().unwrap().TerminatePod(&pod.id, containerIds).await?;        
        }

        return Ok(())
    }

    pub async fn Start(&self) -> Result<()> {
        let rx = self.agentRx.lock().unwrap().take().unwrap();
        let clone = self.clone();
        *self.state.lock().unwrap() = NodeAgentState::Registering;
        SetNodeStatus(&self.node).await?;
        NODEAGENT_STORE.get().unwrap().CreateNode(&*self.node.node.lock().unwrap())?;

        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });
        
        return Ok(())
    }

    pub async fn Stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_one();
    }

    pub fn State(&self) -> NodeAgentState {
        return *self.state.lock().unwrap();
    }

    pub fn IncrNodeRevision(&self) -> i64 {
        let revision = self.node.revision.fetch_add(1, Ordering::SeqCst);
        self.node.node.lock().unwrap().metadata.resource_version = Some(format!("{}", revision));
        self.node.node.lock().unwrap().metadata.annotations.as_mut().unwrap().insert(
            AnnotationNodeMgrNodeRevision.to_string(), 
            format!("{}", revision)
        );
        return revision;
    }

    pub async fn Process(&self, mut rx: mpsc::Receiver<NodeAgentMsg>) -> Result<()> {
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                _ = interval.tick() => {
                    if self.State() == NodeAgentState::Registered {
                        if IsNodeStatusReady(&self.node) {
                            info!("Node is ready");
                            *self.state.lock().unwrap() = NodeAgentState::Ready;
                            SetNodeStatus(&self.node).await?;
                            self.node.node.lock().unwrap().status.as_mut().unwrap().phase = Some(format!("{}", NodeRunning));
                            NODEAGENT_STORE.get().unwrap().UpdateNode(&self.node)?;
                        }
                    } else {
                        SetNodeStatus(&self.node).await?;
                        NODEAGENT_STORE.get().unwrap().UpdateNode(&self.node)?;
                    }
                }
                msg = rx.recv() => {
                    if let Some(msg) = msg {
                        self.NodeHandler(&msg).await?;
                    } else {
                        break;
                    }
                }
            }
        }
        
        return Ok(())
    }

    pub async fn NodeHandler(&self, msg: &NodeAgentMsg) -> Result<()> {
        match msg {
            NodeAgentMsg::NodeMgrMsg(msg) => {
                return self.ProcessNodeMgrMsg(msg).await;
            }
            NodeAgentMsg::PodStatusChange(msg) => {
                let qpod = msg.pod.clone();
                if qpod.PodState() == PodState::Cleanup {
                    self.CleanupPodStoreAndAgent(&qpod).await?;
                    NODEAGENT_STORE.get().unwrap().DeletePod(&qpod)?;
                    qpod.SetPodState(PodState::Deleted);
                } else if qpod.PodState() != PodState::Deleted {
                    NODEAGENT_STORE.get().unwrap().UpdatePod(&qpod)?;
                }

                
            }
            NodeAgentMsg::NodeUpdate => {
                SetNodeStatus(&self.node).await?;
                //NODEAGENT_STORE.get().unwrap().UpdateNode(&*self.node.node.lock().unwrap())?;
            }
            _ => {
                error!("NodeHandler Received unknown message {:?}", msg);
            }
        }

        return Ok(())
    }

    pub async fn ProcessNodeMgrMsg(&self, msg: &NmMsg::NodeAgentMessage) -> Result<()> {
        let body = msg.message_body.as_ref().unwrap();
        match body {
            NmMsg::node_agent_message::MessageBody::NodeConfiguration(msg) => {
                return self.OnNodeConfigurationCommand(&msg).await;
            }
            NmMsg::node_agent_message::MessageBody::NodeFullSync(msg) => {
                return self.OnNodeFullSyncCommand(&msg);
            }
            NmMsg::node_agent_message::MessageBody::PodCreate(msg) => {
                return self.OnPodCreateCmd(&msg).await;
            }
            NmMsg::node_agent_message::MessageBody::PodTerminate(msg) => {
                return self.OnPodTerminateCommand(&msg);
            }
            NmMsg::node_agent_message::MessageBody::PodState(_) => {
                //self.Notify(NodeAgentMsg::NodeMgrMsg(msg.clone()));
                return Ok(())
            }
            _ => {
                warn!("NodeAgentMessage {:?} is not handled by actor", msg);
                return Ok(())
            }
        }
    } 

    pub async fn CleanupPodStoreAndAgent(&self, pod: &QuarkPod) -> Result<()> {
        info!("Cleanup pod actor and store pod {} state {:?}", pod.PodId(), pod.PodState());
        let podId = pod.lock().unwrap().id.clone();
        let agent = self.pods.lock().unwrap().remove(&podId);
        match agent {
            Some(pa) => {
                pa.Stop().await?;
            }
            None => ()
        }

        self.node.pods.lock().unwrap().remove(&podId);
        return Ok(())
    }

    pub fn OnNodeFullSyncCommand(&self, _msg: &NmMsg::NodeFullSync) -> Result<()> {
        //let msg = BuildNodeAgentNodeState(&self.node, self.node.revision.load(Ordering::SeqCst))?;
        //self.Notify(NodeAgentMsg::NodeMgrMsg(msg));
        return Ok(())
    }

    pub async fn NodeConfigure(&self, node: k8s::Node) -> Result<()> {
        if *self.state.lock().unwrap() == NodeAgentState::Registering {
            //return Err(Error::CommonError(format!("node is not in registering state, it does not expect configuration change after registering")));
         

            info!("received node spec from nodemgr: {:?}", &node);
            ValidateNodeSpec(&node)?;

            self.node.node.lock().unwrap().spec = node.spec.clone();
            if NodeSpecPodCidrChanged(&*self.node.node.lock().unwrap(), &node) {
                if self.node.pods.lock().unwrap().len() > 0 {
                    return Err(Error::CommonError(format!("change pod cidr when node has pods is not allowed, should not happen")));
                }
                // TODO, set up pod cidr
            }

            *self.state.lock().unwrap() = NodeAgentState::Registered;
            SetNodeStatus(&self.node).await?;
            self.node.node.lock().unwrap().status.as_mut().unwrap().phase = Some(format!("{}", NodePending));
        }

        NODEAGENT_STORE.get().unwrap().UpdateNode(&self.node)?;
        
        return Ok(())
    }

    pub async fn OnNodeConfigurationCommand(&self, msg: &NmMsg::NodeConfiguration) -> Result<()> {
        if *self.state.lock().unwrap() != NodeAgentState::Registering {
            return Err(Error::CommonError(format!("node is not in registering state, it does not expect configuration change after registering")));
        }

        let node = NodeFromString(&msg.node)?;
        return self.NodeConfigure(node).await;
    }

     pub fn CreatePod(&self, pod: &k8s::Pod, configMap: &k8s::ConfigMap) -> Result<()> {
        let podId =  K8SUtil::PodId(&pod);
        if self.State() != NodeAgentState::Ready {
            let inner = QuarkPodInner {
                id: K8SUtil::PodId(&pod),
                podState: PodState::Cleanup,
                isDaemon: false,
                pod: Arc::new(RwLock::new(pod.clone())),
                configMap: None,
                runtimePod: None,
                containers: BTreeMap::new(),
                lastTransitionTime: SystemTime::now(),
            };
            let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
            NODEAGENT_STORE.get().unwrap().CreatePod(&qpod)?;
            NODEAGENT_STORE.get().unwrap().DeletePod(&qpod)?;
            return Err(Error::CommonError("Node is not in ready state to create a new pod".to_string()));
        }
       
        let hasPod = self.node.pods.lock().unwrap().get(&podId).is_some();
        if !hasPod {
            let podAgent = match self.CreatePodAgent(PodState::Creating, &pod, &Some(configMap.clone()), false) {
                Ok(a) => a,
                Err(e) => {
                    let inner = QuarkPodInner {
                        id: K8SUtil::PodId(&pod),
                        podState: PodState::Cleanup,
                        isDaemon: false,
                        pod: Arc::new(RwLock::new(pod.clone())),
                        configMap: None,
                        runtimePod: None,
                        containers: BTreeMap::new(),
                        lastTransitionTime: SystemTime::now(),
                    };
                    let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
                    NODEAGENT_STORE.get().unwrap().CreatePod(&qpod)?;
                    NODEAGENT_STORE.get().unwrap().DeletePod(&qpod)?;
                    return Err(e);
                }
            };

            podAgent.Start()?;
            let qpod = podAgent.pod.clone();
            NODEAGENT_STORE.get().unwrap().CreatePod(&qpod)?;
            podAgent.Send(NodeAgentMsg::PodCreate( PodCreate {
                pod: qpod,
            }))?;
        } else {
            error!("Pod: {} already exist", podId);
        }
        return Ok(())
    }

    pub async fn OnPodCreateCmd(&self, msg: &NmMsg::PodCreate) -> Result<()> {
        let pod = PodFromString(&msg.pod)?;
        let configMap = ConfigMapFromString(&msg.config_map)?;
        self.CreatePod(&pod, &configMap)?;
        return Ok(())
    }

    pub fn TerminatePod(&self, podId: &str) -> Result<()> {
        let qpod = self.node.pods.lock().unwrap().get(podId).cloned();
        match qpod {
            None => return Err(Error::CommonError(format!("Pod: {} does not exist, nodemgr is not in sync", podId))),
            Some(qpod) => {
                let mut podAgent = self.pods.lock().unwrap().get(podId).cloned();
                if qpod.PodInTerminating() && podAgent.is_some() {
                    // avoid repeating termination if pod is in terminating process and has a pod actor work on it
			        return Ok(())
                }

                // todo: when will it happen?
                if podAgent.is_none() {
                    warn!("Pod actor {} already exit, create a new one", podId);
                    podAgent = Some(self.StartPodAgent(&qpod)?);
                }

                let podAgent = podAgent.unwrap();
                podAgent.Send(NodeAgentMsg::PodTerminate)?;
            }
        }

        return Ok(())
    }

    pub fn OnPodTerminateCommand(&self, msg: &NmMsg::PodTerminate) -> Result<()> {
        let podId = &msg.pod_identifier;
        return self.TerminatePod(podId);
    }

    pub fn BuildAQuarkPod(&self, state: PodState, pod: &k8s::Pod, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<QuarkPod> {
        ValidatePodSpec(pod)?;

        let mut pod = pod.clone();
        if pod.metadata.uid.is_none() {
            pod.metadata.uid = Some(Uuid::new_v4().to_string());
        }

        if pod.metadata.annotations.is_none() {
            pod.metadata.annotations = Some(BTreeMap::new());
        }
        
        let inner = QuarkPodInner {
            id: K8SUtil::PodId(&pod),
            podState: state,
            isDaemon: isDaemon,
            pod: Arc::new(RwLock::new(pod)),
            configMap: configMap.clone(),
            runtimePod: None,
            containers: BTreeMap::new(),
            lastTransitionTime: SystemTime::now(), 
        };

        let annotationsNone = inner.pod.read().unwrap().metadata.annotations.is_none();
        if annotationsNone {
            inner.pod.write().unwrap().metadata.annotations = Some(BTreeMap::new());
        }

        let nodeId = K8SUtil::NodeId(&(*self.node.node.lock().unwrap()));
        inner.pod.write().unwrap().metadata.annotations.as_mut().unwrap().insert(AnnotationNodeMgrNode.to_string(), nodeId);

        let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
        let k8snode = self.node.node.lock().unwrap().clone();
        qpod.SetPodStatus(Some(&k8snode));
        
        return Ok(qpod);
    }

    pub fn StartPodAgent(&self, qpod: &QuarkPod) -> Result<PodAgent> {
        qpod.Pod().write().unwrap().status.as_mut().unwrap().host_ip = Some(self.node.node.lock().unwrap().status.as_ref().unwrap().addresses.as_ref().unwrap()[0].address.clone());
        qpod.Pod().write().unwrap().metadata.annotations.as_mut().unwrap().insert(AnnotationNodeMgrNode.to_string(), K8SUtil::NodeId(&(*self.node.node.lock().unwrap())));

        let podAgent = PodAgent::New(self.agentChann.clone(), qpod, &self.node.nodeConfig);
        let podId = qpod.lock().unwrap().id.clone();
        self.node.pods.lock().unwrap().insert(podId.clone(), qpod.clone());
        self.pods.lock().unwrap().insert(podId, podAgent.clone());
        return Ok(podAgent);
    }

    pub fn CreatePodAgent(&self, state: PodState, pod: &k8s::Pod, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<PodAgent> {
        let qpod = self.BuildAQuarkPod(state, pod, configMap, isDaemon)?;

        let podAgent = self.StartPodAgent(&qpod)?;
        return Ok(podAgent);
    }
}

pub fn InitK8sNode() -> Result<k8s::Node> {
    let hostname = hostname::get()?.to_str().unwrap().to_string();
    let mut node = k8s::Node {
        metadata: ObjectMeta  {
            name: Some(hostname),
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
    pub fn NewQuarkNode(nodeConfig: &NodeConfiguration) -> Result<QuarkNode> {
        let k8sNode = InitK8sNode()?;
        
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

pub async fn Run(nodeConfig: NodeConfiguration) -> Result<NodeAgent> {
    NodeAgent::CleanPods().await?;
    
    let quarkNode = QuarkNode::NewQuarkNode(&nodeConfig)?;
    let nodeAgent = NodeAgent::New(&quarkNode)?;
    nodeAgent.Start().await?;
    return Ok(nodeAgent);
}