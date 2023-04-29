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
use k8s_openapi::api::core::v1 as k8s;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time};
use qobjs::k8s_util::K8SUtil;
use crate::node_status::IsNodeStatusReady;

//use qobjs::runtime_types::QuarkNode;
use qobjs::runtime_types::*;
use qobjs::common::*;
use qobjs::pb_gen::node_mgr_pb::{self as NmMsg};

use crate::message::BuildFornaxcoreGrpcPodState;
use crate::nm_svc::{NodeAgentMsg, PodCreate};
use crate::node_status::SetNodeStatus;
use crate::pod::*;
use crate::NETWORK_PROVIDER;
use crate::runtime::k8s_labels::*;
use crate::message::*;

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
    pub supervisor: mpsc::Sender<NodeAgentMsg>,
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
    pub fn New(supervisor: mpsc::Sender<NodeAgentMsg>, node: &QuarkNode) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<NodeAgentMsg>(30);
        
        let inner = NodeAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            state: Mutex::new(NodeAgentState::Initializing),
            node: node.clone(),
            supervisor: supervisor,
            agentChann: tx,
            agentRx: Mutex::new(Some(rx)),
            pods: Mutex::new(BTreeMap::new()),
        };

        let agent = NodeAgent(Arc::new(inner));
        return Ok(agent);
    }

    pub async fn Start(&self) -> Result<()> {
        let rx = self.agentRx.lock().unwrap().take().unwrap();
        let clone = self.clone();
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
            AnnotationFornaxCoreNodeRevision.to_string(), 
            format!("{}", revision)
        );
        return revision;
    }

    pub fn SaveAndNotifyPodState(&self, pod: &QuarkPod) {
        if pod.PodInTransitState() {
            let revision = self.IncrNodeRevision();
            pod.Pod().write().unwrap().metadata.resource_version = Some(revision.to_string());
            pod.Pod().write().unwrap().metadata.annotations.as_mut().unwrap().insert(
                AnnotationFornaxCoreNodeRevision.to_string(), 
                format!("{}", revision)
            );
            
            self.Notify(NodeAgentMsg::NodeMgrMsg(
                BuildFornaxcoreGrpcPodState(revision, pod).unwrap()
            ));
        }
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
                    let state = self.State();

                    if state == NodeAgentState::Registering {
                        info!("Start node registry node spec {:?}", self.node.node.lock().unwrap());
                        
                        let messsageType = NmMsg::MessageType::NodeRegister;
                        let rev = self.node.revision.load(Ordering::SeqCst);
                        let nr = NmMsg::NodeRegistry {
                            node_revision: rev,
                            node: NodeToString(&*self.node.node.lock().unwrap())?,
                        };

                        let msg = NmMsg::FornaxCoreMessage {
                            message_type: messsageType as i32,
                            message_body: Some(NmMsg::fornax_core_message::MessageBody::NodeRegistry(nr)),
                            ..Default::default()
                        };

                        self.Notify(NodeAgentMsg::NodeMgrMsg(msg));
                    }

                    if self.State() == NodeAgentState::Registered {
                        continue;
                    }

                    SetNodeStatus(&self.node).await?;
                    if IsNodeStatusReady(&self.node) {
                        info!("Node is ready, tell Fornax core");
                        let revision = self.IncrNodeRevision();
                        self.node.node.lock().unwrap().metadata.resource_version = Some(format!("{}", revision));
                        self.node.node.lock().unwrap().status.as_mut().unwrap().phase = Some(format!("{}", NodeRunning));
                        *self.state.lock().unwrap() = NodeAgentState::Ready;
                        
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
                return self.ProcessNodeMgrMsg(msg);
            }
            NodeAgentMsg::PodStatusChange(msg) => {
                let qpod = msg.pod.clone();
                if qpod.PodState() == PodState::Cleanup {
                    self.CleanupPodStoreAndAgent(&qpod).await?;
                }
                self.SaveAndNotifyPodState(&qpod);
            }
            NodeAgentMsg::NodeUpdate => {
                SetNodeStatus(&self.node).await?;
                let revision = self.node.revision.load(Ordering::SeqCst);
                self.Notify(NodeAgentMsg::NodeMgrMsg(
                BuildFornaxGrpcNodeState(&self.node, revision)?));
            }
            _ => {
                error!("NodeHandler Received unknown message {:?}", msg);
            }
        }

        return Ok(())
    }

    pub fn ProcessNodeMgrMsg(&self, msg: &NmMsg::FornaxCoreMessage) -> Result<()> {
        let body = msg.message_body.as_ref().unwrap();
        match body {
            NmMsg::fornax_core_message::MessageBody::NodeConfiguration(msg) => {
                return self.OnNodeConfigurationCommand(&msg);
            }
            NmMsg::fornax_core_message::MessageBody::NodeFullSync(msg) => {
                return self.OnNodeFullSyncCommand(&msg);
            }
            NmMsg::fornax_core_message::MessageBody::PodCreate(msg) => {
                return self.OnPodCreateCmd(&msg);
            }
            NmMsg::fornax_core_message::MessageBody::PodTerminate(msg) => {
                return self.OnPodTerminateCommand(&msg);
            }
            NmMsg::fornax_core_message::MessageBody::PodState(_) => {
                self.Notify(NodeAgentMsg::NodeMgrMsg(msg.clone()));
                return Ok(())
            }
            _ => {
                warn!("FornaxCoreMessage {:?} is not handled by actor", msg);
                return Ok(())
            }
        }
    }

    pub fn Notify(&self, msg: NodeAgentMsg) {
        self.supervisor.try_send(msg).unwrap();
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
        let msg = BuildFornaxGrpcNodeState(&self.node, self.node.revision.load(Ordering::SeqCst))?;
        self.Notify(NodeAgentMsg::NodeMgrMsg(msg));
        return Ok(())
    }

    pub fn OnNodeConfigurationCommand(&self, msg: &NmMsg::NodeConfiguration) -> Result<()> {
        if *self.state.lock().unwrap() == NodeAgentState::Registering {
            return Err(Error::CommonError(format!("node is not in registering state, it does not expect configuration change after registering")));
        }

        let node = NodeFromString(&msg.node)?;
        info!("received node spec from fornaxcore: {:?}", &node);
        ValidateNodeSpec(&node)?;

        self.node.node.lock().unwrap().spec = node.spec.clone();
        if NodeSpecPodCidrChanged(&*self.node.node.lock().unwrap(), &node) {
            if self.node.pods.lock().unwrap().len() > 0 {
                return Err(Error::CommonError(format!("change pod cidr when node has pods is not allowed, should not happen")));
            }
            // TODO, set up pod cidr
        }

        *self.state.lock().unwrap() = NodeAgentState::Registered;

        // todo: ....
        /*
        	// start go routine to check node status until it is ready
	go func() {
		for {
			// finish if node has initialized
			if n.state != NodeStateRegistered {
				break
			}

			// check node runtime dependencies, send node ready message if node is ready for new pod
			SetNodeStatus(n.node, n.dependencies)
			if IsNodeStatusReady(n.node) {
				klog.InfoS("Node is ready, tell Fornax core")
				// bump revision to let Fornax core to update node status
				revision := n.incrementNodeRevision()
				n.node.V1Node.ResourceVersion = fmt.Sprint(revision)
				n.node.V1Node.Status.Phase = v1.NodeRunning
				n.notify(n.fornoxCoreRef, BuildFornaxGrpcNodeReady(n.node, revision))
				n.state = NodeStateReady
				// do not periodically report, fornaxcore will send syncup request when it determine resyncup is required
				// n.startStateReport()
			} else {
				time.Sleep(5 * time.Second)
			}
		}
	}()
         */

        return Ok(())
    }

    pub fn OnPodCreateCmd(&self, msg: &NmMsg::PodCreate) -> Result<()> {
        let pod = PodFromString(&msg.pod)?;
        if self.State() != NodeAgentState::Ready {
            let inner = QuarkPodInner {
                id: K8SUtil::PodId(&pod),
                podState: PodState::Cleanup,
                isDaemon: false,
                pod: Arc::new(RwLock::new(pod)),
                configMap: None,
                runtimePod: None,
                containers: BTreeMap::new(),
                lastTransitionTime: SystemTime::now(),
            };
            let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
            self.SaveAndNotifyPodState(&qpod);
            return Err(Error::CommonError("Node is not in ready state to create a new pod".to_string()));
        }

        let podId = &msg.pod_identifier;
        let configMap = ConfigMapFromString(&msg.config_map)?;

        let hasPod = self.node.pods.lock().unwrap().get(podId).is_some();
        if !hasPod {
            let podAgent = self.CreatePodAgent(PodState::Creating, &pod, &Some(configMap), false)?;
            let inner = QuarkPodInner {
                id: K8SUtil::PodId(&pod),
                podState: PodState::Cleanup,
                isDaemon: false,
                pod: Arc::new(RwLock::new(pod)),
                configMap: None,
                runtimePod: None,
                containers: BTreeMap::new(),
                lastTransitionTime: SystemTime::now(),
            };
            let qpod = QuarkPod(Arc::new(Mutex::new(inner)));

            self.SaveAndNotifyPodState(&qpod);
            podAgent.Send(NodeAgentMsg::PodCreate( PodCreate {
                pod: qpod,
            }))?;
        } else {
            error!("Pod: {} already exist", podId);
        }
        return Ok(())
    }

    pub fn OnPodTerminateCommand(&self, msg: &NmMsg::PodTerminate) -> Result<()> {
        let podId = &msg.pod_identifier;
        let qpod = self.node.pods.lock().unwrap().get(podId).cloned();
        match qpod {
            None => return Err(Error::CommonError(format!("Pod: {} does not exist, Fornax core is not in sync", podId))),
            Some(qpod) => {
                let mut podAgent = self.pods.lock().unwrap().get(podId).cloned();
                if qpod.PodInTerminating() && podAgent.is_some() {
                    // avoid repeating termination if pod is in terminating process and has a pod actor work on it
			        return Ok(())
                }

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

    pub fn BuildAQuarkPod(&self, state: PodState, pod: &k8s::Pod, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<QuarkPod> {
        ValidatePodSpec(pod)?;

        let inner = QuarkPodInner {
            id: K8SUtil::PodId(pod),
            podState: state,
            isDaemon: isDaemon,
            pod: Arc::new(RwLock::new(pod.clone())),
            configMap: configMap.clone(),
            runtimePod: None,
            containers: BTreeMap::new(),
            lastTransitionTime: SystemTime::now(), 
        };

        let nodeId = K8SUtil::NodeId(&(*self.node.node.lock().unwrap()));
        inner.pod.write().unwrap().metadata.annotations.as_mut().unwrap().insert(AnnotationFornaxCoreNode.to_string(), nodeId);

        let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
        let k8snode = self.node.node.lock().unwrap().clone();
        qpod.SetPodStatus(Some(&k8snode));
        
        return Ok(qpod);
    }

    pub fn StartPodAgent(&self, qpod: &QuarkPod) -> Result<PodAgent> {
        qpod.Pod().write().unwrap().status.as_mut().unwrap().host_ip = Some(self.node.node.lock().unwrap().status.as_ref().unwrap().addresses.as_ref().unwrap()[0].address.clone());
        qpod.Pod().write().unwrap().metadata.annotations.as_mut().unwrap().insert(AnnotationFornaxCoreNode.to_string(), K8SUtil::NodeId(&(*self.node.node.lock().unwrap())));

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

pub struct QuarkNodeInner {
    pub nodeConfig: NodeConfiguration,
    pub node: Mutex<k8s::Node>,
    pub revision: AtomicI64,
    pub pods: Mutex<BTreeMap<String, QuarkPod>>, 
}

#[derive(Clone)]
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

pub async fn Run(nodeConfig: NodeConfiguration) -> Result<mpsc::Receiver<NodeAgentMsg>> {
    let (tx, rx) = mpsc::channel::<NodeAgentMsg>(30);
        
    let quarkNode = QuarkNode::NewQuarkNode(&nodeConfig)?;
    let nodeAgent = NodeAgent::New(tx, &quarkNode)?;
    nodeAgent.Start().await?;
    return Ok(rx);
}