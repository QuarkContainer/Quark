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
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, Duration};
use qshare::consts::{NodePending, DefaultNodeFuncLogFolder, AnnotationNodeMgrNode, AnnotationNodeMgrNodeRevision, NodeRunning, DefaultDomainName};
use qshare::node::PodDef;
use tokio::sync::{mpsc, Notify};
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use core::ops::Deref;
use tokio::time;


use qshare::k8s;
use qshare::node::*;

//use qobjs::runtime_types::QuarkNode;
use qshare::common::*;

use crate::pod_mgr::{NAMESPACE_MGR, RUNTIME_MGR};
use crate::pod_mgr::node_status::{SetNodeStatus, IsNodeStatusReady};
use crate::QLET_CONFIG;

use super::pm_msg::{NodeAgentMsg, PodCreate};
use super::{qnode::*, ConfigName, QLET_STORE};
use super::pod_agent::*;
use super::NODEAGENT_STORE;
use super::qpod::*;
use super::NODE_READY_NOTIFY;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PmAgentState {
    Initializing,
    Initialized,
    Registering,
    Registered,
    Ready,
}

pub struct PmAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub state: Mutex<PmAgentState>,

    pub node: QuarkNode,
    pub agentChann: mpsc::Sender<NodeAgentMsg>,
    pub agentRx: Mutex<Option<mpsc::Receiver<NodeAgentMsg>>>,
    
    pub pods: Mutex<BTreeMap<String, PodAgent>>,
}

#[derive(Clone)]
pub struct PmAgent(pub Arc<PmAgentInner>);

impl Deref for PmAgent {
    type Target = Arc<PmAgentInner>;

    fn deref(&self) -> &Arc<PmAgentInner> {
        &self.0
    }
}

impl PmAgent {
    pub fn Send(&self, msg: NodeAgentMsg) -> Result<()> {
        match self.agentChann.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_) => {
                return Err(Error::CommonError(format!("PodAgent send message fail")));
            }
        }
    }
}

impl PmAgent {
    pub fn New(node: &QuarkNode) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<NodeAgentMsg>(30);
        
        let inner = PmAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            state: Mutex::new(PmAgentState::Initializing),
            node: node.clone(),
            agentChann: tx,
            agentRx: Mutex::new(Some(rx)),
            pods: Mutex::new(BTreeMap::new()),
        };

        let agent = PmAgent(Arc::new(inner));
        return Ok(agent);
    }

    // clean up all qserverless created pods, used when nodeagent restarted
    // work around when the nodeagent persistent store is not ready
    pub async fn CleanPods(nodename: &str) -> Result<()> {
        let pods = RUNTIME_MGR.get().unwrap().GetPods().await?;
        let nodeName = &format!("{}/{}", DefaultDomainName, nodename);
        let sysConfigName = ConfigName();
        for p in pods {
            let pod = p.sandbox.unwrap();

            // skip call sandbox which not created by qserverless
            match pod.annotations.get(AnnotationNodeMgrNode) {
                None => continue,
                Some(n) => {
                    if !(sysConfigName == "product" || n == nodeName) {
                        continue;
                    }
                }
            }

            info!("node {} removing pod {} annotations {:#?}", nodename, &pod.id, &pod.annotations);
            
            let containers = RUNTIME_MGR.get().unwrap().GetPodContainers(&pod.id).await?;
            let mut containerIds = Vec::new();
            for container in containers {
                containerIds.push(container.id);
            }
            match RUNTIME_MGR.get().unwrap().TerminatePod(&pod.id, containerIds).await {
                Ok(()) => (),
                Err(e) => {
                    error!("fail to TerminatePod {} with error {:?}", &pod.id, e);
                }
            }
                   
        }

        return Ok(())
    }

    pub async fn Start(&self) -> Result<()> {
        let rx = self.agentRx.lock().unwrap().take().unwrap();
        let clone = self.clone();
        *self.state.lock().unwrap() = PmAgentState::Registering;
        SetNodeStatus(&self.node).await?;
        NODEAGENT_STORE.CreateNode(&*self.node.node.lock().unwrap())?;
        QLET_STORE.get().unwrap().CreateNode(&self.node)?;

        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });
        
        return Ok(())
    }

    pub async fn Stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_one();
    }

    pub fn State(&self) -> PmAgentState {
        return *self.state.lock().unwrap();
    }

    pub fn IncrNodeRevision(&self) -> i64 {
        let revision = self.node.revision.fetch_add(1, Ordering::SeqCst);
        self.node.node.lock().unwrap().resource_version = format!("{}", revision);
        self.node.node.lock().unwrap().annotations.insert(
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
                    if self.State() == PmAgentState::Registered || self.State() == PmAgentState::Registering {
                        if IsNodeStatusReady(&self.node) {
                            NODE_READY_NOTIFY.notify_waiters();
                            info!("Node {} is ready", &QLET_CONFIG.nodeName);
                            *self.state.lock().unwrap() = PmAgentState::Ready;
                            SetNodeStatus(&self.node).await?;
                            self.node.node.lock().unwrap().status.phase = format!("{}", NodeRunning);
                            QLET_STORE.get().unwrap().UpdateNode(&self.node)?;
                            NODEAGENT_STORE.UpdateNode(&self.node)?;
                        }
                    } else {
                        SetNodeStatus(&self.node).await?;
                        QLET_STORE.get().unwrap().UpdateNode(&self.node)?;
                        NODEAGENT_STORE.UpdateNode(&self.node)?;
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
            NodeAgentMsg::PodStatusChange(msg) => {
                let qpod = msg.pod.clone();
                if qpod.PodState() == PodState::Cleanup {
                    self.CleanupPodStoreAndAgent(&qpod).await?;
                    NODEAGENT_STORE.DeletePod(&qpod)?;
                    QLET_STORE.get().unwrap().RemovePod(&qpod)?;
                    qpod.SetPodState(PodState::Deleted);
                } else if qpod.PodState() != PodState::Deleted {
                    NODEAGENT_STORE.UpdatePod(&qpod)?;
                    QLET_STORE.get().unwrap().UpdatePod(&qpod)?;
                }
            }
            NodeAgentMsg::NodeUpdate => {
                SetNodeStatus(&self.node).await?;
                //NODEAGENT_STORE.UpdateNode(&*self.node.node.lock().unwrap())?;
            }
            _ => {
                error!("NodeHandler Received unknown message {:?}", msg);
            }
        }

        return Ok(())
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

    pub async fn NodeConfigure(&self, node: &Node) -> Result<()> {
        if *self.state.lock().unwrap() == PmAgentState::Registering {
            info!("received node spec from nodemgr: {:?}", &node);
            ValidateNodeSpec(&node)?;

            {
                let mut nodelock = self.node.node.lock().unwrap();
                nodelock.node_ip = node.node_ip.clone();
                nodelock.pod_cidr = node.pod_cidr.clone();
                nodelock.unschedulable = node.unschedulable;
            }

            if NodeSpecPodCidrChanged(&*self.node.node.lock().unwrap(), &node) {
                if self.node.pods.lock().unwrap().len() > 0 {
                    return Err(Error::CommonError(format!("change pod cidr when node has pods is not allowed, should not happen")));
                }
                // TODO, set up pod cidr
            }

            *self.state.lock().unwrap() = PmAgentState::Registered;
            SetNodeStatus(&self.node).await?;
            self.node.node.lock().unwrap().status.phase = format!("{}", NodePending);
        }

        NODEAGENT_STORE.UpdateNode(&self.node)?;
        QLET_STORE.get().unwrap().UpdateNode(&self.node)?;
                            
        return Ok(())
    }

     pub fn CreatePod(&self, pod: &PodDef, configMap: &k8s::ConfigMap) -> Result<IpAddress> {
        let podId =  pod.PodId();
        if self.State() != PmAgentState::Ready {
            let inner = QuarkPodInner {
                id: podId.clone(),
                podState: PodState::Cleanup,
                isDaemon: false,
                pod: Arc::new(RwLock::new(pod.clone())),
                configMap: None,
                runtimePod: None,
                containers: BTreeMap::new(),
                lastTransitionTime: SystemTime::now(),
            };
            let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
            NODEAGENT_STORE.CreatePod(&qpod)?;
            QLET_STORE.get().unwrap().CreatePod(&qpod)?;
            NODEAGENT_STORE.DeletePod(&qpod)?;
            QLET_STORE.get().unwrap().RemovePod(&qpod)?;
            return Err(Error::CommonError("Node is not in ready state to create a new pod".to_string()));
        }
       
        let hasPod = self.node.pods.lock().unwrap().get(&podId).is_some();
        if !hasPod {
            let podAgent = match self.CreatePodAgent(PodState::Creating, &pod, &Some(configMap.clone()), false) {
                Ok(a) => a,
                Err(e) => {
                    let inner = QuarkPodInner {
                        id: pod.PodId(),
                        podState: PodState::Cleanup,
                        isDaemon: false,
                        pod: Arc::new(RwLock::new(pod.clone())),
                        configMap: None,
                        runtimePod: None,
                        containers: BTreeMap::new(),
                        lastTransitionTime: SystemTime::now(),
                    };
                    let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
                    NODEAGENT_STORE.CreatePod(&qpod)?;
                    QLET_STORE.get().unwrap().CreatePod(&qpod)?;
                    NODEAGENT_STORE.DeletePod(&qpod)?;
                    QLET_STORE.get().unwrap().RemovePod(&qpod)?;
                    return Err(e);
                }
            };

            let uid = pod.uid.clone();
            let namespace = pod.PodNamespace();
            let podname = pod.name.clone();
            
            let addr = NAMESPACE_MGR.NewPodSandbox(&namespace, &uid, &podname)?;

            podAgent.Start()?;
            let qpod = podAgent.pod.clone();
            qpod.Pod().write().unwrap().ipAddr = addr.0;
            NODEAGENT_STORE.CreatePod(&qpod)?;
            QLET_STORE.get().unwrap().CreatePod(&qpod)?;
            podAgent.Send(NodeAgentMsg::PodCreate( PodCreate {
                pod: qpod,
            }))?;

            return Ok(addr)
        } else {
            return Err(Error::CommonError(format!("Pod: {} already exist", podId)));
        }
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

    pub fn ReadFuncLog(&self, namespace: &str, funcName: &str, _offset: usize, _len: usize) -> Result<String> {
        let filename = format!("{}/func/{}/{}.log", DefaultNodeFuncLogFolder, namespace, funcName);
        let content = std::fs::read_to_string(&filename)?;
        return Ok(content)
    }

    pub fn BuildQuarkPod(&self, state: PodState, pod: &PodDef, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<QuarkPod> {
        let pod = pod.clone();
        
        let inner = QuarkPodInner {
            id: pod.PodId(),
            podState: state,
            isDaemon: isDaemon,
            pod: Arc::new(RwLock::new(pod)),
            configMap: configMap.clone(),
            runtimePod: None,
            containers: BTreeMap::new(),
            lastTransitionTime: SystemTime::now(), 
        };

        let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
        let node = self.node.node.lock().unwrap().clone();
        qpod.SetPodStatus(Some(&node));
        
        return Ok(qpod);
    }

    pub fn StartPodAgent(&self, qpod: &QuarkPod) -> Result<PodAgent> {
        qpod.Pod().write().unwrap().status.host_ip = self.node.node.lock().unwrap().status.addresses[0].address.clone();
        let nodeId = self.node.node.lock().unwrap().NodeId();
        qpod.Pod().write().unwrap().annotations.insert(AnnotationNodeMgrNode.to_string(), nodeId);

        let podAgent = PodAgent::New(self.agentChann.clone(), qpod, &self.node.nodeConfig);
        let podId = qpod.lock().unwrap().id.clone();
        self.node.pods.lock().unwrap().insert(podId.clone(), qpod.clone());
        self.pods.lock().unwrap().insert(podId, podAgent.clone());
        return Ok(podAgent);
    }

    pub fn CreatePodAgent(&self, state: PodState, pod: &PodDef, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<PodAgent> {
        let qpod = self.BuildQuarkPod(state, pod, configMap, isDaemon)?;

        let podAgent = self.StartPodAgent(&qpod)?;
        return Ok(podAgent);
    }
}