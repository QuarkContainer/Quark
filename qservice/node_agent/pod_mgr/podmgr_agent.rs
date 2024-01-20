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
use uuid::Uuid;
use tokio::sync::{mpsc, Notify};
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use core::ops::Deref;
use tokio::time;


use qshare::k8s;
use qshare::k8s_util::K8SUtil;

//use qobjs::runtime_types::QuarkNode;
use qshare::common::*;

use crate::pod_mgr::{RUNTIME_MGR, NODEAGENT_CONFIG};
use crate::pod_mgr::node_status::{SetNodeStatus, IsNodeStatusReady};

use super::pm_msg::{NodeAgentMsg, PodCreate};
use super::{qnode::*, ConfigName};
use super::pod_agent::*;
use super::NODEAGENT_STORE;
use super::qpod::{QuarkPod, PodState, QuarkPodInner};
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
                    if self.State() == PmAgentState::Registered {
                        if IsNodeStatusReady(&self.node) {
                            NODE_READY_NOTIFY.notify_waiters();
                            info!("Node {} is ready", NODEAGENT_CONFIG.NodeName());
                            *self.state.lock().unwrap() = PmAgentState::Ready;
                            SetNodeStatus(&self.node).await?;
                            self.node.node.lock().unwrap().status.as_mut().unwrap().phase = Some(format!("{}", NodeRunning));
                            NODEAGENT_STORE.UpdateNode(&self.node)?;
                        }
                    } else {
                        SetNodeStatus(&self.node).await?;
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
                    qpod.SetPodState(PodState::Deleted);
                } else if qpod.PodState() != PodState::Deleted {
                    NODEAGENT_STORE.UpdatePod(&qpod)?;
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

    pub async fn NodeConfigure(&self, node: k8s::Node) -> Result<()> {
        if *self.state.lock().unwrap() == PmAgentState::Registering {
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

            *self.state.lock().unwrap() = PmAgentState::Registered;
            SetNodeStatus(&self.node).await?;
            self.node.node.lock().unwrap().status.as_mut().unwrap().phase = Some(format!("{}", NodePending));
        }

        NODEAGENT_STORE.UpdateNode(&self.node)?;
        
        return Ok(())
    }

     pub fn CreatePod(&self, pod: &k8s::Pod, configMap: &k8s::ConfigMap) -> Result<()> {
        let podId =  K8SUtil::PodId(&pod);
        if self.State() != PmAgentState::Ready {
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
            NODEAGENT_STORE.CreatePod(&qpod)?;
            NODEAGENT_STORE.DeletePod(&qpod)?;
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
                    NODEAGENT_STORE.CreatePod(&qpod)?;
                    NODEAGENT_STORE.DeletePod(&qpod)?;
                    return Err(e);
                }
            };

            podAgent.Start()?;
            let qpod = podAgent.pod.clone();
            NODEAGENT_STORE.CreatePod(&qpod)?;
            podAgent.Send(NodeAgentMsg::PodCreate( PodCreate {
                pod: qpod,
            }))?;
        } else {
            error!("Pod: {} already exist", podId);
        }
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

    pub fn ReadFuncLog(&self, namespace: &str, funcName: &str, _offset: usize, _len: usize) -> Result<String> {
        let filename = format!("{}/func/{}/{}.log", DefaultNodeFuncLogFolder, namespace, funcName);
        let content = std::fs::read_to_string(&filename)?;
        return Ok(content)
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

        //let nodeId = K8SUtil::NodeId(&(*self.node.node.lock().unwrap()));
        //inner.pod.write().unwrap().metadata.annotations.as_mut().unwrap().insert(AnnotationNodeMgrNode.to_string(), nodeId);

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