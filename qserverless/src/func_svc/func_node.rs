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
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::SystemTime;
use core::ops::Deref;
use tokio::sync::Mutex as TMutex;
use tonic::Streaming;
use tokio::sync::mpsc;
use std::result::Result as SResult;
use tonic::Status;
use tokio::task::JoinHandle;

use qobjs::k8s;
use qobjs::ObjectMeta;
use qobjs::common::*;
use qobjs::func;
use tokio::sync::Notify;

use crate::FUNC_CALL_MGR;
use crate::FUNC_NODE_MGR;
use crate::PACKAGE_MGR;
use crate::func_call::FuncCall;
use crate::func_call::FuncCallInner;
use crate::func_call::FuncCallId;
use crate::func_call::FuncCallState;
use crate::func_pod::*;
use crate::message::FuncNodeMsg;
use crate::package::Package;
use crate::package::PackageId;

#[derive(Debug, Clone)]
pub enum FuncNodeState {
    WaitingConn, // master node waiting for nodeagent connection
    Running(mpsc::Sender<SResult<func::FuncSvcMsg, Status>>), // get the registration message
}

#[derive(Debug)]
pub struct FuncNodeInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub nodeName: String,
    pub state: FuncNodeState,
    pub FuncPods: BTreeMap<FuncPodId, FuncPod>,
    pub ProcessingCallerFuncCalls: BTreeSet<FuncCallId>, 
    pub PengingProcessingCallerFuncCalls: BTreeSet<FuncCallId>, 
    // when node start up, funcalls which assumed processing by other nodes
    pub CheckingProcessingCallerFuncCalls: BTreeSet<FuncCallId>, 
    
    pub ProcessingCalleeFuncCalls: BTreeSet<FuncCallId>,
    pub CheckingCalleeFuncCalls: BTreeSet<FuncCallId>,

    pub tx: Option<mpsc::Sender<SResult<func::FuncSvcMsg, Status>>>,
    pub internMsgTx: mpsc::Sender<FuncNodeMsg>,

    pub processorHandler: Option<JoinHandle<()>>,
    pub internMsgRx: Option<mpsc::Receiver<FuncNodeMsg>>,
}

impl FuncNodeInner {

}

#[derive(Debug, Clone)]
pub struct FuncNode(Arc<Mutex<FuncNodeInner>>);

impl Deref for FuncNode {
    type Target = Arc<Mutex<FuncNodeInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncNodeInner>> {
        &self.0
    }
}


impl FuncNode {
    //pub fn New(name: &str) -> Result<()> {}

    pub fn State(&self) -> FuncNodeState {
        return self.lock().unwrap().state.clone();
    }

    pub fn Close(&self) {
        let notify = self.lock().unwrap().closeNotify.clone();
        notify.notify_waiters();
    }
    
    pub fn SetState(&self, state: FuncNodeState) {
        self.lock().unwrap().state = state;
    }

    // check whether the node is processing the func call
    // return: true: yes or possible(the node is in the waiting state and need confirm)
    // false: No
    pub fn CheckFuncCallProcessor(&self, funcCallId: &FuncCallId) -> bool {
        let mut inner = self.lock().unwrap();
        match &inner.state {
            FuncNodeState::WaitingConn => {
                inner.CheckingCalleeFuncCalls.insert(funcCallId.clone());
                return true;
            }
            FuncNodeState::Running(_) => {
                return inner.ProcessingCalleeFuncCalls.contains(funcCallId);
            }
        }
    }

    pub fn ConfirmFuncCallPrococessing(&self, funcCallId: &FuncCallId) {
        let mut inner = self.lock().unwrap();
        match &inner.state {
            FuncNodeState::WaitingConn => {
                inner.PengingProcessingCallerFuncCalls.insert(funcCallId.clone());
            }
            FuncNodeState::Running(_) => {
                if inner.PengingProcessingCallerFuncCalls.remove(funcCallId) {
                    // confirm the funcall  is processing by another node
                    inner.ProcessingCalleeFuncCalls.insert(funcCallId.clone());
                }
            }
        }
    }

    // this will be called in 2 situations
    // 1. When the nodeagent reconnect to the FuncService, todo: need to handle this scenario
    // 2. When the Master FuncService die, the nodeagent sent FuncAgentRegisterReq to another FuncService
    pub fn OnNodeRegiste(&self, regReq: func::FuncAgentRegisterReq) -> Result<()> {    
        let mut inner = self.lock().unwrap();
        for funccall in regReq.func_calls {
            let packageId = PackageId {
                namespace: funccall.namespace.clone(),
                packageName: funccall.package_name.clone(),
            };
            let funcCallId = FuncCallId {
                packageId: packageId.clone(),
                funcName: funccall.func_name.clone(),
            };

            let package = match PACKAGE_MGR.Get(&packageId) {
                None => {
                    error!("StartProcess get invalid package {:?}", packageId);
                    continue;
                }
                Some(p) => p,
            };

            let state = if funccall.callee_node_id.len() == 0 {
                inner.PengingProcessingCallerFuncCalls.insert(funcCallId.clone());
                FuncCallState::Scheduling
            } else {
                match FUNC_NODE_MGR.Get(&funccall.callee_node_id) {
                    None => {
                        error!("StartProcess get invalid node name  {:?}", &funccall.callee_node_id);
                        FuncCallState::Scheduling
                    }
                    Some(node) => {
                        if node.CheckFuncCallProcessor(&funcCallId) {
                            FuncCallState::Scheduled(funccall.callee_node_id.clone())
                        } else {
                            FuncCallState::Scheduling
                        }
                    }
                }
            };

            let _funcall = match FUNC_CALL_MGR.Get(&funcCallId) {
                None => {
                    let funcCallContextInner = FuncCallInner {
                        id: funcCallId.clone(),
                        package: package.clone(),
                        callerNode: self.clone(),
                        state: Mutex::new(state),
                        parameters: funccall.parameters.clone(),
                        priority: funccall.priority,
                        // todo: covert from funccall.createtime
                        createTime: SystemTime::now(),
                    };
    
                    let funcall = FuncCall(Arc::new(funcCallContextInner));
                    FUNC_CALL_MGR.Add(&funcall);
                    funcall
                }
                Some(funccall) => {
                    *funccall.state.lock().unwrap() = state;
                    funccall
                }
            };

            // todo: if the funcall is in scheduling state, schedule it
        }
        
        for podStatus in regReq.func_pods {
            let packageId = PackageId {
                namespace: podStatus.namespace.clone(),
                packageName: podStatus.package_name.clone(),
            };
            let package = match PACKAGE_MGR.Get(&packageId) {
                None => {
                    error!("StartProcess get invalid package {:?}", packageId);
                    continue;
                }
                Some(p) => p,
            };
            let state = match podStatus.state {
                n if func::FuncPodState::Creating as i32 == n => FuncPodState::Creating(SystemTime::now()),
                n if func::FuncPodState::Keepalive as i32 == n => FuncPodState::Keepalive(SystemTime::now()),
                n if func::FuncPodState::Running as i32 == n => {
                    let node = match FUNC_NODE_MGR.Get(&podStatus.func_caller_node_id) {
                        None => {
                            error!("StartProcess get invalid node name  {:?}", &podStatus.func_caller_node_id);
                            continue;
                        }
                        Some(n) => n,
                    };

                    let funcCallId = FuncCallId {
                        packageId: packageId.clone(),
                        funcName: podStatus.func_name.clone(),
                    };
                    
                    node.ConfirmFuncCallPrococessing(&funcCallId);
                    FuncPodState::Running(funcCallId)
                }
                _ => panic!("get unexpected podStatus.state {}", podStatus.state)
            };

            let funcPodId = FuncPodId {
                packageId: packageId.clone(),
                podName: podStatus.pod_name.clone(),
                nodeName: inner.nodeName.clone(),
            };

            let pod : k8s::Pod = serde_json::from_str(&podStatus.pod)?;
            match inner.FuncPods.get(&funcPodId) {
                None => {
                    let podInner = FuncPodInner {
                        podName: podStatus.pod_name.clone(),
                        package: package,
                        node: self.clone(),
                        state: Mutex::new(state),
                        pod: Mutex::new(pod),
                    };
                    let funcPod = FuncPod(Arc::new(podInner));
                    inner.FuncPods.insert(funcPodId, funcPod);

                }
                Some(funcPod) => {
                    *funcPod.state.lock().unwrap() = state;
                    *funcPod.pod.lock().unwrap() = pod;
                }
            }
        }
        
        return Ok(())
    }

    pub fn CreateProcessor(&self, regReq: func::FuncAgentRegisterReq, rx: Streaming<func::FuncSvcMsg>, tx: mpsc::Sender<SResult<func::FuncSvcMsg, Status>>) -> Result<()> {
        let node = self.clone();

        let handler = tokio::spawn(async move {
            node.StartProcess(regReq, rx, tx).await.unwrap();
        });

        self.lock().unwrap().processorHandler = Some(handler);
        return Ok(())
    }

    pub async fn ProcessInternalMsg(&self, _msg: FuncNodeMsg) -> Result<()> {
        return Ok(())
    }

    pub async fn ProcessNodeAgentMsg(&self, msg: func::FuncSvcMsg) -> Result<()> {
        //let msgId = msg.msg_id;
        let body = match msg.event_body {
            None => panic!("ProcessNodeAgentMsg get none eventbody"),
            Some(b) => b,
        };
        match body {
            /*
            func::func_svc_msg::EventBody::FuncAgentRegisterReq(req) => {
            // processing
            self.SetState(FuncNodeState::Running(tx));
            } */
            _ => {
                error!("didn't get FuncAgentRegisterReq message instead {:?}", body);
                return Err(Error::CommonError(format!("didn't get FuncAgentRegisterReq message instead {:?}", body)));
            }
        }
    }

    pub async fn StartProcess(
        &self, 
        regReq: func::FuncAgentRegisterReq, 
        rx: Streaming<func::FuncSvcMsg>, 
        tx: mpsc::Sender<SResult<func::FuncSvcMsg, Status>>
    ) -> Result<()> {
        let state = self.State();
        match state {
            FuncNodeState::WaitingConn => (),
            _ => {
                self.Close();
            }
        }

        let handler = self.lock().unwrap().processorHandler.take();
        match handler {
            None => (),
            Some(handler) => {
                handler.await?;
            }
        };

        let mut internMsgRx = self.lock().unwrap().internMsgRx.take().expect("StartProcess doesn't get internal rx channel");
        defer!(self.SetState(FuncNodeState::WaitingConn));
        let mut rx = rx;
        let closeNotify = self.lock().unwrap().closeNotify.clone();

        self.SetState(FuncNodeState::Running(tx));

        {
            // OnNodeRegiste needs to access another node, 
            // to avoid deadlock, use global mutex to serialize the process 
            let _l = FUNC_NODE_MGR.initlock.lock().await;                
            self.OnNodeRegiste(regReq)?;
        }

        loop {
            tokio::select! {
                _ = closeNotify.notified() => {
                    break;
                }
                interalMsg = internMsgRx.recv() => {
                    match interalMsg {
                        None => {
                            panic!("FuncNode::StartProcess expect None internal message");
                        }
                        Some(msg) => {
                            self.ProcessInternalMsg(msg).await?;
                        }
                    }
                    
                }
                msg = rx.message() => {
                    let msg : func::FuncSvcMsg = match msg {
                        Err(e) => {
                            error!("FuncNode get error message {:?}", e);
                            break;
                        }
                        Ok(m) => {
                            match m {
                                None => {
                                    error!("FuncNode get None message");
                                    break;
                                }
                                Some(m) => m,
                            }
                        }
                    };
                    self.ProcessNodeAgentMsg(msg).await?;
                }
            }
        }

        self.lock().unwrap().internMsgRx = Some(internMsgRx);

        return Ok(())
    }

    pub fn NewFuncPod(&self, package: &Package) -> Result<FuncPod> {
        let uid = uuid::Uuid::new_v4().to_string();
        let inner = FuncPodInner {
            podName: uid.clone(),
            package: package.clone(),
            node: self.clone(),
            state: Mutex::new(FuncPodState::Creating(SystemTime::now())),
            pod: Mutex::new(k8s::Pod {
                metadata: ObjectMeta { 
                    namespace: Some(package.Namespace()),
                    name: Some(package.Name()),
                    uid: Some(uid.clone()),
                    ..Default::default()
                },
                spec: Some(package.Spec()),
                ..Default::default()
            })
        };

        let pod = FuncPod(Arc::new(inner));
        self.CreatePod(&pod)?;

        let funcPodId = FuncPodId {
            packageId: package.PackageId(),
            podName: uid,
            nodeName: self.NodeName(),
        };

        self.lock().unwrap().FuncPods.insert(funcPodId, pod.clone());

        return Ok(pod);
    }

    pub fn NodeName(&self) -> String {
        return self.lock().unwrap().nodeName.clone();
    }

    pub fn CreatePod(&self, _pod: &FuncPod) -> Result<()> {
        unimplemented!();
    }
}

#[derive(Debug, Clone)]
pub struct NodeAgentConnection {

}

pub struct FuncNodeMgr {
    pub initlock: TMutex<()>, // used for node register process to avoid deadlock
    pub nodes: Mutex<BTreeMap<String, FuncNode>>,
}

impl FuncNodeMgr {
    pub fn New() -> Self {
        return Self {
            initlock: TMutex::new(()),
            nodes: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn Get(&self, nodeName: &str) -> Option<FuncNode> {
        return self.nodes.lock().unwrap().get(nodeName).cloned();
    }
}