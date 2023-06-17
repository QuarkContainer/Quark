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
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::SystemTime;
use core::ops::Deref;
use qobjs::utility::SystemTimeProto;
use tokio::sync::Mutex as TMutex;
use tonic::Streaming;
use tokio::sync::mpsc;
use std::result::Result as SResult;
use tonic::Status;
use tokio::task::JoinHandle;

use qobjs::common::*;
use qobjs::func;
use tokio::sync::Notify;

use crate::FUNC_CALL_MGR;
use crate::FUNC_NODE_MGR;
use crate::FUNC_POD_MGR;
use crate::FUNC_SVC_MGR;
use crate::PACKAGE_MGR;
use crate::func_call::FuncCall;
use crate::func_call::FuncCallInner;
use crate::func_call::FuncCallState;
use crate::func_pod::*;
use crate::message::FuncNodeMsg;
use crate::package::PackageId;

#[derive(Debug, Clone)]
pub enum FuncNodeState {
    WaitingConn, // master node waiting for nodeagent connection
    Running, // get the registration message
}

#[derive(Debug)]
pub struct FuncNodeInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub nodeName: String,
    pub state: FuncNodeState,
    pub funcPods: BTreeMap<String, FuncPod>,
    pub callerFuncCalls: BTreeMap<String, FuncCall>, 
    pub calleeFuncCalls: BTreeMap<String, FuncCall>, 

    pub internMsgTx: mpsc::Sender<FuncNodeMsg>,
    pub internMsgRx: Option<mpsc::Receiver<FuncNodeMsg>>,

    pub processorHandler: Option<JoinHandle<()>>,
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
    pub fn New(nodeName: &str) -> Self {
        let (tx, rx) = mpsc::channel(30);
        let inner = FuncNodeInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            nodeName: nodeName.to_string(),
            state: FuncNodeState::WaitingConn,
            funcPods: BTreeMap::new(),
            callerFuncCalls: BTreeMap::new(),
            calleeFuncCalls: BTreeMap::new(),
            internMsgTx: tx,
            internMsgRx: Some(rx),
            processorHandler: None,
        };

        return Self(Arc::new(Mutex::new(inner)));
    }

    //pub fn New(name: &str) -> Result<()> {}
    pub fn Send(&self, msg: FuncNodeMsg) -> Result<()> {
        match self.lock().unwrap().internMsgTx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_) => return Err(Error::MpscSendFail),
        }
    }

    pub fn State(&self) -> FuncNodeState {
        return self.lock().unwrap().state.clone();
    }

    pub fn IsRunning(&self) -> bool {
        match self.State() {
            FuncNodeState::WaitingConn => return false,
            _ => return true
        };
    }

    pub fn Close(&self) {
        let notify = self.lock().unwrap().closeNotify.clone();
        notify.notify_waiters();
    }
    
    pub fn SetState(&self, state: FuncNodeState) {
        self.lock().unwrap().state = state;
    }

    pub fn HasCallerFuncCall(&self, funcCallId: &str) -> bool {
        let inner = self.lock().unwrap();
        return inner.callerFuncCalls.contains_key(funcCallId);
    }

    pub fn HasCalleeFuncCall(&self, funcCallId: &str) -> bool {
        let inner = self.lock().unwrap();
        return inner.calleeFuncCalls.contains_key(funcCallId);
    }

    // this will be called in 2 situations
    // 1. When the nodeagent reconnect to the FuncService, todo: need to handle this scenario
    // 2. When the Master FuncService die, the nodeagent sent FuncAgentRegisterReq to another FuncService
    pub fn OnNodeRegister(&self, regReq: func::FuncAgentRegisterReq) -> Result<()> {    
        let mut inner = self.lock().unwrap();
        for funccall in regReq.callee_calls {
            let packageId = PackageId {
                namespace: funccall.namespace.clone(),
                packageName: funccall.package_name.clone(),
            };
            let funcCallId = funccall.func_name.clone();

            let package = match PACKAGE_MGR.Get(&packageId) {
                Err(_) => {
                    error!("StartProcess get invalid package {:?}", packageId);
                    continue;
                }
                Ok(p) => p,
            };

            let funcCallInner = FuncCallInner {
                jobId: funccall.job_id.clone(),
                id: funcCallId.clone(),
                package: package.clone(),
                funcName: funccall.func_name.clone(),

                callerNodeId: funccall.caller_node_id.clone(),
                callerFuncPodId: funccall.caller_pod_id.clone(),
                calleeNodeId: Mutex::new(funccall.caller_node_id.clone()),
                calleeFuncPodId: Mutex::new(funccall.caller_pod_id.clone()),
                state: Mutex::new(FuncCallState::Cancelling),
                parameters: funccall.parameters.clone(),
                priority: funccall.priority as usize,
                createTime: SystemTimeProto::FromTimestamp(funccall.createtime.as_ref().unwrap()).ToSystemTime(),
            };

            let funcCall = FuncCall(Arc::new(funcCallInner));
            
            FUNC_CALL_MGR.RegisteCallee(&funcCall)?;
        }

        for funccall in regReq.caller_calls {
            let packageId = PackageId {
                namespace: funccall.namespace.clone(),
                packageName: funccall.package_name.clone(),
            };
            let funcCallId = funccall.func_name.clone();

            let package = match PACKAGE_MGR.Get(&packageId) {
                Err(_) => {
                    error!("StartProcess get invalid package {:?}", packageId);
                    continue;
                }
                Ok(p) => p,
            };

            let funcCallInner = FuncCallInner {
                jobId: funccall.job_id.clone(),
                id: funcCallId.clone(),
                package: package.clone(),
                funcName: funccall.func_name.clone(),

                callerNodeId: funccall.caller_node_id.clone(),
                callerFuncPodId: funccall.caller_pod_id.clone(),
                calleeNodeId: Mutex::new(funccall.caller_node_id.clone()),
                calleeFuncPodId: Mutex::new(funccall.caller_pod_id.clone()),
                state: Mutex::new(FuncCallState::Cancelling),
                parameters: funccall.parameters.clone(),
                priority: funccall.priority as usize,
                createTime: SystemTimeProto::FromTimestamp(funccall.createtime.as_ref().unwrap()).ToSystemTime(),
            };

            let funcCall = FuncCall(Arc::new(funcCallInner));
            
            // todo: handle the situation when the result is ready
            let _ret = FUNC_CALL_MGR.RegisteCaller(&funcCall)?;
        }
        
        for podStatus in regReq.func_pods {
            let package  = if !podStatus.client_mode {
                let packageId = PackageId {
                    namespace: podStatus.namespace.clone(),
                    packageName: podStatus.package_name.clone(),
                };
                let package = match PACKAGE_MGR.Get(&packageId) {
                    Err(_) => {
                        error!("StartProcess get invalid package {:?}", packageId);
                        continue;
                    }
                    Ok(p) => p,
                };
                Some(package)
            } else {
                None
            };
            
            let state = match podStatus.state {
                n if func::FuncPodState::Idle as i32 == n => FuncPodState::Idle(SystemTime::now()),
                n if func::FuncPodState::Running as i32 == n => {
                    FuncPodState::Running(podStatus.func_call_id.clone())
                }
                _ => panic!("get unexpected podStatus.state {}", podStatus.state)
            };

            let funcPodId = podStatus.func_pod_id.clone();

            match inner.funcPods.get(&funcPodId) {
                None => {
                    let podInner = FuncPodInner {
                        podName: podStatus.func_pod_id.clone(),
                        package: package,
                        node: self.clone(),
                        state: Mutex::new(state),
                        clientMode: podStatus.client_mode,
                    };
                    let funcPod = FuncPod(Arc::new(podInner));
                    inner.funcPods.insert(funcPodId, funcPod);

                }
                Some(funcPod) => {
                    *funcPod.state.lock().unwrap() = state;
                }
            }
        }
        
        return Ok(())
    }

    pub async fn CreateProcessor(&self, regReq: func::FuncAgentRegisterReq, rx: Streaming<func::FuncSvcMsg>, tx: mpsc::Sender<SResult<func::FuncSvcMsg, Status>>) -> Result<()> {
        let node = self.clone();
        let ready = Arc::new(Notify::new());
        let readyclone = ready.clone();
        let handler = tokio::spawn(async move {
            node.StartProcess(regReq, rx, tx, readyclone).await.unwrap();
        });

        ready.notified().await;
        self.lock().unwrap().processorHandler = Some(handler);
        return Ok(())
    }

    // get funccall req from nodeagent
    pub fn OnFuncSvcCallReq(&self, req: func::FuncSvcCallReq) -> Result<()> {
        let packageId = PackageId {
            namespace: req.namespace.clone(),
            packageName: req.package_name.clone(),
        };
        let package = match PACKAGE_MGR.Get(&packageId) {
            Ok(p) => p,
            Err(_) => {
                let resp = func::FuncSvcCallResp {
                    id: req.id,
                    error: format!("FuncCall fail as package {:?} doesn't exist", &packageId),
                    resp: String::new(),
                    caller_node_id: req.caller_node_id.clone(),
                    caller_pod_id: req.caller_pod_id.clone(),
                    callee_node_id: req.callee_node_id.clone(),
                    callee_pod_id: req.callee_node_id.clone(),
                };
                return self.Send(FuncNodeMsg::FuncCallResp(resp));
            }
        };
        let inner = FuncCallInner {
            jobId: req.job_id.clone(),
            id: req.id.clone(),
            package: package,
            funcName: req.func_name.clone(),
            callerNodeId: req.caller_node_id.clone(),
            callerFuncPodId: req.caller_pod_id.clone(),
            calleeNodeId: Mutex::new(req.callee_node_id.clone()),
            calleeFuncPodId: Mutex::new(req.callee_pod_id.clone()),
            state: Mutex::new(FuncCallState::Scheduling(SystemTime::now())),
            parameters: req.parameters.clone(),
            priority: req.priority as usize,
            createTime: SystemTimeProto::FromTimestamp(&req.createtime.as_ref().unwrap()).ToSystemTime(),
        };

        let funcCall = FuncCall(Arc::new(inner));
        self.lock().unwrap().callerFuncCalls.insert(req.id.clone(), funcCall.clone());
        FUNC_SVC_MGR.lock().unwrap().OnNewFuncCall(&funcCall)?;
        
        return Ok(())
    }

    // get funccall response from nodeagent
    pub fn OnFuncSvcCallResp(&self, resp: func::FuncSvcCallResp) -> Result<()> {
        error!("OnFuncSvcCallResp ... {:?}", &resp);
        self.lock().unwrap().calleeFuncCalls.remove(&resp.id);
        let callerNode = FUNC_NODE_MGR.Get(&resp.caller_node_id)?;
        let pod = FUNC_POD_MGR.Get(&resp.callee_pod_id)?;
        *pod.state.lock().unwrap() = FuncPodState::Idle(SystemTime::now());
        callerNode.Send(FuncNodeMsg::FuncCallResp(resp))?;
        FUNC_SVC_MGR.lock().unwrap().OnFreePod(&pod)?;
        return Ok(())
    }

    // get new funcpod from nodeagent
    pub fn OnFuncPodConnReq(&self, req: func::FuncPodConnReq) -> Result<()> {
        let package = if !req.client_mode {
            let packageId = PackageId {
                namespace: req.namespace.clone(),
                packageName: req.package_name.clone(),
            };
            error!("OnFuncPodConnReq {:?} ...", &packageId);
            
            let package = match PACKAGE_MGR.Get(&packageId) {
                Ok(p) => {
                    let resp = func::FuncPodConnResp {
                        func_pod_id: req.func_pod_id.clone(),
                        error: String::new(),
                    };
                    self.Send(FuncNodeMsg::FuncPodConnResp(resp))?;
                    p
                }
                Err(_) => {
                    let resp = func::FuncPodConnResp {
                        func_pod_id: req.func_pod_id ,
                        error: format!("FuncCall fail as package {:?} doesn't exist", &packageId),
                    };
                    return self.Send(FuncNodeMsg::FuncPodConnResp(resp));
                }
            };

            Some(package)
        } else {
            let resp = func::FuncPodConnResp {
                func_pod_id: req.func_pod_id.clone(),
                error: String::new(),
            };
            self.Send(FuncNodeMsg::FuncPodConnResp(resp))?;
            None
        };

        let funcPodInner = FuncPodInner {
            podName: req.func_pod_id.clone(),
            package : package.clone(),
            node: self.clone(),
            clientMode: req.client_mode,
            state: Mutex::new(FuncPodState::Idle(SystemTime::now()))
        };

        let funcPod = FuncPod(Arc::new(funcPodInner));
        self.lock().unwrap().funcPods.insert(req.func_pod_id.clone(), funcPod.clone());
        FUNC_POD_MGR.Add(&funcPod);
        if !funcPod.clientMode {
            FUNC_SVC_MGR.lock().unwrap().OnFreePod(&funcPod)?;
        }
        
        return Ok(())
    }

    pub fn OnNodeDisconnected(&self) -> Result<()> {
        info!("node {} disconnected", self.NodeName());
        let funcPodIds : Vec<_> = self.lock().unwrap().funcPods.keys().cloned().collect(); 
        for funcPodId in funcPodIds {
            info!("node {} disconnected funcpod {} lost", self.NodeName(), &funcPodId);
            self.OnFuncPodDisconnReq(&funcPodId)?;
        }

        {
            let mut inner = self.lock().unwrap();
            loop {
                match inner.callerFuncCalls.pop_first() {
                    None => break,
                    Some((_, call)) => {
                        call.SetState(FuncCallState::Cancelling);
                    }
                }
            }
        }
        
        return Ok(())
    }

    // when a pod is disconnected
    pub fn OnFuncPodDisconnReq(&self, funcPodId: &str) -> Result<()> {
        let funcPod = match self.lock().unwrap().funcPods.remove(funcPodId) {
            None => {
                error!("OnFuncPodDisconnReq can't find pod {:?}", funcPodId);
                return Ok(())
            }
            Some(pod) => pod
        };
        FUNC_POD_MGR.Remove(funcPodId)?;
                
        match funcPod.state.lock().unwrap().clone() {
            FuncPodState::Idle(_) => (),
            FuncPodState::Running(callId) => {
                let funcCall = match self.lock().unwrap().calleeFuncCalls.remove(&callId) {
                    None => {
                        return Err(Error::CommonError(format!("OnFuncPodDisconnReq can't find funcall {}", callId)));
                    }
                    Some(c) => c
                };
                let callerNode = FUNC_NODE_MGR.Get(&funcCall.callerNodeId)?;
                let resp = func::FuncSvcCallResp {
                    id: funcCall.id.clone(),
                    error: format!("funcpod {} disconnect ", funcPodId),
                    resp: String::new(),
                    caller_node_id: funcCall.callerNodeId.clone(),
                    caller_pod_id: funcCall.callerFuncPodId.clone(),
                    callee_node_id: funcCall.calleeNodeId.lock().unwrap().clone(),
                    callee_pod_id: funcCall.calleeFuncPodId.lock().unwrap().clone(),
                };
                callerNode.Send(FuncNodeMsg::FuncCallResp(resp))?;
            }
            _ => {}
        }

        *funcPod.state.lock().unwrap() = FuncPodState::Dead;
        if !funcPod.clientMode {
            FUNC_SVC_MGR.lock().unwrap().OnPodExit(&funcPod)?;
        }
        
        return Ok(())
    }

    // get message from nodeagent
    pub async fn ProcessNodeAgentMsg(&self, msg: func::FuncSvcMsg) -> Result<()> {
        error!("ProcessNodeAgentMsg from node {} msg {:?}", self.NodeName(), &msg);
        let body = match msg.event_body {
            None => panic!("ProcessNodeAgentMsg get none eventbody"),
            Some(b) => b,
        };
        match body {
            func::func_svc_msg::EventBody::FuncSvcCallReq(req) => {
                return self.OnFuncSvcCallReq(req);
            } 
            func::func_svc_msg::EventBody::FuncSvcCallResp(resp) => {
                return self.OnFuncSvcCallResp(resp);
            } 
            func::func_svc_msg::EventBody::FuncPodConnReq(req) => {
                return self.OnFuncPodConnReq(req);
            } 
            func::func_svc_msg::EventBody::FuncPodDisconnReq(req) => {
                return self.OnFuncPodDisconnReq(&req.func_pod_id);
            } 
            _ => {
                error!("didn't get FuncAgentRegisterReq message instead {:?}", body);
                return Err(Error::CommonError(format!("didn't get FuncAgentRegisterReq message instead {:?}", body)));
            }
        }
    }

    // get funccall req from another func node
    pub fn OnFuncCall(&self, call: FuncCall, tx: &mpsc::Sender<SResult<func::FuncSvcMsg, Status>>) -> Result<()> {
        let req = func::FuncSvcCallReq {
            job_id: call.jobId.clone(),
            id: call.id.clone(),
            namespace: call.package.Namespace(),
            package_name: call.package.Name(),
            func_name: call.funcName.clone(),
            parameters: call.parameters.clone(),
            priority: call.priority as u64,
            createtime: Some(SystemTimeProto::FromSystemTime(call.createTime).ToTimeStamp()),
            caller_node_id: call.callerNodeId.clone(),
            caller_pod_id: call.callerFuncPodId.clone(),
            callee_node_id: call.calleeNodeId.lock().unwrap().clone(),
            callee_pod_id: call.calleeFuncPodId.lock().unwrap().clone(),
        };

        self.SendToNodeAgent(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallReq(req))
        }, tx);

        self.lock().unwrap().calleeFuncCalls.insert(call.id.clone(), call.clone());

        return Ok(())
    }

    // get funccall resp from another func node
    pub fn OnFuncCallResp(&self, resp: func::FuncSvcCallResp, tx: &mpsc::Sender<SResult<func::FuncSvcMsg, Status>>) -> Result<()> {
        self.lock().unwrap().calleeFuncCalls.remove(&resp.id);
        self.SendToNodeAgent(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallResp(resp))
        }, tx);

        return Ok(())
    }

    // get message from another funcnode
    pub fn OnNodeMsg(&self, msg: FuncNodeMsg, tx: &mpsc::Sender<SResult<func::FuncSvcMsg, Status>>) -> Result<()> {
        match msg {
            FuncNodeMsg::FuncCall(funcCall) => {
                return self.OnFuncCall(funcCall, tx);
            }
            FuncNodeMsg::FuncCallResp(resp) => {
                self.OnFuncCallResp(resp, tx)?;
                return Ok(())
            }
            FuncNodeMsg::FuncPodConnResp(resp) => {
                self.SendToNodeAgent(func::FuncSvcMsg {
                    event_body: Some(func::func_svc_msg::EventBody::FuncPodConnResp(resp))
                }, tx);
                return Ok(())
            }
        }
    }

    // send funccall resp from the nodeagent
    pub fn SendToNodeAgent(&self, msg: func::FuncSvcMsg, tx: &mpsc::Sender<SResult<func::FuncSvcMsg, Status>>) {
        error!("SendToNodeAgent to node {} msg {:?}", self.NodeName(), &msg);
        match tx.try_send(Ok(msg)) {
            Ok(()) => (),
            Err(_) => {
                error!("FuncNode send message fail, disconnecting...");
                return;
            }
        }
    }

    pub async fn StartProcess(
        &self, 
        regReq: func::FuncAgentRegisterReq, 
        rx: Streaming<func::FuncSvcMsg>, 
        tx: mpsc::Sender<SResult<func::FuncSvcMsg, Status>>,
        ready: Arc<Notify>
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

        ready.notify_waiters();
        let mut internMsgRx = self.lock().unwrap().internMsgRx.take().expect("StartProcess doesn't get internal rx channel");
        defer!(self.SetState(FuncNodeState::WaitingConn));
        let mut rx = rx;
        let closeNotify = self.lock().unwrap().closeNotify.clone();

        self.SetState(FuncNodeState::Running);

        {
            // OnNodeRegiste needs to access another node, 
            // to avoid deadlock, use global mutex to serialize the process 
            let _l = FUNC_NODE_MGR.initlock.lock().await;                
            self.OnNodeRegister(regReq)?;
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
                            self.OnNodeMsg(msg, &tx)?;
                        }
                    }
                    
                }
                msg = rx.message() => {
                    let msg : func::FuncSvcMsg = match msg {
                        Err(e) => {
                            error!("FuncNode {} disconnected get error message {:?}", self.NodeName(), e);
                            self.OnNodeDisconnected()?;
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

    pub fn NodeName(&self) -> String {
        return self.lock().unwrap().nodeName.clone();
    }

    pub fn AddPod(&self, _pod: &FuncPod) -> Result<()> {
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
        let ret = Self {
            initlock: TMutex::new(()),
            nodes: Mutex::new(BTreeMap::new()),
        };

        return ret;
    }

    pub fn Get(&self, nodeName: &str) -> Result<FuncNode> {
        match self.nodes.lock().unwrap().get(nodeName) {
            None => return Err(Error::ENOENT(format!("can't get node {}", nodeName))),
            Some(n) => Ok(n.clone()),
        }
    }

    pub fn Insert(&self, nodeName: &str, node: &FuncNode) {
        self.nodes.lock().unwrap().insert(nodeName.to_owned(), node.clone());
    }
}