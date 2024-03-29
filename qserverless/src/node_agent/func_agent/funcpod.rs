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
use std::sync::{Arc, atomic::AtomicBool};
use core::ops::Deref;
use std::time::SystemTime;
use qobjs::utility::SystemTimeProto;
use tokio::sync::{mpsc, Notify};
use std::result::Result as SResult;

use qobjs::func;
use qobjs::common::*;
use qobjs::func::func_agent_msg::EventBody;

use crate::FUNC_SVC_CLIENT;
use crate::blobstore::blob_session::BlobSession;

use super::func_agent::{FuncAgent, FuncCall, FuncCallContext, FuncCallInner};


#[derive(Debug, Clone)]
pub enum funcPodState {
    Idle,
    Running(FuncCall), // handling FuncCallId
}

impl funcPodState {
    pub fn State(&self) -> func::FuncPodState {
        match self {
            Self::Idle => func::FuncPodState::Idle,
            Self::Running(_) => func::FuncPodState::Running,
        }
    }

    pub fn FuncCallId(&self) -> String {
        match self {
            Self::Idle => String::new(),
            Self::Running(funcCall) => funcCall.id.clone(),
        }
    }
}

#[derive(Debug)]
pub struct FuncPodInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub nodeId: String,
    pub funcPodId: String,
    pub namespace: String,
    pub packageName: String,
    pub state: Mutex<funcPodState>,
    pub clientMode: bool,

    pub agentChann: mpsc::Sender<SResult<func::FuncAgentMsg, tonic::Status>>,

    // func call id to funcCall
    pub callerCalls: Mutex<BTreeMap<String, FuncCall>>,
    pub callContexts: Mutex<BTreeMap<String, FuncCallContext>>,
    
    pub blobSession: BlobSession,
    pub funcAgent: FuncAgent,
}

#[derive(Debug, Clone)]
pub struct FuncPod(pub Arc<FuncPodInner>);

impl Deref for FuncPod {
    type Target = Arc<FuncPodInner>;

    fn deref(&self) -> &Arc<FuncPodInner> {
        &self.0
    }
}

impl FuncPod {
    pub fn New(
        funcAgent: &FuncAgent,
        registerMsg: &func::FuncPodRegisterReq, 
        stream: tonic::Streaming<func::FuncAgentMsg>,
        agentTx: mpsc::Sender<SResult<func::FuncAgentMsg, tonic::Status>>) 
    -> Result<Self> {
        let funcPodId = registerMsg.func_pod_id.clone();
        let inner = FuncPodInner {
            nodeId: funcAgent.nodeId.clone(),
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            funcPodId: funcPodId.clone(),
            namespace: registerMsg.namespace.to_string(),
            packageName: registerMsg.package_name.to_string(),
            clientMode: registerMsg.client_mode,
            state: Mutex::new(funcPodState::Idle),
            agentChann: agentTx,
            callerCalls: Mutex::new(BTreeMap::new()),
            callContexts: Mutex::new(BTreeMap::new()),
            blobSession: BlobSession::New(&funcAgent.blobSvcAddr),
            funcAgent: funcAgent.clone(),
        };
        let instance = FuncPod(Arc::new(inner));
        let clone = instance.clone();
        tokio::spawn(async move {
            clone.Process(stream).await.unwrap();
        });

        return Ok(instance);
    }

    // get funcagentcallreq from funcpod
    pub fn OnFuncAgentCallReq(&self, callerFuncPodId: &str, req: func::FuncAgentCallReq) -> Result<()> {
        let createTime = SystemTime::now();
        let inner = FuncCallInner {
            jobId: req.job_id.clone(),
            id: req.id.clone(),
            callerNodeId: self.nodeId.clone(),
            callerFuncPodId: callerFuncPodId.to_string(),
            calleeNodeId: String::new(),
            calleeFuncPodId: String::new(), // not assigned,
            namespace: req.namespace.clone(),
            packageName: req.package_name.clone(),
            funcName: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            callerFuncCallId: req.caller_func_id.clone(),
            priority: req.priority as usize,
            createTime: createTime,
            callType: req.call_type,
        };

        let funcCall = FuncCall(Arc::new(inner));
        self.callerCalls.lock().unwrap().insert(req.id.clone(), funcCall);
        
        let protoTime = SystemTimeProto::FromSystemTime(createTime);
        let req = func::FuncSvcCallReq {
            job_id: req.job_id.clone(),
            id: req.id.clone(),
            namespace: req.namespace.to_string(),
            package_name: req.package_name.to_string(),
            func_name: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            caller_func_id: req.caller_func_id.clone(),
            priority: req.priority,
            createtime: Some(protoTime.ToTimeStamp()),
            caller_node_id: self.nodeId.clone(),
            caller_pod_id: callerFuncPodId.to_string(),
            callee_node_id: String::new(),
            callee_pod_id: String::new(),
            call_type: req.call_type,
        };

        FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallReq(req))
        })?;

        return Ok(())
    }

    pub fn State(&self) -> funcPodState {
        return self.state.lock().unwrap().clone();
    }

    pub fn SetState(&self, state: funcPodState) {
        *self.state.lock().unwrap() = state;
    }

    pub fn ToGrpcType(&self) -> func::FuncPodStatus {
        return func::FuncPodStatus {
            func_pod_id: self.funcPodId.clone(),
            namespace: self.namespace.clone(),
            package_name: self.packageName.clone(),
            state: self.State().State() as i32,
            func_call_id: self.State().FuncCallId(),
            client_mode: self.clientMode,
        }
    }

    pub fn Send(&self, msg: func::FuncAgentMsg) -> Result<()> {
        match self.agentChann.try_send(Ok(msg)) {
            Ok(()) => return Ok(()),
            Err(_) => return Err(Error::MpscSendFail),
        }
    }

    pub async fn OnBlobOpenReq(&self, msgId: u64, msg: func::BlobOpenReq) -> Result<()> {
        let resp = match self.blobSession.Open(&msg.svc_addr, &self.namespace, &msg.name).await {
            Ok((id, b)) => {
                let inner = b.lock().unwrap();
                let resp = func::BlobOpenResp {
                    id: id,
                    namespace: inner.namespace.clone(),
                    name: inner.name.clone(),
                    size: inner.size as u64,
                    checksum: inner.checksum.clone(),
                    create_time: Some(SystemTimeProto::FromSystemTime(inner.createTime).ToTimeStamp()),
                    last_access_time: Some(SystemTimeProto::FromSystemTime(inner.lastAccessTime).ToTimeStamp()),
                    error: String::new(),
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobOpenResp(resp))
                }
            }
            Err(e) => {
                let resp = func::BlobOpenResp {
                    error: format!("{:?}", e),
                    ..Default::default()
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobOpenResp(resp))
                }
            }
        };

        match self.Send(resp) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }

    pub async fn OnBlobDeleteReq(&self, msgId: u64, msg: func::BlobDeleteReq) -> Result<()> {
        let resp = match self.blobSession.Delete(&msg.svc_addr, &self.namespace, &msg.name).await {
            Ok(()) => {
                let resp = func::BlobDeleteResp {
                    error: String::new(),
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobDeleteResp(resp))
                }
            }
            Err(e) => {
                let resp = func::BlobDeleteResp {
                    error: format!("{:?}", e),
                    ..Default::default()
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobDeleteResp(resp))
                }
            }
        };

        match self.Send(resp) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }

    pub async fn OnBlobReadReq(&self, msgId: u64, msg: func::BlobReadReq) -> Result<()> {
        let mut buf = Vec::with_capacity(msg.len as usize);
        buf.resize(msg.len as usize, 0u8);

        let resp = match self.blobSession.Read(msg.id, msg.len).await {
            Ok(buf) => {
                let resp = func::BlobReadResp {
                    data: buf,
                    error: String::new()
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobReadResp(resp))
                }
            }
            Err(e) => {
                let resp = func::BlobReadResp {
                    data: Vec::new(),
                    error: format!("{:?}", e),
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobReadResp(resp))
                }
            }
        };

        match self.Send(resp) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }

    pub async fn OnBlobSeekReq(&self, msgId: u64, msg: func::BlobSeekReq) -> Result<()> {
        let resp = match self.blobSession.Seek(msg.id, msg.seek_type, msg.pos).await {
            Ok(offset) => {
                let resp = func::BlobSeekResp {
                    offset: offset,
                    error: String::new()
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobSeekResp(resp))
                }
            }
            Err(e) => {
                let resp = func::BlobSeekResp {
                    offset: 0,
                    error: format!("{:?}", e),
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobSeekResp(resp))
                }
            }
        };

        match self.Send(resp) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }
    
    pub async fn OnBlobCloseReq(&self, msgId: u64, msg: func::BlobCloseReq) -> Result<()> {
        let resp = match self.blobSession.Close(msg.id).await {
            Ok(()) => {
                let resp = func::BlobCloseResp {
                    error: String::new()
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobCloseResp(resp))
                }
            }
            Err(e) => {
                let resp = func::BlobCloseResp {
                    error: format!("{:?}", e),
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobCloseResp(resp))
                }
            }
        };

        match self.Send(resp) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }

    pub fn OnBlobCreateReq(&self, msgId: u64, msg: func::BlobCreateReq) -> Result<()> {
        let resp = match self.blobSession.Create(&self.namespace, &msg.name) {
            Ok(id) => {
                let resp = func::BlobCreateResp {
                    id: id,
                    svc_addr: self.funcAgent.blobSvcAddr.clone(),
                    error: String::new()
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobCreateResp(resp))
                }
            }
            Err(e) => {
                let resp = func::BlobCreateResp {
                    id: 0,
                    svc_addr: self.funcAgent.blobSvcAddr.clone(),
                    error: format!("{:?}", e),
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobCreateResp(resp))
                }
            }
        };

        match self.Send(resp) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }

    pub async fn OnBlobWriteReq(&self, msgId: u64, msg: func::BlobWriteReq) -> Result<()> {
        let resp = match self.blobSession.Write(msg.id, &msg.data).await {
            Ok(()) => {
                let resp = func::BlobWriteResp {
                    error: String::new()
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobWriteResp(resp))
                }
            }
            Err(e) => {
                let resp = func::BlobWriteResp {
                    error: format!("{:?}", e),
                };
                func::FuncAgentMsg {
                    msg_id: msgId,
                    event_body: Some(func::func_agent_msg::EventBody::BlobWriteResp(resp))
                }
            }
        };

        match self.Send(resp) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }

    // get funcMsg from pod
    pub async fn OnFuncMsg(&self, msg: func::FuncMsg) -> Result<()> {
        let mut msg = msg;
        msg.src_node_id = self.nodeId.clone();
        msg.src_pod_id = self.funcPodId.clone();
        match self.State() {
            funcPodState::Running(funccall) => {
                msg.src_func_id = funccall.id.clone()
            }
            _ => {
                // it is caller pod
            }
        };

        //error!("OnFuncMsg msg is {:#?}", &msg);
        FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncMsg(msg))
        })?;

        return Ok(())
    }

    pub async fn OnFuncPodMsg(&self, funcPodId: &str, msg: func::FuncAgentMsg) -> Result<()> {
        //error!("OnFuncPodMsg msg is {:#?}", &msg);
        let body = match msg.event_body {
            None => return Err(Error::EINVAL(format!("OnFuncPodMsg None event_body"))),
            Some(b) => b,
        };

        let msgId = msg.msg_id;

        match body {
            EventBody::FuncAgentCallReq(msg) => {
                self.OnFuncAgentCallReq(funcPodId, msg)?;
            }
            EventBody::FuncAgentCallResp(msg) => {
                let call = match self.State() {
                    funcPodState::Idle => {
                        error!("OnFuncAgentCallResp doesn't find funcall id {}", &msg.id);
                        return Ok(())
                    }
                    funcPodState::Running(funccall) => {
                        funccall
                    }
                };
                
                self.SetState(funcPodState::Idle);
                
                let resp = func::FuncSvcCallResp {
                    id: msg.id,
                    res: msg.res,
                    caller_node_id: call.callerNodeId.clone(),
                    caller_pod_id: call.callerFuncPodId.clone(),
                    callee_node_id: self.nodeId.clone(),
                    callee_pod_id: funcPodId.to_owned(),
                };
        
                FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
                    event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallResp(resp))
                })?;
        

                //self.funcAgent.OnFuncAgentCallResp(funcPodId, msg)?;
            }
            EventBody::FuncAgentCallAck(msg) => {
                let call = match self.State() {
                    funcPodState::Idle => {
                        error!("OnFuncAgentCallResp doesn't find funcall id {}", &msg.id);
                        return Ok(())
                    }
                    funcPodState::Running(funccall) => {
                        funccall
                    }
                };

                let ack = func::FuncSvcCallAck {
                    id: msg.id,
                    error: msg.error,
                    caller_node_id: call.callerNodeId.clone(),
                    caller_pod_id: call.callerFuncPodId.clone(),
                    callee_node_id: self.nodeId.clone(),
                    callee_pod_id: funcPodId.to_owned(),
                };

                FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
                    event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallAck(ack))
                })?;
            }
            EventBody::BlobOpenReq(msg) => {
                self.OnBlobOpenReq(msgId, msg).await?;
            }
            EventBody::BlobDeleteReq(msg) => {
                self.OnBlobDeleteReq(msgId, msg).await?;
            }
            EventBody::BlobReadReq(msg) => {
                self.OnBlobReadReq(msgId, msg).await?;
            }
            EventBody::BlobSeekReq(msg) => {
                self.OnBlobSeekReq(msgId, msg).await?;
            }
            EventBody::BlobCreateReq(msg) => {
                self.OnBlobCreateReq(msgId, msg)?;
            }
            EventBody::BlobWriteReq(msg) => {
                self.OnBlobWriteReq(msgId, msg).await?;
            }
            EventBody::BlobCloseReq(msg) => {
                self.OnBlobCloseReq(msgId, msg).await?;
            }
            EventBody::FuncMsg(msg) => {
                self.OnFuncMsg(msg).await?;
            }
            m => {
                error!("get unexpected msg {:?}", m);
            }
        };

        return Ok(())
    }

    pub async fn Process(
        &self, 
        stream: tonic::Streaming<func::FuncAgentMsg>
    ) -> Result<()> {
        let closeNotify = self.closeNotify.clone();
        let mut stream = stream;
        
        loop {
            tokio::select! {
                _ = closeNotify.notified() => {
                    break;
                }
                msg = stream.message() => {
                    let msg : func::FuncAgentMsg = match msg {
                        Err(e) => {
                            error!("FuncPod {} get error message {:?} disconnect...", &self.funcPodId, e);
                            match self.funcAgent.OnFuncPodDisconnect(&self.funcPodId) {
                                Err(e) => {
                                    error!("FuncPod {} disconnect get error message {:?} ", &self.funcPodId, e);
                                }
                                Ok(()) => ()
                            }
                            break;
                        }
                        Ok(m) => {
                            match m {
                                None => {
                                    error!("FuncPod get None message");
                                    break;
                                }
                                Some(m) => m,
                            }
                        }
                    };
                    
                    self.OnFuncPodMsg(&self.funcPodId, msg).await?;
                }
            }
        }

        return Ok(())
    }
}
