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
use std::sync::{Mutex, Arc};
use std::result::Result as SResult;
use std::time::SystemTime;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status};
use tokio::sync::{oneshot, mpsc};
use core::ops::Deref;

use qobjs::{common::*, func::{self, func_agent_msg::EventBody}};

use crate::FUNC_SVC_CLIENT;

use super::funcpod::FuncPod;
use super::funcpod_mgr::FuncPodMgr;

#[derive(Debug)]
pub struct FuncCallInner {
    pub id: String,
    pub callerFuncPodId: String,
    pub calleeFuncPodId: String,
    pub namespace: String,
    pub package: String,
    pub funcName: String,
    pub priority: usize,
    pub parameters: String,
}


#[derive(Debug, Clone)]
pub struct FuncCall(Arc<FuncCallInner>);

impl Deref for FuncCall {
    type Target = Arc<FuncCallInner>;

    fn deref(&self) -> &Arc<FuncCallInner> {
        &self.0
    }
}

#[derive(Debug)]
pub struct FuncCallContext {
    pub respChann: oneshot::Sender<Result<String>>,
}

#[derive(Debug, Default)]
pub struct FuncAgent {
    pub nodeId: String,
    pub callContexts: Mutex<BTreeMap<String, FuncCallContext>>,
    pub funcPodMgr: FuncPodMgr,
    // func instance id to funcCall
    pub callerCalls: Mutex<BTreeMap<String, FuncCall>>,
    // func instance id to funcCall
    pub calleeCalls: Mutex<BTreeMap<String, FuncCall>>,
}

impl FuncAgent {
    pub fn CallResponse(&self, callId: &str, response: Result<String>) -> Result<()> {
        match self.callContexts.lock().unwrap().remove(callId) {
            None => return Err(Error::CommonError(format!("get unepxecet callid {}", callId))),
            Some(context) => {
                context.respChann.send(response).expect("CallResponse fail...");
            }
        };

        return Ok(())
    }

    pub async fn FuncCall(&self, funcCall: &FuncCall) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        self.callContexts.lock().unwrap().insert(funcCall.id.clone(), FuncCallContext {
            respChann: tx,
        });

        let ret = rx.await.expect("get unepxecet oneshot result");
        return ret;
    }

    pub fn OnNewConnection(
        &self, 
        registerMsg: &func::FuncPodRegisterReq, 
        stream: tonic::Streaming<func::FuncAgentMsg>,
        agentTx: mpsc::Sender<SResult<func::FuncAgentMsg, Status>>) 
    -> Result<()> {
        let instanceId = registerMsg.instance_id.clone();
        let instance = FuncPod::New(registerMsg, stream, agentTx)?;
        self.funcPodMgr.AddPod(&instanceId, &instance)?;
        return Ok(());
    }

    pub fn OnFuncAgentCallReq(&self, callerId: &str, req: func::FuncAgentCallReq) -> Result<()> {
        let inner = FuncCallInner {
            id: req.id.clone(),
            callerFuncPodId: callerId.to_string(),
            calleeFuncPodId: String::new(), // not assigned,
            namespace: req.namespace.clone(),
            package: req.package.clone(),
            funcName: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            priority: req.priority as usize,
        };

        let funcCall = FuncCall(Arc::new(inner));
        self.callerCalls.lock().unwrap().insert(req.id.clone(), funcCall);

        let createTime = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
        let req = func::FuncSvcCallReq {
            id: req.id.clone(),
            namespace: req.namespace.to_string(),
            package_name: req.package.to_string(),
            func_name: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            priority: req.priority,
            createtime: Some(func::Timestamp {
                top:  (createTime >> 64) as u64,
                bottom: createTime as u64,
            }),
            callee_node_id: self.nodeId.clone(),
            calller_pod_id: callerId.to_string(),
            caller_node_id: String::new(),
            callee_pod_id: String::new(),
        };

        FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallReq(req))
        })?;

        return Ok(())
    }

    pub fn OnFuncSvcCallReq(&self, _req: func::FuncSvcCallReq) -> Result<()> {
        unimplemented!();
    }

    pub async fn OnFuncAgentMsg(&self, callerId: &str, msg: func::FuncAgentMsg) -> Result<()> {
        let body = match msg.event_body {
            None => return Err(Error::EINVAL),
            Some(b) => b,
        };

        match body {
            EventBody::FuncAgentCallReq(msg) => {
                self.OnFuncAgentCallReq(callerId, msg)?;
            }
            _ => {

            }
        };

        return Ok(())
    }
}

#[tonic::async_trait]
impl func::func_agent_service_server::FuncAgentService for FuncAgent {
    type StreamProcessStream = ReceiverStream<SResult<func::FuncAgentMsg, Status>>;
    
    async fn stream_process(
        &self,
        request: tonic::Request<tonic::Streaming<func::FuncAgentMsg>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        let mut stream = request.into_inner();
        let msg = stream.message().await.unwrap().unwrap();
        let body = match msg.event_body {
            None => return Err(Status::aborted("get non event_body")),
            Some(body) => body,
        };

        let registerMsg = match body {
            EventBody::FuncPodRegisterReq(req) => {
                req
            }
            _  => {
                return Err(Status::aborted("first message should be FuncPodRegisterResp"));
            }
        };

        let (tx, rx) = mpsc::channel(30);
        self.OnNewConnection(&registerMsg, stream, tx).unwrap();
        return Ok(Response::new(ReceiverStream::new(rx)));
    }
}