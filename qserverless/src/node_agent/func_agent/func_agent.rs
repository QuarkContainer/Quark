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
use std::sync::{Mutex};
use std::result::Result as SResult;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status};
use tokio::sync::{oneshot, mpsc};

use qobjs::{common::*, func::{self, func_agent_msg::EventBody}};

use super::funcpod::FuncPod;
use super::funcpod_mgr::FuncInstMgr;

#[derive(Debug)]
pub struct FuncCall {
    pub id: String,
    pub funcInstanceId: String,
    pub namespace: String,
    pub package: String,
    pub funcName: String,
    pub parameters: String,
}

pub struct FuncCallContext {
    pub respChann: oneshot::Sender<Result<String>>,
}

pub struct FuncAgent {
    pub callContexts: Mutex<BTreeMap<String, FuncCallContext>>,
    pub funcInstMgr: FuncInstMgr,
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
        self.funcInstMgr.AddInstance(&instanceId, &instance)?;
        return Ok(());
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
    
    async fn func_call(
        &self,
        request: tonic::Request<func::FuncAgentCallReq>,
    ) -> SResult<tonic::Response<func::FuncAgentCallResp>, tonic::Status> {
        let req = request.into_inner();
        let funcCall = FuncCall {
            id: uuid::Uuid::new_v4().to_string(),
            funcInstanceId: String::new(),
            namespace: req.namespace.clone(),
            package: req.package.clone(),
            funcName: req.func_name.clone(),
            parameters: req.parameters.clone(),
        };
        
        let resp = match self.FuncCall(&funcCall).await {
            Err(e) => func::FuncAgentCallResp {
                error: format!("{:?}", e),
                resp: String::new(),
            } ,
            Ok(resp) => func::FuncAgentCallResp {
                error: String::new(),
                resp: resp,
            }
        };

        return Ok(Response::new(resp));
    }
}