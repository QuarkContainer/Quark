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
use qobjs::func::func_agent_service_server::FuncAgentServiceServer;
use qobjs::types::*;
use qobjs::utility::SystemTimeProto;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status};
use tokio::sync::{oneshot, mpsc};
use core::ops::Deref;

use qobjs::{common::*, func::{self, func_agent_msg::EventBody}};

use crate::{FUNC_SVC_CLIENT, NODEAGENT_CONFIG, CADVISOR_PROVIDER, NODE_CONFIG};

use super::funcpod::{FuncPod, funcPodState};
use super::funcpod_mgr::FuncPodMgr;

#[derive(Debug)]
pub struct FuncCallInner {
    pub jobId: String,
    pub id: String,
    pub namespace: String,
    pub packageName: String,
    pub funcName: String,
    pub callerFuncCallId: String,
    pub priority: usize,
    pub parameters: String,
    pub createTime: SystemTime,

    pub callerNodeId: String,
    pub callerFuncPodId: String,
    pub calleeNodeId: String,
    pub calleeFuncPodId: String,
    pub callType: i32,
    
}

impl FuncCallInner {
    pub fn ToGrpcType(&self) -> func::FuncSvcCallReq {
        return func::FuncSvcCallReq {
            job_id: self.jobId.clone(),
            id: self.id.clone(),
            namespace: self.namespace.clone(),
            package_name: self.packageName.clone(),
            func_name: self.funcName.clone(),
            caller_func_id: self.callerFuncCallId.clone(),
            priority: self.priority as u64,
            parameters: self.parameters.clone(),
            createtime: Some(SystemTimeProto::FromSystemTime(self.createTime).ToTimeStamp()),
            caller_node_id: self.callerNodeId.clone(),
            caller_pod_id: self.callerFuncPodId.clone(),
            callee_node_id: self.calleeNodeId.clone(),
            callee_pod_id: self.calleeFuncPodId.clone(),
            call_type: self.callType,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FuncCall(pub Arc<FuncCallInner>);

impl Deref for FuncCall {
    type Target = Arc<FuncCallInner>;

    fn deref(&self) -> &Arc<FuncCallInner> {
        &self.0
    }
}

#[derive(Debug)]
pub struct FuncCallContext {
    pub respChann: oneshot::Sender<SResult<String, String>>,
}

#[derive(Debug, Default)]
pub struct FuncAgentInner {
    pub nodeId: String,
    pub blobSvcAddr: String,
    pub callContexts: Mutex<BTreeMap<String, FuncCallContext>>,
    pub funcPodMgr: FuncPodMgr,
    // func instance id to funcCall
    pub callerCalls: Mutex<BTreeMap<String, FuncCall>>,
    // func instance id to funcCall
    pub calleeCalls: Mutex<BTreeMap<String, FuncCall>>,
    pub systemResource: Resource,
    pub reservedResource: Resource,
    pub allocableResource: Resource,
}

#[derive(Debug, Default, Clone)]
pub struct FuncAgent(Arc<FuncAgentInner>);

impl Deref for FuncAgent {
    type Target = Arc<FuncAgentInner>;

    fn deref(&self) -> &Arc<FuncAgentInner> {
        &self.0
    }
}

impl FuncAgent {
    pub fn New(nodeId: &str, blobSvcAddr: &str) -> Self {
        let info = CADVISOR_PROVIDER.get().unwrap().CAdvisorInfo();
        let systemResource = Resource::New(
            info.machineInfo.MemoryCapacity, 
            info.machineInfo.NumCores as _
        );

        let reservedResource = Resource::New(
            NODE_CONFIG.ReservedMem,
            NODE_CONFIG.ReservedCpuCores
        );

        assert!(systemResource.Fullfil(&reservedResource));
        let allocableResource = systemResource - reservedResource;

        let inner = FuncAgentInner {
            nodeId: nodeId.to_string(),
            blobSvcAddr: blobSvcAddr.to_string(),
            systemResource: systemResource,
            reservedResource: reservedResource,
            allocableResource: allocableResource,
            ..Default::default()
        };

        return Self(Arc::new(inner))
    }

    pub fn ToGrpcType(&self) -> func::FuncAgentRegisterReq {
        let mut callerCalls = Vec::new();
        let mut calleeCalls = Vec::new();
        for (_, call) in self.callerCalls.lock().unwrap().iter() {
            callerCalls.push(call.ToGrpcType());
        }
        for (_, call) in self.calleeCalls.lock().unwrap().iter() {
            calleeCalls.push(call.ToGrpcType());
        }
        return func::FuncAgentRegisterReq {
            node_id: self.nodeId.clone(),
            caller_calls: callerCalls,
            callee_calls: calleeCalls,
            func_pods: self.funcPodMgr.ToGrpcType(),
            resource: Some(func::Resource {
                mem: self.allocableResource.mem as _,
                cpu: self.allocableResource.cpu as _,
            })
        }
    }

    // when funcagent get response from funcsvc for direct call
    pub fn CallResponse(&self, funcCallId: &str, response: SResult<String, String>) -> Result<()> {
        match self.callContexts.lock().unwrap().remove(funcCallId) {
            None => return Err(Error::CommonError(format!("get unepxecet callid {}", funcCallId))),
            Some(context) => {
                context.respChann.send(response).expect("CallResponse fail...");
            }
        };
    
        return Ok(())
    }

    pub async fn FuncCall(&self, funcCallReq: func::FuncAgentCallReq) -> SResult<String, String> {
        let (tx, rx) = oneshot::channel();
        let id = funcCallReq.id.clone();
        self.callContexts.lock().unwrap().insert(id.clone(), FuncCallContext {
            respChann: tx,
        });

        // if callerId is "", it is from grpc call directly
        match self.OnFuncAgentCallReq("", funcCallReq) {
            Ok(()) => (),
            Err(e) => {
                self.callContexts.lock().unwrap().remove(&id);
                return Err(format!("{:?}", e));
            }
        }

        let ret = rx.await.expect("get unepxecet oneshot result");
        
        return ret;
    }

    pub fn OnNewConnection(
        &self, 
        registerMsg: &func::FuncPodRegisterReq, 
        stream: tonic::Streaming<func::FuncAgentMsg>,
        agentTx: mpsc::Sender<SResult<func::FuncAgentMsg, Status>>) 
    -> Result<()> {
        let funcPodId = registerMsg.func_pod_id.clone();
        let funcPod = FuncPod::New(&self, registerMsg, stream, agentTx)?;
        self.funcPodMgr.AddPod(&funcPodId, &funcPod)?;

        let msg = func::FuncPodConnReq {
            func_pod_id: registerMsg.func_pod_id.clone(),
            namespace: registerMsg.namespace.clone(),
            package_name: registerMsg.package_name.clone(),
            client_mode: registerMsg.client_mode,
        };

        FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncPodConnReq(msg))
        })?;

        return Ok(());
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

    pub fn OnFuncPodDisconnect(&self, funcPodId: &str) -> Result<()> {
        let pod = self.funcPodMgr.RemovePod(funcPodId)?;

        match &pod.State() {
            funcPodState::Idle => (),
            funcPodState::Running(calleeFuncCall) => {
                let resp: func::FuncSvcCallResp = func::FuncSvcCallResp {
                    id: calleeFuncCall.id.clone(),
                    error: format!("func pod {} unexpected disconnect  ... maybe network issue", funcPodId),
                    resp: String::new(),
                    caller_node_id: calleeFuncCall.callerNodeId.clone(),
                    caller_pod_id: calleeFuncCall.callerFuncPodId.clone(),
                    callee_node_id: calleeFuncCall.calleeNodeId.clone(),
                    callee_pod_id: calleeFuncCall.calleeFuncPodId.clone(),
                };
        
                FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
                    event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallResp(resp))
                })?;
            }
        }

        let req = func::FuncPodDisconnReq {
            func_pod_id: funcPodId.to_string(),
        };

        FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncPodDisconnReq(req))
        })?;

        return Ok(())
    }

    // get funcCall msg from func_svc, forward it to the target funcpod
    pub fn OnFuncSvcCallReq(&self, req: func::FuncSvcCallReq) -> Result<()> {
        let funcPod = self.funcPodMgr.GetPod(&req.callee_pod_id)?;

        let createTimeProto = SystemTimeProto::FromTimestamp(req.createtime.as_ref().unwrap());

        let inner = FuncCallInner {
            jobId: req.job_id.clone(),
            id: req.id.clone(),
            callerNodeId: req.caller_node_id.clone(),
            callerFuncPodId: req.caller_pod_id.clone(),
            calleeNodeId: req.callee_node_id.clone(),
            calleeFuncPodId: req.callee_pod_id, // not assigned,
            namespace: req.namespace.clone(),
            packageName: req.package_name.clone(),
            funcName: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            callerFuncCallId: req.caller_func_id.clone(),
            priority: req.priority as usize,
            createTime: createTimeProto.ToSystemTime(),
            callType: req.call_type,
        }; 

        let funcCall = FuncCall(Arc::new(inner));
        //self.calleeCalls.lock().unwrap().insert(req.id.clone(), funcCall);

        let req = func::FuncAgentCallReq {
            job_id: req.job_id.clone(),
            id: req.id.clone(),
            namespace: req.namespace.clone(),
            package_name: req.package_name.clone(),
            func_name: req.func_name.clone(),
            parameters: req.parameters.clone(),
            priority: req.priority,
            caller_node_id: req.caller_node_id.clone(),
            caller_pod_id: req.caller_pod_id.clone(),
            caller_func_id: req.caller_func_id.clone(),
            call_type: req.call_type
        };

        funcPod.SetState(funcPodState::Running(funcCall));

        funcPod.Send(func::FuncAgentMsg {
            msg_id: 0,
            event_body: Some(func::func_agent_msg::EventBody::FuncAgentCallReq(req))
        })?;

        return Ok(())
        
    }

    // when the func_agent get callresponse from funcSvc
    pub fn OnFuncSvcCallResp(&self, resp: func::FuncSvcCallResp) -> Result<()> {
        let id = resp.id.clone();
        let callerPodId = resp.caller_pod_id.clone();
        
        if callerPodId.len() == 0 { // it is from grpc gateway direct call
            let resp = if resp.error.len() == 0 {
                Ok(resp.resp)
            } else {
                Err(resp.error)
            };
            self.CallResponse(&id, resp)?;
        } else {
            let resp = func::FuncAgentCallResp {
                id: id,
                resp: resp.resp,
                error: resp.error,
            };
            // send response to the caller pod
            self.funcPodMgr.SendTo(&callerPodId, func::FuncAgentMsg {
                msg_id: 0,
                event_body: Some(func::func_agent_msg::EventBody::FuncAgentCallResp(resp)),
            })?;
        }
        
        return Ok(())
    }

    // when the func_agent get call ack from funcSvc
    pub fn OnFuncSvcCallAck(&self, ack: func::FuncSvcCallAck) -> Result<()> {
        let id = ack.id.clone();
        let callerPodId = ack.caller_pod_id.clone();
        
        let resp = func::FuncAgentCallAck {
            id: id,
            error: ack.error,
            callee_node_id: ack.callee_node_id,
            callee_pod_id: ack.callee_pod_id,
            caller_node_id: ack.caller_node_id,
            caller_pod_id: ack.caller_pod_id,
        };

        // send response to the caller pod
        self.funcPodMgr.SendTo(&callerPodId, func::FuncAgentMsg {
            msg_id: 0,
            event_body: Some(func::func_agent_msg::EventBody::FuncAgentCallAck(resp)),
        })?;
        
        return Ok(())
    }

    pub fn OnFuncSvcFuncMsg(&self, msg: func::FuncMsg) -> Result<()> { 
        let dstPod = msg.dst_pod_id.clone();
        match self.funcPodMgr.SendTo(&dstPod, func::FuncAgentMsg {
            msg_id: 0,
            event_body: Some(func::func_agent_msg::EventBody::FuncMsg(msg.clone())),
        }) {
            Err(e) => {
                // silience drop
                error!("OnFuncSvcFuncMsg send funcmsg {:?} fail with error {:?}", msg, e);
            }
            Ok(()) => ()
        }

        return Ok(())
    }

    // get msg from func_svc
    pub async fn OnFuncSvcMsg(&self, msg: func::FuncSvcMsg) -> Result<()> {
        //error!("OnFuncSvcMsg msg {:#?}", &msg);
        let body = match msg.event_body {
            None => return Err(Error::EINVAL(format!("OneFuncSvcMsg has None event_body"))),
            Some(b) => b,
        };

        match body {
            func::func_svc_msg::EventBody::FuncSvcCallReq(msg) => {
                match self.OnFuncSvcCallReq(msg.clone()) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        // try remove caller caller if it exists
                        self.callerCalls.lock().unwrap().remove(&msg.id);
                        
                        let resp = func::FuncSvcCallResp {
                            id: msg.id.clone(),
                            error: format!("funccall {} get error {:?}", msg.id, e),
                            resp: String::new(),
                            caller_node_id: msg.caller_node_id.clone(),
                            caller_pod_id: msg.caller_pod_id.clone(),
                            callee_node_id: msg.callee_node_id.clone(),
                            callee_pod_id: msg.callee_pod_id.clone(),
                        };
                
                        FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
                            event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallResp(resp))
                        })?;
                    }
                }
            }
            func::func_svc_msg::EventBody::FuncSvcCallResp(msg) => {
                self.OnFuncSvcCallResp(msg)?;
            }
            func::func_svc_msg::EventBody::FuncSvcCallAck(msg) => {
                self.OnFuncSvcCallAck(msg)?;
            }
            func::func_svc_msg::EventBody::FuncMsg(msg) => {
                self.OnFuncSvcFuncMsg(msg)?;
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
            msg  => {
                return Err(Status::aborted(format!("first message should be FuncPodRegisterReq but get {:?}", msg)));
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
        let id = req.id.clone();
        let res = self.FuncCall(req).await;
        let resp = match res {
            Ok(response) => func::FuncAgentCallResp {
                id: id,
                resp: response,
                error: String::new(),
            },
            Err(err) => func::FuncAgentCallResp {
                id: id,
                resp: String::new(),
                error: err,
            }
        };
        
        return Ok(Response::new(resp));
    }
}

pub async fn FuncAgentGrpcService(funcAgent: &FuncAgent) -> Result<()> {
    use tonic::transport::Server;
    use std::path::Path;
    use tokio::net::UnixListener;
    use tokio_stream::wrappers::UnixListenerStream;

    let path = NODEAGENT_CONFIG.FuncAgentSvcSocketAddr();

    info!("FuncAgentGrpcService path is {}", &path);
    std::fs::create_dir_all(Path::new(&path).parent().unwrap())?;
    std::fs::remove_file(Path::new(&path)).ok();
    let listener = UnixListener::bind(path).unwrap();
    let stream = UnixListenerStream::new(listener);

    let funcSvcFuture = Server::builder()
        .add_service(FuncAgentServiceServer::new(funcAgent.clone()))
        .serve_with_incoming(stream);

    let gatewayAddr = format!("0.0.0.0:{}", GATEWAY_PORT);
    let GatewaySvcFuture = Server::builder()
        .add_service(FuncAgentServiceServer::new(funcAgent.clone()))
        .serve(gatewayAddr.parse().unwrap());

    tokio::select! {
        _ = funcSvcFuture => {}
        _ = GatewaySvcFuture => {}
    }

    Ok(())
}