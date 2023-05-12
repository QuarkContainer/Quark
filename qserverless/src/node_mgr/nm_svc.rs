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

use std::{collections::BTreeMap, sync::Mutex};

use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status, Request};
use tokio::sync::mpsc;
use core::ops::Deref;
use std::sync::Arc;
use std::result::Result as SResult;

use k8s_openapi::api::core::v1 as k8s;

use qobjs::pb_gen::nm as nm_svc;
use qobjs::pb_gen::node_mgr as NodeMgr;
use qobjs::service_directory as sd;
use qobjs::common::Result;

use crate::na_client::*;

#[derive(Debug)]
pub struct NodeMgrSvcInner {
    pub clients: Mutex<BTreeMap<String, NodeAgentClient>>,
    pub agentsChann: mpsc::Sender<SrvMsg>,
    pub processChannel: Option<mpsc::Receiver<SrvMsg>>,
}


// NodeMgrSvc direct connect to NodeAgent
#[derive(Debug, Clone)]
pub struct NodeMgrSvc(Arc<NodeMgrSvcInner>);

impl Deref for NodeMgrSvc {
    type Target = Arc<NodeMgrSvcInner>;

    fn deref(&self) -> &Arc<NodeMgrSvcInner> {
        &self.0
    }
}

impl NodeMgrSvc {
    pub fn New() -> Self {
        let (tx, rx) = mpsc::channel(30);

        let inner = NodeMgrSvcInner {
            clients: Mutex::new(BTreeMap::new()),
            agentsChann: tx,
            processChannel: Some(rx),
        };

        return Self(Arc::new(inner));
    }

    pub fn NodeAgent(&self, nodeId: &str) -> Option<NodeAgentClient> {
        return self.clients.lock().unwrap().get(nodeId).cloned();
    }
}

#[derive(Debug)]
pub enum SrvMsg {
    AgentClose(String),
    AgentConnect((nm_svc::NodeRegistry, mpsc::Sender<SResult<nm_svc::NodeAgentMessage, Status>>)),
    AgentMsg((String, Result<nm_svc::NodeAgentMessage>)),
}

#[tonic::async_trait]
impl nm_svc::node_agent_service_server::NodeAgentService for NodeMgrSvc {
    type StreamProcessStream = ReceiverStream<SResult<nm_svc::NodeAgentReq, Status>>;
    async fn stream_process(
        &self,
        request: tonic::Request<tonic::Streaming<nm_svc::NodeAgentRespMsg>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        let stream = request.into_inner();
        let (tx, rx) = mpsc::channel(30);
        let client = NodeAgentClient::New(self, stream, tx);
        tokio::spawn(async move {
            client.Process().await;
        });
        return Ok(Response::new(ReceiverStream::new(rx)));
    }
}

#[tonic::async_trait]
impl NodeMgr::node_mgr_service_server::NodeMgrService for NodeMgrSvc {
    async fn create_pod(
        &self,
        request: tonic::Request<NodeMgr::CreatePodReq>,
    ) -> SResult<tonic::Response<NodeMgr::CreatePodResp>, tonic::Status> {
        let req = request.get_ref();
        let pod: k8s::Pod = match serde_json::from_str(&req.pod) {
            Err(_e) => {
                return Ok(Response::new(NodeMgr::CreatePodResp {
                    error: format!("pod json is not valid {}", &req.pod),
                }))
            }
            Ok(p) => p,
        };

        let configmap: k8s::ConfigMap = match serde_json::from_str(&req.config_map) {
            Err(_e) => {
                return Ok(Response::new(NodeMgr::CreatePodResp {
                    error: format!("config_map json is not valid {}", &req.config_map),
                }))
            }
            Ok(p) => p,
        };

        match crate::NM_CACHE.get().unwrap().CreatePod(&req.node, &pod, &configmap).await {
            Err(e) => {
                return Ok(Response::new(NodeMgr::CreatePodResp {
                    error: format!("create pod fail with error {:?}", e),
                }))
            }
            Ok(()) =>  {
                return Ok(Response::new(NodeMgr::CreatePodResp {
                    error: String::new(),
                }))
            }
        }
    }
}

#[tonic::async_trait]
impl sd::service_directory_service_server::ServiceDirectoryService for NodeMgrSvc {
    // This is to verify the grpc server is working.
    // 1. go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
    // 2. Launch the grpc server
    // 3. grpcurl -plaintext -proto resilience_function/proto/service_directory.proto -d '{"client_name": "a client"}' [::]:50071 service_directory.ServiceDirectoryService/TestPing
    async fn test_ping(
        &self,
        request: Request<sd::TestRequestMessage>,
    ) -> SResult<Response<sd::TestResponseMessage>, Status> {
        error!("Request from {:?}", request.remote_addr());

        let response = sd::TestResponseMessage {
            server_name: "Server".to_owned(),
        };
        Ok(Response::new(response))
    }

    async fn put(
        &self,
        _request: Request<sd::PutRequestMessage>,
    ) -> SResult<Response<sd::PutResponseMessage>, Status> {
        let response = sd::PutResponseMessage {
            error: "NodeMgr doesn't support Put".to_owned(),
            ..Default::default()
        };
        Ok(Response::new(response))
    }

    async fn create(
        &self,
        _request: Request<sd::CreateRequestMessage>,
    ) -> SResult<Response<sd::CreateResponseMessage>, Status> {
        let response = sd::CreateResponseMessage {
            error: "NodeMgr doesn't support create".to_owned(),
            ..Default::default()
        };
        Ok(Response::new(response))
    }

    async fn get(
        &self,
        request: Request<sd::GetRequestMessage>,
    ) -> SResult<Response<sd::GetResponseMessage>, Status> {
        //info!("get Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match crate::NM_CACHE.get().unwrap().GetCacher(&req.obj_type) {
            None => {
                return Ok(Response::new(sd::GetResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    obj: None,
                }))
            }
            Some(c) => c,
        };

        match cacher.Get(&req.namespace, &req.name, req.revision).await {
            Err(e) => {
                return Ok(Response::new(sd::GetResponseMessage {
                    error: format!("Fail: {:?}", e),
                    obj: None,
                }))
            }
            Ok(o) => {
                return Ok(Response::new(sd::GetResponseMessage {
                    error: "".into(),
                    obj: match o {
                        None => None,
                        Some(o) => Some(o.Obj()),
                    },
                }))
            }
        }
    }

    async fn delete(
        &self,
        _request: Request<sd::DeleteRequestMessage>,
    ) -> SResult<Response<sd::DeleteResponseMessage>, Status> {
        let response = sd::DeleteResponseMessage {
            error: "NodeMgr doesn't support delete".to_owned(),
            ..Default::default()
        };
        Ok(Response::new(response))
    }

    async fn update(
        &self,
        _request: Request<sd::UpdateRequestMessage>,
    ) -> SResult<Response<sd::UpdateResponseMessage>, Status> {
        let response = sd::UpdateResponseMessage {
            error: "NodeMgr doesn't support update".to_owned(),
            ..Default::default()
        };
        Ok(Response::new(response))
    }

    async fn list(
        &self,
        _request: Request<sd::ListRequestMessage>,
    ) -> SResult<Response<sd::ListResponseMessage>, Status> {
        unimplemented!()
    }

    type WatchStream = std::pin::Pin<Box<dyn futures::Stream<Item = SResult<sd::WEvent, Status>> + Send>>;

    async fn watch(
        &self,
        _request: Request<sd::WatchRequestMessage>,
    ) -> SResult<Response<Self::WatchStream>, Status> {
        unimplemented!()
    }
}