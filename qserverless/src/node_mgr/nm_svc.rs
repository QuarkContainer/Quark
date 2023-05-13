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

use qobjs::k8s;

use qobjs::nm as nm_svc;
use qobjs::node_mgr as NodeMgr;
use qobjs::service_directory as sd;
use qobjs::common::Result;
use qobjs::selection_predicate::*;
use qobjs::selector::*;
use qobjs::types::*;

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
        request: Request<sd::ListRequestMessage>,
    ) -> SResult<Response<sd::ListResponseMessage>, Status> {
        let req = request.get_ref();
        let cacher = match crate::NM_CACHE.get().unwrap().GetCacher(&req.obj_type) {
            None => {
                return Ok(Response::new(sd::ListResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                    objs: Vec::new(),
                }))
            }
            Some(c) => c,
        };

        let labelSelector = match Selector::Parse(&req.label_selector) {
            Err(e) => {
                return Ok(Response::new(sd::ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };
        let fieldSelector = match Selector::Parse(&req.field_selector) {
            Err(e) => {
                return Ok(Response::new(sd::ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };

        let opts = ListOption {
            revision: req.revision,
            revisionMatch: RevisionMatch::Exact,
            predicate: SelectionPredicate {
                label: labelSelector,
                field: fieldSelector,
                limit: 00,
                continue_: None,
            },
        };

        match cacher.List(&req.namespace, &opts).await {
            Err(e) => {
                return Ok(Response::new(sd::ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(resp) => {
                let mut objs = Vec::new();
                for o in resp.objs {
                    objs.push(o.Obj());
                }
                return Ok(Response::new(sd::ListResponseMessage {
                    error: "".into(),
                    revision: resp.revision,
                    objs: objs,
                }));
            }
        }
    }

    type WatchStream = std::pin::Pin<Box<dyn futures::Stream<Item = SResult<sd::WEvent, Status>> + Send>>;

    async fn watch(
        &self,
        request: Request<sd::WatchRequestMessage>,
    ) -> SResult<Response<Self::WatchStream>, Status> {
        let (tx, rx) = mpsc::channel(200);
        let stream = ReceiverStream::new(rx);

        tokio::spawn(async move {
            let req = request.get_ref();
            let cacher = match crate::NM_CACHE.get().unwrap().GetCacher(&req.obj_type) {
                None => {
                    tx.send(Err(Status::invalid_argument(&format!(
                        "doesn't support obj type {}",
                        &req.obj_type
                    ))))
                    .await
                    .ok();
                    return;
                }
                Some(c) => c,
            };

            let labelSelector = match Selector::Parse(&req.label_selector) {
                Err(e) => {
                    tx.send(Err(Status::invalid_argument(&format!("Fail: {:?}", e))))
                        .await
                        .ok();
                    
                    return;
                }
                Ok(s) => s,
            };
            let fieldSelector = match Selector::Parse(&req.field_selector) {
                Err(e) => {
                    tx.send(Err(Status::invalid_argument(&format!("Fail: {:?}", e))))
                        .await
                        .ok();
                    return;
                }
                Ok(s) => s,
            };

            let predicate = SelectionPredicate {
                label: labelSelector,
                field: fieldSelector,
                limit: 00,
                continue_: None,
            };

            match cacher.Watch(&req.namespace, req.revision, predicate) {
                Err(e) => {
                    tx.send(Err(Status::invalid_argument(&format!("Fail: {:?}", e))))
                        .await
                        .ok();
                    return;
                }
                Ok(mut w) => loop {
                    let event = w.stream.recv().await;
                    match event {
                        None => return,
                        Some(event) => {
                            let eventType = match event.type_ {
                                EventType::None => 0,
                                EventType::Added => 1,
                                EventType::Modified => 2,
                                EventType::Deleted => 3,
                                EventType::Error(s) => {
                                    tx.send(Err(Status::invalid_argument(&format!(
                                        "Fail: {:?}",
                                        s
                                    ))))
                                    .await
                                    .ok();
                                    return;
                                }
                            };
                            let we = sd::WEvent {
                                event_type: eventType,
                                obj: Some(event.obj.Obj()),
                            };
                            match tx.send(Ok(we)).await {
                                Ok(()) => (),
                                Err(e) => {
                                    tx.send(Err(Status::invalid_argument(&format!(
                                        "Fail: {:?}",
                                        e
                                    ))))
                                    .await
                                    .ok();
                                    return;
                                }
                            }
                        }
                    }
                },
            }
        });

        return Ok(Response::new(Box::pin(stream) as Self::WatchStream));
    }
}


pub async fn GrpcService() -> Result<()> {
    use tonic::transport::Server;
    use qobjs::service_directory::service_directory_service_server::ServiceDirectoryServiceServer;

    let svc = NodeMgrSvc::New();

    let sdFuture = Server::builder()
        .add_service(ServiceDirectoryServiceServer::new(svc.clone()))
        .serve("127.0.0.1:8890".parse().unwrap());

    let nodeAgentSvc = qobjs::nm::node_agent_service_server::NodeAgentServiceServer::new(svc.clone());
    let naFuture = Server::builder()
        .add_service(nodeAgentSvc)
        .serve("127.0.0.1:8888".parse().unwrap());

    let nodeMgrSvc: NodeMgr::node_mgr_service_server::NodeMgrServiceServer<NodeMgrSvc> = NodeMgr::node_mgr_service_server::NodeMgrServiceServer::new(svc.clone());
    let nmFuture = Server::builder()
        .add_service(nodeMgrSvc)
        .serve("127.0.0.1:8889".parse().unwrap());

    info!("nodemgr start ...");
    tokio::select! {
        _ = sdFuture => {}
        _ = naFuture => {}
        _ = nmFuture => {}
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use qobjs::{nm_client::*, cacher_client::CacherClient};

    #[actix_rt::test]
    async fn NMTestBody() {
        let cacheClient = CacherClient::New("http://127.0.0.1:8890".into()).await.unwrap();

        println!("nodelist is {:?}", cacheClient.List("node", "", &ListOption::default()).await.unwrap());

        let mut nodeWs = cacheClient
            .Watch("node", "", &ListOption::default())
            .await.unwrap();

        let mut podWs = cacheClient
            .Watch("pod", "", &ListOption::default())
            .await.unwrap();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    event = nodeWs.Next() => println!("node event is {:#?}", event),
                    event = podWs.Next() => println!("pod event is {:#?}", event),
                }
            }
        });

        //println!("node event is {:?}", nodeWs.Next().await.unwrap());
        
        //println!("pod event is {:#?}", podWs.Next().await.unwrap());
        let list = cacheClient.List("pod", "default", &ListOption::default()).await.unwrap();
        println!("list1 is {:?}", list);

        let client = NodeMgrClient::New("http://127.0.0.1:8889".into()).await.unwrap();
        let podstr = r#"
        {
            "apiVersion":"v1",
            "kind":"Pod",
            "metadata":{
                "name":"nginx",
                "namespace": "default"
            },
            "spec":{
                "hostNetwork": true,
                "containers":[
                    {
                        "name":"nginx",
                        "image":"nginx:alpine",
                        "ports":[
                            {
                                "containerPort": 80,
                                "hostIP": "192.168.0.22",
                                "hostPort": 88
                            }
                        ]
                    }
                ]
            }
        }"#;

        let pod : k8s::Pod = serde_json::from_str(podstr).unwrap();
        let configMap = k8s::ConfigMap::default();

        client.CreatePod("qserverless.quarksoft.io/brad-desktop", &pod, &configMap).await.unwrap();

        let list = cacheClient.List("pod", "default", &ListOption::default()).await.unwrap();
        println!("list2 is {:?}", list);

        std::thread::sleep(std::time::Duration::from_secs(1));

        assert!(false);
        
        /*let ws = cacheClient
            .Watch("pod", "default", &ListOption::default())
            .await.unwrap();*/


    }

}