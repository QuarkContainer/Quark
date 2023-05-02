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

use futures_util::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status};
use tokio::sync::mpsc;
use core::ops::Deref;
use std::sync::Arc;
use std::result::Result as SResult;

use qobjs::pb_gen::node_mgr_pb::{self as nm_svc};
use qobjs::common::Result;

#[derive(Clone, Debug)]
pub struct NodeAgentStruct {
    pub sender: mpsc::Sender<SResult<nm_svc::FornaxCoreMessage, Status>>,
}

#[derive(Debug)]
pub struct NodeMgrSvcInner {
    pub clients: Mutex<BTreeMap<String, NodeAgentStruct>>,
    pub agentsChann: mpsc::Sender<SrvMsg>,
    pub processChannel: Option<mpsc::Receiver<SrvMsg>>,
}

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
}

#[derive(Debug)]
pub enum SrvMsg {
    AgentClose(String),
    AgentConnect(String),
    AgentMsg(Result<nm_svc::FornaxCoreMessage>),
}


#[tonic::async_trait]
impl nm_svc::fornax_core_service_server::FornaxCoreService for NodeMgrSvc {
    type getMessageStream = ReceiverStream<SResult<nm_svc::FornaxCoreMessage, Status>>;

    async fn get_message(
        &self,
        request: tonic::Request<nm_svc::NodeIdentifier>,
    ) -> SResult<tonic::Response<Self::getMessageStream>, tonic::Status> {
        println!("get_message req {:?}", request);
        let (tx, rx) = mpsc::channel(30);
        
        let msg = nm_svc::NodeFullSync{};
        let msg = nm_svc::FornaxCoreMessage {
            node_identifier: None,
            message_type: nm_svc::MessageType::NodeReady as i32,
            message_body: Some(nm_svc::fornax_core_message::MessageBody::NodeFullSync(msg))
        };
    
        tx.send(Ok(msg)).await.unwrap();

        let na = NodeAgentStruct {
            sender: tx,
        };
        self.clients.lock().unwrap().insert(request.get_ref().identifier.to_string(), na);

        return Ok(Response::new(ReceiverStream::new(rx)));
    }

    async fn put_message(
        &self,
        request: tonic::Request<nm_svc::FornaxCoreMessage>,
    ) -> SResult<tonic::Response<()>, tonic::Status> {
        println!("put_message req {:?}", request);
        return Ok(Response::new(()));
    }

    type StreamMsgStream = ReceiverStream<SResult<nm_svc::FornaxCoreMessage, Status>>;

    async fn stream_msg(
        &self,
        request: tonic::Request<tonic::Streaming<nm_svc::FornaxCoreMessage>>,
    ) -> SResult<tonic::Response<Self::StreamMsgStream>, tonic::Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(30);
        let svc = self.clone();
        tokio::spawn(async move {
            let nodeId;
            if let Some(msg) = stream.next().await {
                match &msg {
                    Ok(m) => {
                        match m.message_body.as_ref().unwrap() {
                            nm_svc::fornax_core_message::MessageBody::NodeRegistry(b) => {
                                nodeId = b.node.clone();
                                svc.clients.lock().unwrap().insert(nodeId.clone(), NodeAgentStruct {
                                    sender: tx,
                                });
                                svc.agentsChann.send(SrvMsg::AgentConnect(nodeId.clone())).await.unwrap();
                            }
                            _ => {
                                error!("get agent connnection get unexpected initia msg {:?}", msg);
                                return;
                            }
                        }
                    }
                    Err(e)  => {
                        error!("get agent connnection error {:?}", e);
                        return;
                    }
                }
            } else {
                error!("get agent connnection disconected");
                return;
            }
            loop {
                if let Some(msg) = stream.next().await {
                    match msg {
                        Ok(m) => svc.agentsChann.send(SrvMsg::AgentMsg(Ok(m))).await.unwrap(),
                        Err(e) => svc.agentsChann.send(SrvMsg::AgentMsg(Err(e.into()))).await.unwrap(),
                    }
                } else {
                    svc.agentsChann.send(SrvMsg::AgentClose(nodeId)).await.unwrap();
                    break;
                }
            }
        });
        
        return Ok(Response::new(ReceiverStream::new(rx)));
    }
}