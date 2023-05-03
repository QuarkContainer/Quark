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
use qobjs::runtime_types::NodeFromString;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status};
use tokio::sync::mpsc;
use core::ops::Deref;
use std::sync::Arc;
use std::result::Result as SResult;

use qobjs::pb_gen::node_mgr_pb::{self as nm_svc};
use qobjs::common::Result;

use crate::node_agent::*;

#[derive(Debug)]
pub struct NodeMgrSvcInner {
    pub clients: Mutex<BTreeMap<String, NodeAgent>>,
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

    pub fn NodeAgent(&self, nodeId: &str) -> Option<NodeAgent> {
        return self.clients.lock().unwrap().get(nodeId).cloned();
    }
 /*
    pub fn OnNodeAgentConnect(&self, msg: SrvMsg) -> Result<()> {
        match msg {
            SrvMsg::AgentConnect((nodeReg, chann)) => {
                let k8sNode = NodeFromString(&nodeReg.node)?;
                let nodeId = nodeReg.identifier.clone();
                let rev = nodeReg.node_revision;

                match self.NodeAgent(nodeId) {
                    None => 
                }
            }
        }
    }
 */
}

#[derive(Debug)]
pub enum SrvMsg {
    AgentClose(String),
    AgentConnect((nm_svc::NodeRegistry, mpsc::Sender<SResult<nm_svc::NodeAgentMessage, Status>>)),
    AgentMsg((String, Result<nm_svc::NodeAgentMessage>)),
}


#[tonic::async_trait]
impl nm_svc::node_agent_service_server::NodeAgentService for NodeMgrSvc {
    type StreamMsgStream = ReceiverStream<SResult<nm_svc::NodeAgentMessage, Status>>;

    async fn stream_msg(
        &self,
        request: tonic::Request<tonic::Streaming<nm_svc::NodeAgentMessage>>,
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
                            nm_svc::node_agent_message::MessageBody::NodeRegistry(b) => {
                                nodeId = b.node.clone();
                                svc.agentsChann.send(SrvMsg::AgentConnect((b.clone(), tx))).await.unwrap();
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
                        Ok(m) => svc.agentsChann.send(SrvMsg::AgentMsg((nodeId.clone(), Ok(m)))).await.unwrap(),
                        Err(e) => svc.agentsChann.send(SrvMsg::AgentMsg((nodeId.clone(), Err(e.into())))).await.unwrap(),
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