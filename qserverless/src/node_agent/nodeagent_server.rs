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

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use core::ops::Deref;

use qobjs::nm::NodeAgentReq;
use qobjs::runtime_types::ConfigMapFromString;
use qobjs::runtime_types::NodeFromString;
use qobjs::runtime_types::PodFromString;
use tokio::sync::Notify;
use tonic::Streaming;
use tokio::sync::mpsc;
use tokio::time;

use qobjs::nm as NmMsg;
use qobjs::common::*;

use crate::NODEAGENT_STORE;
use crate::node::NodeAgent;

pub struct NodeAgentServerMgrInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub nodeMgrAddress: Vec<String>,
}

pub struct NodeAgentServerMgr(Arc<NodeAgentServerMgrInner>);

impl Deref for NodeAgentServerMgr {
    type Target = Arc<NodeAgentServerMgrInner>;

    fn deref(&self) -> &Arc<NodeAgentServerMgrInner> {
        &self.0
    }
}

impl NodeAgentServerMgr {
    pub fn New(nodeMgrSvcAddrs: Vec<String>) -> Self {
        let inner = NodeAgentServerMgrInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            nodeMgrAddress: nodeMgrSvcAddrs,
        };

        return Self(Arc::new(inner));
    }

    pub async fn Process(&self, nodeAgent: &NodeAgent) -> Result<()> {
        let mut futures = Vec::new();
        for addr in &self.nodeMgrAddress {
            futures.push(self.ProcessNodeAgentServer(addr, nodeAgent));
        }

        futures::future::join_all(futures).await;
        return Ok(())
    }

    pub fn Close(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }

    pub async fn ProcessNodeAgentServer(&self, addr: &str, nodeAgent: &NodeAgent) {
        loop {
            let server = match NodeAgentServer::New(addr, nodeAgent).await {
                Ok(s) => s,
                Err(e) => {
                    error!("ProcessNodeAgentServer::Connect {} fail with error {:?}", addr, e);
                    time::sleep(time::Duration::from_secs(5)).await; // retry connect to the nodemgr
                    continue;
                }
            };

            info!("ProcessNodeAgentServer::Connect {} successfully", addr);
            tokio::select! {
                ret = server.Process(nodeAgent) => {
                    match ret {
                        Ok(()) => break,
                        Err(e) => {
                            error!("ProcessNodeAgentServer::Process fail with error {:?}", e);
                            continue;
                        }
                    }
                }
                _ = self.closeNotify.notified() => {
                    server.closeNotify.notify_waiters();
                }
            }
        }
    }
}

pub struct NodeAgentServer {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub rx: Mutex<Option<Streaming<NmMsg::NodeAgentReq>>>,
    pub tx: mpsc::Sender<NmMsg::NodeAgentRespMsg>,
    pub nodeAgent: NodeAgent,
}

impl NodeAgentServer {
    pub async fn New(nodeMgrAddr: &str, nodeAgent: &NodeAgent) -> Result<Self> {
        let mut client = NmMsg::node_agent_service_client::NodeAgentServiceClient::connect(nodeMgrAddr.to_string()).await?;
        let (tx, rx) = mpsc::channel(30);
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response = client.stream_process(stream).await?;
    
        let inbound: Streaming<NodeAgentReq> = response.into_inner();
        
        let ret = Self {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            rx: Mutex::new(Some(inbound)),
            tx: tx,
            nodeAgent: nodeAgent.clone(),
        };

        return Ok(ret);
    }

    pub async fn Process(&self, nodeAgent: &NodeAgent) -> Result<()> {
        let mut rx = self.rx.lock().unwrap().take().unwrap();

        let revision = self.NodeInitialization(nodeAgent, &mut rx).await?;
        let mut watchStream = NODEAGENT_STORE.get().unwrap().Watch(revision)?;

        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                event = watchStream.stream.recv() => {
                    let event = event.unwrap();
                    
                    let eventBody = event.ToGrpcEvent()?;
                    let msg = NmMsg::NodeAgentRespMsg {
                        message_body: Some(NmMsg::node_agent_resp_msg::MessageBody::NodeAgentStreamMsg(NmMsg::NodeAgentStreamMsg{
                            event_body: Some(eventBody)
                        })),
                    };
            
                    self.tx.send(msg).await.unwrap();
                }
                msg = rx.message() => {
                    let msg = match msg {
                        Err(e) => return Err(Error::CommonError(format!("QServer::Process get error {:?}", e))),
                        Ok(m) => m,
                    };
    
                    let req = match msg {
                        None => break,
                        Some(msg) => msg,
                    };
    
                    let reqId = req.request_id;
                    let resp = match self.ReqHandler(req, nodeAgent).await {
                        Err(e) => {
                            NmMsg::NodeAgentResp {
                                request_id: reqId,
                                error: format!("{:?}", e),
                                message_body: None,
                            }
                        }
                        Ok(body) => {
                            NmMsg::NodeAgentResp {
                                request_id: reqId,
                                error: "".to_owned(),
                                message_body: Some(body),
                            }
                        }
                    };
                    let msg = NmMsg::NodeAgentRespMsg {
                        message_body: Some(NmMsg::node_agent_resp_msg::MessageBody::NodeAgentResp(resp)),
                    };
            
                    self.tx.send(msg).await.unwrap();
                }
            }
        }

        return Ok(())
    }

    pub async fn NodeInitialization(&self, nodeAgent: &NodeAgent, rx: &mut Streaming<NodeAgentReq>) -> Result<i64> {
        let node = NODEAGENT_STORE.get().unwrap().GetNode();
        let list = NODEAGENT_STORE.get().unwrap().List();
        let mut pods = Vec::new();
        for pod in list.pods {
            pods.push(serde_json::to_string(&pod)?);
        }

        let nodeRegister = NmMsg::NodeRegister {
            revision: list.revision,
            node: serde_json::to_string(&node)?,
            pods: pods,
        };

        let msg = NmMsg::NodeAgentStreamMsg {
            event_body: Some(NmMsg::node_agent_stream_msg::EventBody::NodeRegister(nodeRegister))
        };

        self.SendStreamMessage(msg).await?;

        let msg = rx.message().await;

        let msg = match msg {
            Err(e) => return Err(Error::CommonError(format!("QServer::Process get error {:?}", e))),
            Ok(m) => m,
        };

        let req = match msg {
            None => return Err(Error::CommonError(format!("NodeInitialization fail with server close"))),
            Some(msg) => msg,
        };

        let reqId = req.request_id;
        let resp = match self.ReqHandler(req, nodeAgent).await {
            Err(e) => {
                NmMsg::NodeAgentResp {
                    request_id: reqId,
                    error: format!("{:?}", e),
                    message_body: None,
                }
            }
            Ok(body) => {
                NmMsg::NodeAgentResp {
                    request_id: reqId,
                    error: "".to_owned(),
                    message_body: Some(body),
                }
            }
        };
        let msg = NmMsg::NodeAgentRespMsg {
            message_body: Some(NmMsg::node_agent_resp_msg::MessageBody::NodeAgentResp(resp)),
        };

        self.tx.send(msg).await.unwrap();
        return Ok(list.revision)
    }

    pub fn CreatePod(&self, req: NmMsg::CreatePodReq, nodeAgent: &NodeAgent) -> Result<()> {
        let pod = PodFromString(&req.pod)?;
        let configMap = ConfigMapFromString(&req.config_map)?;
        nodeAgent.CreatePod(&pod, &configMap)?;
        return Ok(())
    }

    pub async fn ReqHandler(&self, req: NmMsg::NodeAgentReq, nodeAgent: &NodeAgent) -> Result<NmMsg::node_agent_resp::MessageBody> {
        error!("ReqHandler 1 {:?}", &req);
        let body: NmMsg::node_agent_req::MessageBody = req.message_body.unwrap();
        match body {
            NmMsg::node_agent_req::MessageBody::NodeConfigReq(req) => {
                let node = NodeFromString(&req.node)?;
                nodeAgent.NodeConfigure(node).await?;
                return Ok(NmMsg::node_agent_resp::MessageBody::NodeConfigResp(NmMsg::NodeConfigResp{}));
            }
            NmMsg::node_agent_req::MessageBody::CreatePodReq(req) => {
                self.CreatePod(req, nodeAgent)?;
                return Ok(NmMsg::node_agent_resp::MessageBody::CreatePodResp(NmMsg::CreatePodResp{}));
            }
            NmMsg::node_agent_req::MessageBody::TerminatePodReq(req) => {
                let podId = &req.pod_id;
                nodeAgent.TerminatePod(podId)?;
                return Ok(NmMsg::node_agent_resp::MessageBody::TerminatePodResp(NmMsg::TerminatePodResp{}));
            }
            NmMsg::node_agent_req::MessageBody::ReadFuncLogReq(req) => {
                match nodeAgent.ReadFuncLog(&req.namespace, &req.func_name, req.offset as usize, req.len as usize) {
                    Err(e) => {
                        return Ok(NmMsg::node_agent_resp::MessageBody::ReadFuncLogResp(NmMsg::ReadFuncLogResp{
                            error: format!("{:?}", e),
                            ..Default::default()
                        }));
                    }
                    Ok(content) => {
                        return Ok(NmMsg::node_agent_resp::MessageBody::ReadFuncLogResp(NmMsg::ReadFuncLogResp{
                            content: content,
                            ..Default::default()
                        }));
                    }
                }
                
            }
        }
    }

    pub async fn SendStreamMessage(&self, msg: NmMsg::NodeAgentStreamMsg) -> Result<()> {
        let msg = NmMsg::NodeAgentRespMsg {
            message_body: Some(NmMsg::node_agent_resp_msg::MessageBody::NodeAgentStreamMsg(msg)),
        };

        self.tx.send(msg).await.unwrap();

        return Ok(())
    }
}
