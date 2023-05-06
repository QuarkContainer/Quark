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

use qobjs::pb_gen::node_mgr_pb::NodeAgentReq;
use tokio::sync::Notify;
use tonic::Streaming;
use tokio::sync::mpsc;

use qobjs::pb_gen::node_mgr_pb as NmMsg;
use qobjs::common::*;

use crate::node::NodeAgent;


pub struct NodeAgentServer {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub rx: Mutex<Option<Streaming<NmMsg::NodeAgentReq>>>,
    pub tx: mpsc::Sender<NmMsg::NodeAgentRespMsg>,
    pub nodeAgent: NodeAgent,
}

impl NodeAgentServer {
    pub async fn New(nodeAgent: NodeAgent) -> Result<Self> {
        let mut client = NmMsg::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888").await?;
        let (tx, rx) = mpsc::channel(30);
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response = client.stream_process(stream).await?;
    
        let inbound: Streaming<NodeAgentReq> = response.into_inner();
        
        let ret = Self {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            rx: Mutex::new(Some(inbound)),
            tx: tx,
            nodeAgent: nodeAgent,
        };

        return Ok(ret);
    }

    pub async fn Process(&self) -> Result<()> {
        let mut rx = self.rx.lock().unwrap().take().unwrap();
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
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
                    let mut resp = self.ReqHandler(req)?;
                    resp.request_id = reqId;
                    let msg = NmMsg::NodeAgentRespMsg {
                        error: "".to_string(),
                        message_body: Some(NmMsg::node_agent_resp_msg::MessageBody::NodeAgentResp(resp)),
                    };
            
                    self.tx.send(msg).await.unwrap();
            
                    return Ok(())
                }
            }
        }
        return Ok(())
    }

    pub fn ReqHandler(&self, _req: NmMsg::NodeAgentReq) -> Result<NmMsg::NodeAgentResp> {
        unimplemented!();
    }

    pub async fn SendStreamMessage(&self, msg: NmMsg::NodeAgentStreamMsg) -> Result<()> {
        let msg = NmMsg::NodeAgentRespMsg {
            error: "".to_string(),
            message_body: Some(NmMsg::node_agent_resp_msg::MessageBody::NodeAgentStreamMsg(msg)),
        };

        self.tx.send(msg).await.unwrap();

        return Ok(())
    }
}
