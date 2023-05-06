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
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use tokio::sync::Notify;
use tonic::Streaming;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::pb_gen::node_mgr_pb as NmMsg;
use crate::common::*;

pub struct QServer {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub rx: Mutex<Option<Streaming<NmMsg::NodeAgentReq>>>,
    pub tx: mpsc::Sender<NmMsg::NodeAgentRespMsg>,
}

impl QServer {
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

pub struct QClient {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub rx: Mutex<Option<Streaming<NmMsg::NodeAgentRespMsg>>>,
    pub tx: mpsc::Sender<NmMsg::NodeAgentReq>,
    pub pendingReqs: Mutex<BTreeMap<u64, oneshot::Sender<NmMsg::NodeAgentResp>>>,
    pub nextReqId: AtomicU64,
}

impl QClient {
    pub async fn Process(&self) -> Result<()> {
        let mut rx = self.rx.lock().unwrap().take().unwrap();
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                msg = rx.message() => {
                    match msg {
                        Err(e) => return Err(Error::CommonError(format!("QClient::Process rx message fail {:?}", e))),
                        Ok(msg) => {
                            match msg {
                                None => break,
                                Some(msg) => {
                                    match msg.message_body.unwrap() {
                                        NmMsg::node_agent_resp_msg::MessageBody::NodeAgentResp(resp) => {
                                            let reqId = resp.request_id;
                                            let chann = self.pendingReqs.lock().unwrap().remove(&reqId);
                                            match chann {
                                                None => error!("QClient::Process get none exist response {:?}", resp),
                                                Some(chann) => {
                                                    match chann.send(resp) {
                                                        Ok(()) => (),
                                                        Err(e) => {
                                                            error!("QClient::Process send messaage fail response {:?}", e);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        NmMsg::node_agent_resp_msg::MessageBody::NodeAgentStreamMsg(_msg) => {

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(())
    }

    pub fn ReqId(&self) -> u64 {
        return self.nextReqId.fetch_add(1, Ordering::Release) + 1;
    }

    pub async fn Call(&mut self, req: NmMsg::NodeAgentReq) -> Result<NmMsg::NodeAgentResp> {
        let reqId = self.ReqId();
        let mut req = req;
        req.request_id = reqId;
        let (tx, rx) = oneshot::channel::<NmMsg::NodeAgentResp>();

        self.pendingReqs.lock().unwrap().insert(reqId, tx);
        match self.tx.send(req).await {
            Ok(()) => (),
            Err(e) => {
                return Err(Error::CommonError(format!("QClient::Call send fail with error {:?}", e)));
            }
        }
        
        let resp = match rx.await {
            Ok(r) => r,
            Err(e) => return Err(Error::CommonError(format!("QClient::Call recv fail with error {:?}", e))),
        };
        return Ok(resp);
    }
}