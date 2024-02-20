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

use crate::nm as NmMsg;
use crate::common::*;

pub struct QServer {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub rx: Mutex<Option<Streaming<NmMsg::NodeAgentReq>>>,
    pub tx: mpsc::Sender<NmMsg::NodeAgentRespMsg>,
}
/*
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
*/