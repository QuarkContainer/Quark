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
use core::ops::Deref;
use std::sync::atomic::AtomicBool;

use tokio::sync::Notify;
use tokio::sync::mpsc;

use qobjs::func;
use qobjs::func::FuncAgentMsg;
use qobjs::common::*;

use crate::FUNC_CALL_MGR;

#[derive(Debug)]
pub struct FuncAgentClientInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub agentChann: mpsc::Sender<FuncAgentMsg>,}

#[derive(Debug, Clone)]
pub struct FuncAgentClient(Arc<FuncAgentClientInner>);

impl Deref for FuncAgentClient {
    type Target = Arc<FuncAgentClientInner>;

    fn deref(&self) -> &Arc<FuncAgentClientInner> {
        &self.0
    }
}


impl FuncAgentClient {
    pub async fn Init(agentAddr: &str) -> Result<Self> {
        let mut client = {
            let client;
            loop {
                match func::func_agent_service_client::FuncAgentServiceClient::connect(agentAddr.to_string()).await {
                    Ok(c) => {
                        client = c;
                        break;
                    }
                    Err(_) => ()
                }
            }
            client
        };

        let (tx, rx) = mpsc::channel(0);
        tx.try_send(func::FuncAgentMsg {
            event_body: Some(func::func_agent_msg::EventBody::FuncPodRegisterReq(FUNC_CALL_MGR.funcMgr.RegisteMsg())),
        }).unwrap();

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response = client.stream_process(stream).await?;
    
        let stream = response.into_inner();

        let inner = FuncAgentClientInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            agentChann: tx,
        };

        let ret = Self(Arc::new(inner));
        let clone = ret.clone();
        tokio::spawn(async move {
            clone.Process(stream).await.unwrap();
        });

        return Ok(ret);
    }

    pub fn Send(&self, msg: func::FuncAgentMsg) {
        self.agentChann.try_send(msg).unwrap();
    }

    pub async fn OnFuncAgentMsg(&self, _msg: func::FuncAgentMsg) -> Result<()> {
        unimplemented!();
    }

    pub async fn Process(&self, stream: tonic::Streaming<FuncAgentMsg>) -> Result<()> {
        let mut stream = stream;
        
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, std::sync::atomic::Ordering::SeqCst);
                    break;
                }
                msg = stream.message() => {
                    let msg : func::FuncAgentMsg = match msg {
                        Err(e) => {
                            error!("FuncNode get error message {:?}", e);
                            break;
                        }
                        Ok(m) => {
                            match m {
                                None => {
                                    error!("FuncNode get None message");
                                    break;
                                }
                                Some(m) => m,
                            }
                        }
                    };

                    self.OnFuncAgentMsg(msg).await?;
                }

            }
        }

        return Ok(())
    }
}