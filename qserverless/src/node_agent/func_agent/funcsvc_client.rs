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
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use core::ops::Deref;
use std::time::Duration;

use qobjs::func;
use qobjs::common::*;
use qobjs::func::FuncSvcMsg;
use tokio::sync::Notify;
use tokio::sync::mpsc;

use crate::FUNC_AGENT;

#[derive(Debug, Clone)]
pub struct FuncSvcClient {
    //pub closeNotify: Arc<Notify>,
    //pub stop: AtomicBool,

    pub agentChann: mpsc::Sender<FuncSvcMsg>,
    pub stream: Arc<Mutex<Option<tonic::Streaming<func::FuncSvcMsg>>>>,
}

impl FuncSvcClient {
    pub async fn New(svcAddr: &str, registerReq: func::FuncAgentRegisterReq) -> Result<Self> {
        let mut client = {
            let client;
            loop {
                match func::func_svc_service_client::FuncSvcServiceClient::connect(svcAddr.to_string()).await {
                    Ok(c) => {
                        client = c;
                        break;
                    }
                    Err(e) => {
                        error!("can't connect to funcsvc {}, {:?}", svcAddr, e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            }
            client
        };

        let (tx, rx) = mpsc::channel(30);
        tx.try_send(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncAgentRegisterReq(registerReq)),
        }).unwrap();

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response = client.stream_process(stream).await?;
    
        let stream = response.into_inner();

        let ret = FuncSvcClient {
            agentChann: tx,
            stream: Arc::new(Mutex::new(Some(stream))),
        };

        return Ok(ret)
    }

    pub async fn Send(&self, msg: &func::FuncSvcMsg) -> Result<()> {
        match self.agentChann.send_timeout(msg.clone(), Duration::from_secs(1)).await {
            Ok(()) => return Ok(()),
            Err(_t) => return Err(Error::MpscSendFail),
        }
    }
}

#[derive(Debug)]
pub struct FuncSvcClientMgrInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub agentChann: mpsc::Sender<FuncSvcMsg>,
    pub svcAddr: String,
}

#[derive(Debug, Clone)]
pub struct FuncSvcClientMgr(Arc<FuncSvcClientMgrInner>);

impl Deref for FuncSvcClientMgr {
    type Target = Arc<FuncSvcClientMgrInner>;

    fn deref(&self) -> &Arc<FuncSvcClientMgrInner> {
        &self.0
    }
}

impl FuncSvcClientMgr {
    // todo: handle master/slave FuncSvc
    pub fn New(svcAddr: &str) -> Self {
        let (tx, rx) = mpsc::channel(100);
        
        let inner = FuncSvcClientMgrInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            agentChann: tx,
            svcAddr: svcAddr.to_string(),
        };

        let ret = Self(Arc::new(inner));
        let clone = ret.clone();
        let _future = tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return ret;
    }

    pub fn Send(&self, msg: func::FuncSvcMsg) -> Result<()> {
        match self.agentChann.try_send(msg) {
            Err(_) => return Err(Error::MpscSendFail),
            Ok(()) => return Ok(())
        }
    }

    pub async fn Process(&self, rx: mpsc::Receiver<FuncSvcMsg>) -> Result<()> {
        let registerMsg = FUNC_AGENT.ToGrpcType();
        let mut client = FuncSvcClient::New(&self.svcAddr, registerMsg).await?;
        let mut currentMsg;
        let mut stream = client.stream.lock().unwrap().take().unwrap();
        let mut rx = rx;
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, std::sync::atomic::Ordering::SeqCst);
                    break;
                }
                msg = rx.recv() => {
                    currentMsg = match msg {
                        None => break,
                        Some(m) => m,
                    };
                    loop {
                        match client.Send(&currentMsg).await {
                            Ok(()) => {
                                break;
                            }
                            Err(_) => {
                                // reconnect
                                client = FuncSvcClient::New(&self.svcAddr, func::FuncAgentRegisterReq::default()).await?;
                                stream = client.stream.lock().unwrap().take().unwrap();
                            }
                        }
                    }
                }
                msg = stream.message() => {
                    let msg : func::FuncSvcMsg = match msg {
                        Err(e) => {
                            error!("FuncSvcClient get error message {:?}", e);
                            break;
                        }
                        Ok(m) => {
                            match m {
                                None => {
                                    error!("FuncSvcClientMgr get None message");
                                    break;
                                }
                                Some(m) => m,
                            }
                        }
                    };

                    FUNC_AGENT.OnFuncSvcMsg(msg).await?;
                }

            }
        }

        return Ok(())
    }
}


