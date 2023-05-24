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

use std::sync::{Arc, atomic::AtomicBool};
use core::ops::Deref;
use tokio::sync::{mpsc, Notify};
use std::result::Result as SResult;

use qobjs::func;
use qobjs::common::*;

use super::funcagent_msg::FuncPodMsg;

#[derive(Debug)]
pub enum InstanceState {
    Idle,
    Running(u64), // handling FuncCallId
}

#[derive(Debug)]
pub struct FuncPodInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub instanceId: String,
    pub state: InstanceState,

    pub agentChann: mpsc::Sender<FuncPodMsg>,
}

#[derive(Debug, Clone)]
pub struct FuncPod(pub Arc<FuncPodInner>);

impl Deref for FuncPod {
    type Target = Arc<FuncPodInner>;

    fn deref(&self) -> &Arc<FuncPodInner> {
        &self.0
    }
}

impl FuncPod {
    pub fn New(
        registerMsg: &func::FuncPodRegisterReq, 
        stream: tonic::Streaming<func::FuncAgentMsg>,
        agentTx: mpsc::Sender<SResult<func::FuncAgentMsg, tonic::Status>>) 
    -> Result<Self> {
        let (tx, rx) = mpsc::channel(30);
        let instanceId = registerMsg.instance_id.clone();
        let inner = FuncPodInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            instanceId: instanceId.clone(),
            state: InstanceState::Idle,
            agentChann: tx,
        };
        let instance = FuncPod(Arc::new(inner));
        let clone = instance.clone();
        tokio::spawn(async move {
            clone.Process(stream, agentTx, rx).await.unwrap();
        });

        return Ok(instance);
    }

    pub async fn ProcessInternalMsg(&self, _msg: FuncPodMsg, _tx: &mpsc::Sender<SResult<func::FuncAgentMsg, tonic::Status>>) -> Result<()> {
        unimplemented!();
    }

    pub async fn ProcessExternalMsg(&self, _msg: func::FuncAgentMsg, _tx: &mpsc::Sender<SResult<func::FuncAgentMsg, tonic::Status>>) -> Result<()> {
        unimplemented!();
    }

    pub async fn Process(
        &self, 
        stream: tonic::Streaming<func::FuncAgentMsg>,
        tx: mpsc::Sender<SResult<func::FuncAgentMsg, tonic::Status>>,
        agentrx: mpsc::Receiver<FuncPodMsg>
    ) -> Result<()> {
        let closeNotify = self.closeNotify.clone();
        let mut stream = stream;
        let mut agentrx = agentrx;

        loop {
            tokio::select! {
                _ = closeNotify.notified() => {
                    break;
                }
                interalMsg = agentrx.recv() => {
                    match interalMsg {
                        None => {
                            panic!("FuncPod::StartProcess expect Pod internal message");
                        }
                        Some(msg) => {
                            self.ProcessInternalMsg(msg, &tx).await?;
                        }
                    }
                    
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
                    self.ProcessExternalMsg(msg, &tx).await?;
                }
            }
        }

        return Ok(())
    }
}
