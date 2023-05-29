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

use crate::FUNC_AGENT;


#[derive(Debug)]
pub enum funcPodState {
    Idle,
    Running(String), // handling FuncCallId
}

impl funcPodState {
    pub fn State(&self) -> func::FuncPodState {
        match self {
            Self::Idle => func::FuncPodState::Idle,
            Self::Running(_) => func::FuncPodState::Running,
        }
    }

    pub fn FuncCallId(&self) -> String {
        match self {
            Self::Idle => String::new(),
            Self::Running(id) => id.clone(),
        }
    }
}

#[derive(Debug)]
pub struct FuncPodInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub funcPodId: String,
    pub namespace: String,
    pub packageName: String,
    pub state: funcPodState,

    pub agentChann: mpsc::Sender<SResult<func::FuncAgentMsg, tonic::Status>>,
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
        let funcPodId = registerMsg.func_pod_id.clone();
        let inner = FuncPodInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            funcPodId: funcPodId.clone(),
            namespace: registerMsg.namespace.to_string(),
            packageName: registerMsg.package_name.to_string(),
            state: funcPodState::Idle,
            agentChann: agentTx,
        };
        let instance = FuncPod(Arc::new(inner));
        let clone = instance.clone();
        tokio::spawn(async move {
            clone.Process(stream).await.unwrap();
        });

        return Ok(instance);
    }

    pub fn ToGrpcType(&self) -> func::FuncPodStatus {
        return func::FuncPodStatus {
            func_pod_id: self.funcPodId.clone(),
            namespace: self.namespace.clone(),
            package_name: self.packageName.clone(),
            state: self.state.State() as i32,
            func_call_id: self.state.FuncCallId(),
        }
    }

    pub fn Send(&self, msg: func::FuncAgentMsg) -> Result<()> {
        match self.agentChann.try_send(Ok(msg)) {
            Ok(()) => return Ok(()),
            Err(_) => return Err(Error::MpscSendFail),
        }
    }

    pub async fn Process(
        &self, 
        stream: tonic::Streaming<func::FuncAgentMsg>
    ) -> Result<()> {
        let closeNotify = self.closeNotify.clone();
        let mut stream = stream;

        loop {
            tokio::select! {
                _ = closeNotify.notified() => {
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
                    
                    FUNC_AGENT.OnFuncAgentMsg(&self.funcPodId, msg).await?;
                }
            }
        }

        return Ok(())
    }
}
