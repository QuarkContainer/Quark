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
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::SystemTime;
use core::ops::Deref;
use tonic::Streaming;
use tokio::sync::mpsc;
use std::result::Result as SResult;
use tonic::Status;

use qobjs::k8s;
use qobjs::ObjectMeta;
use qobjs::common::*;
use qobjs::func;
use tokio::sync::Notify;

use crate::func_context::FuncCallContext;
use crate::func_pod::*;
use crate::package::Package;

#[derive(Debug, Clone)]
pub enum FuncNodeState {
    WaitingConn, // master node waiting for nodeagent connection
    Connected, // get the connection
    Running(mpsc::Sender<SResult<func::FuncSvcMsg, Status>>), // get the registration message
}

#[derive(Debug)]
pub struct FuncNodeInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub name: String,
    pub state: FuncNodeState,
    pub conn: Option<NodeAgentConnection>,
    pub FuncPods: BTreeMap<String, FuncPod>,
    pub CallerFuncCalls: BTreeMap<String, FuncCallContext>,
    pub CalleeFuncCalls: BTreeMap<String, String>,
    pub tx: Option<mpsc::Sender<SResult<func::FuncSvcMsg, Status>>>,
}

#[derive(Debug, Clone)]
pub struct FuncNode(Arc<Mutex<FuncNodeInner>>);

impl Deref for FuncNode {
    type Target = Arc<Mutex<FuncNodeInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncNodeInner>> {
        &self.0
    }
}


impl FuncNode {
    //pub fn New(name: &str) -> Result<()> {}

    pub fn State(&self) -> FuncNodeState {
        return self.lock().unwrap().state.clone();
    }

    pub fn Close(&self) {
        let notify = self.lock().unwrap().closeNotify.clone();
        notify.notify_waiters();
    }
    
    pub fn SetState(&self, state: FuncNodeState) {
        self.lock().unwrap().state = state;
    }

    pub async fn StartProess(&self, rx: Streaming<func::FuncSvcMsg>, tx: mpsc::Sender<SResult<func::FuncSvcMsg, Status>>) -> Result<()> {
        let state = self.State();
        match state {
            FuncNodeState::WaitingConn => (),
            _ => {
                self.Close();
                self.SetState(FuncNodeState::Connected);
            }
        }
        
        defer!(self.SetState(FuncNodeState::WaitingConn));
        let mut rx = rx;
        let closeNotify = self.lock().unwrap().closeNotify.clone();

        tokio::select! {
            _ = closeNotify.notified() => {
                return Ok(());
            }
            msg = rx.message() => {
                let msg : func::FuncSvcMsg = msg?.unwrap();
                let body = match msg.event_body {
                    None => return Ok(()),
                    Some(b) => b,
                };
                match body {
                    func::func_svc_msg::EventBody::FuncAgentRegisterReq(_req) => {
                        // processing
                        self.SetState(FuncNodeState::Running(tx));
                    }
                    _ => {
                        error!("didn't get FuncAgentRegisterReq message instead {:?}", body);
                            
                        return Err(Error::CommonError(format!("didn't get FuncAgentRegisterReq message instead {:?}", body)));
                    }
                }
            }
        }

        loop {
            tokio::select! {
                _ = closeNotify.notified() => {
                    break;
                }
                msg = rx.message() => {
                    let msg : func::FuncSvcMsg = msg?.unwrap();
                    //let msgId = msg.msg_id;
                    let body = match msg.event_body {
                        None => return Ok(()),
                        Some(b) => b,
                    };
                    match body {
                        _ => {
                            error!("didn't get FuncAgentRegisterReq message instead {:?}", body);
                            return Err(Error::CommonError(format!("didn't get FuncAgentRegisterReq message instead {:?}", body)));
                        }
                    }
                }
            }
        }

        return Ok(())
    }

    pub fn NewFuncPod(&self, package: &Package) -> Result<FuncPod> {
        let uid = uuid::Uuid::new_v4().to_string();
        let inner = FuncPodInner {
            id: uid.clone(),
            package: package.clone(),
            node: self.clone(),
            state: Mutex::new(FuncPodState::Creating(SystemTime::now())),
            pod: Mutex::new(k8s::Pod {
                metadata: ObjectMeta { 
                    namespace: Some(package.Namespace()),
                    name: Some(package.Name()),
                    uid: Some(uid.clone()),
                    ..Default::default()
                },
                spec: Some(package.Spec()),
                ..Default::default()
            })
        };

        let pod = FuncPod(Arc::new(inner));
        self.CreatePod(&pod)?;

        self.lock().unwrap().FuncPods.insert(uid, pod.clone());

        return Ok(pod);
    }

    pub fn CreatePod(&self, _pod: &FuncPod) -> Result<()> {
        unimplemented!();
    }
}

#[derive(Debug, Clone)]
pub struct NodeAgentConnection {

}

pub struct FuncNodeMgr {
    pub nodes: BTreeMap<String, FuncNode>,
}