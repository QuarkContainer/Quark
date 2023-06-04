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
use std::collections::BTreeMap;
use std::sync::Mutex;

use core::ops::Deref;

use tokio::sync::mpsc;
use tokio::sync::oneshot;

use qobjs::func;
use qobjs::common::*;

use crate::FUNC_AGENT_CLIENT;
use crate::func_def::QSResult;
use crate::func_mgr::FuncMgr;

pub struct FuncCallInner {
    pub id: String,
    pub namespace: String,
    pub packageName: String,
    pub funcName: String,
    pub parameters: String,
    pub priority: String,
}

pub struct FuncCall(Arc<FuncCallInner>);

impl Deref for FuncCall {
    type Target = Arc<FuncCallInner>;

    fn deref(&self) -> &Arc<FuncCallInner> {
        &self.0
    }
}

pub struct FuncCallMgr {
    pub callerCalls: Mutex<BTreeMap<String, oneshot::Sender<QSResult>>>,
    pub funcMgr: Arc<FuncMgr>,
    pub reqSender: mpsc::Sender<func::FuncAgentCallReq>,
    pub reqRecv: Mutex<Option<mpsc::Receiver<func::FuncAgentCallReq>>>,
}

impl FuncCallMgr {
    pub fn Init() -> Self {
        let (tx, rx) = mpsc::channel(30);
        return Self {
            callerCalls: Mutex::new(BTreeMap::new()),
            funcMgr: Arc::new(FuncMgr::Init()),
            reqSender: tx,
            reqRecv: Mutex::new(Some(rx)),
        }
    }
    
    pub async fn RemoteCall(
        &self, 
        namespace: String,
        packageName: String,
        funcName: String,
        parameters: String,
        priority: usize,
    ) -> QSResult {
        let id = uuid::Uuid::new_v4().to_string();
        let req = func::FuncAgentCallReq {
            id: id.clone(),
            namespace: namespace,
            package_name: packageName,
            func_name: funcName,
            parameters: parameters,
            priority: priority as u64,
        };

        let (tx, rx) = oneshot::channel();
        self.callerCalls.lock().unwrap().insert(id.clone(), tx);

        let msg = func::FuncAgentMsg {
            msg_id: 0,
            event_body: Some(func::func_agent_msg::EventBody::FuncAgentCallReq(req)),
        };
        
        FUNC_AGENT_CLIENT.get().unwrap().Send(msg);

        let res = rx.await.expect("func call fail");
        return res;
    }

    pub fn CallResponse(&self, id: &str, res: QSResult) -> Result<()> {
        let tx = match self.callerCalls.lock().unwrap().remove(id) {
            None => return Err(Error::ENOENT(format!("FuncCallMgr::CallResponse can't find the id {}", id))),
            Some(tx) => tx,
        };

        tx.send(res).unwrap();
        return Ok(())
    }

    pub async fn Process(&self) -> Result<()> {
        let mut rx = match self.reqRecv.lock().unwrap().take() {
            None => return Err(Error::CommonError("FuncCallMgr process can't run twice".to_owned())),
            Some(rx) => rx,
        };

        let funcMgr = self.funcMgr.clone();
        loop {
            let call = match rx.recv().await {
                None => break,
                Some(req) => req,
            };

            let id = call.id.clone();
            let res = funcMgr.Call(&call.func_name, &call.parameters).await;
            let resp = match res {
                Err(e) => {
                    func::FuncAgentCallResp {
                        id: id,
                        resp: String::new(),
                        error: e,
                    }
                }
                Ok(resp) => {
                    func::FuncAgentCallResp {
                        id: id,
                        resp: resp,
                        error: String::new(),
                    }
                }
            };

            let msg = func::FuncAgentMsg {
                msg_id: 0,
                event_body: Some(func::func_agent_msg::EventBody::FuncAgentCallResp(resp))
            };

            FUNC_AGENT_CLIENT.get().unwrap().Send(msg);
        }
        
        return Ok(())
    }

    pub fn LocalCall(&self, call: &func::FuncAgentCallReq) -> Result<()>{
        match self.reqSender.try_send(call.clone()) {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::MpscSendFail),
        }
    }
}
