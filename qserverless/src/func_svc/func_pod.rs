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
use std::time::SystemTime;
use core::ops::Deref;

use qobjs::audit::func_audit::FuncStateFail;
use qobjs::audit::func_audit::FuncStateSuccess;
use qobjs::common::*;
use qobjs::types::*;
use qobjs::func;

use crate::AUDIT_AGENT;
use crate::FUNC_NODE_MGR;
use crate::FUNC_SVC_MGR;
use crate::func_call::FuncCall;
use crate::package::*;
use crate::func_node::*;
use crate::message::FuncNodeMsg;

#[derive(Debug, Clone)]
pub struct FuncPodId {
    pub packageId: PackageId,
    pub podName: String,
    pub nodeName: String,
}

impl ToString for FuncPodId {
    fn to_string(&self) -> String {
        return format!("{}/{}@{}", &self.packageId.to_string(), &self.podName, &self.nodeName);
    }
}

impl Ord for FuncPodId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.packageId == other.packageId {
            return other.podName.cmp(&self.podName);
        }

        return other.packageId.cmp(&other.packageId);
    }
}

impl PartialOrd for FuncPodId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for FuncPodId {
    fn eq(&self, other: &Self) -> bool {
        return self.packageId == other.packageId && self.podName == other.podName;
    }
}

impl Eq for FuncPodId {}

#[derive(Debug, Clone)]
pub enum FuncPodState {
    Idle(SystemTime), // IdleTime
    Running(FuncCall), // the funcCallId
    Exiting,
    Dead,
}

impl FuncPodState {
    pub fn IsDead(&self) -> bool {
        match self {
            Self::Dead => return true,
            _ => return false,
        }
    }

    pub fn IsExiting(&self) -> bool {
        match self {
            Self::Exiting => return true,
            _ => return false,
        }
    }
}

#[derive(Debug)]
pub struct FuncPodInner {
    pub namespace: String,
    pub podName: String,
    pub package: Option<Package>,
    pub node: FuncNode,
    pub clientMode: bool,
    pub state: Mutex<FuncPodState>,
    pub callerFuncCalls: Mutex<BTreeMap<String, FuncCall>>, 
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
    pub fn ScheduleFuncCall(&self, funcCall: &FuncCall) -> Result<()> {
        *self.state.lock().unwrap() = FuncPodState::Running(funcCall.clone());
        *funcCall.calleeNodeId.lock().unwrap() = self.node.NodeName();
        *funcCall.calleeFuncPodId.lock().unwrap() = self.podName.clone();
        AUDIT_AGENT.AssignFunc(
            &funcCall.id, 
            &self.node.NodeName(),
        )?;
        self.node.Send(FuncNodeMsg::FuncCall(funcCall.clone()))?;
        return Ok(())
    }

    pub fn SetKeepalive(&self) -> SystemTime {
        let curr: SystemTime = SystemTime::now();
        *self.state.lock().unwrap() = FuncPodState::Idle(curr);
        return curr;
    }

    pub fn KeepaliveTime(&self) -> Result<SystemTime> {
        let state = self.state.lock().unwrap();
        match *state {
            FuncPodState::Idle(curr) => return Ok(curr),
            _ => return Err(Error::CommonError(format!("IdleTime invalid func pod state {:?}", state))),
        }
    }

    pub fn IsDead(&self) -> bool {
        return self.state.lock().unwrap().IsDead();
    }

    pub fn IsExiting(&self) -> bool {
        return self.state.lock().unwrap().IsExiting();
    }

    pub fn OnFuncSvcCallReq(&self, funccall: &FuncCall) -> Result<()> {
        self.callerFuncCalls.lock().unwrap().insert(funccall.id.clone(), funccall.clone());
        return Ok(())
    }

    // get funccall response from nodeagent
    pub fn OnFuncSvcCallResp(&self, resp: func::FuncSvcCallResp) -> Result<()> {
        match resp.res.as_ref() {
            None => {
                error!("get none func res {:#?}", resp);
            }
            Some(res) => {
                let funcRes : FuncRes = res.clone().into();
                let state = match funcRes {
                    FuncRes::Resp(_) => FuncStateSuccess,
                    FuncRes::Error(_) => FuncStateFail,
                };
        
                AUDIT_AGENT.FinishFunc(
                    &resp.id, 
                    state
                )?;
            }
        };
        
        
        let callerNode = FUNC_NODE_MGR.Get(&resp.caller_node_id)?;
        callerNode.Send(FuncNodeMsg::FuncCallResp(resp))?;
        *self.state.lock().unwrap() = FuncPodState::Idle(SystemTime::now());
        
        FUNC_SVC_MGR.lock().unwrap().OnFreePod(self, false)?;
        return Ok(())
    } 

        // get funccall ack from nodeagent
        pub fn OnFuncSvcCallAck(&self, ack: func::FuncSvcCallAck) -> Result<()> {
            let callerNode = FUNC_NODE_MGR.Get(&ack.caller_node_id)?;
            callerNode.Send(FuncNodeMsg::FuncCallAck(ack))?;
            return Ok(())
        } 

    // get funccall resp from another func node
    pub fn OnFuncCallResp(&self, resp: &func::FuncSvcCallResp) -> Result<()> {
        self.callerFuncCalls.lock().unwrap().remove(&resp.caller_pod_id);
        return Ok(())
    }
    
    // when a pod is disconnected
    pub fn OnFuncPodDisconnReq(&self) -> Result<()> {
        match self.state.lock().unwrap().clone() {
            FuncPodState::Idle(_) => (),
            FuncPodState::Running(funcCall) => {
                let callerNode = FUNC_NODE_MGR.Get(&funcCall.callerNodeId)?;
                let funcRes = FuncRes::NewError(
                    FuncErrSource::System, 
                    format!("funcpod {} disconnect ", &self.podName)
                );

                let resp = func::FuncSvcCallResp {
                    id: funcCall.id.clone(),
                    res: Some(funcRes.ToGrpc()),
                    caller_node_id: funcCall.callerNodeId.clone(),
                    caller_pod_id: funcCall.callerFuncPodId.clone(),
                    callee_node_id: funcCall.calleeNodeId.lock().unwrap().clone(),
                    callee_pod_id: funcCall.calleeFuncPodId.lock().unwrap().clone(),
                };
                callerNode.Send(FuncNodeMsg::FuncCallResp(resp))?;
            }
            _ => {}
        }

        *self.state.lock().unwrap() = FuncPodState::Dead;
        if !self.clientMode {
            FUNC_SVC_MGR.lock().unwrap().OnPodExit(self)?;
        }

        return Ok(())
    }

}

pub struct FuncPodMgr {
    pub pods: Mutex<BTreeMap<String, FuncPod>>,
}

impl FuncPodMgr {
    pub fn New() -> Self {
        return Self {
            pods: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn Get(&self, podName: &str) -> Result<FuncPod> {
        match self.pods.lock().unwrap().get(podName) {
            None => return Err(Error::ENOENT(format!("FuncPodMgr::Get can't find pod {}", podName))),
            Some(p) => return Ok(p.clone()),
        }
    }

    pub fn Remove(&self, podName: &str) -> Result<FuncPod> {
        match self.pods.lock().unwrap().remove(podName) {
            None => return Err(Error::ENOENT(format!("FuncPodMgr::Remove can't find pod {}", podName))),
            Some(p) => return Ok(p.clone()),
        }
    }

    pub fn Add(&self, funcPod: &FuncPod) {
        self.pods.lock().unwrap().insert(funcPod.podName.clone(), funcPod.clone());
    }
}