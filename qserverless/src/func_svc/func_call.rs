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

use core::ops::Deref;
use std::sync::{Arc, Mutex, Weak}; 
use std::time::SystemTime;
use std::collections::BTreeMap;

use qobjs::common::*;

use crate::scheduler::Resource;
use crate::package::*;

#[derive(Debug, Clone)]
pub struct FuncCallId {
    pub packageId: PackageId,
    pub funcName: String,
}

impl Ord for FuncCallId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.packageId == other.packageId {
            return other.funcName.cmp(&self.funcName);
        }

        return other.packageId.cmp(&other.packageId);
    }
}

impl PartialOrd for FuncCallId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for FuncCallId {
    fn eq(&self, other: &Self) -> bool {
        return self.packageId == other.packageId && self.funcName == other.funcName;
    }
}

impl Eq for FuncCallId {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncCallResult {
    Err(String),
    Ok(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncCallState {
    // FuncSvc get func request and put in pending queue
    Scheduling(SystemTime),
    // FuncSvc scheduled the Func to one FuncAgent
    Scheduled, // the content is callee NodeId
    // there is callee, but caller is not online
    PendingCaller(SystemTime),
    // there is Caller, but callee is not online
    PendingCallee(SystemTime), 
    // there is callee and result is done, waiting for caller
    PendingCallerWithResult((SystemTime, FuncCallResult)),
    //
    Cancelling,
}

impl FuncCallState {
    pub fn IsCancelling(&self) -> bool {
        match self {
            FuncCallState::Cancelling => return true,
            _ => return false,
        }
    }

    pub fn IsScheduling(&self) -> bool {
        match self {
            FuncCallState::Scheduling(_) => return true,
            _ => return false,
        }
    }
}

#[derive(Debug)]
pub struct FuncCallInner {
    pub id: String,
    pub package: Package,
    pub funcName: String,
    
    pub callerNodeId: String,
    pub callerFuncPodId: String,
    pub calleeNodeId: Mutex<String>,
    pub calleeFuncPodId: Mutex<String>,
    pub state: Mutex<FuncCallState>,

    pub parameters: String,
    pub priority: usize,
    pub createTime: SystemTime,
}

#[derive(Debug, Clone)]
pub struct FuncCall(pub Arc<FuncCallInner>);

impl Deref for FuncCall {
    type Target = Arc<FuncCallInner>;

    fn deref(&self) -> &Arc<FuncCallInner> {
        &self.0
    }
}

impl FuncCall {
    pub fn Downgrade(&self) -> FuncCallWeak {
        return FuncCallWeak(Arc::downgrade(&self.0));
    }

    pub fn ReqResource(&self) -> Resource {
        return self.package.ReqResource();
    }

    pub fn Priority(&self) -> usize {
        return self.priority as usize;
    }

    pub fn Package(&self) -> Package {
        return self.package.clone();
    }

    pub fn SetState(&self, state: FuncCallState) {
        *self.state.lock().unwrap() = state;
    }

    pub fn Match(&self, _other: &FuncCall) -> bool {
        // todo:
        return true;
    }

}

#[derive(Debug, Clone)]
pub struct FuncCallWeak(pub Weak<FuncCallInner>);

impl Deref for FuncCallWeak {
    type Target = Weak<FuncCallInner>;

    fn deref(&self) -> &Weak<FuncCallInner> {
        &self.0
    }
}

impl FuncCallWeak {
    pub fn Upgrade(&self) -> Option<FuncCall> {
        match self.0.upgrade() {
            None => None,
            Some(d) => {
                return Some(FuncCall(d));
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct FuncCallMgrInner {
    pub funcCalls: BTreeMap<String, FuncCall>,
    // the function callee has finished with result, but the caller is not online
    pub pendingResultFuncCalls: BTreeMap<SystemTime, String>,
    // the function is running by some callee, but the caller is not online
    pub pendingCallerFuncCalls: BTreeMap<SystemTime, String>,
    // the function apppears in caller and shows running, but callee node is not online
    pub pendingCalleeFuncCalls: BTreeMap<SystemTime, String>,
}

#[derive(Debug, Default)]
pub struct FuncCallMgr(Arc<Mutex<FuncCallMgrInner>>);

impl Deref for FuncCallMgr {
    type Target = Arc<Mutex<FuncCallMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncCallMgrInner>> {
        &self.0
    }
}

impl FuncCallMgr {
    // when a node register in func service, register the list of funccall it is working on
    pub fn RegisteCallee(&self, funcCall: &FuncCall) -> Result<()> {
        let mut inner = self.lock().unwrap();
        let now = SystemTime::now();
        match inner.funcCalls.get(&funcCall.id).cloned() {
            None => {
                funcCall.SetState(FuncCallState::PendingCaller(now));
                inner.pendingCallerFuncCalls.insert(now, funcCall.id.clone());
                inner.funcCalls.insert(funcCall.id.clone(), funcCall.clone());
                return Ok(())
            }
            Some(curr) => {
                if !curr.Match(funcCall) {
                    error!("find unmatched func call...");
                    return Ok(())
                }
                {
                    let mut statlock = curr.state.lock().unwrap();
                    match *statlock {
                        FuncCallState::PendingCallee(time) => {
                            if *curr.calleeFuncPodId.lock().unwrap() != *funcCall.calleeFuncPodId.lock().unwrap() {
                                return Ok(())
                            }
                            inner.pendingCalleeFuncCalls.remove(&time);
                        }
                        _ => {
                            // the func has no callee before, just drop it silently
                            return Ok(())
                        }
                    }
                    *statlock = FuncCallState::Scheduled;
                }
                
            }
        }

        return Ok(())
    }

    pub fn RegisteCaller(&self, funcCall: &FuncCall) -> Result<Option<FuncCallResult>> {
        let mut inner = self.lock().unwrap();
        let now = SystemTime::now();
        match inner.funcCalls.get(&funcCall.id).cloned() {
            None => {
                funcCall.SetState(FuncCallState::PendingCallee(now));
                inner.pendingCalleeFuncCalls.insert(now, funcCall.id.clone());
                inner.funcCalls.insert(funcCall.id.clone(), funcCall.clone());
                return Ok(None)
            }
            Some(curr) => {
                if !curr.Match(funcCall) {
                    error!("find unmatched func call...");
                    return Ok(None)
                }
                {
                    let mut statlock = curr.state.lock().unwrap();
                    match &*statlock {
                        FuncCallState::PendingCaller(time) => {
                            if *curr.callerFuncPodId != *funcCall.callerFuncPodId {
                                return Ok(None)
                            }
                            inner.pendingCallerFuncCalls.remove(&time);
                        }
                        FuncCallState::PendingCallerWithResult((time, result)) => {
                            inner.pendingResultFuncCalls.remove(&time);
                            return Ok(Some(result.clone()));
                        }
                        _ => {
                            // the func has no callee before, just drop it silently
                            return Ok(None)
                        }
                    }
                    *statlock = FuncCallState::Scheduled;
                }
                
            }
        }

        return Ok(None)
    }
}