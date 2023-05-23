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
use std::{sync::{Arc, Mutex, Weak}, time::SystemTime, collections::BTreeMap};

use crate::{scheduler::Resource, func_node::FuncNode};
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

#[derive(Debug, Clone)]
pub enum FuncCallState {
    // FuncSvc get func request and put in pending queue
    Scheduling,
    // FuncSvc scheduled the Func to one FuncAgent
    Scheduled(String),
    // Callee FuncAgent complete the request and return result to caller
    Complete(String),
}

#[derive(Debug)]
pub struct FuncCallContextInner {
    pub id: FuncCallId,
    pub package: Package,
    pub callerNode: FuncNode,

    pub state: FuncCallState,

    pub parameters: String,
    pub priority: i32,
    pub createTime: SystemTime,
}

#[derive(Debug, Clone)]
pub struct FuncCallContext(pub Arc<Mutex<FuncCallContextInner>>);

impl Deref for FuncCallContext {
    type Target = Arc<Mutex<FuncCallContextInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncCallContextInner>> {
        &self.0
    }
}

impl FuncCallContext {
    pub fn Downgrade(&self) -> FuncCallContextWeak {
        return FuncCallContextWeak(Arc::downgrade(&self.0));
    }

    pub fn ReqResource(&self) -> Resource {
        return self.lock().unwrap().package.ReqResource();
    }

    pub fn Id(&self) -> FuncCallId {
        return self.lock().unwrap().id.clone();
    }
}

#[derive(Debug, Clone)]
pub struct FuncCallContextWeak(pub Weak<Mutex<FuncCallContextInner>>);

impl Deref for FuncCallContextWeak {
    type Target = Weak<Mutex<FuncCallContextInner>>;

    fn deref(&self) -> &Weak<Mutex<FuncCallContextInner>> {
        &self.0
    }
}

impl FuncCallContextWeak {
    pub fn Upgrade(&self) -> Option<FuncCallContext> {
        match self.0.upgrade() {
            None => None,
            Some(d) => {
                return Some(FuncCallContext(d));
            }
        }
    }
}

pub struct FuncCallMgr {
    pub funcCalls: Mutex<BTreeMap<FuncCallId, FuncCallContext>>,
}

impl FuncCallMgr {
    pub fn New() -> Self {
        return Self {
            funcCalls: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn Add(&self, funcCall: &FuncCallContext) {
        self.funcCalls.lock().unwrap().insert(funcCall.Id(), funcCall.clone());
    }

    pub fn Get(&self, id: &FuncCallId) -> Option<FuncCallContext> {
        return self.funcCalls.lock().unwrap().get(id).cloned();
    }
}