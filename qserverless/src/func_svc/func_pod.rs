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

use qobjs::common::*;

use crate::func_call::FuncCall;
use crate::package::*;
use crate::func_node::*;

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

#[derive(Debug)]
pub enum FuncPodState {
    Idle(SystemTime), // IdleTime
    Running(String), 
}

#[derive(Debug)]
pub struct FuncPodInner {
    pub podName: String,
    pub package: Package,
    pub node: FuncNode,
    pub state: Mutex<FuncPodState>,
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
    pub fn RunTask(&self, _task: &FuncCall) -> Result<()> {
        unimplemented!();
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
}