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
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;
use std::collections::BTreeMap;
use core::ops::Deref;
use tokio::sync::Notify;

use qobjs::common::*;

use crate::func_agent::funcpod::FuncPod;

use super::func_agent::FuncCall;


#[derive(Debug, Default)]
pub struct FuncInstMgrInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,
    pub currInstanceId: AtomicU64,

    pub instances: Mutex<BTreeMap<String, FuncPod>>,
    // func instance id to funcCall
    pub ingressCall: Mutex<BTreeMap<String, FuncCall>>,
    // func instance id to funcCall
    pub egressCall: Mutex<BTreeMap<String, FuncCall>>,
}

#[derive(Debug, Default, Clone)]
pub struct FuncInstMgr(pub Arc<FuncInstMgrInner>);

impl Deref for FuncInstMgr {
    type Target = Arc<FuncInstMgrInner>;

    fn deref(&self) -> &Arc<FuncInstMgrInner> {
        &self.0
    }
}

impl FuncInstMgr {
    pub fn AddInstance(&self, id: &str, instance: &FuncPod) -> Result<()> {
        self.instances.lock().unwrap().insert(id.to_string(), instance.clone());
        return Ok(())
    }
}

