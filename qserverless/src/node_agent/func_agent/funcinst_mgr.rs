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

use std::{sync::{Arc, atomic::{AtomicBool, AtomicU64}}, collections::BTreeMap};
use core::ops::Deref;
use tokio::sync::Notify;

use crate::func_agent::funcinst::FuncInstance;
use crate::func_agent::func_call::*;

pub struct FuncInstMgrInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,
    pub currInstanceId: AtomicU64,

    pub instances: BTreeMap<u64, FuncInstance>,
    // func instance id to funcCall
    pub ingressCall: BTreeMap<u64, FuncCall>,
    // func instance id to funcCall
    pub egressCall: BTreeMap<u64, FuncCall>,
}

#[derive(Clone)]
pub struct FuncInstMgr(pub Arc<FuncInstMgrInner>);

impl Deref for FuncInstMgr {
    type Target = Arc<FuncInstMgrInner>;

    fn deref(&self) -> &Arc<FuncInstMgrInner> {
        &self.0
    }
}

