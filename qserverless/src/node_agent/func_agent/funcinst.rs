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

use qobjs::func;


#[derive(Debug)]
pub enum InstanceState {
    Idle,
    Running(u64), // handling FuncCallId
}

pub struct FuncInstanceInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub id: u64,
    pub instanceId: String,
    pub state: InstanceState,
    pub agentChann: mpsc::Sender<func::FuncAgentMsg>,
}

#[derive(Clone)]
pub struct FuncInstance(pub Arc<FuncInstanceInner>);

impl Deref for FuncInstance {
    type Target = Arc<FuncInstanceInner>;

    fn deref(&self) -> &Arc<FuncInstanceInner> {
        &self.0
    }
}

