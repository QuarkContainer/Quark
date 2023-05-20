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

use crate::func_context::*;
use crate::scheduler::*;

pub struct Node {
    pub id: String,
    pub conn: Option<FuncConn>,
    pub totalResource: Resource,

    pub remainResource: Resource,
}

#[derive(Debug)]
// Connection with FuncAgent
pub struct FuncConnInner {
    // AgentId won't change after function agent restart
    pub agentId: String, 

    // AgentUid will change after function agent restart, 
    // if so, all the running ingress functions of agent will be re-scheduled
    // and all the running egress functions will be cancelled
    pub agentUid: String, 

    // func instance id to funcCall
    pub funcCalls: BTreeMap<u64, FuncCallContext>,
    

    
}

pub struct FuncConn {

}