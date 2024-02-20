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

use qobjs::func;

use crate::func_pod::FuncPod;
use crate::func_call::FuncCall;

#[derive(Debug)]
pub enum FuncNodeMsg {
    FuncCall(FuncCall),
    FuncCallResp(func::FuncSvcCallResp),
    FuncCallAck(func::FuncSvcCallAck),
    FuncPodConnResp(func::FuncPodConnResp),
    FuncMsg(func::FuncMsg),
}

pub struct FuncCalleeMsg {
    pub pod: FuncPod,
    pub funcCall: FuncCall,
}

pub struct FuncAgentRegister {
    pub id: String,
}