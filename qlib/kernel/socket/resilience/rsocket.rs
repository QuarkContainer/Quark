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

use alloc::sync::Arc;
use core::ops::Deref;

use crate::qlib::kernel::fs::file::*;
//use super::message::*;

pub struct ResilienceSocketInner {
    pub connection: Arc<File>,
    pub nextReqId: u64,
    pub nextSessionId: u64,
}

pub struct ResilienceSocketOpsInner {
    pub sessionId: u64,
}

#[derive(Clone)]
pub struct ResilienceSocketOps(pub Arc<ResilienceSocketOpsInner>);

impl Deref for ResilienceSocketOps {
    type Target = Arc<ResilienceSocketOpsInner>;

    fn deref(&self) -> &Arc<ResilienceSocketOpsInner> {
        &self.0
    }
}


// impl ResilienceSocketOps {
//     pub fn Write(&self, _funcName: &[u8], buf: &[u8]) {
//         let msgCall = UserFuncCall::New(buf);
//         msgCall.UserData();
//     }
// }
