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

use std::{result::Result as SResult, sync::atomic::AtomicI64};
use std::sync::Arc;
use tonic::Status;
use qobjs::pb_gen::node_mgr_pb::{self as nm_svc};
use tokio::sync::mpsc;
use core::ops::Deref;

use qobjs::common::*;

#[derive(Debug)]
pub struct NodeAgentInner {
    pub sender: mpsc::Sender<SResult<nm_svc::NodeAgentMessage, Status>>,
    pub revision: AtomicI64,
}

#[derive(Clone, Debug)]
pub struct NodeAgent(Arc<NodeAgentInner>);

impl Deref for NodeAgent {
    type Target = Arc<NodeAgentInner>;

    fn deref(&self) -> &Arc<NodeAgentInner> {
        &self.0
    }
}

impl NodeAgent {
    //pub fn New()

    pub async fn Send(&self, msg: nm_svc::NodeAgentMessage) -> Result<()>{
        match self.sender.send(Ok(msg)).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                return Err(Error::CommonError(format!("NodeAgent Send error {:?}", e)));
            }
        }
    }


}