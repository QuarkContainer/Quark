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
use std::time::Duration;

use qobjs::common::*;
use qobjs::runtime_types::QuarkContainer;
use qobjs::runtime_types::QuarkPod;
use qobjs::nm::*;

use crate::container::*;
use crate::pod::*;
use crate::node::*;

pub enum MsgType {
    Node,
    Pod,
    Container,
}

#[derive(Debug)]
pub enum NodeAgentMsg {
    NodeUpdate,
    PodSandboxCreated(PodSandboxCreated),
    PodSandboxReady(PodSandboxReady),
    PodContainerCreated(PodContainerCreated),
    PodContainerStarted(PodContainerStarted),
    PodContainerStandy(PodContainerStandy),
    PodContainerUnhealthy(PodContainerUnhealthy),
    PodContainerReady(PodContainerReady),
    PodContainerStarting(PodContainerStarting),
    PodContainerStopping(PodContainerStopping),
    PodContainerStopped(PodContainerStopped),
    PodContainerTerminated(PodContainerTerminated),
    PodContainerFailed(PodContainerFailed),
    PodTerminate,
    PodHibernate,
    PodCreate(PodCreate),
    PodCleanup(PodCleanup),
    PodStatusChange(PodStatusChange),
    PodOOM(PodOOM),
    HouseKeeping,
    NodeMgrMsg(NodeAgentMessage)
}

#[derive(Debug)]
pub struct PodSandboxCreated {
    pub pod: QuarkPod,
}

#[derive(Debug)]
pub struct PodSandboxReady {
    pub pod: QuarkPod,
}

#[derive(Debug)]
pub struct PodContainerCreated {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerStarted {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerStandy {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerUnhealthy {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerReady {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerStarting {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerStopping {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
    pub gracePeriod: Duration,
}

#[derive(Debug)]
pub struct PodContainerStopped {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerTerminated {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodContainerFailed {
    pub pod: QuarkPod,
    pub container: QuarkContainer,
}

#[derive(Debug)]
pub struct PodCreate {
    pub pod: QuarkPod,
}

#[derive(Debug)]
pub struct PodCleanup {
    pub pod: QuarkPod,
}

#[derive(Debug)]
pub struct PodStatusChange {
    pub pod: QuarkPod,
}

#[derive(Debug)]
pub struct PodOOM {
    pub pod: QuarkPod,
}

pub struct NodeAgentSvc {
    pub node: NodeAgent,
    pub pods: BTreeMap<String, PodAgent>,
    pub containers: BTreeMap<String, PodContainerAgent>,
}

impl NodeAgentSvc {
    pub fn Send(&self, msgType: MsgType, agentId: &str, msg: NodeAgentMsg) -> Result<()> {
        match msgType {
            MsgType::Node => return self.node.Send(msg),
            MsgType::Pod => {
                match self.pods.get(agentId) {
                    None => return Err(Error::CommonError(format!("NodeAgentSvc::Send can't find pod {}", agentId))),
                    Some(agent) => {
                        return agent.Send(msg);
                    }
                }
            }
            MsgType::Container => {
                match self.containers.get(agentId) {
                    None => return Err(Error::CommonError(format!("NodeAgentSvc::Send can't find container {}", agentId))),
                    Some(agent) => {
                        return agent.Send(msg);
                    }
                }
            }
        }
    }
}