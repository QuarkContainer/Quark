// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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

use std::time::Duration;

use super::{qcontainer::QuarkContainer, qpod::QuarkPod};

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
