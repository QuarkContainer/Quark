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


use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::crictl as cri;

pub enum PodType {
    Normal,
    Python(String)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RuntimePod {
    pub id: String,
    pub IPs: Vec<String>,
    pub sandboxConfig: Option<cri::PodSandboxConfig>,
    pub sandbox: Option<cri::PodSandbox>,
}

// Pod is a sandbox container and a group of containers.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RuntimeContainer {
    pub id: String,
    pub containerConfig: Option<cri::ContainerConfig>,
    pub container: cri::Container,
}

// PodStatus represents the status of the pod and its containers.
#[derive(Serialize, Deserialize, Debug)]
pub struct RuntimePodStatus {
    pub sandboxStatus: cri::PodSandboxStatus,
    pub containerStatus: BTreeMap<String, cri::ContainerStatus>
}

// ContainerStatus represents the cri status of a container and nodeagent status.
#[derive(Serialize, Deserialize, Debug)]
pub struct RuntimeContainerStatus {
    pub status: cri::ContainerStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeContainerState {
    Creating,
    Created,
    Stopping,
    Stopped,
    Terminated,
    Terminating,
    Running,
    Started,
}
