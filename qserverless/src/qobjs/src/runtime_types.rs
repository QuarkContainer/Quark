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

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::SystemTime;
use std::sync::Arc;
use core::ops::Deref;
use std::sync::Mutex;

use k8s_openapi::api::core::v1 as k8s;

use crate::pb_gen::v1alpha2 as cri;
use crate::k8s_util::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct RuntimePod {
    pub id: String,
    pub IPs: Vec<String>,
    pub sandboxConfig: Option<cri::PodSandboxConfig>,
    pub sandbox: cri::PodSandbox,
}

// Pod is a sandbox container and a group of containers.
#[derive(Serialize, Deserialize, Debug)]
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

// ContainerStatus represents the cri status of a container and fornax status.
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

#[derive(Serialize, Deserialize, Debug)]
pub struct QuarkContainerInner {
    pub state: RuntimeContainerState,
    pub initContainer: bool,
    pub spec: k8s::Container,
    pub runtimeContainer: RuntimeContainer,
    pub containerStatus: Option<cri::ContainerStatus>,
}

#[derive(Debug, Clone)]
pub struct QuarkContainer(pub Arc<Mutex<QuarkContainerInner>>);

impl Deref for QuarkContainer {
    type Target = Arc<Mutex<QuarkContainerInner>>;

    fn deref(&self) -> &Arc<Mutex<QuarkContainerInner>> {
        &self.0
    }
}

impl QuarkContainer {
    pub fn State(&self) -> RuntimeContainerState {
        return self.lock().unwrap().state
    }

    pub fn SetState(&self, state: RuntimeContainerState) {
        self.lock().unwrap().state = state;
    }

    pub fn ContainerName(&self) -> String {
        return self.lock().unwrap().spec.name.clone();
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum PodState {
    Creating,
    // a preserved state for standby pod use
    Created,
    // a state when startup and readiness probe passed
    Running,
    // a state preserved for use to evacuate session
    Evacuating,
    Terminating,
    // a normal pod exit status when fornax core request to terminate
    Terminated,
    // a abnormal pod exit status when pod met unexpected condtion
    Failed,
    // pod artifacts are cleaned, eg. pod dir, cgroup// pod artifacts are cleaned, eg. pod dir, cgroup
    Cleanup,
}

#[derive(Debug)]
pub struct QuarkPodInner {
    pub id: String,
    pub podState: PodState,
    pub isDaemon: bool,
    pub pod: Arc<k8s::Pod>,
    pub configMap: k8s::ConfigMap,
    pub runtimePod: Arc<RuntimePod>,
    pub containers: BTreeMap<String, QuarkContainer>,
    pub lastTransitionTime: SystemTime,
}

#[derive(Debug, Clone)]
pub struct QuarkPod(Arc<Mutex<QuarkPodInner>>);

impl Deref for QuarkPod {
    type Target = Arc<Mutex<QuarkPodInner>>;

    fn deref(&self) -> &Arc<Mutex<QuarkPodInner>> {
        &self.0
    }
}

impl QuarkPod {
    pub fn Pod(&self) -> Arc<k8s::Pod> {
        return self.lock().unwrap().pod.clone();
    }

    pub fn PodId(&self) -> String {
        return K8SUtil::PodId(&self.lock().unwrap().pod);
    }

    pub fn RuntimePod(&self) -> Arc<RuntimePod> {
        return self.lock().unwrap().runtimePod.clone();
    }
}

pub const DefaultFornaxCoreNodeNameSpace: &str = "qserverless.quark.io";
pub const DefaultDomainName: &str = "qserverless.quark.io";

// NodePending means the node has been created/added by the system, but not configured.
pub const NodePending : &str = "Pending";
// NodeRunning means the node has been configured and has Kubernetes components running.
pub const NodeRunning : &str = "Running";
// NodeTerminated means the node has been removed from the cluster.
pub const NodeTerminated : &str = "Terminated";


// NodeReady means kubelet is healthy and ready to accept pods.
pub const NodeReady: &str = "Ready";
// NodeMemoryPressure means the kubelet is under pressure due to insufficient available memory.
pub const NodeMemoryPressure: &str = "MemoryPressure";
// NodeDiskPressure means the kubelet is under pressure due to insufficient available disk.
pub const NodeDiskPressure: &str = "DiskPressure";
// NodePIDPressure means the kubelet is under pressure due to insufficient available PID.
pub const NodePIDPressure: &str = "PIDPressure";
// NodeNetworkUnavailable means that network for the node is not correctly configured.
pub const NodeNetworkUnavailable: &str = "NetworkUnavailable";

// These are valid condition statuses. "ConditionTrue" means a resource is in the condition.
// "ConditionFalse" means a resource is not in the condition. "ConditionUnknown" means kubernetes
// can't decide if a resource is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
pub const ConditionTrue    : &str = "True";
pub const ConditionFalse   : &str = "False";
pub const ConditionUnknown : &str = "Unknown";

