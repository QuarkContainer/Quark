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

use k8s_openapi::apimachinery::pkg::apis::meta::v1::Time;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::SystemTime;
use std::sync::{Arc, RwLock};
use core::ops::Deref;
use std::sync::Mutex;
use chrono::prelude::*;

use k8s_openapi::api::core::v1 as k8s;

use crate::common::*;
use crate::v1alpha2 as cri;
use crate::k8s_util::*;

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


#[derive(Serialize, Deserialize, Debug, Clone)]
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

    pub fn InitContainer(&self) -> bool {
        return self.lock().unwrap().initContainer;
    }

    pub fn ContainerExit(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerExit(&status); 
    }

    pub fn ContainerExitNormal(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerExitNormal(&status); 
    }

    pub fn ContainerExitAbnormal(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerExitAbnormal(&status); 
    }

    pub fn ContainerStatusUnknown(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerStatusUnknown(&status); 
    }

    pub fn ContainerRunning(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerRunning(&status); 
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
    // a normal pod exit status when nodemgr request to terminate
    Terminated,
    // a abnormal pod exit status when pod met unexpected condtion
    Failed,
    // pod artifacts are cleaned, eg. pod dir, cgroup// pod artifacts are cleaned, eg. pod dir, cgroup
    Cleanup,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QuarkPodJson {
    pub id: String,
    pub podState: PodState,
    pub isDaemon: bool,
    pub pod: k8s::Pod,
    pub configMap: Option<k8s::ConfigMap>,
    pub runtimePod: Option<RuntimePod>,
    pub containers: BTreeMap<String, QuarkContainerInner>,
    pub lastTransitionTime: SystemTime,
}

#[derive(Debug)]
pub struct QuarkPodInner {
    pub id: String,
    pub podState: PodState,
    pub isDaemon: bool,
    pub pod: Arc<RwLock<k8s::Pod>>,
    pub configMap: Option<k8s::ConfigMap>,
    pub runtimePod: Option<Arc<RuntimePod>>,
    pub containers: BTreeMap<String, QuarkContainer>,
    pub lastTransitionTime: SystemTime,
}

// PodPending means the pod has been accepted by the system, but one or more of the containers
// has not been started. This includes time before being bound to a node, as well as time spent
// pulling images onto the host.
pub const PodPending: &str = "Pending";
// PodRunning means the pod has been bound to a node and all of the containers have been started.
// At least one container is still running or is in the process of being restarted.
pub const PodRunning: &str = "Running";
// PodSucceeded means that all containers in the pod have voluntarily terminated
// with a container exit code of 0, and the system is not going to restart any of these containers.
pub const PodSucceeded: &str = "Succeeded";
// PodFailed means that all containers in the pod have terminated, and at least one container has
// terminated in a failure (exited with a non-zero exit code or was stopped by the system).
pub const PodFailed: &str = "Failed";
// PodUnknown means that for some reason the state of the pod could not be obtained, typically due
// to an error in communicating with the host of the pod.
// Deprecated: It isn't being set since 2015 (74da3b14b0c0f658b3bb8d2def5094686d0e9095)
pub const PodUnknown: &str = "Unknown";

#[derive(Debug, Clone)]
pub struct QuarkPod(pub Arc<Mutex<QuarkPodInner>>);

impl Deref for QuarkPod {
    type Target = Arc<Mutex<QuarkPodInner>>;

    fn deref(&self) -> &Arc<Mutex<QuarkPodInner>> {
        &self.0
    }
}

impl QuarkPod {
    pub fn ToQuarkPodJson(&self) -> QuarkPodJson {
        let inner = self.lock().unwrap();
        let mut map = BTreeMap::new();
        for (k, v) in &inner.containers {
            map.insert(k.clone(), v.lock().unwrap().clone());
        }

        let pod = inner.pod.read().unwrap().clone();

        let runtimePod = match &inner.runtimePod {
            None => None,
            Some(p) => Some(p.as_ref().clone())
        };

        return QuarkPodJson {
            id: inner.id.clone(),
            podState: inner.podState,
            isDaemon: inner.isDaemon,
            pod: pod,
            configMap: inner.configMap.clone(),
            runtimePod: runtimePod,
            containers: map,
            lastTransitionTime: inner.lastTransitionTime,
        }
    }

    pub fn FromQuarkPodJosn(pod: QuarkPodJson) -> Self {
        let inner = QuarkPodInner {
            id: pod.id,
            podState: pod.podState,
            isDaemon: pod.isDaemon,
            pod: Arc::new(RwLock::new(pod.pod)),
            configMap: pod.configMap,
            runtimePod: match pod.runtimePod {
                None => None,
                Some(p) => Some(Arc::new(p))
            },
            containers: {
                let mut map = BTreeMap::new();
                for (k, v) in pod.containers {
                    map.insert(k, QuarkContainer(Arc::new(Mutex::new(v))));
                }
                map
            },
            lastTransitionTime: pod.lastTransitionTime,
        };

        return Self(Arc::new(Mutex::new(inner)))
    }

    pub fn Pod(&self) -> Arc<RwLock<k8s::Pod>> {
        return self.lock().unwrap().pod.clone();
    }

    pub fn PodId(&self) -> String {
        return K8SUtil::PodId(&self.Pod().read().unwrap());
    }

    pub fn RuntimePod(&self) -> Option<Arc<RuntimePod>> {
        return self.lock().unwrap().runtimePod.clone();
    }

    pub fn PodState(&self) -> PodState {
        return self.lock().unwrap().podState;
    }

    pub fn SetPodState(&self, state: PodState) {
        self.lock().unwrap().podState = state;
    }

    pub fn PodInTerminating(&self) -> bool {
        let state = self.PodState();
        return state == PodState::Terminating 
            || state == PodState::Terminated 
            || state == PodState::Cleanup;
    }

    pub fn PodCreated(&self) -> bool {
        let state = self.PodState();
        return state != PodState::Creating || self.RuntimePod().is_some();
    }

    pub fn PodInTransitState(&self) -> bool {
        let state = self.PodState();
        return state == PodState::Creating 
            || state == PodState::Evacuating 
            || state == PodState::Terminating
            || state == PodState::Created;
    }

    pub fn ToK8sPhase(&self) -> String {
        let state = self.PodState();
        let mut podPhase;
        match state {
            PodState::Creating => podPhase = PodPending.to_string(),
            PodState::Created => podPhase = PodPending.to_string(),
            PodState::Running => podPhase = PodRunning.to_string(),
            PodState::Terminating => podPhase = PodUnknown.to_string(),
            PodState::Cleanup => {
                /*if fppod.RuntimePod == nil || fppod.RuntimePod.Sandbox == nil {
                    podPhase = v1.PodFailed
                } else {
                    podPhase = v1.PodSucceeded
                }*/
                podPhase = PodSucceeded.to_string();
            }
            PodState::Terminated => {
                /*if fppod.RuntimePod == nil || fppod.RuntimePod.Sandbox == nil {
                    podPhase = v1.PodFailed
                } else {
                    podPhase = v1.PodSucceeded
                }*/
                podPhase = PodSucceeded.to_string();
            }
            PodState::Failed => return PodFailed.to_string(),
            _ => {
                podPhase = PodUnknown.to_string();
            }
        }

        for v in self.lock().unwrap().containers.values() {
            let status = v.lock().unwrap().containerStatus.clone();
            if ContainerExitAbnormal(&status) {
                podPhase = PodFailed.to_string();
                break;
            }
        }

        return podPhase;
    }

    pub fn SetPodStatus(&self, node: Option<&k8s::Node>) {
        let pod = self.Pod();
        
        let mut podStatus = pod.read().unwrap().status.clone().unwrap_or(k8s::PodStatus::default());
        podStatus.phase = Some(self.ToK8sPhase());

        if let Some(node) = node {
            podStatus.host_ip = Some(node.status.as_ref().unwrap().addresses.as_ref().unwrap()[0].address.clone());
        }   
    
        let hasDeleted = self.lock().unwrap().pod.read().unwrap().metadata.deletion_timestamp.is_some();
        if (podStatus.phase.as_ref().unwrap() == PodSucceeded 
            || podStatus.phase.as_ref().unwrap() == PodFailed)
            &&  !hasDeleted {
                let utc: DateTime<Utc> = Utc::now();
                self.lock().unwrap().pod.write().unwrap().metadata.deletion_timestamp = Some(Time(utc));
        }

        if podStatus.conditions.is_none() {
            podStatus.conditions = None;
        }
        podStatus.conditions = Some(self.GetPodConditions());

        if self.RuntimePod().is_some() {
            if self.RuntimePod().as_ref().unwrap().IPs.len() > 0 {
                podStatus.pod_ip = Some(self.RuntimePod().as_ref().unwrap().IPs[0].clone());
                let mut podIps = Vec::new();
                for v in &self.RuntimePod().as_ref().unwrap().IPs {
                    podIps.push(k8s::PodIP{
                        ip: Some(v.to_string()),
                    })
                }
                podStatus.pod_ips = Some(podIps);
            }
        }

        if podStatus.start_time.is_none() {
            let utc = Utc::now();
            podStatus.start_time = Some(Time(utc));
        }

        self.Pod().write().unwrap().status = Some(podStatus);
    }

    pub fn Containers(&self) -> Vec<QuarkContainer> {
        let containers: Vec<QuarkContainer> = self.lock().unwrap().containers.values().cloned().collect();
        return containers;
    }

    pub fn GetPodConditions(&self) -> Vec<k8s::PodCondition> {
        let mut conditions = BTreeMap::new();

        let utc: DateTime<Utc> = Utc::now();
        let mut initReadyCondition = k8s::PodCondition {
            type_: PodInitialized.to_string(),
            status: ConditionUnknown.to_string(),
            last_probe_time: Some(Time(utc)),
            ..Default::default()
        };
        
        let mut containerReadyCondition = k8s::PodCondition {
            type_: ContainersReady.to_string(),
            status: ConditionUnknown.to_string(),
            last_probe_time: Some(Time(utc)),
            ..Default::default()
        };
        
        let mut podReadyCondition = k8s::PodCondition {
            type_: PodReady.to_string(),
            status: ConditionUnknown.to_string(),
            last_probe_time: Some(Time(utc)),
            ..Default::default()
        };
        
        let podScheduledCondition = k8s::PodCondition {
            type_: PodScheduled.to_string(),
            status: ConditionTrue.to_string(),
            last_probe_time: Some(Time(utc)),
            ..Default::default()
        };


        let mut allInitContainerNormal = true;
        let containers: Vec<QuarkContainer> = self.lock().unwrap().containers.values().cloned().collect();
        for v in &containers {
            let c = v.lock().unwrap();
            if c.initContainer {
                if !ContainerExit(&c.containerStatus) {
                    containerReadyCondition.status = ConditionFalse.to_string();
                    containerReadyCondition.message = Some("init container not finished yet".to_string());
                    containerReadyCondition.reason = Some("init container not finished yet".to_string());
                    allInitContainerNormal = false;
                    break;
                }

                if ContainerExitAbnormal(&c.containerStatus) { 
                    containerReadyCondition.status = ConditionFalse.to_string();
                    containerReadyCondition.message = Some("init container exit abnormally".to_string());
                    containerReadyCondition.reason = Some("init container exit abnormally".to_string());
                    allInitContainerNormal = false;
                    break;
                }
                allInitContainerNormal = allInitContainerNormal && ContainerExitNormal(&c.containerStatus);
            }
        }

        if allInitContainerNormal {
            initReadyCondition.status = ConditionTrue.to_string();
            initReadyCondition.message = Some("init container exit normally".to_string());
            initReadyCondition.reason = Some("init container exit normally".to_string());
        }
        
        
        let mut allContainerNormal = true;
        //let mut allContainerReady = true;

        // todo: fix
        /* 
        if fppod.RuntimePod == nil || (len(fppod.Pod.Spec.Containers)+len(fppod.Pod.Spec.InitContainers) != len(fppod.RuntimePod.Containers)) {
		containerReadyCondition.Status = v1.ConditionFalse
		containerReadyCondition.Message = "missing some containers"
		containerReadyCondition.Reason = "missing some containers"
	} else { */
        for v in &containers {
            let c = v.lock().unwrap();
            if !c.initContainer {
                if !ContainerRunning(&c.containerStatus) {
                    containerReadyCondition.status = ConditionFalse.to_string();
                    containerReadyCondition.message = Some("one container is not running".to_string());
                    containerReadyCondition.reason = Some("one container is not running".to_string());
                    allContainerNormal = false;
                    //allContainerReady = false;
                    break;
                }

                allContainerNormal = allContainerNormal && ContainerRunning(&c.containerStatus);
                //allContainerReady = allContainerReady && c.state == RuntimeContainerState::Running;
            }
        }

        if allContainerNormal {
            containerReadyCondition.status = ConditionTrue.to_string();
            containerReadyCondition.message = Some("all pod containers are running".to_string());
            containerReadyCondition.reason = Some("all pod containers are running".to_string());
        }

        if &containerReadyCondition.status == ConditionTrue && &initReadyCondition.status == ConditionTrue {
            if self.PodState() == PodState::Running {
                podReadyCondition.status = ConditionTrue.to_string();
                podReadyCondition.message = Some("all pod containers are ready".to_string());
                podReadyCondition.reason = Some("all pod containers are ready".to_string());
            }
        } else {
            podReadyCondition.status = ConditionFalse.to_string();
            podReadyCondition.message = Some("some pod containers are not running".to_string());
            podReadyCondition.reason = Some("some pod containers are not running".to_string());
        }

        conditions.insert(PodInitialized.to_string(), initReadyCondition);
        conditions.insert(ContainersReady.to_string(), containerReadyCondition);
        conditions.insert(PodReady.to_string(), podReadyCondition);
        conditions.insert(PodScheduled.to_string(), podScheduledCondition);

        let pod = self.Pod();

        if pod.read().unwrap().status.is_none() {
            pod.write().unwrap().status = Some(k8s::PodStatus {
                conditions: Some(Vec::new()),
                ..Default::default()
            });
        }

        for oldCondition in pod.read().unwrap().status.as_ref().unwrap().conditions.as_ref().unwrap() {
            match conditions.get_mut(&oldCondition.type_) {
                Some(newCondition) => {
                    if oldCondition.status != newCondition.status {
                        newCondition.last_transition_time = oldCondition.last_probe_time.clone();
                    }
                }
                None => ()
            }
        }

        let newConditions = conditions.values().cloned().collect::<Vec<_>>();

        return newConditions;
    }
}


// ContainersReady indicates whether all containers in the pod are ready.
pub const ContainersReady: &str = "ContainersReady";
// PodInitialized means that all init containers in the pod have started successfully.
pub const PodInitialized: &str = "Initialized";
// PodReady means the pod is able to service requests and should be added to the
// load balancing pools of all matching services.
pub const PodReady: &str = "Ready";
// PodScheduled represents status of the scheduling process for this pod.
pub const PodScheduled: &str = "PodScheduled";

pub fn ContainerExit(status: &Option<cri::ContainerStatus>) -> bool {
    if let Some(status) = status {
        return status.finished_at != 0 && status.state == cri::ContainerState::ContainerCreated as i32;
    }

    return false;
}

pub fn ContainerExitNormal(status: &Option<cri::ContainerStatus>) -> bool {
    return ContainerExit(status) && status.as_ref().unwrap().exit_code == 0;
}

pub fn ContainerExitAbnormal(status: &Option<cri::ContainerStatus>) -> bool {
    return ContainerExit(status) && status.as_ref().unwrap().exit_code != 0;
}

pub fn ContainerStatusUnknown(status: &Option<cri::ContainerStatus>) -> bool {
    if let Some(status) = status {
        return status.state == cri::ContainerState::ContainerUnknown as i32;
    }

    return false;
}

pub fn ContainerRunning(status: &Option<cri::ContainerStatus>) -> bool {
    if let Some(status) = status {
        return status.state == cri::ContainerState::ContainerRunning as i32;
    }

    return false;
}


pub const DefaultNodeMgrNodeNameSpace: &str = "qserverless.quarksoft.io";
pub const DefaultDomainName: &str = "qserverless.quarksoft.io";

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

pub fn PodToString(pod: &k8s::Pod) -> Result<String> {
    let s = serde_json::to_string(pod)?;
    return Ok(s);
}

pub fn PodFromString(s: &str) -> Result<k8s::Pod> {
    let p: k8s::Pod = serde_json::from_str(s)?;
    return Ok(p);
}

pub fn ConfigMapToString(o: &k8s::ConfigMap) -> Result<String> {
    let s = serde_json::to_string(o)?;
    return Ok(s);
}

pub fn ConfigMapFromString(s: &str) -> Result<k8s::ConfigMap> {
    let p: k8s::ConfigMap = serde_json::from_str(s)?;
    return Ok(p);
}

pub fn NodeToString(o: &k8s::Node) -> Result<String> {
    let s = serde_json::to_string(o)?;
    return Ok(s);
}

pub fn NodeFromString(s: &str) -> Result<k8s::Node> {
    let p: k8s::Node = serde_json::from_str(s)?;
    return Ok(p);
}