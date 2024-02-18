// Copyright (c) 2023 Quark Container Authors / 2018 The gVisor Authors.
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

use qshare::node::*;
use qshare::types::RuntimePod;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::SystemTime;
use std::ops::Deref;

use qshare::common::*;
use qshare::k8s;
use qshare::consts::*;

use super::qcontainer::*;

pub fn ValidatePodSpec(_pod: &k8s::Pod) -> Result<()> {
    return Ok(());
}

pub fn ValidateConfigMap(_configMap: &k8s::ConfigMap) -> Result<()> {
    return Ok(());
}

pub fn ValidateSecret(_secret: &k8s::Secret) -> Result<()> {
    return Ok(());
}


// PodInitialized means that all init containers in the pod have started successfully.
pub const PodInitialized: &str = "Initialized";
// PodReady means the pod is able to service requests and should be added to the
// load balancing pools of all matching services.
pub const PodReady: &str = "Ready";
// PodScheduled represents status of the scheduling process for this pod.
pub const PodScheduled: &str = "PodScheduled";

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
    //
    Deleted,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QuarkPodJson {
    pub id: String,
    pub podState: PodState,
    pub isDaemon: bool,
    pub pod: PodDef,
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
    pub pod: Arc<RwLock<PodDef>>,
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

    pub fn Pod(&self) -> Arc<RwLock<PodDef>> {
        return self.lock().unwrap().pod.clone();
    }

    pub fn PodId(&self) -> String {
        return self.Pod().read().unwrap().PodId();
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

    pub fn ToPhase(&self) -> String {
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

    pub fn SetPodStatus(&self, node: Option<&Node>) {
        let pod = self.Pod();
        
        pod.write().unwrap().status.phase = self.ToPhase();
        if let Some(node) = node {
            pod.write().unwrap().status.host_ip = node.status.addresses[0].address.clone();
        }   
    
        let hasDeleted = pod.write().unwrap().deletion_timestamp.is_some();
        let phase = pod.write().unwrap().status.phase.clone();
        if (phase == PodSucceeded 
            || phase == PodFailed)
            &&  !hasDeleted {
                pod.write().unwrap().deletion_timestamp = Some(SystemTime::now());
        }

        pod.write().unwrap().status.conditions = self.GetPodConditions();

        if self.RuntimePod().is_some() {
            if self.RuntimePod().as_ref().unwrap().IPs.len() > 0 {
                let mut podIps = Vec::new();
                for v in &self.RuntimePod().as_ref().unwrap().IPs {
                    podIps.push(v.clone())
                }
                pod.write().unwrap().status.pod_ips = podIps;
            }
        }

        if pod.write().unwrap().status.start_time.is_none() {
            pod.write().unwrap().status.start_time = Some(SystemTime::now());
        }
    }

    pub fn Containers(&self) -> Vec<QuarkContainer> {
        let containers: Vec<QuarkContainer> = self.lock().unwrap().containers.values().cloned().collect();
        return containers;
    }

    pub fn GetPodConditions(&self) -> Vec<PodCondition> {
        let mut conditions = BTreeMap::new();

        let time = SystemTime::now();
        let mut initReadyCondition = PodCondition {
            type_: PodInitialized.to_string(),
            status: ConditionUnknown.to_string(),
            last_probe_time: time,
            last_transition_time: time,
            reason: "".to_owned(),
            message: "".to_owned(),
        };
        
        let mut containerReadyCondition = PodCondition {
            type_: ContainersReady.to_string(),
            status: ConditionUnknown.to_string(),
            last_probe_time: time,
            last_transition_time: time,
            reason: "".to_owned(),
            message: "".to_owned(),
        };
        
        let mut podReadyCondition = PodCondition {
            type_: PodReady.to_string(),
            status: ConditionUnknown.to_string(),
            last_probe_time: time,
            last_transition_time: time,
            reason: "".to_owned(),
            message: "".to_owned(),
        };
        
        let podScheduledCondition = PodCondition {
            type_: PodScheduled.to_string(),
            status: ConditionTrue.to_string(),
            last_probe_time: time,
            last_transition_time: time,
            reason: "".to_owned(),
            message: "".to_owned(),
        };


        let mut allInitContainerNormal = true;
        let containers: Vec<QuarkContainer> = self.lock().unwrap().containers.values().cloned().collect();
        for v in &containers {
            let c = v.lock().unwrap();
            if c.initContainer {
                if !ContainerExit(&c.containerStatus) {
                    containerReadyCondition.status = ConditionFalse.to_string();
                    containerReadyCondition.message = "init container not finished yet".to_string();
                    containerReadyCondition.reason = "init container not finished yet".to_string();
                    allInitContainerNormal = false;
                    break;
                }

                if ContainerExitAbnormal(&c.containerStatus) { 
                    containerReadyCondition.status = ConditionFalse.to_string();
                    containerReadyCondition.message = "init container exit abnormally".to_string();
                    containerReadyCondition.reason = "init container exit abnormally".to_string();
                    allInitContainerNormal = false;
                    break;
                }
                allInitContainerNormal = allInitContainerNormal && ContainerExitNormal(&c.containerStatus);
            }
        }

        if allInitContainerNormal {
            initReadyCondition.status = ConditionTrue.to_string();
            initReadyCondition.message = "init container exit normally".to_string();
            initReadyCondition.reason = "init container exit normally".to_string();
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
                    containerReadyCondition.message = "one container is not running".to_string();
                    containerReadyCondition.reason = "one container is not running".to_string();
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
            containerReadyCondition.message = "all pod containers are running".to_string();
            containerReadyCondition.reason = "all pod containers are running".to_string();
        }

        if &containerReadyCondition.status == ConditionTrue && &initReadyCondition.status == ConditionTrue {
            if self.PodState() == PodState::Running {
                podReadyCondition.status = ConditionTrue.to_string();
                podReadyCondition.message = "all pod containers are ready".to_string();
                podReadyCondition.reason = "all pod containers are ready".to_string();
            }
        } else {
            podReadyCondition.status = ConditionFalse.to_string();
            podReadyCondition.message = "some pod containers are not running".to_string();
            podReadyCondition.reason = "some pod containers are not running".to_string();
        }

        conditions.insert(PodInitialized.to_string(), initReadyCondition);
        conditions.insert(ContainersReady.to_string(), containerReadyCondition);
        conditions.insert(PodReady.to_string(), podReadyCondition);
        conditions.insert(PodScheduled.to_string(), podScheduledCondition);

        let pod = self.Pod();

        for oldCondition in &pod.read().unwrap().status.conditions {
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

