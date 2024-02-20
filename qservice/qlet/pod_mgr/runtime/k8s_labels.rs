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
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use qshare::node::ContainerDef;
use serde::Deserialize;

use k8s_openapi::api::core::v1::{self as k8s, ContainerPort};
use qshare::node::PodDef;
use qshare::common::*;
use qshare::consts::*;

pub struct LabeledPodSandboxInfo {
	// Labels from v1.Pod
	pub labels: BTreeMap<String, String>,
	pub podName: String,
	pub podNamespace: String,
	pub podUID: String,
}

pub struct AnnotatedPodSandboxInfo {
	// Annotations from v1.Pod
	pub annotations: BTreeMap<String, String>,
}

pub struct LabeledContainerInfo {
	pub containerName: String,
	pub podName: String,
	pub podNamespace: String,
	pub podUID: String,
}

#[derive(Debug)]

pub enum TerminationMessagePolicy {
 	// TerminationMessageReadFile is the default behavior and will set the container status message to
	// the contents of the container's terminationMessagePath when the container exits.
	TerminationMessageReadFile, // TerminationMessagePolicy = "File"
	// TerminationMessageFallbackToLogsOnError will read the most recent contents of the container logs
	// for the container status message when the container exits with an error and the
	// terminationMessagePath has no contents.
	TerminationMessageFallbackToLogsOnError, // TerminationMessagePolicy = "FallbackToLogsOnError"   
}

#[derive(Debug, Default)]
pub struct AnnotatedContainerInfo {
	pub hash: u64,
	pub restartCount: i32,
	pub podDeletionGracePeriod: i64,
	pub podTerminationGracePeriod: i64,
	pub terminationMessagePath: String,
	pub terminationMessagePolicy: String,
	pub preStopHandler: k8s::LifecycleHandler, 
	pub containerPorts: Vec<ContainerPort>,
}

pub fn NewPodAnnotations(pod: &PodDef) -> HashMap<String, String> {
	let mut a = HashMap::new();

	for (k, v) in &pod.annotations {
		a.insert(k.to_string(), v.to_string());
	}
	
	return a;
}

pub fn NewPodLabels(pod: &PodDef) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    labels.insert(KUBERNETES_POD_NAME_LABEL.to_string(), pod.name.clone());
    labels.insert(KUBERNETES_POD_NAMESPACE_LABEL.to_string(), pod.namespace.clone());
    labels.insert(KUBERNETES_POD_UIDLABEL.to_string(), pod.uid.clone());
    
	for (k, v) in &pod.labels {
		labels.insert(k.to_string(), v.to_string());
	}

    return labels;
}

// HashContainer returns the hash of the container. It is used to compare
// the running container with its desired spec.
pub fn HashContainer(container: &ContainerDef) -> u64 {
	use std::collections::hash_map::DefaultHasher;
	use std::hash::{Hash, Hasher};

	let mut s = DefaultHasher::new();
	let j = serde_json::to_string(&container).unwrap();

	Hash::hash_slice(j.as_bytes(), &mut s);
	let ret = s.finish();
	return ret;
}

pub fn NewContainerLabels(container: &ContainerDef, pod: &Arc<RwLock<PodDef>>) -> HashMap<String, String> {
	let pod = &pod.read().unwrap();
    let mut labels = HashMap::new();
    labels.insert(KUBERNETES_POD_NAME_LABEL.to_string(), pod.name.clone());
    labels.insert(KUBERNETES_POD_NAMESPACE_LABEL.to_string(), pod.namespace.clone());
    labels.insert(KUBERNETES_POD_UIDLABEL.to_string(), pod.name.clone());
	labels.insert(KUBERNETES_CONTAINER_NAME_LABEL.to_string(), container.name.clone());

	return labels;
}

pub fn NewContainerAnnotations(container: &ContainerDef, pod: &Arc<RwLock<PodDef>>, restartCount: i32, overrideAnnotations: &BTreeMap<String, String>) -> HashMap<String, String> {
	let pod = &pod.read().unwrap();
    let mut annotations = HashMap::new();

	annotations.insert(CONTAINER_HASH_LABEL.to_string(), format!("{}", HashContainer(container)));
	annotations.insert(CONTAINER_RESTART_COUNT_LABEL.to_string(), format!("{}", restartCount));
	// annotations.insert(CONTAINER_TERMINATION_MESSAGE_PATH_LABEL.to_string(), container.termination_message_path.as_deref().unwrap_or("").to_string());
	// annotations.insert(CONTAINER_TERMINATION_MESSAGE_POLICY_LABEL.to_string(), container.termination_message_policy.as_deref().unwrap_or("").to_string());

	if let Some(s) = &pod.deletion_grace_period_seconds {
		annotations.insert(POD_DELETION_GRACE_PERIOD_LABEL.to_string(), format!("{}", *s));
	}

	if let Some(s) = &pod.termination_grace_period_seconds {
		annotations.insert(POD_DELETION_GRACE_PERIOD_LABEL.to_string(), format!("{}", *s));
	}

	// if container.lifecycle.is_some() && container.lifecycle.as_ref().unwrap().pre_stop.is_some() {
	// 	match serde_json::to_string(container.lifecycle.as_ref().unwrap().pre_stop.as_ref().unwrap()) {
	// 		Ok(s) => {
	// 			annotations.insert(CONTAINER_PRE_STOP_HANDLER_LABEL.to_string(), s);
	// 		}
	// 		Err(e) => {
	// 			error!("Unable to marshal lifecycle PreStop handler for container containerName {} pod {} error {:?}", 
	// 				container.name, K8SUtil::PodId(pod), e);
	// 		}
	// 	}
	// }

	// if let Some(ports) = &container.ports {
	// 	if ports.len() > 0 {
	// 		match serde_json::to_string(ports) {
	// 			Ok(s) => {
	// 				annotations.insert(CONTAINER_PORTS_LABEL.to_string(), s);
	// 			}
	// 			Err(e) => {
	// 				error!("Unable to marshal container ports for container containerName {} pod {} error {:?}", 
	// 					container.name, K8SUtil::PodId(pod), e);
	// 			}
	// 		}
	// 	}
	// }

	for (k, v) in overrideAnnotations {
		annotations.insert(k.to_string(), v.to_string());
	}

	return annotations;
}

pub fn GetPodSandboxInfoFromLabels(labels: &BTreeMap<String, String>) -> LabeledPodSandboxInfo {
	let mut podSandboxInfo = LabeledPodSandboxInfo {
		labels: BTreeMap::new(),
		podName: GetStringValueFromLabel(labels, KUBERNETES_POD_NAME_LABEL),
		podNamespace: GetStringValueFromLabel(labels, KUBERNETES_POD_NAMESPACE_LABEL),
		podUID: GetStringValueFromLabel(labels, KUBERNETES_POD_UIDLABEL),
	};

	for (k, v) in labels {
		if k!=KUBERNETES_POD_NAME_LABEL && k!=KUBERNETES_POD_NAMESPACE_LABEL && k!=KUBERNETES_POD_UIDLABEL {
			podSandboxInfo.labels.insert(k.to_string(), v.to_string());
		}
	}

	return podSandboxInfo
}

pub fn GetPodSandboxInfoFromAnnotations(annotations: BTreeMap<String, String>) -> AnnotatedPodSandboxInfo {
	return AnnotatedPodSandboxInfo {
		annotations: annotations,
	}
}

pub fn GetContainerInfoFromLabels(labels: &BTreeMap<String, String>) -> LabeledContainerInfo {
	return LabeledContainerInfo {
		podName: GetStringValueFromLabel(labels, KUBERNETES_POD_NAME_LABEL),
		podNamespace: GetStringValueFromLabel(labels, KUBERNETES_POD_NAMESPACE_LABEL),
		podUID: GetStringValueFromLabel(labels, KUBERNETES_POD_UIDLABEL),
		containerName: GetStringValueFromLabel(labels, KUBERNETES_CONTAINER_NAME_LABEL),
	}
}

pub fn GetContainerInfoFromAnnotations(annotations: &BTreeMap<String, String>) -> AnnotatedContainerInfo {
	let mut containerInfo = AnnotatedContainerInfo {
		terminationMessagePath: GetStringValueFromLabel(annotations, CONTAINER_TERMINATION_MESSAGE_PATH_LABEL),
		terminationMessagePolicy: GetStringValueFromLabel(annotations, CONTAINER_TERMINATION_MESSAGE_POLICY_LABEL),
		..Default::default()
	};

	match GetUint64ValueFromLabel(annotations, CONTAINER_HASH_LABEL) {
		Ok(v) => containerInfo.hash = v,
		Err(_) => error!("Unable to get label value from annotations label {} annotations {:?}", CONTAINER_HASH_LABEL, annotations),
	}

	match GetIntValueFromLabel(annotations, CONTAINER_RESTART_COUNT_LABEL) {
		Ok(v) => containerInfo.restartCount = v,
		Err(_) => error!("Unable to get label value from annotations label {} annotations {:?}", CONTAINER_RESTART_COUNT_LABEL, annotations),
	}

	match GetInt64PointerFromLabel(annotations, POD_DELETION_GRACE_PERIOD_LABEL) {
		Ok(v) => containerInfo.podTerminationGracePeriod = v,
		Err(_) => error!("Unable to get label value from annotations label {} annotations {:?}", POD_DELETION_GRACE_PERIOD_LABEL, annotations),
	}

	match GetJSONObjectFromLabel::<Vec<ContainerPort>>(annotations, CONTAINER_PORTS_LABEL) {
		Ok(v) => containerInfo.containerPorts = v,
		Err(_) => error!("Unable to get label value from annotations label {} annotations {:?}", CONTAINER_PORTS_LABEL, annotations),
	}

	return containerInfo
}

pub fn GetStringValueFromLabel(labels: &BTreeMap<String, String>, label: &str) -> String {
	match labels.get(label) {
		Some(v) => return v.to_string(),
		None => {
			// Do not report error, because there should be many old containers without label now.
			info!("Container doesn't have requested label, it may be an old or invalid container label {}", label);
			return "".to_string();
		}
	}
}

pub fn GetIntValueFromLabel(labels: &BTreeMap<String, String>, label: &str) -> Result<i32> {
	match labels.get(label) {
		Some(v) => {
			match v.parse::<i32>() {
				Ok(v) => return Ok(v),
				Err(_e) => Err(Error::CommonError(format!("GetIntValueFromLabel val {} is not valid", v))),
			}
		}
		None => {
			// Do not report error, because there should be many old containers without label now.
			info!("Container doesn't have requested label, it may be an old or invalid container label {}", label);
			return Ok(0);
		}
	}
}

pub fn GetUint64ValueFromLabel(labels: &BTreeMap<String, String>, label: &str) -> Result<u64> {
	match labels.get(label) {
		Some(v) => {
			match u64::from_str_radix(v, 16) {
				Ok(v) => return Ok(v),
				Err(_e) => return Err(Error::CommonError(format!("GetUint64ValueFromLabel val {} is not valid", v))),
			}
		}
		None => {
			// Do not report error, because there should be many old containers without label now.
			info!("Container doesn't have requested label, it may be an old or invalid container label {}", label);
			return Ok(0);
		}
	}
}

pub fn GetInt64PointerFromLabel(labels: &BTreeMap<String, String>, label: &str) -> Result<i64> {
	match labels.get(label) {
		Some(v) => {
			match i64::from_str_radix(v, 16) {
				Ok(v) => return Ok(v),
				Err(_e) => return Err(Error::CommonError(format!("GetInt64PointerFromLabel val {} is not valid", v))),
			}
		}
		None => {
			// Do not report error, because there should be many old containers without label now.
			info!("Container doesn't have requested label, it may be an old or invalid container label {}", label);
			return Ok(0);
		}
	}
}

pub fn GetJSONObjectFromLabel<T: for<'a> Deserialize<'a>>(labels: &BTreeMap<String, String>, label: &str) -> Result<T> {
	match labels.get(label) {
		Some(v) => {
			let p: T = serde_json::from_str(v)?;
			return Ok(p);
		}
		None => {
			// Do not report error, because there should be many old containers without label now.
			info!("Container doesn't have requested label, it may be an old or invalid container label {}", label);
			return Err(Error::CommonError(format!("getJSONObjectFromLabel not exist label {}", label)));
		}
	}
}