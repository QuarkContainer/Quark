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

use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
use k8s_openapi::api::core::v1 as k8s;

pub struct K8SUtil {}

impl K8SUtil {
    pub fn PodId(pod: &k8s::Pod) -> String {
        return Self::Id(&pod.metadata);
    }

    pub fn NodeId(node: &k8s::Node) -> String {
        return Self::Id(&node.metadata);
    }

    pub fn Id(meta: &ObjectMeta) -> String {
        return format!("Namespace:{},Name:{},UID:{}", 
            K8SUtil::Namespace(&meta),
            K8SUtil::Name(&meta),
            K8SUtil::Uid(&meta));
    }

    pub fn Namespace(meta: &ObjectMeta) -> String {
        match &meta.namespace {
            None => return "".to_string(),
            Some(s) => return s.clone(),
        }
    }

    pub fn Name(meta: &ObjectMeta) -> String {
        match &meta.name {
            None => return "".to_string(),
            Some(s) => return s.clone(),
        }
    }

    pub fn Uid(meta: &ObjectMeta) -> String {
        match &meta.uid {
            None => return "".to_string(),
            Some(s) => return s.clone(),
        }
    }
}

// PullPolicy describes a policy for if/when to pull a container image
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct PullPolicy {}

impl PullPolicy {
    // PullAlways means that kubelet always attempts to pull the latest image. Container will fail If the pull fails.
	pub const PullAlways: &str = "Always";

    // PullNever means that kubelet never pulls an image, but only uses a local image. Container will fail if the image isn't present
	pub const PullNever: &str = "Never";
	
    // PullIfNotPresent means that kubelet pulls if the image isn't present on disk. Container will fail if the image isn't present and the pull fails.
	pub const PullIfNotPresent: &str = "IfNotPresent";
}