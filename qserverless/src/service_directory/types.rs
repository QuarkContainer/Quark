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

//use serde::{Deserialize, Serialize};  

use crate::shared::common::*;

use super::etcd_store::*;
use crate::etcd_store::DataObject;

pub trait DeepCopy {
    fn DeepCopy(&self) -> Self;
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Spec {
    // NodeName is a request to schedule this pod onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this pod onto that node, assuming that it fits resource
	// requirements.
	// +optional
	pub nodename: String,
	// Specifies the hostname of the Pod.
	// If not specified, the pod's hostname will be set to a system-defined value.
	// +optional
	pub hostname: String,
}

impl DeepCopy for Pod {
    fn DeepCopy(&self) -> Self {
        return Pod {
            spec: Spec {
                nodename: self.spec.nodename.clone(),
                hostname: self.spec.hostname.clone(),
            }
        } 
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Pod {
    spec: Spec
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Podset {
    // NodeName is a request to schedule this pod onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this pod onto that node, assuming that it fits resource
	// requirements.
	// +optional
	pub nodename: String,
	// Specifies the hostname of the Pod.
	// If not specified, the pod's hostname will be set to a system-defined value.
	// +optional
	pub hostname: String,
}

impl DeepCopy for Podset {
    fn DeepCopy(&self) -> Self {
        return Self {
            nodename: self.nodename.clone(),
            hostname: self.hostname.clone(),
        }
    }
}

impl DataObject {
    pub fn NewPod(namespace: &str, name: &str, nodeName: &str, hostName: &str) -> Result<Self> {
        let pod = Pod {
            spec: Spec {
                nodename: nodeName.to_string(),
                hostname: hostName.to_string(),
            }
        };

        let meta = MetaDataInner {
            namespace: namespace.to_string(),
            name: name.to_string(),
            ..Default::default()
        };

        let mut obj = meta.ToObject();
        obj.val =serde_json::to_string(&pod)?;


        return Ok(obj.into())
    }
}