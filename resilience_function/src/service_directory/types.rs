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
use super::service_directory::*;

pub struct QType {
    pub metadata: MetaDataInner,
    pub data: QTypes,
}

impl QType {
    pub fn NewPod(namespace: &str, name: &str) -> Self {
        let meta = MetaDataInner {
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            ..Default::default()
        };

        return Self::New(meta, QTypes::Pod(Pod::default()));
    }

    pub fn Prefix(&self) -> &str {
        return self.data.Prefix();
    }

    pub fn StoreKey(&self) -> String {
        return "/".to_owned() + self.Prefix() + "/" + &self.metadata.namespace + "/" + &self.metadata.name;
    }

    pub fn New(metadata: MetaDataInner, data: QTypes) -> Self {
        return Self {
            metadata: metadata,
            data: data,
        };
    }

    pub fn Deserialize(obj: Object) -> Result<Self> {
        let data: QTypes = serde_json::from_slice(&obj.val)?;
        let metadata: MetaDataInner = MetaDataInner::New(&obj);
        return Ok(Self {
            metadata: metadata,
            data: data,
        })
    }

    pub fn Serialize(&self) -> Result<Object> {
        let mut obj = self.metadata.ToObject();
        obj.val = serde_json::to_vec(&self.data)?;
        return Ok(obj)
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Pod {
    // NodeName is a request to schedule this pod onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this pod onto that node, assuming that it fits resource
	// requirements.
	// +optional
	pub NodeName: String,
	// Specifies the hostname of the Pod.
	// If not specified, the pod's hostname will be set to a system-defined value.
	// +optional
	pub Hostname: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Podset {
    // NodeName is a request to schedule this pod onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this pod onto that node, assuming that it fits resource
	// requirements.
	// +optional
	pub NodeName: String,
	// Specifies the hostname of the Pod.
	// If not specified, the pod's hostname will be set to a system-defined value.
	// +optional
	pub Hostname: String,
}


#[derive(Debug, Serialize, Deserialize)]
pub enum QTypes {
    Pod(Pod),
    Podset(Podset)
}

impl QTypes {
    pub fn Prefix(&self) -> &str {
        match self {
            Self::Pod(_) => return "pods",
            Self::Podset(_) => return "podsets",
        }
    }
}