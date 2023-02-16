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

use std::collections::BTreeMap;

use crate::shared::common::*;

use super::etcd_store::*;
use super::service_directory::*;

#[derive(Debug, PartialEq, Eq)]
pub struct QType {
    pub metadata: MetaDataInner,
    pub data: QTypes,
}

impl DeepCopy for QType {
    fn DeepCopy(&self) -> Self {
        return Self {
            metadata: self.metadata.Copy(),
            data: self.data.DeepCopy(),
        }
    }
}

impl QType {
    pub fn NewPod(namespace: &str, name: &str) -> Self {
        let meta = MetaDataInner {
            namespace: namespace.to_string(),
            name: name.to_string(),
            fields: [
                ("metadata.name".to_owned(), name.to_owned()),
                ("metadata.namespace".to_owned(), namespace.to_owned()),

            ].into_iter().collect::<BTreeMap<String, String>>().into(),
            ..Default::default()
        };

        return Self::New(meta, QTypes::Pod(Pod::default()));
    }

    pub fn NewPodWithData(namespace: &str, name: &str, pod: Pod) -> Self {
        let meta = MetaDataInner {
            namespace: namespace.to_string(),
            name: name.to_string(),
            fields: [
                ("metadata.name".to_owned(), name.to_owned()),
                ("metadata.namespace".to_owned(), namespace.to_owned()),

            ].into_iter().collect::<BTreeMap<String, String>>().into(),
            ..Default::default()
        };

        return Self::New(meta, QTypes::Pod(pod));
    }

    pub fn Prefix(&self) -> &str {
        return self.data.Prefix();
    }

    pub fn StoreKey(&self) -> String {
        return self.Prefix().to_string() + "/" + &self.metadata.namespace + "/" + &self.metadata.name;
    }

    pub fn New(metadata: MetaDataInner, data: QTypes) -> Self {
        return Self {
            metadata: metadata,
            data: data,
        };
    }

    pub fn Decode(obj: &DataObject) -> Result<Self> {
        let data: QTypes = serde_json::from_str(&obj.obj.val)?;
        let mut metadata: MetaDataInner = MetaDataInner::New(&obj.obj);
        metadata.reversion = obj.reversion;
        return Ok(Self {
            metadata: metadata,
            data: data,
        })
    }

    pub fn Encode(&self) -> Result<Object> {
        let mut obj = self.metadata.ToObject();
        obj.val = serde_json::to_string(&self.data)?;
        return Ok(obj)
    }

    pub fn DataObj(&self) -> Result<DataObject> {
        let obj = self.Encode()?;
        let mut dataObj : DataObjInner = obj.into();
        dataObj.reversion = self.metadata.reversion;
        return Ok(dataObj.into())
    }
}

pub trait DeepCopy {
    fn DeepCopy(&self) -> Self;
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
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

impl DeepCopy for Pod {
    fn DeepCopy(&self) -> Self {
        return Self {
            NodeName: self.NodeName.clone(),
            Hostname: self.Hostname.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
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

impl DeepCopy for Podset {
    fn DeepCopy(&self) -> Self {
        return Self {
            NodeName: self.NodeName.clone(),
            Hostname: self.Hostname.clone(),
        }
    }
}


#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum QTypes {
    Pod(Pod),
    Podset(Podset)
}

impl DeepCopy for QTypes {
    fn DeepCopy(&self) -> Self {
        match self {
            Self::Pod(item) => return Self::Pod(item.DeepCopy()),
            Self::Podset(item) => Self::Podset(item.DeepCopy()),
        }
    }
}

impl QTypes {
    pub fn Prefix(&self) -> &str {
        match self {
            Self::Pod(_) => return "pods",
            Self::Podset(_) => return "podsets",
        }
    }
}