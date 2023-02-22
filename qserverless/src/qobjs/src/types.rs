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

use serde_derive::{Deserialize, Serialize};  
use std::sync::atomic::AtomicI64;
use std::{sync::Arc, collections::BTreeMap};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering;

use prost::Message;

use crate::common::*;
use crate::selector::Labels;
use crate::service_directory::*;
use crate::selection_predicate::*;


pub trait DeepCopy {
    fn DeepCopy(&self) -> Self;
}


#[derive(Debug, PartialEq, Eq, Clone)]
pub enum EventType {
    None,
    Added,
    Modified,
    Deleted,
    Error(String),
}

impl EventType {
    pub fn DeepCopy(&self) -> Self {
        match self {
            Self::None => return Self::None,
            Self::Added => return Self::Added,
            Self::Modified => return Self::Modified,
            Self::Deleted => return Self::Deleted,
            Self::Error(str) => return Self::Error(str.to_string()),
        }
    }
}

impl Default for EventType {
    fn default() -> Self {
        return Self::None;
    }
}

#[derive(Debug)]
pub struct WatchEvent {
    pub type_: EventType,

    // Object is:
	//  * If Type is Added or Modified: the new state of the object.
	//  * If Type is Deleted: the state of the object immediately before deletion.
	//  * If Type is Error:
    pub obj: DataObject,
}


#[derive(Debug, Default)]
pub struct MetaDataInner {
    pub kind: String,
    pub namespace: String,
    pub name: String,
    pub lables: Labels,
    pub annotations: Labels,
    
    // revision number set by etcd
    pub reversion: AtomicI64,
}

impl DeepCopy for MetaDataInner {
    fn DeepCopy(&self) -> Self {
        return self.Copy();
    }
}

impl PartialEq for MetaDataInner {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind &&
        self.namespace == other.namespace &&
        self.lables == other.lables &&
        self.annotations == other.annotations &&
        self.reversion.load(Ordering::Relaxed) == other.reversion.load(Ordering::Relaxed)
    }
}
impl Eq for MetaDataInner {}

impl MetaDataInner {
    pub fn New(item: &Object) -> Self {
        let mut lables = BTreeMap::new();
        for l in &item.labels {
            lables.insert(l.key.clone(), l.val.clone());
        }

        let mut annotations = BTreeMap::new();
        for l in &item.annotations {
            annotations.insert(l.key.clone(), l.val.clone());
        }

        let inner = MetaDataInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: lables.into(),
            annotations: annotations.into(),
            reversion: AtomicI64::new(0),
        };

        return inner;
    }

    pub fn Key(&self) -> String {
        return format!("{}/{}", &self.namespace, &self.name);
    }

    pub fn Revision(&self) -> i64 {
        return self.reversion.load(Ordering::Relaxed);
    }

    pub fn SetRevision(&self, rev: i64) {
        return self.reversion.store(rev, Ordering::SeqCst);
    }

    pub fn Copy(&self) -> Self {
        return Self {
            kind: self.kind.clone(),
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            lables: self.lables.Copy(),
            annotations: self.annotations.Copy(),
            reversion: AtomicI64::new(self.Revision()),
        }
    }

    pub fn ToObject(&self) -> Object {
        let mut obj = Object::default();
        obj.kind = self.kind.clone();
        obj.namespace = self.namespace.clone();
        obj.name = self.name.clone();
        obj.labels = self.lables.ToVec();

        return obj;
    }
}

#[derive(Debug, Default)]
pub struct DataObjInner {
    pub metadata: MetaDataInner,

    pub obj: Object,
}

impl PartialEq for DataObjInner {
    fn eq(&self, other: &Self) -> bool {
        return self.metadata == other.metadata && self.obj.val == other.obj.val;
    }
}
impl Eq for DataObjInner {}

impl Deref for DataObjInner {
    type Target = MetaDataInner;

    fn deref(&self) -> &MetaDataInner {
        &self.metadata
    }
}

impl DeepCopy for DataObjInner {
    fn DeepCopy(&self) -> Self {
        return Self {
            metadata: self.metadata.DeepCopy(),
            obj: self.obj.clone(),
        }
    }
}

impl DerefMut for DataObjInner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.metadata
    }
}

impl From<Object> for DataObjInner {
    fn from(item: Object) -> Self {
        let metadata : MetaDataInner = MetaDataInner::New(&item);

        let inner = Self {
            metadata: metadata,
            obj: item,
        };

        return inner;
    }
}

#[derive(Debug, Default)]
pub struct DataObjList {
    pub objs: Vec<DataObject>,
    pub revision: i64,
    pub continue_: Option<Continue>,
    pub remainCount: i64,
}

impl DataObjList {
    pub fn New(objs: Vec<DataObject>, revision: i64, continue_: Option<Continue>, remainCount: i64) -> Self {
        return Self {
            objs: objs,
            revision:  revision,
            continue_: continue_,
            remainCount: remainCount,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct DataObject(Arc<DataObjInner>);

impl PartialEq for DataObject {
    fn eq(&self, other: &Self) -> bool {
        return self.0 == other.0;
    }
}
impl Eq for DataObject {}

impl From<Object> for DataObject {
    fn from(item: Object) -> Self {
        let inner = item.into();

        return Self(Arc::new(inner));
    }
}

impl From<DataObjInner> for DataObject {
    fn from(inner: DataObjInner) -> Self {
        return Self(Arc::new(inner));
    }
}

impl Deref for DataObject {
    type Target = Arc<DataObjInner>;

    fn deref(&self) -> &Arc<DataObjInner> {
        &self.0
    }
}

impl DeepCopy for DataObject {
    fn DeepCopy(&self) -> Self {
        return Self(Arc::new(self.0.DeepCopy()));
    }
}

impl DataObject {
    pub fn Namespace(&self) -> String {
        return self.metadata.namespace.clone();
    }

    pub fn Name(&self) -> String {
        return self.metadata.name.clone();
    }

    pub fn Key(&self) -> String {
        return self.metadata.Key();
    }

    pub fn Obj(&self) -> Object {
        return self.obj.clone();
    }

    pub fn Revision(&self) -> i64 {
        return self.metadata.Revision();
    }

    pub fn Decode(buf: &[u8]) -> Result<Self> {
        let obj = Object::decode(buf)?;
        return Ok(Self(Arc::new(obj.into())))
    }

    pub fn Encode(&self) -> Result<Vec<u8>> {
        let mut buf : Vec<u8> = Vec::new();
        buf.reserve(self.obj.encoded_len());
        self.obj.encode(&mut buf)?;
        return Ok(buf)
    }

    pub fn Labels(&self) -> Labels {
        let lables = self.lables.clone();
        return lables
    }

    pub fn SetRevision(&self, rev: i64) {
        self.metadata.SetRevision(rev)
    }
    
}

impl Object {
    pub fn Encode(&self) -> Result<Vec<u8>> {
        let mut buf : Vec<u8> = Vec::new();
        buf.reserve(self.encoded_len());
        self.encode(&mut buf)?;
        return Ok(buf)
    }

    pub fn Decode(buf: &[u8]) -> Result<Self> {
        let o = Self::decode(buf)?;
        return Ok(o)
    }
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