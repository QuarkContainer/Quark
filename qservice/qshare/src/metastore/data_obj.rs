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
// limitations under the License.// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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
use std::{sync::Arc, collections::BTreeMap};
use std::ops::{Deref};
use std::ops::Add;
use std::ops::Sub;
use regex::Regex;

use prost::Message;

use crate::common::*;
use crate::qmeta::*;
use crate::ObjectMeta;
use crate::k8s;
// use crate::system_types::{FuncPackage, FuncPackageSpec};
// use crate::func;

use super::selector::Labels;
use super::selection_predicate::*;


pub const QMETASVC_PORT : u16 = 8890; 
pub const GATEWAY_PORT : u16 = 8889;
pub const NODEMGRSVC_ADDR : &str = "127.0.0.1:8888";
pub const STATESVC_ADDR : &str = "127.0.0.1:8890";
pub const FUNCSVC_ADDR : &str = "127.0.0.1:8891";
pub const AUDITDB_ADDR: &str = "postgresql://audit_user:123456@localhost/auditdb";
pub const OBJECTDB_ADDR: &str = "postgresql://blob_user:123456@localhost/blobdb";

pub const QUARK_POD : &str = "qpod";
pub const QUARK_NODE : &str = "qnode";

pub const POD_DELETION_GRACE_PERIOD_LABEL           : &str = "io.kubernetes.pod.deletionGracePeriod";
pub const POD_TERMINATION_GRACE_PERIOD_LABEL        : &str = "io.kubernetes.pod.terminationGracePeriod";

pub const CONTAINER_HASH_LABEL                      : &str = "io.kubernetes.container.hash";
pub const CONTAINER_RESTART_COUNT_LABEL             : &str = "io.kubernetes.container.restartCount";
pub const CONTAINER_TERMINATION_MESSAGE_PATH_LABEL  : &str = "io.kubernetes.container.terminationMessagePath";
pub const CONTAINER_TERMINATION_MESSAGE_POLICY_LABEL: &str = "io.kubernetes.container.terminationMessagePolicy";
pub const CONTAINER_PRE_STOP_HANDLER_LABEL          : &str = "io.kubernetes.container.preStopHandler";
pub const CONTAINER_PORTS_LABEL                     : &str = "io.kubernetes.container.ports";

pub const KUBERNETES_POD_NAME_LABEL         : &str = "io.kubernetes.pod.name";
pub const KUBERNETES_POD_NAMESPACE_LABEL    : &str = "io.kubernetes.pod.namespace";
pub const KUBERNETES_POD_UIDLABEL           : &str = "io.kubernetes.pod.uid";
pub const KUBERNETES_CONTAINER_NAME_LABEL   : &str = "io.kubernetes.container.name";

pub const LabelNodeMgrNodeDaemon              : &str = "daemon.qserverless.quarksoft.io";
pub const LabelNodeMgrApplication             : &str = "application.core.qserverless.quarksoft.io";
pub const AnnotationNodeMgrNode               : &str = "node.qserverless.quarksoft.io";
pub const AnnotationNodeMgrPod                : &str = "pod.qserverless.quarksoft.io";
pub const AnnotationNodeMgrCreationUnixMicro  : &str = "create.unixmicro.core.qserverless.quarksoft.io";
pub const AnnotationNodeMgrSessionService     : &str = "sessionservice.core.qserverless.quarksoft.io";
pub const AnnotationNodeMgrApplicationSession : &str = "applicationsession.core.qserverless.quarksoft.io";
pub const AnnotationNodeMgrNodeRevision       : &str = "noderevision.core.qserverless.quarksoft.io";
pub const AnnotationNodeMgrHibernatePod       : &str = "hibernatepod.core.qserverless.quarksoft.io";
pub const AnnotationNodeMgrSessionServicePod  : &str = "sessionservicepod.core.qserverless.quarksoft.io";
pub const AnnotationFuncPodPackageName        : &str = "packagename.qserverless.quarksoft.io";
pub const AnnotationFuncPodPackageType        : &str = "packagetype.qserverless.quarksoft.io";
pub const AnnotationFuncPodPyPackageId        : &str = "pypackageid.qserverless.quarksoft.io";
pub const EnvVarNodeMgrPodId                  : &str = "qserverless_podid";
pub const EnvVarNodeMgrNamespace              : &str = "qserverless_namespace";
pub const EnvVarNodeMgrPackageId              : &str = "qserverless_packageid";
pub const EnvVarNodeAgentAddr                 : &str = "qserverless_nodeagentaddr";
pub const DefaultNodeAgentAddr                : &str = "unix:///var/lib/quark/nodeagent/sock";
pub const DefaultNodeFuncLogFolder            : &str = "/var/log/quark";

pub const BLOB_LOCAL_HOST                     : &str = "local";

pub const ERROR_SOURCE_SYSTEM: i32 = 1;
pub const ERROR_SOURCE_USER: i32 = 2;

#[derive(Debug, Clone)]
pub struct FuncErr {
    pub source: i32,
    pub error: String,
}

pub enum FuncErrSource {
    System = 1,
    User = 2,
}

// #[derive(Debug)]
// pub enum FuncRes {
//     Error(FuncErr),
//     Resp(String)
// }

// impl From<func::FuncRes> for FuncRes {
//     fn from(item: func::FuncRes) -> Self {
//         let res = item.res.unwrap();
//         match res {
//             func::func_res::Res::Error(error) => {
//                 return Self::NewErrorWithCode(error.source, error.error)
//             }
//             func::func_res::Res::Resp(resp) => {
//                 return Self::NewResponse(resp)
//             }
//         }
//     }
// }

// impl FuncRes {
//     pub fn NewError(source: FuncErrSource, error: String) -> Self {
//         return Self::Error(FuncErr {
//             source: source as i32,
//             error: error
//         })
//     }

//     pub fn NewErrorWithCode(source: i32, error: String) -> Self {
//         return Self::Error(FuncErr {
//             source: source,
//             error: error
//         })
//     }

//     pub fn ToResult(&self) -> SResult<String, FuncErr> {
//         match self {
//             FuncRes::Resp(resp) => return Ok(resp.clone()),
//             FuncRes::Error(e) => return Err(e.clone()),
//         }
//     }

//     pub fn NewResponse(response: String) -> Self {
//         return Self::Resp(response)
//     }

//     pub fn ToGrpc(&self) -> func::FuncRes {
//         let res = match self {
//             Self::Error(e) => {
//                 let error = func::Error {
//                     source: e.source,
//                     error: e.error.clone(),
//                 };
//                 func::func_res::Res::Error(error)
//             }
//             Self::Resp(r) => {
//                 func::func_res::Res::Resp(r.clone())
//             }
//         };

//         return func::FuncRes {
//             res: Some(res)
//         }
//     }

//     pub fn Success(&self) -> bool {
//         match self {
//             Self::Error(_) => false,
//             Self::Resp(_) => true,
//         }
//     }
// }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaEvent {
    pub type_: EventType,
    pub inInitialList: bool,

    pub obj: DataObject,
    pub oldObj: Option<DataObject>,
}

#[derive(Debug)]
pub struct WatchEvent {
    pub type_: EventType,

    pub obj: DataObject,
}

#[derive(Debug, Default)]
pub struct DataObjectInner {
    pub kind: String,
    pub namespace: String,
    pub name: String,
    pub lables: Labels,
    pub annotations: Labels,

    // Revision of the Channel
    pub channelRev: i64,

    // revision number set by creator of object such as etcd
    pub revision: i64,

    pub data: String,
}

impl PartialEq for DataObjectInner {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind &&
        self.namespace == other.namespace &&
        self.lables == other.lables &&
        self.annotations == other.annotations &&
        self.revision == other.revision &&
        self.data == other.data
    }
}
impl Eq for DataObjectInner {}

impl DataObjectInner {
    pub fn CopyWithRev(&self, channelRev: i64, revision: i64) -> Self {
        return Self {
            kind: self.kind.clone(),
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            lables: self.lables.Copy(),
            annotations: self.annotations.Copy(),
            channelRev: channelRev,
            revision: revision,
            data: self.data.clone(),
        }
    }
}

impl DeepCopy for DataObjectInner {
    fn DeepCopy(&self) -> Self {
        return Self {
            kind: self.kind.clone(),
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            lables: self.lables.Copy(),
            annotations: self.annotations.Copy(),
            channelRev: self.channelRev,
            revision:self.revision,
            data: self.data.clone(),
        }
    }
}

impl From<&Object> for DataObjectInner {
    fn from(item: &Object) -> Self {
        let mut lables = BTreeMap::new();
        for l in &item.labels {
            lables.insert(l.key.clone(), l.val.clone());
        }

        let mut annotations = BTreeMap::new();
        for l in &item.annotations {
            annotations.insert(l.key.clone(), l.val.clone());
        }

        let inner = DataObjectInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: lables.into(),
            annotations: annotations.into(),
            channelRev: 0,
            revision: 0,
            data: item.data.clone(),
        };

        return inner;
    }
}

impl From<&Obj> for DataObjectInner {
    fn from(item: &Obj) -> Self {
        let mut lables = BTreeMap::new();
        for l in &item.labels {
            lables.insert(l.key.clone(), l.val.clone());
        }

        let mut annotations = BTreeMap::new();
        for l in &item.annotations {
            annotations.insert(l.key.clone(), l.val.clone());
        }

        let inner = DataObjectInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: lables.into(),
            annotations: annotations.into(),
            channelRev: item.channel_rev,
            revision: item.revision,
            data: item.data.clone(),
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
pub struct DataObject(pub Arc<DataObjectInner>);

impl PartialEq for DataObject {
    fn eq(&self, other: &Self) -> bool {
        return self.0 == other.0;
    }
}
impl Eq for DataObject {}

impl From<DataObjectInner> for DataObject {
    fn from(inner: DataObjectInner) -> Self {
        return Self(Arc::new(inner));
    }
}

impl From<&Obj> for DataObject {
    fn from(item: &Obj) -> Self {
        return Self::NewFromObj(item);
    }
}

impl Deref for DataObject {
    type Target = Arc<DataObjectInner>;

    fn deref(&self) -> &Arc<DataObjectInner> {
        &self.0
    }
}

impl DeepCopy for DataObject {
    fn DeepCopy(&self) -> Self {
        return Self(Arc::new(self.0.DeepCopy()));
    }
}

impl DataObject {
    pub fn NewFromK8sObj(kind: &str, meta: &ObjectMeta, data: String) -> Self {
        let mut annotations = BTreeMap::new();
        match &meta.annotations {
            None => (),
            Some(labels) => {
                for (k, v) in labels {
                    annotations.insert(k.to_string(), v.to_string());
                }
            }
        }

        let mut labels = BTreeMap::new();
        match &meta.labels {
            None => (),
            Some(l) => {
                for (k, v) in l {
                    labels.insert(k.to_string(), v.to_string());
                }
            }
        }

        let inner = DataObjectInner {
            kind: kind.to_string(),
            namespace: meta.namespace.as_deref().unwrap_or("").to_string(),
            name: meta.name.as_deref().unwrap_or("").to_string(),
            annotations: annotations.into(),
            lables: labels.into(),
            channelRev: 0,
            revision: meta.resource_version.as_deref().unwrap_or("0").parse::<i64>().unwrap_or(0),
            data: data,
        };

        return Self(Arc::new(inner));
    }

    pub fn NewFromObj(item: &Obj) -> Self {
        let mut lables = BTreeMap::new();
        for l in &item.labels {
            lables.insert(l.key.clone(), l.val.clone());
        }

        let mut annotations = BTreeMap::new();
        for l in &item.annotations {
            annotations.insert(l.key.clone(), l.val.clone());
        }

        let inner = DataObjectInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: lables.into(),
            annotations: annotations.into(),
            channelRev: item.channel_rev,
            revision: item.revision,
            data: item.data.clone(),
        };

        return inner.into();
    }

    pub fn CopyWithRev(&self, channelRev: i64, revision: i64) -> Self {
        return Self(Arc::new(self.0.CopyWithRev(channelRev, revision)));
    }

    pub fn NewFromObject(item: &Object, channelRev: i64, revision: i64) -> Self {
        let mut lables = BTreeMap::new();
        for l in &item.labels {
            lables.insert(l.key.clone(), l.val.clone());
        }

        let mut annotations = BTreeMap::new();
        for l in &item.annotations {
            annotations.insert(l.key.clone(), l.val.clone());
        }

        let inner = DataObjectInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: lables.into(),
            annotations: annotations.into(),
            channelRev: channelRev,
            revision: revision,
            data: item.data.clone(),
        };

        return inner.into();
    }

    pub fn Namespace(&self) -> String {
        return self.namespace.clone();
    }

    pub fn Name(&self) -> String {
        return self.name.clone();
    }

    pub fn Key(&self) -> String {
        return format!("{}/{}", &self.namespace, &self.name);
    }

    pub fn StoreKey(&self) -> String {
        return format!("{}/{}/{}", &self.kind, &self.namespace, &self.name);
    }

    pub fn Object(&self) -> Object {
        return Object {
            kind: self.kind.clone(),
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            labels: self.lables.ToVec(),
            annotations: self.annotations.ToVec(),
            data: self.data.clone(),
        }
    }

    pub fn Obj(&self) -> Obj {
        return Obj {
            kind: self.kind.clone(),
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            channel_rev: self.channelRev,
            revision: self.Revision(),
            labels: self.lables.ToVec(),
            annotations: self.annotations.ToVec(),
            data: self.data.clone(),
        }
    }

    pub fn Revision(&self) -> i64 {
        return self.revision;
    }

    /*pub fn Decode(buf: &[u8]) -> Result<Self> {
        let obj = Obj::decode(buf)?;
        return Ok(Self::NewFromObj(&obj))
    }*/

    pub fn Encode(&self) -> Result<Vec<u8>> {
        let mut buf : Vec<u8> = Vec::new();
        let obj = self.Object();
        buf.reserve(obj.encoded_len());
        obj.encode(&mut buf)?;
        return Ok(buf)
    }

    pub fn Labels(&self) -> Labels {
        let lables = self.lables.clone();
        return lables
    }
}

impl Object {
    pub fn DeepCopy(&self) -> Self {
        return Object { 
            kind: self.kind.to_string(), 
            namespace: self.namespace.to_string(), 
            name: self.name.to_string(), 
            labels: self.labels.clone(), 
            annotations: self.annotations.clone(), 
            data:self.data.to_string () 
        }
    }

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

/********************** it is test type after this **************************** */

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
    pub fn NewPod(namespace: &str, name: &str) -> Result<Self> {
        let mut pod = k8s::Pod::default();
        pod.metadata.namespace = Some(namespace.to_string());
        pod.metadata.name = Some(name.to_string());
        

        let podStr = serde_json::to_string(&pod)?;
        let obj = DataObject::NewFromK8sObj("pod", &pod.metadata, podStr);

        return Ok(obj)
    }

    pub fn NewFuncPackage1(namespace: &str, name: &str) -> Result<Self> {
        let podSpecStr = r#"
        {
            "hostNetwork": true,
            "containers":[
                {
                    "name":"func_pod",
                    "image":"localhost:5000/func_pod:latest"
                }
            ]
        }"#;
        let podSpec: k8s::PodSpec = serde_json::from_str(podSpecStr)?;
        let package = FuncPackage1 {
            metadata: ObjectMeta { 
                namespace: Some(namespace.to_string()),
                name: Some(name.to_string()),
                ..Default::default()
            },
            spec: FuncPackageSpec1 {
                template: podSpec
            },
        };
        let packageStr = serde_json::to_string(&package)?;
        let obj = DataObject::NewFromK8sObj("package", &package.metadata, packageStr);
        return Ok(obj);
    }

    pub fn NewFuncPyPackage(namespace: &str, name: &str) -> Result<Self> {
        let podSpecStr = r#"
        {
            "hostNetwork": true,
            "containers":[
                {
                    "name":"py_pod",
                    "image":"localhost:5000/py_pod:latest"
                }
            ]
        }"#;
        let podSpec: k8s::PodSpec = serde_json::from_str(podSpecStr)?;
        let package = FuncPackage1 {
            metadata: ObjectMeta { 
                namespace: Some(namespace.to_string()),
                name: Some(name.to_string()),
                ..Default::default()
            },
            spec: FuncPackageSpec1 {
                template: podSpec
            },
        };
        let packageStr = serde_json::to_string_pretty(&package)?;
        error!("packageStr is {}", packageStr);
        let obj = DataObject::NewFromK8sObj("package", &package.metadata, packageStr);
        return Ok(obj);
    }
}


#[derive(Debug, Clone)]
pub struct PackageId {
    pub namespace: String,
    pub packageName: String,
}

impl ToString for PackageId {
    fn to_string(&self) -> String {
        return format!("{}/{}", &self.namespace, &self.packageName);
    }
}


impl PackageId {
    pub fn New(packetIdStr: &str) -> Result<Self> {
        let strs : Vec<&str> = packetIdStr.splitn(2, "/").collect();
        if strs.len() != 2 {
            return Err(Error::CommonError(format!("invalid PackageId str {:?}", packetIdStr)));
        }

        return Ok(Self {
            namespace: strs[0].to_string(),
            packageName: strs[1].to_string(),
        })
    }
}

impl Ord for PackageId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.namespace == other.namespace {
            return other.packageName.cmp(&self.packageName);
        }

        return other.namespace.cmp(&other.namespace);
    }
}

impl PartialOrd for PackageId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PackageId {
    fn eq(&self, other: &Self) -> bool {
        return self.namespace == other.namespace && self.packageName == other.packageName;
    }
}

impl Eq for PackageId {}


#[derive(Debug, Clone, Copy)]
pub struct Resource {
    pub mem: i64, // 
    pub cpu: i32,
}

impl Resource {
    pub fn New(memory: u64, cpucores: u32) -> Self {
        return Resource {
            mem: memory as _,
            cpu: (cpucores * 1000) as _,
        }
    }

    pub fn NewFromStr(memStr: &str, cpuStr: &str) -> Result<Self> {
        let mem = Self::ParseMemoryString(memStr)?;
        let cpu = Self::ParseCpuStr(cpuStr)?;
        return Ok(Resource {
            mem: mem as _,
            cpu: cpu as _,
        })
    }

    pub fn ParseMemoryString(memStr: &str) -> Result<u64> {
        let re = Regex::new(r"(\d+)([KMG]i?)?")?;
        if let Some(captures) = re.captures(memStr) {
            let value: u64 = captures.get(1).unwrap().as_str().parse()?;
            let unit = captures.get(2).map(|m| m.as_str()).unwrap_or("");
    
            match unit {
                "" => Ok(value),
                "K" | "Ki" => Ok(value * 1024),
                "M" | "Mi" => Ok(value * 1024 * 1024),
                "G" | "Gi" => Ok(value * 1024 * 1024 * 1024),
                _ => return Err(Error::CommonError(format!("ParseMemoryString fail with memory {:?}", memStr))),
            }
        } else {
            return Err(Error::CommonError(format!("ParseMemoryString fail with memory {:?}", memStr)));
        }
    }

    pub fn ParseCpuStr(cpuStr: &str) -> Result<u32> {
        let re = Regex::new(r"(\d+)(m)?")?;
        if let Some(captures) = re.captures(cpuStr) {
            let value: u32 = captures.get(1).unwrap().as_str().parse()?;
            let milli = captures.get(2).is_some();
            if milli {
                Ok(value)
            } else {
                Ok(value * 1000)
            }
        } else {
            return Err(Error::CommonError(format!("ParseCpuStr fail with cpuStr {:?}", cpuStr)));
        }
    }
}

impl Default for Resource {
    fn default() -> Self {
        return Self {
            mem: 0,
            cpu: 0,
        }
    }
}

impl Add for Resource {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            mem: self.mem + other.mem,
            cpu: self.cpu + other.cpu,
        }
    }
}

impl Sub for Resource {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert!(self.mem >= other.mem && self.cpu >= other.cpu, "Resource::sub {:?} - {:?}", &self, &other);
        Self {
            mem: self.mem - other.mem,
            cpu: self.cpu - other.cpu,
        }
    }
}

impl Resource {
    pub fn Fullfil(&self, req: &Self) -> bool {
        return req.mem <= self.mem && req.cpu <= self.cpu;
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct FuncPackage1 {
    pub metadata: ObjectMeta,
    pub spec: FuncPackageSpec1,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct FuncPackageSpec1 {
    pub template: k8s::PodSpec,
}