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

use super::common::*;
use super::k8s;

pub const QMETASVC_PORT : u16 = 8890; 
pub const GATEWAY_PORT : u16 = 8889;
pub const NODEMGRSVC_ADDR : &str = "127.0.0.1:8888";
pub const QMETASVC_ADDR : &str = "127.0.0.1:8890";
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



pub const DefaultNodeMgrNodeNameSpace: &str = "qserverless.io";
pub const DefaultDomainName: &str = "qserverless.io";

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