// Copyright (c) 2023 Quark Container Authors 
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

use std::ops::Deref;
use std::{collections::BTreeMap, sync::Arc, time::SystemTime};
use k8s_openapi::api::core::v1::Volume;
use serde::{Deserialize, Serialize};

use crate::cadvisor_types::MachineInfo;
use crate::k8s;
use crate::metastore::data_obj::{DataObject, DataObjectInner};
use crate::common::*;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeSystemInfo {
    /// The Architecture reported by the node
    pub architecture: String,

    /// ContainerRuntime Version reported by the node through runtime remote API (e.g. containerd://1.4.2).
    pub container_runtime_version: String,

    /// Kernel Version reported by the node from 'uname -r' (e.g. 3.16.0-0.bpo.4-amd64).
    pub kernel_version: String,

    /// MachineID reported by the node. For unique machine identification in the cluster this field is preferred. Learn more from man(5) machine-id: http://man7.org/linux/man-pages/man5/machine-id.5.html
    pub machine_id: String,

    /// The Operating System reported by the node
    pub operating_system: String,

    /// OS Image reported by the node from /etc/os-release (e.g. Debian GNU/Linux 7 (wheezy)).
    pub os_image: String,

    pub system_uuid: String,

    pub boot_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeAddress {
    /// The node address.
    pub address: String,

    /// Node address type, one of Hostname, ExternalIP or InternalIP.
    pub type_: String,
}

// CPU, in cores. (500m = .5 cores)
pub const ResourceCPU :&str = "cpu";
// Memory, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
pub const ResourceMemory: &str = "memory";
// Volume size, in bytes (e,g. 5Gi = 5GiB = 5 * 1024 * 1024 * 1024)
pub const ResourceStorage: &str = "storage";
// Local ephemeral storage, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
// The resource name for ResourceEphemeralStorage is alpha and it can change across releases.
pub const ResourceEphemeralStorage: &str = "ephemeral-storage";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Quantity(pub i64);

/// NodeCondition contains condition information for a node.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeCondition {
    /// Last time we got an update on a given condition.
    pub last_heartbeat_time: Option<SystemTime>,

    /// Last time the condition transit from one status to another.
    pub last_transition_time: Option<SystemTime>,

    /// Human readable message indicating details about last transition.
    pub message: Option<String>,

    /// (brief) reason for the condition's last transition.
    pub reason: Option<String>,

    /// Status of the condition, one of True, False, Unknown.
    pub status: String,

    /// Type of node condition.
    pub type_: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContainerImage {
    /// Names by which this image is known. e.g. \["kubernetes.example/hyperkube:v1.0.7", "cloud-vendor.registry.example/cloud-vendor/hyperkube:v1.0.7"\]
    pub names: Vec<String>,

    /// The size of the image in bytes.
    pub size_bytes: i64,
}

/// ObjectMeta is metadata that all persisted resources must have, which includes all objects users must create.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObjectMeta {
    pub name: String,
    pub namespace: String,
    pub uid: String,
    pub resource_version: String,
    pub labels: BTreeMap<String, String>,
    /// Annotations is an unstructured key value map stored with a resource that may be set by external tools to store and retrieve arbitrary metadata. They are not queryable and should be preserved when modifying objects. More info: http://kubernetes.io/docs/user-guide/annotations
    pub annotations: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    /////////////// metadata //////////////////////////
    pub name: String,
    pub tenant: String,
    pub namespace: String,
    pub uid: String,
    pub resource_version: String,
    pub labels: BTreeMap<String, String>,
    /// Annotations is an unstructured key value map stored with a resource that may be set by external tools to store and retrieve arbitrary metadata. They are not queryable and should be preserved when modifying objects. More info: http://kubernetes.io/docs/user-guide/annotations
    pub annotations: BTreeMap<String, String>,

    //////////////// spec ////////////////////
    pub node_ip: String,

    /// PodCIDR represents the pod IP range assigned to the node.
    pub pod_cidr: String,

    /// Unschedulable controls node schedulability of new pods. By default, node is schedulable. More info: https://kubernetes.io/docs/concepts/nodes/node/#manual-node-administration
    pub unschedulable: bool,

    //pub spec: NodeDef,
    pub status: NodeStatus,
}

impl Node {
    pub fn NodeId(&self) -> String {
        return format!("{}/{}", &self.namespace, &self.name);
    }

    pub fn ToString(&self) -> String {
        return serde_json::to_string_pretty(self).unwrap();
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeDef {
    pub node_ip: String,

    /// PodCIDR represents the pod IP range assigned to the node.
    pub pod_cidr: String,

    /// Unschedulable controls node schedulability of new pods. By default, node is schedulable. More info: https://kubernetes.io/docs/concepts/nodes/node/#manual-node-administration
    pub unschedulable: bool,
}

/// NodeStatus is information about the current status of a node.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeStatus {
    pub phase: String,
    
    /// List of addresses reachable to the node
    pub addresses: Vec<NodeAddress>,

    /// Allocatable represents the resources of a node that are available for scheduling. Defaults to Capacity.
    pub allocatable: BTreeMap<String, Quantity>,

    /// Capacity represents the total resources of a node
    pub capacity: std::collections::BTreeMap<String, Quantity>,

    /// Conditions is an array of current observed node conditions. 
    pub conditions: Vec<NodeCondition>,

    // pub config: QletConfig,

    /// List of container images on this node
    pub images: Vec<ContainerImage>,

    /// Set of ids/uuids to uniquely identify the node
    pub node_info: NodeSystemInfo,

    // /// List of volumes that are attached to the node.
    // pub volumes_attached: Option<Vec<crate::api::core::v1::AttachedVolume>>,

    // /// List of attachable volumes in use (mounted) by the node.
    // pub volumes_in_use: Option<Vec<String>>,
}

pub fn ResourceListFromMachineInfo(info: &Arc<MachineInfo>) -> BTreeMap<String, Quantity> {
    let mut map = BTreeMap::new();
    map.insert(ResourceCPU.to_string(), Quantity(info.NumCores as i64 * 1000));
    map.insert(ResourceMemory.to_string(), Quantity(info.MemoryCapacity as i64));
    return map;
}

#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// The Architecture reported by the node
    pub architecture: String,

    /// Boot ID reported by the node.
    pub boot_id: String,

    /// ContainerRuntime Version reported by the node through runtime remote API (e.g. containerd://1.4.2).
    pub container_runtime_version: String,

    /// Kernel Version reported by the node from 'uname -r' (e.g. 3.16.0-0.bpo.4-amd64).
    pub kernel_version: String,

    /// MachineID reported by the node. For unique machine identification in the cluster this field is preferred. Learn more from man(5) machine-id: http://man7.org/linux/man-pages/man5/machine-id.5.html
    pub machine_id: String,

    /// The Operating System reported by the node
    pub operating_system: String,

    //// SystemUUID reported by the node. For unique machine identification MachineID is preferred. This field is specific to Red Hat hosts https://access.redhat.com/documentation/en-us/red_hat_subscription_management/1/html/rhsm/uuid
    pub system_uuid: String,

    /// Capacity represents the total resources of a node
    pub capacity: std::collections::BTreeMap<String, Quantity>,
}

impl NodeInfo {
    pub fn NewFromCadvisorInfo(info: &MachineInfo) -> Self {
        let mut map = BTreeMap::new();
        map.insert(ResourceCPU.to_string(), Quantity(info.NumCores as i64 * 1000));
        map.insert(ResourceMemory.to_string(), Quantity(info.MemoryCapacity as i64));
    

        return Self {
            architecture: "amd64".to_owned(),
            boot_id: info.BootID.clone(),
            container_runtime_version: "".to_owned(), // todo:...
            kernel_version: "".to_owned(), // todo:...
            machine_id: info.MachineID.clone(),
            operating_system: "linux".to_owned(),
            system_uuid: info.SystemUUID.clone(),    
            capacity: map,   
        }
    }
}

pub struct QNodeInner {
    
}

pub enum FuncDef {
    PythonFuncDef(PythonFuncDef),
}

pub struct PythonFuncDef {
    pub environment: String,
    pub envs: Vec<(String, String)>,
    pub workingDir: Option<String>,
    pub funcName: String,
    pub initArgments: String,

    pub resourceReq: BTreeMap<String, Quantity>,
}

pub struct Environment {
    pub image: String,
    pub envs: BTreeMap<String, String>,
    pub commands: Vec<String>,
    pub args: Vec<String>,
    pub working_dir: String,  
    pub volume_mounts: Vec<VolumeMount>,

    pub overhead: BTreeMap<String, Quantity>,
}

pub struct EnvDeployment {
    pub environment: String,
    pub resource: BTreeMap<String, Quantity>,
}

pub struct FuncServiceSpec {
    pub environments: BTreeMap<String, Environment>,
    pub functions: BTreeMap<String, FuncDef>,
    pub httpEntryFunc: String, // entry function name
}

pub struct FuncServiceDeployConfig {
    pub envDeployments: BTreeMap<String, EnvDeployment>, // envDeployName --> EnvDeployment
    pub funcMapping: BTreeMap<String, String>, // funcName --> PodName
}

pub struct FuncServiceDeployment {
    pub envDeployments: BTreeMap<String, PodDef>, // podname --> PodDef
}

pub struct FuncServiceInstance {

}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct PodDef {
    pub tenant: String,
    pub namespace: String,
    pub name: String,
    pub uid: String,
    pub resource_version: String,
    pub labels: BTreeMap<String, String>,
    pub annotations: BTreeMap<String, String>,

    pub init_containers: Vec<ContainerDef>,
    pub containers: Vec<ContainerDef>,
    pub volumes: Vec<Volume>,
    pub host_network: bool,
    pub host_name: String,
    pub host_ipc: bool,
    pub host_pid: bool,
    pub share_process_namespace: bool,
    pub overhead: BTreeMap<String, Quantity>,
    pub deletion_timestamp: Option<SystemTime>,
    pub deletion_grace_period_seconds: Option<i32>,
    pub termination_grace_period_seconds: Option<i32>,
    pub runtime_class_name: Option<String>,
    pub security_context: Option<k8s::PodSecurityContext>,
    pub ipAddr: u32,
    
    pub status: PodStatus,
}

impl PodDef {
    pub const KEY: &'static str = "pod";

    pub fn PodId(&self) -> String {
        return format!("{}/{}/{}", &self.tenant, &self.namespace, &self.name);
    }

    pub fn PodNamespace(&self) -> String {
        return format!("{}/{}", &self.tenant, &self.namespace);
    }

    pub fn ToString(&self) -> String {
        return serde_json::to_string_pretty(self).unwrap();
    }

    pub fn FromDataObject(obj: DataObject) -> Result<Self> {
        let spec = match serde_json::from_str::<Self>(&obj.data) {
            Err(e) => return Err(Error::CommonError(format!("FuncPackageSpec::FromDataObject {:?}", e))),
            Ok(s) => s
        };
        return Ok(spec);
    }

    pub fn DataObject(&self) -> DataObject {
        let inner = DataObjectInner {
            kind: Self::KEY.to_owned(),
            tenant: self.tenant.clone(),
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            data: serde_json::to_string_pretty(&self).unwrap(),
            ..Default::default()
        };

        return inner.into();
    }
}

#[derive(Debug, Default, Clone)]
pub struct PodDefBox(Arc<PodDef>);

impl Deref for PodDefBox {
    type Target = Arc<PodDef>;

    fn deref(&self) -> &Arc<PodDef> {
        &self.0
    }
}

impl From<PodDef> for PodDefBox {
    fn from(item: PodDef) -> Self {
        return Self(Arc::new(item))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PodCondition {
    /// Last time we probed the condition.
    pub last_probe_time: SystemTime,

    /// Last time the condition transitioned from one status to another.
    pub last_transition_time: SystemTime,

    /// Human-readable message indicating details about last transition.
    pub message: String,

    /// Unique, one-word, CamelCase reason for the condition's last transition.
    pub reason: String,

    /// Status is the status of the condition. Can be True, False, Unknown. More info: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#pod-conditions
    pub status: String,

    /// Type is the type of the condition. More info: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#pod-conditions
    pub type_: String,
}

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct PodStatus {
    pub host_ip: String,
    pub pod_ip: String,
    pub pod_ips: Vec<String>,
    pub phase: String,
    pub conditions: Vec<PodCondition>,
    pub start_time: Option<SystemTime>,
}

/// VolumeMount describes a mounting of a Volume within a container.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Path within the container at which the volume should be mounted.  Must not contain ':'.
    pub mount_path: String,

    /// mountPropagation determines how mounts are propagated from the host to container and the other way around. When not set, MountPropagationNone is used. This field is beta in 1.10.
    pub mount_propagation: Option<String>,

    /// This must match the Name of a Volume.
    pub name: String,

    /// Mounted read-only if true, read-write otherwise (false or unspecified). Defaults to false.
    pub read_only: Option<bool>,

    /// Path within the volume from which the container's volume should be mounted. Defaults to "" (volume's root).
    pub sub_path: Option<String>,

    /// Expanded path within the volume from which the container's volume should be mounted. Behaves similarly to SubPath but environment variable references $(VAR_NAME) are expanded using the container's environment. Defaults to "" (volume's root). SubPathExpr and SubPath are mutually exclusive.
    pub sub_path_expr: Option<String>,
}

/// ResourceRequirements describes the compute resource requirements.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Limits describes the maximum amount of compute resources allowed. More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
    pub limits: BTreeMap<String, Quantity>,

    /// Requests describes the minimum amount of compute resources required. If Requests is omitted for a container, it defaults to Limits if that is explicitly specified, otherwise to an implementation-defined value. More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
    pub requests: BTreeMap<String, Quantity>,
}

/// ContainerPort represents a network port in a single container.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContainerPort {
    /// Number of port to expose on the pod's IP address. This must be a valid port number, 0 \< x \< 65536.
    pub container_port: i32,

    /// What host IP to bind the external port to.
    pub host_ip: Option<String>,

    /// Number of port to expose on the host. If specified, this must be a valid port number, 0 \< x \< 65536. If HostNetwork is specified, this must match ContainerPort. Most containers do not need this.
    pub host_port: Option<i32>,

    /// If specified, this must be an IANA_SVC_NAME and unique within the pod. Each named port in a pod must have a unique name. Name for the port that can be referred to by services.
    pub name: Option<String>,

    /// Protocol for port. Must be UDP, TCP, or SCTP. Defaults to "TCP".
    ///
    pub protocol: Option<String>,
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
pub struct ContainerDef {
    pub name: String,
    pub image: String,
    pub envs: BTreeMap<String, String>,
    pub commands: Vec<String>,
    pub args: Vec<String>,
    pub working_dir: String,   
    pub volume_mounts: Vec<VolumeMount>,
    pub stdin: bool,
    pub stdin_once: bool,
    pub resources: ResourceRequirements,
    pub ports: Vec<ContainerPort>,
    pub lifecycle: Option<Lifecycle>,
    /// Periodic probe of container service readiness. Container will be removed from service endpoints if the probe fails. Cannot be updated. More info: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#container-probes
    pub readiness_probe: Option<Probe>,
}

/// ExecAction describes a "run in container" action.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecAction {
    /// Command is the command line to execute inside the container, the working directory for the command  is root ('/') in the container's filesystem. The command is simply exec'd, it is not run inside a shell, so traditional shell instructions ('|', etc) won't work. To use a shell, you need to explicitly call out to that shell. Exit status of 0 is treated as live/healthy and non-zero is unhealthy.
    pub command: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HTTPHeader {
    /// The header field name
    pub name: String,

    /// The header field value
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IntOrString {
    Int(i32),
    String(String),
}

impl Default for IntOrString {
    fn default() -> Self {
        IntOrString::Int(0)
    }
}

/// HTTPGetAction describes an action based on HTTP Get requests.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct HTTPGetAction {
    /// Host name to connect to, defaults to the pod IP. You probably want to set "Host" in httpHeaders instead.
    pub host: Option<String>,

    /// Custom headers to set in the request. HTTP allows repeated headers.
    pub http_headers: Vec<HTTPHeader>,

    /// Path to access on the HTTP server.
    pub path: Option<String>,

    /// Name or number of the port to access on the container. Number must be in the range 1 to 65535. Name must be an IANA_SVC_NAME.
    pub port: IntOrString,

    /// Scheme to use for connecting to the host. Defaults to HTTP.
    ///
    pub scheme: String,
}

/// TCPSocketAction describes an action based on opening a socket
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TCPSocketAction {
    pub host: String,

    /// Number or name of the port to access on the container. Number must be in the range 1 to 65535. Name must be an IANA_SVC_NAME.
    pub port: IntOrString,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LifecycleHandler {
    /// Exec specifies the action to take.
    pub exec: Option<ExecAction>,

    /// HTTPGet specifies the http request to perform.
    pub http_get: Option<HTTPGetAction>,

    /// Deprecated. TCPSocket is NOT supported as a LifecycleHandler and kept for the backward compatibility. There are no validation of this field and lifecycle hooks will fail in runtime when tcp handler is specified.
    pub tcp_socket: Option<TCPSocketAction>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Lifecycle {
    /// PostStart is called immediately after a container is created. If the handler fails, the container is terminated and restarted according to its restart policy. Other management of the container blocks until the hook completes. More info: https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/#container-hooks
    pub post_start: Option<LifecycleHandler>,

    /// PreStop is called immediately before a container is terminated due to an API request or management event such as liveness/startup probe failure, preemption, resource contention, etc. The handler is not called if the container crashes or exits. The Pod's termination grace period countdown begins before the PreStop hook is executed. Regardless of the outcome of the handler, the container will eventually terminate within the Pod's termination grace period (unless delayed by finalizers). Other management of the container blocks until the hook completes or until the termination grace period is reached. More info: https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/#container-hooks
    pub pre_stop: Option<LifecycleHandler>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GRPCAction {
    /// Port number of the gRPC service. Number must be in the range 1 to 65535.
    pub port: i32,

    /// Service is the name of the service to place in the gRPC HealthCheckRequest (see https://github.com/grpc/grpc/blob/master/doc/health-checking.md).
    ///
    /// If this is not specified, the default behavior is defined by gRPC.
    pub service: Option<String>,
}

/// Probe describes a health check to be performed against a container to determine whether it is alive or ready to receive traffic.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Probe {
    /// Exec specifies the action to take.
    pub exec: Option<ExecAction>,

    /// Minimum consecutive failures for the probe to be considered failed after having succeeded. Defaults to 3. Minimum value is 1.
    pub failure_threshold: Option<i32>,

    /// GRPC specifies an action involving a GRPC port. This is a beta field and requires enabling GRPCContainerProbe feature gate.
    pub grpc: Option<GRPCAction>,

    /// HTTPGet specifies the http request to perform.
    pub http_get: Option<HTTPGetAction>,

    /// Number of seconds after the container has started before liveness probes are initiated. More info: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#container-probes
    pub initial_delay_seconds: Option<i32>,

    /// How often (in seconds) to perform the probe. Default to 10 seconds. Minimum value is 1.
    pub period_seconds: Option<i32>,

    /// Minimum consecutive successes for the probe to be considered successful after having failed. Defaults to 1. Must be 1 for liveness and startup. Minimum value is 1.
    pub success_threshold: Option<i32>,

    /// TCPSocket specifies an action involving a TCP port.
    pub tcp_socket: Option<TCPSocketAction>,

    /// Optional duration in seconds the pod needs to terminate gracefully upon probe failure. The grace period is the duration in seconds after the processes running in the pod are sent a termination signal and the time when the processes are forcibly halted with a kill signal. Set this value longer than the expected cleanup time for your process. If this value is nil, the pod's terminationGracePeriodSeconds will be used. Otherwise, this value overrides the value provided by the pod spec. Value must be non-negative integer. The value zero indicates stop immediately via the kill signal (no opportunity to shut down). This is a beta field and requires enabling ProbeTerminationGracePeriod feature gate. Minimum value is 1. spec.terminationGracePeriodSeconds is used if unset.
    pub termination_grace_period_seconds: Option<i64>,

    /// Number of seconds after which the probe times out. Defaults to 1 second. Minimum value is 1. More info: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#container-probes
    pub timeout_seconds: Option<i32>,
}