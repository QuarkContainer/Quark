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

use std::{collections::BTreeMap, sync::Arc};

use crate::{cadvisor_types::MachineInfo, qlet_config::QletConfig};

#[derive(Clone, Debug, Default)]
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
}

#[derive(Clone, Debug, Default, PartialEq)]
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Quantity(pub i64);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeCondition(pub String);


#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContainerImage {
    /// Names by which this image is known. e.g. \["kubernetes.example/hyperkube:v1.0.7", "cloud-vendor.registry.example/cloud-vendor/hyperkube:v1.0.7"\]
    pub names: Vec<String>,

    /// The size of the image in bytes.
    pub size_bytes: i64,
}

/// NodeStatus is information about the current status of a node.
#[derive(Clone, Debug)]
pub struct NodeStatus {
    /// List of addresses reachable to the node
    pub addresses: Vec<NodeAddress>,

    /// Allocatable represents the resources of a node that are available for scheduling. Defaults to Capacity.
    pub allocatable: BTreeMap<String, Quantity>,

    /// Capacity represents the total resources of a node
    pub capacity: std::collections::BTreeMap<String, Quantity>,

    /// Conditions is an array of current observed node conditions. 
    pub conditions: Vec<NodeCondition>,

    pub config: QletConfig,

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