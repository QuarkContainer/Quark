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

use std::collections::HashMap;
use spin::Mutex;

pub struct CtrlConn {
    // socket fd connect to ConnectionMgr
    pub sockfd: i32
}

pub struct CtrlInfo {
    // nodes: node ipaddr --> Node
    pub nodes: Mutex<HashMap<u32, Node>>,

    // subnetmapping: subnet --> node ipaddr
    pub subnetmap: Mutex<HashMap<u32, u32>>,

    // Virtual Endpoints
    pub veps: Mutex<HashMap<VirtualEp, VirtualEpInfo>>,

    // cluster subnet information
    pub clusterSubnetInfo: Mutex<ClusterSubnetInfo>,
}

pub struct ClusterSubnetInfo {
    pub subnet: u32,
    pub netmask: u32,
    pub vipSubnet: u32,
    pub vipNetwork: u32,
}

// from current design, one node has only one subnet even it can have multiple VPC
// for one node, different VPC has to use one subnet,
// todo: support different subnet for different VPC
pub struct Node {
    pub ipAddr: u32,
    pub timestamp: u64,

    // node subnet/mask
    pub subnet: u32,
    pub netmask: u32,
    //pub nodename: String ....
}

pub struct VirtualEp {
    pub vpcId: u32,
    pub ipAddr: u32,
    pub port: u16,
}

pub struct Endpoint {
    pub ipAddr: u32,
    pub port: u16,
}

pub struct VirtualEpInfo {
    pub vep: VirtualEp,
    pub dstEps: Vec<Endpoint>
}
