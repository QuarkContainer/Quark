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

use spin::Mutex;
use std::iter::FromIterator;
use std::net::{IpAddr, Ipv4Addr};
use std::{collections::HashMap, str::FromStr};

use super::qlib::rdma_share::*;

pub struct CtrlConn {
    // socket fd connect to ConnectionMgr
    pub sockfd: i32,
}

pub struct CtrlInfo {
    // nodes: node ipaddr --> Node
    pub nodes: Mutex<HashMap<u32, Node>>,

    // subnetmapping: virtual subnet --> node ipaddr
    pub subnetmap: Mutex<HashMap<u32, u32>>,

    // Virtual Endpoints
    pub veps: Mutex<HashMap<VirtualEp, VirtualEpInfo>>,

    // cluster subnet information
    pub clusterSubnetInfo: Mutex<ClusterSubnetInfo>,

    //Pod Ip to Node Ip
    pub podIpInfo: Mutex<HashMap<u32, u32>>,
}

impl Default for CtrlInfo {
    fn default() -> CtrlInfo {
        let mut nodes: HashMap<u32, Node> = HashMap::new();
        let subnet = u32::from(Ipv4Addr::from_str("172.16.1.0").unwrap());
        let netmask = u32::from(Ipv4Addr::from_str("255.255.255.0").unwrap());
        let lab1ip = u32::from(Ipv4Addr::from_str("172.16.1.8").unwrap());
        let devip = u32::from(Ipv4Addr::from_str("172.16.1.6").unwrap());
        nodes.insert(
            lab1ip,
            Node {
                ipAddr: lab1ip,
                timestamp: 1234,
                subnet: subnet,
                netmask: netmask,
            },
        );

        nodes.insert(
            devip,
            Node {
                ipAddr: devip,
                timestamp: 5678,
                subnet: subnet,
                netmask: netmask,
            },
        );

        CtrlInfo {
            nodes: Mutex::new(nodes),
            subnetmap: Mutex::new(HashMap::new()),
            veps: Mutex::new(HashMap::new()),
            clusterSubnetInfo: Mutex::new(ClusterSubnetInfo {
                subnet: 0,
                netmask: 0,
                vipSubnet: 0,
                vipNetmask: 0,
            }),
            //134654144 -> 100733100
            podIpInfo: Mutex::new(HashMap::from_iter([(
                u32::from(Ipv4Addr::from_str("192.168.6.8").unwrap()).to_be(),
                (u32::from(Ipv4Addr::from_str("172.16.1.6").unwrap()).to_be()),
            )])),
        }
    }
}

pub struct ClusterSubnetInfo {
    pub subnet: u32,
    pub netmask: u32,
    pub vipSubnet: u32,
    pub vipNetmask: u32,
}

// virual subnet
// cluster: 10.1.0.0/16
// node1: 10.1.1.0/24
// node2: 10.1.2.0/24
// node3: 10.1.3.0/24

// from current design, one node has only one subnet even it can have multiple VPC
// for one node, different VPC has to use one subnet,
// todo: support different subnet for different VPC
#[derive(Default)]
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
    pub dstPort: u16, //target port??
}

pub struct VirtualEpInfo {
    pub vep: VirtualEp,
    pub dstEps: Vec<Endpoint>,
}
