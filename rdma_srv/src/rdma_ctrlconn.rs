// Copyright (c) 2021 Quark Container Authors
//
// Licensed un&der the Apache License, Version 2.0 (the "License");
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
use std::cell::RefCell;
use std::iter::FromIterator;
use std::net::{IpAddr, Ipv4Addr};
use std::{collections::HashMap, collections::HashSet, str::FromStr};
use super::common::*;
use std::os::unix::io::{AsRawFd, RawFd};

use super::qlib::rdma_share::*;

pub struct CtrlConn {
    // socket fd connect to ConnectionMgr
    pub sockfd: i32,
}

pub struct CtrlInfo {
    // nodes: node ipaddr --> Node
    pub nodes: Mutex<HashMap<u32, Node>>,

    // pods: pod ipaddr --> Pod
    pub pods: Mutex<HashMap<u32, Pod>>,

    // containerids: containerid --> ip
    pub containerids: Mutex<HashMap<String, u32>>,

    //ip --> podId
    pub ipToPodIdMappings: Mutex<HashMap<u32, String>>,

    // subnetmapping: virtual subnet --> node ipaddr
    pub subnetmap: Mutex<HashMap<u32, u32>>,

    // Virtual Endpoints
    pub veps: Mutex<HashMap<VirtualEp, VirtualEpInfo>>,

    // cluster subnet information
    pub clusterSubnetInfo: Mutex<ClusterSubnetInfo>,

    //Pod Ip to Node Ip
    pub podIpInfo: Mutex<HashMap<u32, u32>>,

    // Hashmap for file descriptors so that different handling can be dispatched.
    pub fds: Mutex<HashMap<i32, Srv_FdType>>,

    // Hostname of the node
    pub hostname: Mutex<String>,

    // Timestamp of the node
    pub timestamp: Mutex<u64>,

    pub epoll_fd: Mutex<RawFd>,

    pub isK8s: bool,

    pub isCMConnected: Mutex<bool>,
}

impl Default for CtrlInfo {
    fn default() -> CtrlInfo {        
        let mut nodes: HashMap<u32, Node> = HashMap::new();
        let pods: HashMap<u32, Pod> = HashMap::new();
        let mut containerids: HashMap<String, u32> = HashMap::new();
        let mut ipToPodIdMappings: HashMap<u32, String> = HashMap::new();

        let isK8s = true;
        if !isK8s {
            let lab1ip = u32::from(Ipv4Addr::from_str("172.16.1.43").unwrap()).to_be();
            let node1 = Node {
                hostname: String::from("lab 1"),
                ipAddr: lab1ip,
                timestamp: 0,
                subnet: u32::from(Ipv4Addr::from_str("192.168.2.0").unwrap()),
                netmask: u32::from(Ipv4Addr::from_str("255.255.255.0").unwrap()),
                resource_version: 0,                
            };
            let lab2ip = u32::from(Ipv4Addr::from_str("172.16.1.99").unwrap()).to_be();
            let node2 = Node {
                hostname: String::from("lab 2"),
                ipAddr: lab2ip,
                timestamp: 0,
                subnet: u32::from(Ipv4Addr::from_str("192.168.1.0").unwrap()),
                netmask: u32::from(Ipv4Addr::from_str("255.255.255.0").unwrap()),
                resource_version: 0,
            };
            nodes.insert(lab1ip, node1);
            nodes.insert(lab2ip, node2);
            containerids.insert("server".to_string(), u32::from(Ipv4Addr::from_str("192.168.2.8").unwrap()).to_be());
            containerids.insert("client".to_string(), u32::from(Ipv4Addr::from_str("192.168.1.8").unwrap()).to_be());
            ipToPodIdMappings.insert(u32::from(Ipv4Addr::from_str("192.168.2.8").unwrap()).to_be(), "server".to_string());
            ipToPodIdMappings.insert(u32::from(Ipv4Addr::from_str("192.168.1.8").unwrap()).to_be(), "client".to_string());
        }

        CtrlInfo {
            nodes: Mutex::new(nodes),
            pods: Mutex::new(pods),
            containerids: Mutex::new(containerids),
            ipToPodIdMappings: Mutex::new(ipToPodIdMappings),
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
                u32::from(Ipv4Addr::from_str("192.168.6.8").unwrap()).to_le(),
                (u32::from(Ipv4Addr::from_str("172.16.1.43").unwrap()).to_be()),
            )])),
            fds: Mutex::new(HashMap::new()),
            hostname: Mutex::new(String::new()),
            timestamp: Mutex::new(0),
            epoll_fd: Mutex::new(0),
            isK8s: isK8s,
            isCMConnected: Mutex::new(false),
        }
    }
}

impl CtrlInfo{
    pub fn fds_insert(&self, fd: i32, fdType: Srv_FdType){
        let mut fds = self.fds.lock();
        fds.insert(fd, fdType);
    }

    pub fn fds_get(&self, fd: i32) -> Srv_FdType {
        let fds = self.fds.lock();
        fds.get(&fd).unwrap().clone()
    }

    pub fn hostname_set(&self, value: String) {
        let mut hostname = self.hostname.lock();
        *hostname = value;
    }

    pub fn hostname_get(&self) -> String {
        self.hostname.lock().clone()
    }

    pub fn timestamp_set(&self, value: u64) {
        let mut timestamp = self.timestamp.lock();
        *timestamp = value;
    }

    pub fn timestamp_get(&self) -> u64 {
        self.timestamp.lock().clone()
    }

    pub fn isCMConnected_set(&self, value: bool) {
        let mut isCMConnected = self.isCMConnected.lock();
        *isCMConnected = value;
    }

    pub fn isCMConnected_get(&self) -> bool {
        self.isCMConnected.lock().clone()
    }

    pub fn get_node_ips_for_connecting(&self) -> HashSet<u32> {
        let mut set: HashSet<u32> = HashSet::new();
        let timestamp = self.timestamp_get();
        for (_, node) in self.nodes.lock().iter() {
            debug!("get_node_ips_for_connecting, node: {:?} node.timestamp: {} timestamp: {}", node, node.timestamp, timestamp);
            if node.timestamp < timestamp {
                debug!("get_node_ips_for_connecting, node.ipAddr: {}", node.ipAddr);
                set.insert(node.ipAddr);
            }
        }
        set
    }

    pub fn get_node_ip_by_pod_ip(&self, ip: &u32) -> Option<u32> {
        for (_, node) in self.nodes.lock().iter() {
            // if !self.isK8s {
            //     return Some(node.ipAddr);
            // }
            if node.netmask & *ip == node.subnet {
                return Some(node.ipAddr);
            }
        }
        None
    }

    pub fn epoll_fd_set(&self, value: RawFd) {
        let mut epoll_fd = self.epoll_fd.lock();
        *epoll_fd = value;
    }

    pub fn epoll_fd_get(&self) -> RawFd {
        self.epoll_fd.lock().clone()
    }

    pub fn node_get(&self, ip: u32) -> Node {
        debug!("node_get, ip: {}", ip);
        self.nodes.lock().get(&ip).unwrap().clone()
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
#[derive(Default, Debug, Clone)]
pub struct Node {
    pub ipAddr: u32,
    pub timestamp: u64,
    pub hostname: String,
    pub resource_version: i32,

    // node subnet/mask
    pub subnet: u32,
    pub netmask: u32,
    //pub nodename: String ....
}

#[derive(Default, Debug, Clone)]
pub struct Pod {
    pub key: String,
    pub ip: u32,
    pub node_name: String,
    pub container_id: String,
    pub resource_version: i32,
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
