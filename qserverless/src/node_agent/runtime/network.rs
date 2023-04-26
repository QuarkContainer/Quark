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

use std::net::IpAddr;
use local_ip_address::list_afinet_netifas;
use k8s_openapi::api::core::v1 as k8s;

// NodeHostName identifies a name of the node. Although every node can be assumed
// to have a NodeAddress of this type, its exact syntax and semantics are not
// defined, and are not consistent between different clusters.
pub const NodeHostName : &str = "Hostname";

// NodeInternalIP identifies an IP address which is assigned to one of the node's
// network interfaces. Every node should have at least one address of this type.
//
// An internal IP is normally expected to be reachable from every other node, but
// may not be visible to hosts outside the cluster. By default it is assumed that
// kube-apiserver can reach node internal IPs, though it is possible to configure
// clusters where this is not the case.
//
// NodeInternalIP is the default type of node IP, and does not necessarily imply
// that the IP is ONLY reachable internally. If a node has multiple internal IPs,
// no specific semantics are assigned to the additional IPs.
pub const  NodeInternalIP : &str = "InternalIP";

// NodeExternalIP identifies an IP address which is, in some way, intended to be
// more usable from outside the cluster then an internal IP, though no specific
// semantics are defined. It may be a globally routable IP, though it is not
// required to be.
//
// External IPs may be assigned directly to an interface on the node, like a
// NodeInternalIP, or alternatively, packets sent to the external IP may be NAT'ed
// to an internal node IP rather than being delivered directly (making the IP less
// efficient for node-to-node traffic than a NodeInternalIP).
pub const  NodeExternalIP : &str = "ExternalIP";

// NodeInternalDNS identifies a DNS name which resolves to an IP address which has
// the characteristics of a NodeInternalIP. The IP it resolves to may or may not
// be a listed NodeInternalIP address.
pub const  NodeInternalDNS : &str = "InternalDNS";

// NodeExternalDNS identifies a DNS name which resolves to an IP address which has
// the characteristics of a NodeExternalIP. The IP it resolves to may or may not
// be a listed NodeExternalIP address.
pub const  NodeExternalDNS : &str = "ExternalDNS";

pub struct LocalNetworkAddressProvider{
    pub nodeIPs: Vec<k8s::NodeAddress>,
    pub hostname: String,
}

impl LocalNetworkAddressProvider {
    pub fn Init() -> Self {
        let hostname = hostname::get().unwrap().to_str().unwrap_or("").to_string();
        let ips = GetLocalV4IP();
        let mut addresses = Vec::new();
        for ip in ips {
            let addr = k8s::NodeAddress {
                address: ip.to_string(),
                type_: NodeInternalIP.to_string(),
            };
            addresses.push(addr);
        }
        return Self {
            nodeIPs: addresses,
            hostname: hostname,
        };
    }

    pub fn GetNetAddress(&self) -> Vec<k8s::NodeAddress> {
        return self.nodeIPs.to_vec();
    }
}

pub fn GetLocalV4IP() -> Vec<IpAddr> {
    let network_interfaces = list_afinet_netifas().unwrap();
    let mut ret = Vec::new();
    for (_name, ip) in network_interfaces {
        ret.push(ip)
    }

    return ret;
}