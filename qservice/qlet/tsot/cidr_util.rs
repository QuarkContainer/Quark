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

use std::net::Ipv4Addr;
use std::str::FromStr;

use ipnetwork::Ipv4Network;

use crate::QLET_CONFIG;

pub fn PodCidrNetwork() -> Ipv4Network {
    Ipv4Network::from_str(&QLET_CONFIG.cidr).expect("invalid pod cidr in qlet config")
}

pub fn IsPodIp(ip: u32) -> bool {
    PodCidrNetwork().contains(Ipv4Addr::from(ip))
}

/// Internet / non-pod destinations (pod relay is pod-to-pod only).
pub fn IsEgressIp(ip: u32) -> bool {
    !IsPodIp(ip)
}

pub fn PodCidrNetworkAddr() -> u32 {
    PodCidrNetwork().network().into()
}
