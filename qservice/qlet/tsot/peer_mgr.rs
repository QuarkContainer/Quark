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
use std::sync::Arc; 
use std::sync::RwLock;
use core::ops::Deref;
use std::str::FromStr;

use qshare::common::*;

use crate::QLET_CONFIG;

lazy_static::lazy_static! {
    pub static ref PEER_MGR: PeerMgr = {
        let cidrStr = QLET_CONFIG.cidr.clone();
        let ipv4 = ipnetwork::Ipv4Network::from_str(&cidrStr).unwrap();
        //let localIp = local_ip_address::local_ip().unwrap();
        let pm = PeerMgr::New(ipv4.prefix() as _);
        
        if QLET_CONFIG.singleNodeModel {
            let localIp : u32 = ipnetwork::Ipv4Network::from_str(&QLET_CONFIG.nodeIp).unwrap().ip().into();
            let localPort = QLET_CONFIG.podMgrPort;
            pm.AddPeer(localIp, localPort, ipv4.ip().into()).unwrap();
        }
        pm
    };
}

#[derive(Debug)]
pub struct PeerInner {
    pub hostIp: u32,
    pub port: u16,
    pub cidrAddr: u32,
}

#[derive(Debug, Clone)]
pub struct Peer(Arc<PeerInner>);

impl Peer {
    pub fn New(hostIp: u32, port: u16, cidrAddr: u32) -> Self {
        let inner = PeerInner {
            hostIp: hostIp,
            port: port,
            cidrAddr: cidrAddr
        };

        return Self(Arc::new(inner));
    }
}

impl Deref for Peer {
    type Target = Arc<PeerInner>;

    fn deref(&self) -> &Arc<PeerInner> {
        &self.0
    }
}

#[derive(Debug)]
pub struct PeerMgrInner {
    // map cidrAddr --> Peer
    pub peers: HashMap<u32, Peer>,
    pub maskbits: usize,
    pub mask: u32,
}

#[derive(Debug, Clone)]
pub struct PeerMgr(Arc<RwLock<PeerMgrInner>>);

impl Deref for PeerMgr {
    type Target = Arc<RwLock<PeerMgrInner>>;

    fn deref(&self) -> &Arc<RwLock<PeerMgrInner>> {
        &self.0
    }
}

impl PeerMgr {
    pub fn New(maskbits: usize) -> Self {
        assert!(maskbits < 32);
        let mask = !((1 << maskbits) - 1);
        let inner = PeerMgrInner {
            peers: HashMap::new(),
            maskbits: maskbits,
            mask: mask,
        };

        let mgr = Self(Arc::new(RwLock::new(inner)));
        return mgr;
    }

    pub fn AddPeer(&self, hostIp: u32, port: u16, cidrAddr: u32) -> Result<()> {
        let peer = Peer::New(hostIp, port, cidrAddr);
        let mut inner = self.write().unwrap();
        if inner.peers.contains_key(&cidrAddr) {
            return Err(Error::Exist(format!("PeerMgr::AddPeer get existing peer {:?}", peer)));
        }

        inner.peers.insert(cidrAddr, peer);
        return Ok(())
    }

    pub fn RemovePeer(&self, cidrAddr: u32) -> Result<()> {
        let mut inner = self.write().unwrap();
        match inner.peers.remove(&cidrAddr) {
            None => return Err(Error::NotExist(format!("PeerMgr::RemovePeer peer {:?} not existing", cidrAddr))),
            Some(_peer) => {
                return Ok(())
            }
        }
    }

    pub fn LookforPeer(&self, ip: u32) -> Result<Peer> {
        let inner = self.read().unwrap();
        let cidrAddr = ip & inner.mask;
        match inner.peers.get(&cidrAddr).cloned() {
            None => return Err(Error::NotExist(format!("PeerMgr::LookforPeer peer {:x} doesn't exist", ip))),
            Some(peer) => return Ok(peer.clone()),
        }
    }
}