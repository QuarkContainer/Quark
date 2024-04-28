// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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
// limitations under

use core::ops::Deref;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use qshare::common::*;

use super::pod_broker::PodBroker;

lazy_static::lazy_static! {
    pub static ref POD_BRORKER_MGRS: PodBrokerMgrs = {
        PodBrokerMgrs::default()
    };
}

#[derive(Debug, Default)]
pub struct PodBrokerMgrInner {
    // addr to PodBroker
    pub brokersByAddr: HashMap<u32, PodBroker>,
    pub brokersByName: HashMap<String, PodBroker>,
}

#[derive(Debug, Clone, Default)]
pub struct PodBrokerMgr(Arc<RwLock<PodBrokerMgrInner>>);

impl Deref for PodBrokerMgr {
    type Target = Arc<RwLock<PodBrokerMgrInner>>;

    fn deref(&self) -> &Arc<RwLock<PodBrokerMgrInner>> {
        &self.0
    }
}

impl PodBrokerMgr {
    pub fn AddPodBroker(&self, addr: u32, name: &str, podBroker: PodBroker) -> Result<()> {
        let mut inner = self.write().unwrap();
        if inner.brokersByAddr.contains_key(&addr) {
            return Err(Error::Exist(format!(
                "PodBrokerMgr::AddPodBroker addr {:x?} existing",
                addr
            )));
        }

        inner.brokersByAddr.insert(addr, podBroker.clone());
        inner.brokersByName.insert(name.to_owned(), podBroker);
        return Ok(());
    }

    fn GetBrokerByAddr(&self, addr: u32) -> Result<PodBroker> {
        match self.read().unwrap().brokersByAddr.get(&addr) {
            None => {
                return Err(Error::NotExist(format!(
                    "PodBrokerMgr::GetBrokerByAddr fail with addr {:x?}",
                    addr
                )))
            }
            Some(b) => return Ok(b.clone()),
        }
    }

    fn GetBrokerByName(&self, name: &str) -> Result<PodBroker> {
        match self.read().unwrap().brokersByName.get(name) {
            None => {
                return Err(Error::NotExist(format!(
                    "PodBrokerMgr::GetBrokerByName fail with addr {:?}",
                    name
                )))
            }
            Some(b) => return Ok(b.clone()),
        }
    }

    pub fn RemoveBroker(&self, addr: u32, name: &str) -> Result<()> {
        match self.write().unwrap().brokersByAddr.remove(&addr) {
            None => {
                return Err(Error::NotExist(format!(
                    "PodBrokerMgr::RemoveBroker broker with addr {:x} doesn't exist",
                    addr
                )));
            }
            Some(_) => (),
        }

        match self.write().unwrap().brokersByName.remove(name) {
            None => {
                return Err(Error::NotExist(format!(
                    "PodBrokerMgr::RemoveBroker broker with name {} doesn't exist",
                    name
                )));
            }
            Some(_) => (),
        }

        return Ok(());
    }
}

#[derive(Debug, Default)]
pub struct PodBrokerMgrsInner {
    // namespace to PodBrokerMgr
    pub mgrs: HashMap<String, PodBrokerMgr>,
}

#[derive(Debug, Clone, Default)]
pub struct PodBrokerMgrs(Arc<RwLock<PodBrokerMgrsInner>>);

impl Deref for PodBrokerMgrs {
    type Target = Arc<RwLock<PodBrokerMgrsInner>>;

    fn deref(&self) -> &Arc<RwLock<PodBrokerMgrsInner>> {
        &self.0
    }
}

impl PodBrokerMgrs {
    pub fn AddPodBroker(
        &self,
        namespace: &str,
        addr: u32,
        name: &str,
        broker: PodBroker,
    ) -> Result<()> {
        let mgr = self.GetOrCreateBrokerMgr(namespace)?;
        mgr.AddPodBroker(addr, name, broker)?;
        return Ok(());
    }

    pub fn GetOrCreateBrokerMgr(&self, namespace: &str) -> Result<PodBrokerMgr> {
        let mut inner = self.write().unwrap();
        let mgr = match inner.mgrs.get(namespace) {
            None => {
                let mgr = PodBrokerMgr::default();
                inner.mgrs.insert(namespace.to_owned(), mgr.clone());
                mgr
            }
            Some(mgr) => mgr.clone(),
        };

        return Ok(mgr);
    }

    pub fn HandlePodHibernate(&self, namespace: &str, name: &str) -> Result<()> {
        let broker = self.GetBrokerByName(namespace, name)?;
        broker.HandlePodHibernate()?;

        return Ok(());
    }

    pub fn HandlePodWakeup(&self, namespace: &str, name: &str) -> Result<()> {
        let broker = self.GetBrokerByName(namespace, name)?;
        broker.HandlePodWalkup()?;

        return Ok(());
    }

    pub fn HandlePeerConnect(
        &self,
        namespace: &str,
        dstIp: u32,
        dstPort: u16,
        peerIp: u32,
        peerPort: u16,
        socket: i32,
    ) -> Result<()> {
        let broker = self.GetBrokerByAddr(namespace, dstIp)?;
        broker.HandleNewPeerConnection(peerIp, peerPort, dstPort, socket)?;

        return Ok(());
    }

    pub fn GetBrokerMgr(&self, namespace: &str) -> Result<PodBrokerMgr> {
        match self.read().unwrap().mgrs.get(namespace) {
            None => {
                return Err(Error::NotExist(format!(
                    "PodBrokerMgrs::GetMgr fail with namespace {:x?}",
                    namespace
                )))
            }
            Some(b) => return Ok(b.clone()),
        }
    }

    pub fn GetBrokerByAddr(&self, namespace: &str, addr: u32) -> Result<PodBroker> {
        let mgr = self.GetBrokerMgr(namespace)?;
        let broker = mgr.GetBrokerByAddr(addr)?;
        return Ok(broker);
    }

    pub fn GetBrokerByName(&self, namespace: &str, name: &str) -> Result<PodBroker> {
        let mgr = self.GetBrokerMgr(namespace)?;
        let broker = mgr.GetBrokerByName(name)?;
        return Ok(broker);
    }

    pub fn RemoveBroker(&self, namespace: &str, addr: u32, name: &str) -> Result<()> {
        let mgr = self.GetBrokerMgr(namespace)?;
        return mgr.RemoveBroker(addr, name);
    }
}
