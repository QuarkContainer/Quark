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

use std::{collections::HashMap, sync::{Arc, RwLock}};
use core::ops::Deref;

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
    pub brokers: HashMap<u32, PodBroker>,
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
    pub fn AddPodBroker(&self, addr: u32, podBroker: PodBroker) -> Result<()> {
        let mut inner = self.write().unwrap();
        if inner.brokers.contains_key(&addr) {
            return Err(Error::Exist(format!("PodBrokerMgr::AddPodBroker addr {:x?} existing", addr)));
        }

        inner.brokers.insert(addr, podBroker);
        return Ok(())
    }

    fn GetBroker(&self, addr: u32) -> Result<PodBroker> {
        match self.read().unwrap().brokers.get(&addr) {
            None => return Err(Error::NotExist(format!("PodBrokerMgr::GetBroker fail with addr {:x?}", addr))),
            Some(b) => return Ok(b.clone()),
        }
    }

    pub fn RemoveBroker(&self, addr: u32) -> Result<()> {
        match self.write().unwrap().brokers.remove(&addr) {
            None => {
                return Err(Error::NotExist(format!("PodBrokerMgr::RemoveBroker broker with addr {:x} doesn't exist", addr)));
            }
            Some(_) => return Ok(())
        }
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
    pub fn AddPodBroker(&self, namespace: &str, addr: u32, broker: PodBroker) -> Result<()> {
        let mgr = self.GetOrCreateBrokerMgr(namespace)?;
        mgr.AddPodBroker(addr, broker)?;
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

        return Ok(mgr)
    } 

    pub fn HandlePeerConnect(&self, namespace: &str, targetIp: u32, srcIp: u32, localPort: u16, socket: i32) -> Result<()> {
        let broker = self.GetBroker(namespace, targetIp)?;
        broker.HandleNewPeerConnection(srcIp, localPort, socket)?;
        
        return Ok(())
    }

    pub fn GetBrokerMgr(&self, namespace: &str) -> Result<PodBrokerMgr> {
        match self.read().unwrap().mgrs.get(namespace) {
            None => return Err(Error::NotExist(format!("PodBrokerMgrs::GetMgr fail with namespace {:x?}", namespace))),
            Some(b) => return Ok(b.clone()),
        }
    }

    pub fn GetBroker(&self, namespace: &str, addr: u32) -> Result<PodBroker> {
        let mgr = self.GetBrokerMgr(namespace)?;
        let broker = mgr.GetBroker(addr)?;
        return Ok(broker);
    }

    pub fn RemoveBroker(&self, namespace: &str, addr: u32) -> Result<()> {
        let mgr = self.GetBrokerMgr(namespace)?;
        return mgr.RemoveBroker(addr);
    }
}