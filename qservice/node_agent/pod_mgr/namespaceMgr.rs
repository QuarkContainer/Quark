// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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
use std::ops::Deref;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex;


use qshare::common::*;

use super::cidr::Cidr;
use super::pod_sandbox::PodSandbox;
use super::NODEAGENT_CONFIG;

#[derive(Debug, Clone)]
pub struct NamespaceInner {
    pub namespace: String,

    pub cidr: Cidr,
}

#[derive(Debug, Clone)]
pub struct Namespace(Arc<Mutex<NamespaceInner>>);

impl Deref for Namespace {
    type Target = Arc<Mutex<NamespaceInner>>;

    fn deref(&self) -> &Arc<Mutex<NamespaceInner>> {
        &self.0
    }
}

impl Namespace {
    pub fn New(namespace: &str, addr: u32, maskbits: usize) -> Self {
        let inner = NamespaceInner {
            namespace: namespace.to_owned(),
            cidr: Cidr::New(addr, maskbits),
        };

        return Self(Arc::new(Mutex::new(inner)));
    }

    pub fn NewPodSandboxAddr(&self) -> Result<IpAddress> {
        let inner = self.lock().unwrap();
        let addr = inner.cidr.Allocate()?;

        return Ok(addr)
    }

    pub fn RemovePodSandboxAddr(&self, addr: IpAddress) -> Result<()> {
        let inner = self.lock().unwrap();
        inner.cidr.Free(addr)?;
        return Ok(())
    }
}

#[derive(Debug)]
pub struct NamespaceMgrInner {
    pub namespaces: HashMap<String, Namespace>,
    pub addr: u32,
    pub maskbits: usize,

    pub podSandboxes: HashMap<String, PodSandbox>,
}

impl NamespaceMgrInner {
    pub fn RemovePodSandbox(&mut self, uid: &str) -> Result<()> {
        match self.podSandboxes.remove(uid) {
            None => return Err(Error::NotExist(format!("RemovePodSandbox has no podsandbox with uid {uid}"))),
            Some(sandbox) => {
                let ns = sandbox.lock().unwrap().namespace.clone();
                let addr = sandbox.lock().unwrap().ip.clone();
                let namespace = self.GetNamespace(&ns)?;
                namespace.RemovePodSandboxAddr(addr)?;
                
                return Ok(())
            }
        }
    }

    pub fn GetNamespace(&self, namespace: &str) -> Result<Namespace> {
        match self.namespaces.get(namespace) {
            None => {
                return Err(Error::NotExist(format!("NamespaceMgr namespace {namespace} doesn't exist")))
            }
            Some(ns) => {
                return Ok(ns.clone());
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct NamespaceMgr(Arc<Mutex<NamespaceMgrInner>>);

impl Deref for NamespaceMgr {
    type Target = Arc<Mutex<NamespaceMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<NamespaceMgrInner>> {
        &self.0
    }
}

impl NamespaceMgr {
    pub fn New() -> Self {
        let cidrStr = NODEAGENT_CONFIG.cidr.clone();
        let ipv4 = ipnetwork::Ipv4Network::from_str(&cidrStr).unwrap();

        let inner = NamespaceMgrInner {
            namespaces: HashMap::new(),
            addr: ipv4.ip().into(),
            maskbits: ipv4.prefix() as _,
            podSandboxes: HashMap::new(),
        };

        return Self(Arc::new(Mutex::new(inner)))
    }

    pub fn GetOrCreateNamespace(&self, namespace: &str) -> Namespace {
        let mut inner = self.lock().unwrap();
        match inner.namespaces.get(namespace) {
            None => {
                let ns = Namespace::New(namespace, inner.addr, inner.maskbits);
                inner.namespaces.insert(namespace.to_owned(), ns.clone());
                return ns;
            }
            Some(ns) => {
                return ns.clone();
            }
        }
    }

    pub fn GetNamespace(&self, namespace: &str) -> Result<Namespace> {
        return self.lock().unwrap().GetNamespace(namespace);
    }

    pub fn NewPodSandbox(&self, namespace: &str, uid: &str, name: &str) -> Result<IpAddress> {
        let ns = self.GetOrCreateNamespace(namespace);
        let addr = ns.NewPodSandboxAddr()?;

        let mut inner = self.lock().unwrap();
        let podsandbox = PodSandbox::New(uid, namespace, name, addr);
        match inner.podSandboxes.insert(uid.to_owned(), podsandbox) {
            None => (),
            Some(_) => {
                return Err(Error::Exist(format!("NewPodSandbox exist podsandbox with uid {uid}")))
            }
        }

        return Ok(addr)
    }

    pub fn RemovePodSandbox(&self, uid: &str) -> Result<()> {
        return self.lock().unwrap().RemovePodSandbox(uid);
    }

    pub fn GetPodSandbox(&self, uid: &str) -> Result<PodSandbox> {
        let inner = self.lock().unwrap();
        
        match inner.podSandboxes.get(uid) {
            None => return Err(Error::NotExist(format!("the podsandbox with uid {uid} not exist"))),
            Some(podsandbox) => {
                return Ok(podsandbox.clone())
            }
        }
    }

    pub fn GetPodSandboxAddr(&self, uid: &str) -> Result<IpAddress> {
        let ns = self.GetPodSandbox(uid)?;

        return Ok(ns.lock().unwrap().ip);
    }
}
