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

use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;

use qshare::common::*;
use qshare::node::PodDef;
use qshare::node::PodDefBox;

#[derive(Debug, Default)]
pub struct PodMgrInner {
    pub pods: BTreeMap<String, PodDefBox>,
}

#[derive(Debug, Default, Clone)]
pub struct PodMgr(Arc<Mutex<PodMgrInner>>);

impl Deref for PodMgr {
    type Target = Arc<Mutex<PodMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<PodMgrInner>> {
        &self.0
    }
}

impl PodMgr {
    pub fn GetFuncPods(&self, tenant: &str, namespace: &str, funcname: &str) -> Result<Vec<PodDef>> {
        use std::ops::Bound::*;
        let start = format!("{}/{}/{}_",tenant, namespace, funcname);
        let mut vec = Vec::new();
        for (key, val) in self.lock().unwrap().pods.range::<String, _>((Included(start.clone()), std::ops::Bound::Unbounded)) {
            if key.starts_with(&start) {
                vec.push(val.as_ref().clone());
            } else {
                break;
            }
        }

        return Ok(vec)
    }

    pub fn Add(&self, podDef: PodDef) -> Result<()> {
        let key = format!("{}/{}/{}", &podDef.tenant, &podDef.namespace, &podDef.name);
        let mut inner = self.lock().unwrap();
        if inner.pods.contains_key(&key) {
            return Err(Error::Exist(format!("PodMgr::add {}", key)));
        }

        inner.pods.insert(key, podDef.into());
        
        return Ok(())
    } 

    pub fn Update(&self, podDef: PodDef) -> Result<()> {
        let key = format!("{}/{}/{}", &podDef.tenant, &podDef.namespace, &podDef.name);
        let mut inner = self.lock().unwrap();
        if !inner.pods.contains_key(&key) {
            return Err(Error::NotExist(format!("PodMgr::update {}", key)));
        }

        inner.pods.insert(key, podDef.into());
        
        return Ok(())
    }

    pub fn Remove(&self, podDef: PodDef) -> Result<()> {
        let key = format!("{}/{}/{}", &podDef.tenant, &podDef.namespace, &podDef.name);
        let mut inner = self.lock().unwrap();
        if !inner.pods.contains_key(&key) {
            return Err(Error::NotExist(format!("PodMgr::Remove {}", key)));
        }

        inner.pods.remove(&key);
        
        return Ok(())
    } 
}
