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
use std::sync::RwLock;

use qshare::metastore::cache_store::CacheStore;
use qshare::common::*;

use qshare::etcd::etcd_store::EtcdStore;
use crate::ETCD_OBJECTS;

#[derive(Debug, Default, Clone)]
pub struct SvcDir(Arc<RwLock<SvcDirInner>>);

impl Deref for SvcDir {
    type Target = Arc<RwLock<SvcDirInner>>;

    fn deref(&self) -> &Arc<RwLock<SvcDirInner>> {
        &self.0
    }
}

impl SvcDir {
    pub fn GetCacher(&self, objType: &str) -> Option<CacheStore> {
        return match self.read().unwrap().map.get(objType) {
            None => None,
            Some(c) => Some(c.clone()),
        };
    }
}

#[derive(Debug, Default)]
pub struct SvcDirInner {
    pub map: BTreeMap<String, CacheStore>,
}

impl SvcDirInner {
    pub async fn EtcdInit(&mut self, etcdAddr: &str) -> Result<()> {
        let store = EtcdStore::New(etcdAddr, true).await?;
        for i in 0..ETCD_OBJECTS.len() {
            let t = ETCD_OBJECTS[i];
            let c = CacheStore::New(Arc::new(store.clone()), t, 0).await?;
            self.map.insert(t.to_string(), c);
        }
        
        return Ok(());
    }
}
