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

use lazy_static::lazy_static;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::RwLock as TRwLock;

use qobjs::cacher::*;
use crate::etcd_store::EtcdStore;
use qobjs::common::*;

lazy_static! {
    pub static ref SVC_DIR: SvcDir = SvcDir::default();
    pub static ref CACHE_OBJ_TYPES: Vec<&'static str> = vec!["pod", "podset",];
}

#[derive(Debug, Default)]
pub struct SvcDir(Arc<TRwLock<SvcDirInner>>);

impl Deref for SvcDir {
    type Target = Arc<TRwLock<SvcDirInner>>;

    fn deref(&self) -> &Arc<TRwLock<SvcDirInner>> {
        &self.0
    }
}

impl SvcDir {
    pub async fn GetCacher(&self, objType: &str) -> Option<Cacher> {
        return match self.read().await.map.get(objType) {
            None => None,
            Some(c) => Some(c.clone()),
        };
    }
}

#[derive(Debug, Default)]
pub struct SvcDirInner {
    pub map: BTreeMap<String, Cacher>,
}

impl SvcDirInner {
    pub async fn Init(&mut self, addr: &str) -> Result<()> {
        let store = EtcdStore::New(addr, true).await?;
        for i in 0..CACHE_OBJ_TYPES.len() {
            let t = CACHE_OBJ_TYPES[i];
            let c = Cacher::New(Arc::new(store.clone()), t, 0).await?;
            self.map.insert(t.to_string(), c);
        }
        defer!();

        return Ok(());
    }
}
