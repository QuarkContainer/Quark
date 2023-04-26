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

use std::{collections::BTreeMap, sync::Arc, ops::Deref};

use tokio::sync::{RwLock as TRwLock};

use crate::selection_predicate::ListOption;
use crate::{cacher_client::CacherClient, informer::Informer};
use crate::common::*;

#[derive(Debug)]
pub struct InformerFactoryInner {
    pub client: CacherClient,
    pub namespace: String,
    pub informers: BTreeMap<String, Informer>,
    pub closed: bool,
}

#[derive(Debug, Clone)]
pub struct InformerFactory(Arc<TRwLock<InformerFactoryInner>>);

impl Deref for InformerFactory {
    type Target = Arc<TRwLock<InformerFactoryInner>>;

    fn deref(&self) -> &Arc<TRwLock<InformerFactoryInner>> {
        &self.0
    }
}

impl InformerFactory {
    pub async fn New(addr: &str, namespace: &str) -> Result<Self> {
        let client = CacherClient::New(addr.to_string()).await?;
        let inner = InformerFactoryInner {
            client: client,
            namespace: namespace.to_string(),
            informers: BTreeMap::new(),
            closed: false,
        };

        return Ok(Self(Arc::new(TRwLock::new(inner))))
    }

    pub async fn AddInformer(&self, objType: &str, opts: &ListOption) -> Result<()> {
        let mut inner = self.write().await;
        let informer = Informer::New(&inner.client, objType, &inner.namespace, opts).await?;
        inner.informers.insert(objType.to_string(), informer);
        return Ok(())
    }

    pub async fn RemoveInformer(&self, objType: &str) -> Result<()> {
        let mut inner = self.write().await;
        match inner.informers.remove(objType) {
            None => return Err(Error::NotExist),
            Some(_) => return Ok(())
        }
    }

    pub async fn GetInformer(&self, objType: &str) -> Result<Informer> {
        let inner = self.read().await;
        match inner.informers.get(objType) {
            None => return Err(Error::NotExist),
            Some(i) => return Ok(i.clone())
        }
    }

    pub async fn Closed(&self) -> bool {
        return self.read().await.closed;
    }

    pub async fn Close(&self) -> Result<()> {
        let mut inner = self.write().await;
        for (_, informer) in &inner.informers {
            informer.Close().await?;
        }

        inner.informers.clear();
        inner.closed = true;

        return Ok(())
    }
}