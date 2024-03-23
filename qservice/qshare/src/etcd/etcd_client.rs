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

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use etcd_client::Client;
use tokio::sync::Mutex as TMutex;

use crate::common::*;

#[derive(Clone)]
pub struct EtcdClient {
    pub client: Arc<TMutex<Client>>,
}

impl fmt::Debug for EtcdClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EtcdClient").finish()
    }
}

impl Deref for EtcdClient {
    type Target = Arc<TMutex<Client>>;

    fn deref(&self) -> &Arc<TMutex<Client>> {
        &self.client
    }
}

impl EtcdClient {
    pub async fn NewWithEndpoints(endpoints: &[String]) -> Result<Self> {
        let client = Client::connect(endpoints, None).await?;
        return Ok(Self {
            client: Arc::new(TMutex::new(client)),
        });
    }

    pub fn New(client: Client) -> Self {
        return Self {
            client: Arc::new(TMutex::new(client)),
        };
    }

    pub async fn LeaseGrant(&self, ttl: i64) -> Result<i64> {
        let resp = self
            .client
            .lock()
            .await
            .lease_client()
            .grant(ttl, None)
            .await?;
        return Ok(resp.id());
    }

    pub async fn LeaseRevoke(&self, leaseId: i64) -> Result<()> {
        let _resp = self
            .client
            .lock()
            .await
            .lease_client()
            .revoke(leaseId)
            .await?;
        return Ok(());
    }

    pub async fn LeaseKeepalive(&self, leaseId: i64) -> Result<()> {
        let _resp = self
            .client
            .lock()
            .await
            .lease_client()
            .keep_alive(leaseId)
            .await?;
        return Ok(());
    }
}
