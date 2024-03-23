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

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::Notify;

use qshare::common::*;
use qshare::etcd::etcd_store::EtcdStore;
use qshare::metastore::data_obj::DataObject;
use qshare::metastore::data_obj::DataObjectInner;
use qshare::types::NodeInfo;

pub struct NodeRegister {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub etcdAddresses: Vec<String>,
    pub nodeName: String,
    pub nodeIp: String,
    pub podMgrPort: u16,
    pub tsotSvcPort: u16,
    pub stateSvcPort: u16,
    pub cidr: String,
}

impl NodeRegister {
    pub fn New(
        addresses: &[String],
        nodeName: &str,
        nodeIp: &str,
        podMgrPort: u16,
        tsotSvcPort: u16,
        stateSvcPort: u16,
        cidr: &str,
    ) -> Self {
        let mut etcdAddresses = Vec::new();
        for addr in addresses {
            etcdAddresses.push(addr.clone());
        }
        return Self {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            etcdAddresses: etcdAddresses,
            nodeName: nodeName.to_owned(),
            nodeIp: nodeIp.to_owned(),
            podMgrPort: podMgrPort,
            tsotSvcPort: tsotSvcPort,
            stateSvcPort: stateSvcPort,
            cidr: cidr.to_owned(),
        };
    }

    pub fn Close(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }

    pub fn NodeInfo(&self) -> NodeInfo {
        return NodeInfo {
            nodeName: self.nodeName.clone(),
            nodeIp: self.nodeIp.clone(),
            podMgrPort: self.podMgrPort,
            tsotSvcPort: self.tsotSvcPort,
            stateSvcPort: self.stateSvcPort,
            cidr: self.cidr.clone(),
        };
    }

    pub fn DataObject(&self) -> DataObject {
        let inner = DataObjectInner {
            kind: Self::KEY.to_owned(),
            namespace: "system".to_owned(),
            name: self.nodeName.to_owned(),
            data: serde_json::to_string_pretty(&self.NodeInfo()).unwrap(),
            ..Default::default()
        };

        return inner.into();
    }

    pub const LEASE_TTL: i64 = 2; // seconds
    pub const KEY: &'static str = "node_info";

    pub async fn Process(&self) -> Result<()> {
        let store = EtcdStore::NewWithEndpoints(&self.etcdAddresses, false).await?;

        let leaseId = store.LeaseGrant(Self::LEASE_TTL).await?;
        store.Create(&self.DataObject(), leaseId).await?;

        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    return Ok(());
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                    // keepalive for each 500 ms
                    store.LeaseKeepalive(leaseId).await?;
                }
            }
        }
    }
}
