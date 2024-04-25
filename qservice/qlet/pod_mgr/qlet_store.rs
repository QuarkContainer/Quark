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

use std::collections::BTreeMap;

use qshare::common::*;
use qshare::metastore::cache_store::*;
use qshare::metastore::data_obj::{DataObject, DataObjectInner};
use qshare::metastore::svc_dir::SvcDir;

use crate::pod_mgr::QLET_STORE;
use crate::QLET_CONFIG;

use super::qnode::QuarkNode;
use super::qpod::QuarkPod;

#[derive(Debug)]
pub struct QletStore {
    pub svcDir: SvcDir,
    pub nodeStore: CacheStore,
    pub podStore: CacheStore,
}

impl QletStore {
    pub async fn New() -> Result<Self> {
        let svcDir = SvcDir::default();
        let channelRev = svcDir.ChannelRev();
        let nodeStore = CacheStore::New(None, "node", 0, &channelRev).await?;
        let podStore = CacheStore::New(None, "pod", 0, &channelRev).await?;

        svcDir.AddCacher(nodeStore.clone());
        svcDir.AddCacher(podStore.clone());

        return Ok(Self {
            svcDir: svcDir,
            nodeStore: nodeStore,
            podStore: podStore,
        });
    }

    pub fn ToNodeDataObject(&self, node: &QuarkNode) -> DataObject {
        let mut node = node.node.lock().unwrap();
        let mut labels = BTreeMap::new();
        for (k, v) in &node.labels {
            labels.insert(k.clone(), v.clone());
        }

        let mut annotations = BTreeMap::new();
        for (k, v) in &node.annotations {
            annotations.insert(k.clone(), v.clone());
        }

        let channelRev = self.svcDir.ChannelRev().Next();
        node.resource_version = format!("{}", channelRev);

        let inner = DataObjectInner {
            kind: "node".to_owned(),
            tenant: node.tenant.clone(),
            namespace: node.namespace.clone(),
            name: node.name.clone(),
            lables: labels.into(),
            annotations: annotations.into(),
            channelRev: channelRev,
            revision: channelRev,
            data: node.ToString(),
        };

        return inner.into();
    }

    pub fn CreateNode(&self, node: &QuarkNode) -> Result<()> {
        let obj = self.ToNodeDataObject(node);
        self.nodeStore.Add(&obj)?;
        return Ok(());
    }

    pub fn UpdateNode(&self, node: &QuarkNode) -> Result<()> {
        let obj = self.ToNodeDataObject(node);
        self.nodeStore.Update(&obj)?;
        return Ok(());
    }

    pub fn ToPodDataObject(&self, pod: &QuarkPod) -> DataObject {
        let pod = pod.Pod();
        let mut pod = pod.write().unwrap();
        let mut labels = BTreeMap::new();
        for (k, v) in &pod.labels {
            labels.insert(k.clone(), v.clone());
        }

        let mut annotations = BTreeMap::new();
        for (k, v) in &pod.annotations {
            annotations.insert(k.clone(), v.clone());
        }

        let channelRev = self.svcDir.ChannelRev().Next();
        pod.resource_version = format!("{}", channelRev);

        let inner = DataObjectInner {
            kind: "pod".to_owned(),
            tenant: pod.tenant.clone(),
            namespace: pod.namespace.clone(),
            name: pod.name.clone(),
            lables: labels.into(),
            annotations: annotations.into(),
            channelRev: channelRev,
            revision: channelRev,
            data: pod.ToString(),
        };

        return inner.into();
    }

    pub fn CreatePod(&self, pod: &QuarkPod) -> Result<()> {
        let obj = self.ToPodDataObject(pod);
        self.podStore.Add(&obj)?;
        return Ok(());
    }

    pub fn UpdatePod(&self, pod: &QuarkPod) -> Result<()> {
        // error!("UpdatePod ********** {:?}", pod.PodState());

        let obj = self.ToPodDataObject(pod);
        self.podStore.Update(&obj)?;
        return Ok(());
    }

    pub fn RemovePod(&self, pod: &QuarkPod) -> Result<()> {
        let obj = self.ToPodDataObject(pod);
        self.podStore.Remove(&obj)?;
        return Ok(());
    }
}

pub async fn QletStateService() -> Result<()> {
    use qshare::qmeta::q_meta_service_server::QMetaServiceServer;
    use tonic::transport::Server;

    let localStateSvcAddr = format!("127.0.0.1:{}", QLET_CONFIG.stateSvcPort);

    let stateSvcFuture = Server::builder()
        .add_service(QMetaServiceServer::new(
            QLET_STORE.get().unwrap().svcDir.clone(),
        ))
        .serve(localStateSvcAddr.parse().unwrap());

    info!("state service start ...");
    tokio::select! {
        _ = stateSvcFuture => {}
    }

    Ok(())
}
