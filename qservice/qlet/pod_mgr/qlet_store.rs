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

use super::qnode::QuarkNode;
use super::qpod::QuarkPod;

#[derive(Debug)]
pub struct QletStore {
    pub map: BTreeMap<String, CacheStore>,
    pub channelRev: ChannelRev,
    pub nodeStore: CacheStore,
    pub podStore: CacheStore,
}

impl QletStore {
    pub async fn New() -> Result<Self> {
        let channelRev = ChannelRev::default();
        let mut map = BTreeMap::new();
        let nodeStore = CacheStore::New(None, "node", 0, &channelRev).await?;
        let podStore = CacheStore::New(None, "pod", 0, &channelRev).await?;
        map.insert("node".to_owned(), nodeStore.clone());
        map.insert("pod".to_owned(), nodeStore.clone());

        return Ok(Self {
            map: map,
            channelRev: channelRev,
            nodeStore: nodeStore,
            podStore: podStore,
        })
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

        let channelRev = self.channelRev.Next();
        node.resource_version = format!("{}", channelRev);

        let inner = DataObjectInner {
            kind: "node".to_owned(),
            namespace: node.namespace.clone(),
            name: node.name.clone(),
            lables: labels.into(),
            annotations: annotations.into(),
            channelRev: channelRev,
            revision: channelRev,
            data: node.ToString(),
        };

        return inner.into()
    }

    pub fn CreateNode(&self, node: &QuarkNode) -> Result<()> {
        let obj = self.ToNodeDataObject(node);
        self.nodeStore.Add(&obj)?;
        return Ok(())
    }

    pub fn UpdateNode(&self, node: &QuarkNode) -> Result<()> {
        let obj = self.ToNodeDataObject(node);
        self.nodeStore.Update(&obj)?;
        return Ok(())
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

        let channelRev = self.channelRev.Next();
        pod.resource_version = format!("{}", channelRev);

        let inner = DataObjectInner {
            kind: "node".to_owned(),
            namespace: pod.namespace.clone(),
            name: pod.name.clone(),
            lables: labels.into(),
            annotations: annotations.into(),
            channelRev: channelRev,
            revision: channelRev,
            data: pod.ToString(),
        };

        return inner.into()
    }

    pub fn CreatePod(&self, pod: &QuarkPod) -> Result<()> {
        let obj = self.ToPodDataObject(pod);
        self.podStore.Add(&obj)?;
        return Ok(())
    }

    pub fn UpdatePod(&self, pod: &QuarkPod) -> Result<()> {
        let obj = self.ToPodDataObject(pod);
        self.podStore.Update(&obj)?;
        return Ok(())
    }

    pub fn RemovePod(&self, pod: &QuarkPod) -> Result<()> {
        let obj = self.ToPodDataObject(pod);
        self.podStore.Remove(&obj)?;
        return Ok(())
    }
}

