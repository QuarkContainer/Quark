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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use core::ops::Deref;

use k8s_openapi::{api::core::v1 as k8s, apimachinery::pkg::apis::meta::v1::ObjectMeta};
use qobjs::{types::*, selector::Labels};
use qobjs::common::*;
use qobjs::cacher::*;

pub struct NodeMgrCacheInner {
    pub nodes: Cacher,
    pub pods: Cacher,
    pub nodePods: BTreeMap<String, BTreeSet<String>>,
}

pub struct NodeMgrCache(Arc<RwLock<NodeMgrCacheInner>>);

impl Deref for NodeMgrCache {
    type Target = Arc<RwLock<NodeMgrCacheInner>>;

    fn deref(&self) -> &Arc<RwLock<NodeMgrCacheInner>> {
        &self.0
    }
}

impl NodeMgrCache {
    pub fn MetaToDataObject(kind: &str, meta: &ObjectMeta) -> Result<DataObjectInner> {
        let mut labels = BTreeMap::new();
        if let Some(map) = &meta.labels {
            for (k, v) in map {
                labels.insert(k.to_string(), v.to_string());
            }
        }
        
        let mut annotations = BTreeMap::new();
        if let Some(map) = &meta.annotations {
            for (k, v) in map {
                annotations.insert(k.to_string(), v.to_string());
            }
        }

        let resource_version = meta.resource_version.as_deref().unwrap_or("0").to_string();

        let revision = resource_version.parse::<i64>()?;

        let inner = DataObjectInner {
            kind: kind.to_string(),
            namespace: meta.namespace.as_deref().unwrap_or("").to_string(),
            name: meta.name.as_deref().unwrap_or("").to_string(),
            lables: Labels::NewFromMap(labels),
            annotations: Labels::NewFromMap(annotations),
            reversion: revision,
            data: String::new(),
        };

        return Ok(inner);
    }

    pub fn PodToDataObject(pod: &k8s::Pod) -> Result<DataObject> {
        let mut inner = Self::MetaToDataObject("pod", &pod.metadata)?;
        inner.data = serde_json::to_string(pod)?;
        let dataObj = DataObject(Arc::new(inner));
        return Ok(dataObj)
    }

    pub fn NodeToDataObject(node: &k8s::Node) -> Result<DataObject> {
        let mut inner = Self::MetaToDataObject("node", &node.metadata)?;
        inner.data = serde_json::to_string(node)?;
        let dataObj = DataObject(Arc::new(inner));
        return Ok(dataObj)
    }

    /*pub fn AddPod(&self, pod: &k8s::Pod) -> Result<()> {
        let obj = Self::PodToDataObject(pod)?;
        let mut inner = self.write().unwrap();
        inner.pods.insert(obj.Key(), obj);
        return Ok(())
    }*/

    //pub fn 
}