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

use std::{
    collections::BTreeMap,
    ops::Deref,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use crate::common::*;
use crate::metastore::data_obj::*;

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct NamespaceSpec {
    #[serde(default)]
    pub tenant: String,
    pub namespace: String,
    pub revision: i64,
    pub disable: bool,
}

impl NamespaceSpec {
    pub const KEY: &'static str = "namespace_info";

    pub fn FromDataObject(obj: DataObject) -> Result<Self> {
        let spec = match serde_json::from_str::<Self>(&obj.data) {
            Err(e) => {
                return Err(Error::CommonError(format!(
                    "NamespaceSpec::FromDataObject {:?}",
                    e
                )))
            }
            Ok(s) => s,
        };
        return Ok(spec);
    }

    pub fn DataObject(&self) -> DataObject {
        let inner = DataObjectInner {
            kind: Self::KEY.to_owned(),
            tenant: self.tenant.clone(),
            namespace: "system".to_owned(),
            name: self.namespace.to_owned(),
            data: serde_json::to_string_pretty(&self).unwrap(),
            ..Default::default()
        };

        return inner.into();
    }

    pub fn Key(&self) -> String {
        return format!("{}/{}", &self.tenant, &self.namespace);
    }
}

#[derive(Debug, Default)]
pub struct NamespaceMgrInner {
    pub namespaces: BTreeMap<String, NamespaceSpec>,
}

#[derive(Debug, Default, Clone)]
pub struct NamespaceMgr(Arc<Mutex<NamespaceMgrInner>>);

impl Deref for NamespaceMgr {
    type Target = Arc<Mutex<NamespaceMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<NamespaceMgrInner>> {
        &self.0
    }
}

impl NamespaceMgr {
    pub fn Contains(&self, tenant: &str, namespace: &str) -> bool {
        let podNamespace = format!("{}/{}", tenant, namespace);
        return self.lock().unwrap().namespaces.contains_key(&podNamespace);
    }

    pub fn Add(&self, spec: NamespaceSpec) -> Result<()> {
        let mut inner = self.lock().unwrap();

        let key = spec.Key();

        if inner.namespaces.contains_key(&key) {
            return Err(Error::Exist(format!("NamespaceMgr::AddNamespace {}", &key)));
        };

        inner.namespaces.insert(key, spec);

        return Ok(());
    }

    pub fn Update(&self, spec: NamespaceSpec) -> Result<()> {
        let mut inner = self.lock().unwrap();

        let key = spec.Key();

        if !inner.namespaces.contains_key(&key) {
            return Err(Error::NotExist(format!(
                "NamespaceMgr::UpdateNamespace {}",
                &key
            )));
        };

        inner.namespaces.insert(key, spec);

        return Ok(());
    }
}
