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

use core::ops::Deref;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use crate::common::*;
use crate::metastore::data_obj::*;

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct FuncPackageId {
    pub namespace: String,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct FuncPackageSpec {
    #[serde(default)]
    pub tenant: String,
    pub namespace: String,
    pub name: String,
    pub revision: i64,

    pub image: String,
    pub commands: Vec<String>,
    pub envs: Vec<(String, String)>,

    #[serde(default, rename = "keepalive_policy")]
    pub keepalivePolicy: KeepAlivePolicy,
}

impl FuncPackageSpec {
    pub const KEY: &'static str = "funcpackage";

    pub fn FromDataObject(obj: DataObject) -> Result<Self> {
        let spec = match serde_json::from_str::<Self>(&obj.data) {
            Err(e) => {
                return Err(Error::CommonError(format!(
                    "FuncPackageSpec::FromDataObject {:?}",
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
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            data: serde_json::to_string_pretty(&self).unwrap(),
            ..Default::default()
        };

        return inner.into();
    }

    pub fn ToJson(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }

    pub fn Key(&self) -> String {
        return format!("{}/{}/{}", &self.tenant, &self.namespace, &self.name);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KeepAlivePolicy {
    #[serde(default, rename = "warm_cnt")]
    pub warmCnt: usize, // how many instance keepalive when idle
    #[serde(rename = "keepalive_time")]
    pub keepaliveTime: u64, // keepalive for how many second
}

impl Default for KeepAlivePolicy {
    fn default() -> Self {
        return Self {
            warmCnt: 0,
            keepaliveTime: 10, //keepalive for 10 second when idle
        };
    }
}

#[derive(Debug, Default)]
pub struct FuncPackageInner {
    pub spec: FuncPackageSpec,
}

#[derive(Debug, Default, Clone)]
pub struct FuncPackage(Arc<FuncPackageInner>);

impl Deref for FuncPackage {
    type Target = Arc<FuncPackageInner>;

    fn deref(&self) -> &Arc<FuncPackageInner> {
        &self.0
    }
}

impl FuncPackage {
    pub fn New(spec: FuncPackageSpec) -> Self {
        let inner = FuncPackageInner { spec: spec };

        return Self(Arc::new(inner));
    }
}

#[derive(Debug, Default)]
pub struct FuncPackageMgrInner {
    pub funcPackages: BTreeMap<String, FuncPackage>,
}

#[derive(Debug, Default, Clone)]
pub struct FuncPackageMgr(Arc<Mutex<FuncPackageMgrInner>>);

impl Deref for FuncPackageMgr {
    type Target = Arc<Mutex<FuncPackageMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncPackageMgrInner>> {
        &self.0
    }
}

impl FuncPackageMgr {
    pub fn ContainsFuncPackage(&self, key: &str) -> bool {
        return self.lock().unwrap().funcPackages.contains_key(key);
    }

    pub fn GetFuncPackage(&self, key: &str) -> Result<FuncPackage> {
        match self.lock().unwrap().funcPackages.get(key) {
            None => return Err(Error::NotExist(format!("GetFuncPackage {}", key))),
            Some(p) => return Ok(p.clone()),
        }
    }

    pub fn GetFuncPackages(&self, tenant: &str, namespace: &str) -> Result<Vec<String>> {
        use std::ops::Bound::*;
        let start = format!("{}/{}/", tenant, namespace);
        let mut vec = Vec::new();
        for (key, _) in self
            .lock()
            .unwrap()
            .funcPackages
            .range::<String, _>((Included(start.clone()), Unbounded))
        {
            if key.starts_with(&start) {
                vec.push(key.clone());
            } else {
                break;
            }
        }

        return Ok(vec);
    }

    pub fn Add(&self, spec: FuncPackageSpec) -> Result<()> {
        let key = spec.Key();
        let mut inner = self.lock().unwrap();
        if inner.funcPackages.contains_key(&key) {
            return Err(Error::Exist(format!("FuncPackageMgr::add {}", key)));
        }

        let package = FuncPackage::New(spec);
        inner.funcPackages.insert(key, package);

        return Ok(());
    }

    pub fn Update(&self, spec: FuncPackageSpec) -> Result<()> {
        let key = spec.Key();
        let mut inner = self.lock().unwrap();
        if !inner.funcPackages.contains_key(&key) {
            return Err(Error::NotExist(format!("FuncPackageMgr::Update {}", key)));
        }

        let package = FuncPackage::New(spec);
        inner.funcPackages.insert(key, package);

        // todo: clean all the package instance

        return Ok(());
    }

    pub fn Remove(&self, spec: FuncPackageSpec) -> Result<()> {
        let key = spec.Key();
        let mut inner = self.lock().unwrap();
        if inner.funcPackages.contains_key(&key) {
            return Err(Error::NotExist(format!("FuncPackageMgr::Remove {}", key)));
        }

        inner.funcPackages.remove(&key);

        // todo: clean all the package instance

        return Ok(());
    }
}
