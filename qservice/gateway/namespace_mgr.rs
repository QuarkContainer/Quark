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
use std::sync::Arc;
use std::sync::Mutex;
use core::ops::Deref;

use qshare::metastore::data_obj::*;
use qshare::metastore::informer::EventHandler;
use qshare::metastore::informer::Informer;
use qshare::metastore::informer_factory::InformerFactory;
use qshare::metastore::selection_predicate::ListOption;
use qshare::metastore::store::ThreadSafeStore;
use serde::{Deserialize, Serialize};

use qshare::etcd::etcd_store::EtcdStore;
use qshare::common::*;
use tokio::sync::Notify;

use crate::func_mgr::{FuncPackageMgr, FuncPackageSpec};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct NamespaceSpec {
    pub namespace: String,
    pub revision: i64,
    pub disable: bool,
}

impl NamespaceSpec {
    pub const KEY: &'static str = "namespace_info";

    pub fn FromDataObject(obj: DataObject) -> Result<Self> {
        let spec = match serde_json::from_str::<Self>(&obj.data) {
            Err(e) => return Err(Error::CommonError(format!("NamespaceSpec::FromDataObject {:?}", e))),
            Ok(s) => s
        };
        return Ok(spec);
    }

    pub fn DataObject(&self) -> DataObject {
        let inner = DataObjectInner {
            kind: Self::KEY.to_owned(),
            namespace: "system".to_owned(),
            name: self.namespace.to_owned(),
            data: serde_json::to_string_pretty(&self).unwrap(),
            ..Default::default()
        };

        return inner.into();
    }

    pub fn Key(&self) -> String {
        return self.namespace.clone();
    }
}

#[derive(Debug)]
pub struct NamespaceMgrInner {
    pub funcPackageMgrs: BTreeMap<String, FuncPackageMgr>,
    pub namespaces: BTreeMap<String, NamespaceSpec>,
    
    pub factory: InformerFactory,
    pub namespaceInformer: Informer,
    pub funcPackageInformer: Informer,
}

#[derive(Debug, Clone)]
pub struct NamespaceMgr(Arc<Mutex<NamespaceMgrInner>>);

impl Deref for NamespaceMgr {
    type Target = Arc<Mutex<NamespaceMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<NamespaceMgrInner>> {
        &self.0
    }
}

impl NamespaceMgr {
    pub async fn New(addresses: Vec<String>) -> Result<Self> {
        let factory = InformerFactory::New(addresses, "").await?;
        factory.AddInformer(NamespaceSpec::KEY, &ListOption::default()).await?;
        let namespaceInformer = factory.GetInformer(NamespaceSpec::KEY).await?;
        factory.AddInformer(FuncPackageSpec::KEY, &ListOption::default()).await?;
        let funcPackageInformer = factory.GetInformer(FuncPackageSpec::KEY).await?;
        
        let inner = NamespaceMgrInner {
            funcPackageMgrs: BTreeMap::new(),
            namespaces: BTreeMap::new(),
            factory: factory,
            namespaceInformer: namespaceInformer.clone(),
            funcPackageInformer: funcPackageInformer.clone()
        };

        let mgr = Self(Arc::new(Mutex::new(inner)));
        let _id1 = namespaceInformer.AddEventHandler(Arc::new(mgr.clone())).await?;
        let _id2 = funcPackageInformer.AddEventHandler(Arc::new(mgr.clone())).await?;
        let notify = Arc::new(Notify::new());
        tokio::spawn(async move {
            // todo: handle statesvc crash
            tokio::select! {
                _ = namespaceInformer.Process(notify.clone()) => {

                }
                _ = funcPackageInformer.Process(notify.clone()) => {

                }
            }
        });
        return Ok(mgr)
    }

    pub fn AddNamespace(&self, spec: NamespaceSpec) -> Result<()> {
        let mut inner = self.lock().unwrap();

        let key = spec.Key();

        if inner.namespaces.contains_key(&key) {
            return Err(Error::Exist(format!("NamespaceMgr::AddNamespace {}", &key)));
        };

        inner.namespaces.insert(key, spec);

        return Ok(())
    }

    pub fn UpdateNamespace(&self, spec: NamespaceSpec) -> Result<()> {
        let mut inner = self.lock().unwrap();

        let key = spec.Key();

        if !inner.namespaces.contains_key(&key) {
            return Err(Error::NotExist(format!("NamespaceMgr::UpdateNamespace {}", &key)));
        };

        inner.namespaces.insert(key, spec);

        return Ok(())
    }

    pub fn GetFuncPackageMgr(&self, namespace: &str) -> Result<FuncPackageMgr> {
        let mut inner = self.lock().unwrap();

        let mgr = match inner.funcPackageMgrs.get(namespace) {
            None => {
                let mgr = FuncPackageMgr::default();
                inner.funcPackageMgrs.insert(namespace.to_owned(), mgr.clone());
                mgr
            }
            Some(mgr) => {
                mgr.clone()
            }
        };

        return Ok(mgr);
    }

    pub fn AddFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        let mgr = self.GetFuncPackageMgr(&spec.namespace)?;

        mgr.Add(spec)?;

        return Ok(())
    }

    pub fn UpdateFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        let mgr = self.GetFuncPackageMgr(&spec.namespace)?;

        mgr.Update(spec)?;

        return Ok(())
    }

    pub fn RemoveFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        let mgr = self.GetFuncPackageMgr(&spec.namespace)?;

        mgr.Remove(spec)?;
        return Ok(())
    }

    pub fn ProcessDeltaEvent(&self, event: &DeltaEvent) -> Result<()> {
        let obj = event.obj.clone();
        match &event.type_ {
            EventType::Added => {
                if &obj.kind == FuncPackageSpec::KEY {
                    let spec = FuncPackageSpec::FromDataObject(obj)?;
                    self.AddFuncPackage(spec)?;
                } else if &obj.kind == NamespaceSpec::KEY {
                    let spec: NamespaceSpec = NamespaceSpec::FromDataObject(obj)?;
                    self.AddNamespace(spec)?;
                } else {
                    return Err(Error::CommonError(format!("NamespaceMgr::ProcessDeltaEvent {:?}", event)));
                }
            }
            EventType::Modified => {
                if &obj.kind == FuncPackageSpec::KEY {
                    let spec = FuncPackageSpec::FromDataObject(obj)?;
                    self.UpdateFuncPackage(spec)?;
                } else if &obj.kind == NamespaceSpec::KEY {
                    let spec: NamespaceSpec = NamespaceSpec::FromDataObject(obj)?;
                    self.UpdateNamespace(spec)?;
                } else {
                    return Err(Error::CommonError(format!("NamespaceMgr::ProcessDeltaEvent {:?}", event)));
                }
            }
            EventType::Deleted => {
                if &obj.kind == FuncPackageSpec::KEY {
                    let spec = FuncPackageSpec::FromDataObject(obj)?;
                    self.RemoveFuncPackage(spec)?;
                } else {
                    return Err(Error::CommonError(format!("NamespaceMgr::ProcessDeltaEvent {:?}", event)));
                }
            }
            _o => {
                return Err(Error::CommonError(format!("NamespaceMgr::ProcessDeltaEvent {:?}", event)));
            } 
        }

        return Ok(())
    }
}

impl EventHandler for NamespaceMgr {
    fn handle(&self, _store: &ThreadSafeStore, event: &DeltaEvent) {
        self.ProcessDeltaEvent(event).unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct NamespaceStore {
    pub store: EtcdStore,
}

impl NamespaceStore {
    pub async fn New(endpoints: &[String]) -> Result<Self> {
        let store = EtcdStore::NewWithEndpoints(endpoints, false).await?;

        return Ok(Self {
            store: store,
        });
    }

    pub async fn CreateNamespace(&self, namespace: &NamespaceSpec) -> Result<()> {
        let namespaceObj = namespace.DataObject();
        self.store.Create(&namespaceObj, 0).await?;
        return Ok(())
    }

    pub async fn UpdateNamespace(&self, namespace: &NamespaceSpec) -> Result<()> {
        let namespaceObj = namespace.DataObject();
        self.store.Update(namespace.revision,&namespaceObj).await?;
        return Ok(())
    }

    pub async fn DisasbleNamespace(&self, namespace: &NamespaceSpec) -> Result<()> {
        let namespace = NamespaceSpec {
            namespace: namespace.namespace.clone(),
            revision: namespace.revision,
            disable: true,
        };

        let namespaceObj = namespace.DataObject();
        self.store.Update(namespace.revision,&namespaceObj).await?;
        return Ok(())
    }

    pub async fn CreateFuncPackage(&self, funcPackage: &FuncPackageSpec) -> Result<()> {
        let obj = funcPackage.DataObject();
        self.store.Create(&obj, 0).await?;
        return Ok(())
    }

    pub async fn UpdateFuncPackage(&self, funcPackage: &FuncPackageSpec) -> Result<()> {
        let obj = funcPackage.DataObject();
        self.store.Update(funcPackage.revision,&obj).await?;
        return Ok(())
    }

    pub async fn DropFuncPackage(&self, funcPackage: &FuncPackageSpec) -> Result<()> {
        let obj = funcPackage.DataObject();
        self.store.Delete(&obj.Key(), obj.revision).await?;
        return Ok(())
    }
}