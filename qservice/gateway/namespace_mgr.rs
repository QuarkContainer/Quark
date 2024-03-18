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
use qshare::node::PodDef;
use serde::{Deserialize, Serialize};

use qshare::etcd::etcd_store::EtcdStore;
use qshare::common::*;
use tokio::sync::Notify;

use crate::func_mgr::*;
use crate::pod_mgr::PodMgr;

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
            Err(e) => return Err(Error::CommonError(format!("NamespaceSpec::FromDataObject {:?}", e))),
            Ok(s) => s
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

#[derive(Debug)]
pub struct NamespaceMgrInner {
    pub funcPackageMgr: FuncPackageMgr,
    pub namespaces: Mutex<BTreeMap<String, NamespaceSpec>>,
    pub podMgr: PodMgr,
    
    pub factory: InformerFactory,
    pub namespaceInformer: Informer,
    pub funcPackageInformer: Informer,
}

#[derive(Debug, Clone)]
pub struct NamespaceMgr(Arc<NamespaceMgrInner>);

impl Deref for NamespaceMgr {
    type Target = Arc<NamespaceMgrInner>;

    fn deref(&self) -> &Arc<NamespaceMgrInner> {
        &self.0
    }
}

impl NamespaceMgr {
    pub async fn New(addresses: Vec<String>) -> Result<Self> {
        let factory = InformerFactory::New(addresses, "", "").await?;

        // namespaceSpec
        factory.AddInformer(NamespaceSpec::KEY, &ListOption::default()).await?;
        let namespaceInformer = factory.GetInformer(NamespaceSpec::KEY).await?;

        // funcpackageSpec
        factory.AddInformer(FuncPackageSpec::KEY, &ListOption::default()).await?;
        let funcPackageInformer = factory.GetInformer(FuncPackageSpec::KEY).await?;

        // pod
        factory.AddInformer("pod", &ListOption::default()).await?;
        let podInformer = factory.GetInformer("pod").await?;
        
        let inner = NamespaceMgrInner {
            funcPackageMgr: FuncPackageMgr::default(),
            namespaces: Mutex::new(BTreeMap::new()),
            podMgr: PodMgr::default(),
            factory: factory,
            namespaceInformer: namespaceInformer.clone(),
            funcPackageInformer: funcPackageInformer.clone()
        };

        let mgr = Self(Arc::new(inner));
        let _id1 = namespaceInformer.AddEventHandler(Arc::new(mgr.clone())).await?;
        let _id2 = funcPackageInformer.AddEventHandler(Arc::new(mgr.clone())).await?;
        let _id3 = podInformer.AddEventHandler(Arc::new(mgr.clone())).await?;
        let notify = Arc::new(Notify::new());
        tokio::spawn(async move {
            // todo: handle statesvc crash
            tokio::select! {
                _ = namespaceInformer.Process(notify.clone()) => {

                }
                _ = funcPackageInformer.Process(notify.clone()) => {

                }
                _ = podInformer.Process(notify.clone()) => {

                }
            }
        });
        return Ok(mgr)
    }

    pub fn ContainsNamespace(&self, tenant: &str, namespace: &str) -> bool {
        let podNamespace = format!("{}/{}", tenant, namespace);
        return self.namespaces.lock().unwrap().contains_key(&podNamespace)
    }

    pub fn AddNamespace(&self, spec: NamespaceSpec) -> Result<()> {
        let mut inner = self.namespaces.lock().unwrap();

        let key = spec.Key();

        if inner.contains_key(&key) {
            return Err(Error::Exist(format!("NamespaceMgr::AddNamespace {}", &key)));
        };

        inner.insert(key, spec);

        return Ok(())
    }

    pub fn UpdateNamespace(&self, spec: NamespaceSpec) -> Result<()> {
        let mut inner = self.namespaces.lock().unwrap();

        let key = spec.Key();

        if !inner.contains_key(&key) {
            return Err(Error::NotExist(format!("NamespaceMgr::UpdateNamespace {}", &key)));
        };

        inner.insert(key, spec);

        return Ok(())
    }

    pub fn ContainsFuncPackage(&self, tenant: &str, namespace: &str, name: &str) -> Result<bool> {
        if !self.ContainsNamespace(tenant, namespace) {
            return Err(Error::NotExist(format!("ContainersFuncPackage has no namespace {}/{}", tenant, namespace)));
        }
 
        let key = format!("{}/{}/{}", tenant, namespace, name);
        return Ok(self.funcPackageMgr.ContainsFuncPackage(&key));
    }

    pub fn GetFuncPackage(&self, tenant: &str, namespace: &str, name: &str) -> Result<FuncPackage> {
        let key = format!("{}/{}/{}", tenant, namespace, name);
        return self.funcPackageMgr.GetFuncPackage(&key);
    }

    pub fn GetFuncPackages(&self, tenant: &str, namespace: &str) -> Result<Vec<String>> {
        return self.funcPackageMgr.GetFuncPackages(tenant, namespace);
    }

    pub fn AddFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        self.funcPackageMgr.Add(spec)?;

        return Ok(())
    }

    pub fn UpdateFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        self.funcPackageMgr.Update(spec)?;

        return Ok(())
    }

    pub fn RemoveFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        self.funcPackageMgr.Remove(spec)?;
        return Ok(())
    }

    pub fn GetFuncPods(&self, tenant: &str, namespace: &str, funcName: &str) -> Result<Vec<PodDef>> {
        return self.podMgr.GetFuncPods(tenant, namespace, funcName);
    }

    pub fn ProcessDeltaEvent(&self, event: &DeltaEvent) -> Result<()> {
        let obj = event.obj.clone();
        match &event.type_ {
            EventType::Added => {
                match &obj.kind as &str {
                    FuncPackageSpec::KEY => {
                        let spec = FuncPackageSpec::FromDataObject(obj)?;
                        self.AddFuncPackage(spec)?;
                    }
                    NamespaceSpec::KEY => {
                        let spec: NamespaceSpec = NamespaceSpec::FromDataObject(obj)?;
                        self.AddNamespace(spec)?;
                    }
                    PodDef::KEY => {
                        let podDef = PodDef::FromDataObject(obj)?;
                        self.podMgr.Add(podDef)?;
                    }
                    _ => {
                        return Err(Error::CommonError(format!("NamespaceMgr::ProcessDeltaEvent {:?}", event)));
                    }
                }
            }
            EventType::Modified => {
                match &obj.kind as &str {
                    FuncPackageSpec::KEY => {
                        let spec = FuncPackageSpec::FromDataObject(obj)?;
                        self.UpdateFuncPackage(spec)?;
                    }
                    NamespaceSpec::KEY => {
                        let spec: NamespaceSpec = NamespaceSpec::FromDataObject(obj)?;
                        self.UpdateNamespace(spec)?;
                    }
                    PodDef::KEY => {
                        let podDef = PodDef::FromDataObject(obj)?;
                        self.podMgr.Update(podDef)?;
                    }
                    _ => {
                        return Err(Error::CommonError(format!("NamespaceMgr::ProcessDeltaEvent {:?}", event)));
                    }
                }
            }
            EventType::Deleted => {
                match &obj.kind as &str {
                    FuncPackageSpec::KEY => {
                        let spec = FuncPackageSpec::FromDataObject(obj)?;
                        self.RemoveFuncPackage(spec)?;
                    }
                    PodDef::KEY => {
                        let podDef = PodDef::FromDataObject(obj)?;
                        self.podMgr.Remove(podDef)?;
                    }
                    _ => {
                        return Err(Error::CommonError(format!("NamespaceMgr::ProcessDeltaEvent {:?}", event)));
                    }
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
            tenant: namespace.tenant.clone(),
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
        self.store.Update(funcPackage.revision, &obj).await?;
        return Ok(())
    }

    pub async fn DropFuncPackage(&self, namespace: &str, name: &str, revision: i64) -> Result<()> {
        let key = format!("{}/{}/{}", FuncPackageSpec::KEY, namespace, name);
        self.store.Delete(&key, revision).await?;
        return Ok(())
    }
}