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
use std::sync::Arc;
use tokio::sync::Notify;

use qshare::metastore::data_obj::*;
use qshare::metastore::informer::EventHandler;
use qshare::metastore::informer::Informer;
use qshare::metastore::informer_factory::InformerFactory;
use qshare::metastore::selection_predicate::ListOption;
use qshare::metastore::store::ThreadSafeStore;
use qshare::node::PodDef;

use qshare::common::*;
use qshare::etcd::etcd_store::EtcdStore;
use qshare::obj_mgr::func_mgr::*;
use qshare::obj_mgr::namespace_mgr::*;
use qshare::obj_mgr::pod_mgr::PodMgr;

#[derive(Debug)]
pub struct ObjRepoInner {
    pub funcPackageMgr: FuncPackageMgr,
    pub podMgr: PodMgr,

    pub factory: InformerFactory,
    pub funcPackageInformer: Informer,
}

#[derive(Debug, Clone)]
pub struct ObjRepo(Arc<ObjRepoInner>);

impl Deref for ObjRepo {
    type Target = Arc<ObjRepoInner>;

    fn deref(&self) -> &Arc<ObjRepoInner> {
        &self.0
    }
}

impl ObjRepo {
    pub async fn New(addresses: Vec<String>) -> Result<Self> {
        let factory = InformerFactory::New(addresses, "", "").await?;

        // funcpackageSpec
        factory
            .AddInformer(FuncPackageSpec::KEY, &ListOption::default())
            .await?;
        let funcPackageInformer = factory.GetInformer(FuncPackageSpec::KEY).await?;

        // pod
        factory.AddInformer("pod", &ListOption::default()).await?;
        let podInformer = factory.GetInformer("pod").await?;

        let inner = ObjRepoInner {
            funcPackageMgr: FuncPackageMgr::default(),
            podMgr: PodMgr::default(),
            factory: factory,
            funcPackageInformer: funcPackageInformer.clone(),
        };

        let mgr = Self(Arc::new(inner));
        let _id2 = funcPackageInformer
            .AddEventHandler(Arc::new(mgr.clone()))
            .await?;
        let _id3 = podInformer.AddEventHandler(Arc::new(mgr.clone())).await?;
        let notify = Arc::new(Notify::new());
        tokio::spawn(async move {
            // todo: handle statesvc crash
            tokio::select! {
                _ = funcPackageInformer.Process(notify.clone()) => {

                }
                _ = podInformer.Process(notify.clone()) => {

                }
            }
        });
        return Ok(mgr);
    }

    pub fn ContainsFuncPackage(&self, tenant: &str, namespace: &str, name: &str) -> Result<bool> {
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

        return Ok(());
    }

    pub fn UpdateFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        self.funcPackageMgr.Update(spec)?;

        return Ok(());
    }

    pub fn RemoveFuncPackage(&self, spec: FuncPackageSpec) -> Result<()> {
        self.funcPackageMgr.Remove(spec)?;
        return Ok(());
    }

    pub fn GetFuncPods(
        &self,
        tenant: &str,
        namespace: &str,
        funcName: &str,
    ) -> Result<Vec<PodDef>> {
        return self.podMgr.GetFuncPods(tenant, namespace, funcName);
    }

    pub fn ProcessDeltaEvent(&self, event: &DeltaEvent) -> Result<()> {
        let obj = event.obj.clone();
        match &event.type_ {
            EventType::Added => match &obj.kind as &str {
                FuncPackageSpec::KEY => {
                    let spec = FuncPackageSpec::FromDataObject(obj)?;
                    self.AddFuncPackage(spec)?;
                }
                PodDef::KEY => {
                    let podDef = PodDef::FromDataObject(obj)?;
                    self.podMgr.Add(podDef)?;
                }
                _ => {
                    return Err(Error::CommonError(format!(
                        "NamespaceMgr::ProcessDeltaEvent {:?}",
                        event
                    )));
                }
            },
            EventType::Modified => match &obj.kind as &str {
                FuncPackageSpec::KEY => {
                    let spec = FuncPackageSpec::FromDataObject(obj)?;
                    self.UpdateFuncPackage(spec)?;
                }
                PodDef::KEY => {
                    let podDef = PodDef::FromDataObject(obj)?;
                    self.podMgr.Update(podDef)?;
                }
                _ => {
                    return Err(Error::CommonError(format!(
                        "NamespaceMgr::ProcessDeltaEvent {:?}",
                        event
                    )));
                }
            },
            EventType::Deleted => match &obj.kind as &str {
                FuncPackageSpec::KEY => {
                    let spec = FuncPackageSpec::FromDataObject(obj)?;
                    self.RemoveFuncPackage(spec)?;
                }
                PodDef::KEY => {
                    let podDef = PodDef::FromDataObject(obj)?;
                    self.podMgr.Remove(podDef)?;
                }
                _ => {
                    return Err(Error::CommonError(format!(
                        "NamespaceMgr::ProcessDeltaEvent {:?}",
                        event
                    )));
                }
            },
            _o => {
                return Err(Error::CommonError(format!(
                    "NamespaceMgr::ProcessDeltaEvent {:?}",
                    event
                )));
            }
        }

        return Ok(());
    }
}

impl EventHandler for ObjRepo {
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

        return Ok(Self { store: store });
    }

    pub async fn CreateNamespace(&self, namespace: &NamespaceSpec) -> Result<()> {
        let namespaceObj = namespace.DataObject();
        self.store.Create(&namespaceObj, 0).await?;
        return Ok(());
    }

    pub async fn UpdateNamespace(&self, namespace: &NamespaceSpec) -> Result<()> {
        let namespaceObj = namespace.DataObject();
        self.store.Update(namespace.revision, &namespaceObj).await?;
        return Ok(());
    }

    pub async fn DisasbleNamespace(&self, namespace: &NamespaceSpec) -> Result<()> {
        let namespace = NamespaceSpec {
            tenant: namespace.tenant.clone(),
            namespace: namespace.namespace.clone(),
            revision: namespace.revision,
            disable: true,
        };

        let namespaceObj = namespace.DataObject();
        self.store.Update(namespace.revision, &namespaceObj).await?;
        return Ok(());
    }

    pub async fn CreateFuncPackage(&self, funcPackage: &FuncPackageSpec) -> Result<()> {
        let obj = funcPackage.DataObject();
        self.store.Create(&obj, 0).await?;
        return Ok(());
    }

    pub async fn UpdateFuncPackage(&self, funcPackage: &FuncPackageSpec) -> Result<()> {
        let obj = funcPackage.DataObject();
        self.store.Update(funcPackage.revision, &obj).await?;
        return Ok(());
    }

    pub async fn DropFuncPackage(&self, namespace: &str, name: &str, revision: i64) -> Result<()> {
        let key = format!("{}/{}/{}", FuncPackageSpec::KEY, namespace, name);
        self.store.Delete(&key, revision).await?;
        return Ok(());
    }
}
