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

use qobjs::cacher_client::CacherClient;
use qobjs::common::*;
use qobjs::selection_predicate::ListOption;
use qobjs::system_types::FuncPackage;
use qobjs::zip::ZipMgr;
use qobjs::types::*;

pub struct PackageMgr {
    pub client: CacherClient,
}

impl PackageMgr {
    pub async fn New(qmetaSvcAddr: &str) -> Result<Self> {
        let client = CacherClient::New(qmetaSvcAddr.to_owned()).await?;
        return Ok(Self {
            client: client
        });
    }

    pub async fn ReadObject(&self, namespace: &str, name: &str) -> Result<Vec<u8>> {
        let data = self.client.ReadObject(namespace, name).await?;
        return Ok(data)
    }

    pub async fn ListPackages(&self, namespace: &str) -> Result<Vec<FuncPackage>> {
        let list = self.client.List("package", namespace, &ListOption::default()).await?;

        let mut packages = Vec::new();
        for o in list.objs {
            let funcPackage = serde_json::from_str(&o.data)?;
            packages.push(funcPackage);
        }

        return Ok(packages)
    }

    pub async fn GetPackage(&self, namespace: &str, name: &str) -> Result<FuncPackage> {
        let obj = match self.client.Get("package", namespace, name, 0).await? {
            None => return Err(Error::ENOENT(format!("can't get funcpackage with name {}/{}", namespace, name))),
            Some(o) => o,
        };
        let mut funcPackage: FuncPackage = serde_json::from_str(&obj.data)?;
        funcPackage.metadata.resource_version = Some(format!("{}", obj.reversion));
        return Ok(funcPackage);
    }

    pub async fn CreatePyPackage(&mut self, package: FuncPackage, funcFolder: &str) -> Result<FuncPackage> {
        let namespace = match &package.metadata.namespace {
            None => {
                return Err(Error::CommonError(format!("CreatePyPackage package has no namespace")));
            }
            Some(n) => n.clone(),
        };
        
        let packageName = match &package.metadata.name {
            None => {
                return Err(Error::CommonError(format!("CreatePyPackage package has no name")));
            }
            Some(n) => n.clone(),
        };
        
        let zipfile = ZipMgr::ZipFolder(funcFolder)?;
        
        let objectName = uuid::Uuid::new_v4().to_string();
        match self.client.PutObject(&namespace, &objectName, zipfile).await {
            Err(e) => {
                return Err(Error::CommonError(format!("CreatePyPackage fail with error {:?}", e)))
            }
            Ok(()) => (),
        }

        let mut funcPackage = package;

        if funcPackage.metadata.annotations.is_none() {
            funcPackage.metadata.annotations = Some(BTreeMap::new());
        }

        funcPackage.metadata.annotations.as_mut().unwrap().insert(AnnotationFuncPodPackageType.to_owned(), "python".to_owned());
        funcPackage.metadata.annotations.as_mut().unwrap().insert(AnnotationFuncPodPyPackageId.to_owned(), objectName.clone());
 
        let packageStr = serde_json::to_string(&funcPackage).unwrap();
        let obj = DataObject::NewFromK8sObj("package", &funcPackage.metadata, packageStr.clone());

        match self.client.Get("package", &namespace, &packageName, 0).await {
            Err(_) => (),
            Ok(obj) => {
                match obj {
                    None => (),
                    Some(obj) => {
                        match obj.annotations.get(AnnotationFuncPodPyPackageId) {
                            None => (),
                            Some(objName) => {
                                self.client.DeleteObject(&namespace, objName).await.ok();
                            }
                        }
                    }
                }
            }
        };

        self.client.Delete("package", &namespace, &packageName).await.ok();
        
        match self.client.Create("package", obj.Obj()).await {
            Err(e) => {
                return Err(Error::CommonError(format!("create_py_package: create pakage meta fail with error {:?}", e)));
            }
            Ok(_) => (),
        }

        return Ok(funcPackage)
    }
}