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
use std::result::Result as SResult;

use qobjs::qmeta as qmeta;
use qobjs::system_types::FuncPackage;
use qobjs::blob_mgr::*;
use qobjs::types::*;
use qobjs::cacher_client::CacherClient;
use qobjs::common::*;

pub struct QServerless {
    pub blobMgr: SqlBlobMgr,
    pub cacheClient: CacherClient, 
}

impl QServerless {
    pub async fn New(blobDbAddr: &str, qmetaSvcAddr: &str) -> Result<Self> {
        return Ok(Self {
            blobMgr: SqlBlobMgr::New(blobDbAddr).await?,
            cacheClient: CacherClient::New(qmetaSvcAddr.into()).await?,
        })
    }
}

#[tonic::async_trait]
impl qmeta::q_serverless_server::QServerless for QServerless {
    async fn create_py_package(
        &self,
        request: tonic::Request<qmeta::PyPackageReq>,
    ) -> SResult<tonic::Response<qmeta::PyPackageResp>, tonic::Status> {
        let req = request.get_ref();

        let mut funcPackage: FuncPackage = match serde_json::from_str(&req.package) {
            Err(e) => {
                let response = qmeta::PyPackageResp {
                    error: format!("create_py_package: deserialize package fail with error {:?}", e),
                    ..Default::default()
                };
                return Ok(tonic::Response::new(response));
            }
            Ok(p) => p,
        };

        let namespace = match &funcPackage.metadata.namespace {
            None => {
                let response = qmeta::PyPackageResp {
                    error: format!("create_py_package: package has mo namespace"),
                    ..Default::default()
                };
                return Ok(tonic::Response::new(response));
            }
            Some(n) => n.clone(),
        };

        let pacakgeName = match &funcPackage.metadata.name {
            None => {
                let response = qmeta::PyPackageResp {
                    error: format!("create_py_package: package has no name"),
                    ..Default::default()
                };
                return Ok(tonic::Response::new(response));
            }
            Some(n) => n.clone(),
        };

        let blobId = uuid::Uuid::new_v4().to_string();
        match self.blobMgr.CreateBlob(&blobId, &req.zipfile).await {
            Err(e) => {
                let response = qmeta::PyPackageResp {
                    error: format!("create_py_package: create blob fail with error {:?}", e),
                    ..Default::default()
                };
                return Ok(tonic::Response::new(response));
            }
            Ok(()) => (),
        }

        if funcPackage.metadata.annotations.is_none() {
            funcPackage.metadata.annotations = Some(BTreeMap::new());
        }

        funcPackage.metadata.annotations.as_mut().unwrap().insert(AnnotationFuncPodPackageType.to_owned(), "python".to_owned());
        funcPackage.metadata.annotations.as_mut().unwrap().insert(AnnotationFuncPodPyPackageId.to_owned(), blobId.clone());

        let packageStr = serde_json::to_string(&funcPackage).unwrap();
        let obj = DataObject::NewFromK8sObj("package", &funcPackage.metadata, packageStr.clone());

        self.cacheClient.Delete("package", &namespace, &pacakgeName).await.ok();
        
        match self.cacheClient.Create("package", obj.Obj()).await {
            Err(e) => {
                let response = qmeta::PyPackageResp {
                    error: format!("create_py_package: create pakage meta fail with error {:?}", e),
                    ..Default::default()
                };
                return Ok(tonic::Response::new(response));
            }
            Ok(_) => (),
        }

        let response = qmeta::PyPackageResp {
            package: packageStr,
            ..Default::default()
        };
        return Ok(tonic::Response::new(response));
    }
}
