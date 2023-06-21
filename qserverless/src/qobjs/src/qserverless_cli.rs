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

use tonic::transport::Channel;

use crate::qmeta;
use crate::common::*;
use crate::system_types::FuncPackage;

pub struct QServerlessCli {
    pub client: qmeta::q_serverless_client::QServerlessClient<Channel>,
}

impl QServerlessCli {
    pub async fn New(qserverlessSvcAddr: String) -> Result<Self> {
        let client = qmeta::q_serverless_client::QServerlessClient::connect(qserverlessSvcAddr).await?;
        
        return Ok(Self {
            client: client,
        })
    }

    pub async fn CreatePyPackage(&mut self, package: FuncPackage, zipFile: Vec<u8>) -> Result<FuncPackage> {
        let packageStr = serde_json::to_string(&package)?;
        let req = qmeta::PyPackageReq {
            package: packageStr,
            zipfile: zipFile,
        };

        let response = self.client.create_py_package(tonic::Request::new(req)).await?;
        let resp = response.get_ref();

        if resp.error.len() != 0 {
            return Err(Error::CommonError(resp.error.clone()));
        }

        let package: FuncPackage = serde_json::from_str(&resp.package)?;

        return Ok(package)
    }
}
