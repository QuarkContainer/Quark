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

use qobjs::common::*;
use qobjs::system_types::FuncPackage;
use qobjs::zip::ZipMgr;
use qobjs::qserverless_cli::QServerlessCli;

pub struct PackageMgr {
    pub client: QServerlessCli,
}

impl PackageMgr {
    pub async fn New(qserverlessSvcAddr: &str) -> Result<Self> {
        let client = QServerlessCli::New(qserverlessSvcAddr.to_owned()).await?;
        return Ok(Self {
            client: client
        });
    }

    pub async fn CreatePyPackage(&mut self, package: FuncPackage, funcFolder: &str) -> Result<FuncPackage> {
        // let packageStr = std::fs::read(packageFile)?;
        // let package : FuncPackage = serde_json::from_slice(&packageStr)?;

        let zipfile = ZipMgr::ZipFolder(funcFolder)?;
        return self.client.CreatePyPackage(package, zipfile).await;
    }
}