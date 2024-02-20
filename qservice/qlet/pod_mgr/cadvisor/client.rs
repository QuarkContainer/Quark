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

use reqwest;

use qshare::common::Result;

use qshare::cadvisor_types::{MachineInfo, VersionInfo};
//type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug)]
pub struct NodeCAdvisorInfo {
	//ContainerInfo []*cadvisorv2.ContainerInfo
	pub machineInfo:   MachineInfo,
	//RootFsInfo    *cadvisorv2.FsInfo
	//ImageFsInfo   *cadvisorv2.FsInfo
	pub versionInfo:   VersionInfo,
}

pub struct Client {
    pub baseURL: String,
}

impl Client
{
    pub fn Init() -> Self {
        return Self {
            baseURL: "http://127.0.0.1:8080/api/v2.1".to_owned(),
        }
    }

    pub async fn MachineInfo(&self) -> Result<MachineInfo> {
        let url = self.baseURL.clone() + "/" + "machine";
        let resp = reqwest::get(url).await?.text().await?;
        let mi : MachineInfo = serde_json::from_str(&resp)?;
        
        return Ok(mi);
    }

    pub async fn VersionInfo(&self) -> Result<VersionInfo> {
        return Ok(VersionInfo::default())
    }

    pub async fn GetInfo(&self) -> Result<NodeCAdvisorInfo> {
        let mi = self.MachineInfo().await?;
        let version = self.VersionInfo().await?;
        return Ok(NodeCAdvisorInfo {
            machineInfo: mi,
            versionInfo: version,
        })
    }
}
