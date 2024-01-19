// Copyright (c) 2023 Quark Container Authors / 2018 The gVisor Authors.
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

use once_cell::sync::OnceCell;

use qshare::common::*;
use qshare::crictl;
use qshare::k8s;

use super::runtime::runtime::RuntimeMgr;
use super::runtime::image_mgr::ImageMgr;
use super::cadvisor::provider::CadvisorInfoProvider;

pub static RUNTIME_MGR: OnceCell<RuntimeMgr> = OnceCell::new();
pub static IMAGE_MGR: OnceCell<ImageMgr> = OnceCell::new();
pub static CADVISOR_PROVIDER: OnceCell<CadvisorInfoProvider> = OnceCell::new();

pub struct PodMgr {
    pub cadviceProvider: CadvisorInfoProvider,
    pub runtimeMgr: RuntimeMgr,
    pub imageMgr: ImageMgr,
}

impl PodMgr {
    pub async fn New() -> Result<Self> {
        let cadviceProvider = CadvisorInfoProvider::New().await?;
        let runtimeMgr = RuntimeMgr::New(10).await?;
        let imageMgr = ImageMgr::New(crictl::AuthConfig::default()).await?;

        return Ok(Self {
            cadviceProvider: cadviceProvider,
            runtimeMgr: runtimeMgr,
            imageMgr: imageMgr
        });
    }

    pub fn CreatePodAgent(&self, pod: &k8s::Pod, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<PodAgent> {
        let qpod = self.BuildAQuarkPod(state, pod, configMap, isDaemon)?;

        let podAgent = self.StartPodAgent(&qpod)?;
        return Ok(podAgent);
    }

    // pub async fn CreatePod(&self, pod: &k8s::Pod, configMap: &k8s::ConfigMap) -> Result<()> {
    // }
}