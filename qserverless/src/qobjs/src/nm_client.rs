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

use std::sync::Arc;

use tonic::transport::Channel;
use tokio::sync::Mutex as TMutex;
use k8s_openapi::api::core::v1 as k8s;

use crate::node_mgr::node_mgr_service_client::NodeMgrServiceClient;
use crate::node_mgr as NodeMgr;
use crate::common::*;

#[derive(Debug, Clone)]
pub struct NodeMgrClient {
    client: Arc<TMutex<NodeMgrServiceClient<Channel>>>
}

impl NodeMgrClient {
    pub async fn New(addr: String) -> Result<Self> {
        let client = NodeMgrServiceClient::connect(addr).await?;
        return Ok(Self {
            client: Arc::new(TMutex::new(client))
        });
    }

    pub async fn CreatePod(&self, node: &str, pod: &k8s::Pod, configMap: &k8s::ConfigMap) -> Result<()> {
        let podStr = serde_json::to_string(pod)?;
        let configMapStr = serde_json::to_string(configMap)?;

        let req = NodeMgr::CreatePodReq {
            node: node.to_string(),
            pod: podStr,
            config_map: configMapStr,
        };

        let mut client = self.client.lock().await;
        let resp = client.create_pod(req).await?;
        let resp = resp.get_ref();
        if resp.error.len() == 0 {
            return Ok(())
        }

        return Err(Error::CommonError(resp.error.clone()));
    }
}