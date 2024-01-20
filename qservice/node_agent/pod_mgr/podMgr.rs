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

use std::result::Result as SResult;

use podMgr::podmgr_agent::PmAgent;
use qshare::consts::ConfigMapFromString;
use qshare::consts::NodeFromString;
use qshare::consts::PodFromString;
use qshare::na;
use qshare::crictl;
use qshare::common::*;

use crate::pod_mgr::*;

use super::qnode::QuarkNode;
use super::{CADVISOR_PROVIDER, RUNTIME_MGR};
use super::cadvisor::provider::CadvisorInfoProvider;



pub struct PodMgr {
    pub pmAgent: PmAgent,
}

impl PodMgr {
    pub async fn New() -> Result<Self> {
        CADVISOR_PROVIDER.set(CadvisorInfoProvider::New().await.unwrap()).unwrap();
        RUNTIME_MGR.set(RuntimeMgr::New(10).await.unwrap()).unwrap();
        IMAGE_MGR.set(ImageMgr::New(crictl::AuthConfig::default()).await.unwrap()).unwrap();
    
        let config = &NODE_CONFIG;
        let nodename = &NODEAGENT_CONFIG.NodeName();

        PmAgent::CleanPods(nodename).await?;
    
        let quarkNode = QuarkNode::NewQuarkNode(nodename, &config)?;
        let pmAgent = PmAgent::New(&quarkNode)?;
        pmAgent.Start().await?;
        return Ok(Self {
            pmAgent: pmAgent
        });
    }

    pub fn CreatePod(&self, req: na::CreatePodReq) -> Result<()> {
        let pod = PodFromString(&req.pod)?;
        let configMap = ConfigMapFromString(&req.config_map)?;
        self.pmAgent.CreatePod(&pod, &configMap)?;
        return Ok(())
    }
}

#[tonic::async_trait]
impl na::node_agent_service_server::NodeAgentService for PodMgr {
    async fn create_pod(
        &self,
        request: tonic::Request<na::CreatePodReq>,
    ) -> SResult<tonic::Response<na::CreatePodResp>, tonic::Status> {
        let req = request.into_inner();
        match self.CreatePod(req) {
            Ok(()) => (),
            Err(e) => {
                return Ok(tonic::Response::new(na::CreatePodResp {
                    error: format!("fail: {:?}", e),
                }))
            }
        }
        return Ok(tonic::Response::new(na::CreatePodResp {
            error: "".to_owned()
        }))
    }
    
    async fn terminate_pod(
        &self,
        request: tonic::Request<na::TerminatePodReq>,
    ) -> SResult<tonic::Response<na::TerminatePodResp>, tonic::Status> {
        let req = request.into_inner();
        let podId = &req.pod_id;
        match self.pmAgent.TerminatePod(podId) {
            Err(e) => {
                return Ok(tonic::Response::new(na::TerminatePodResp {
                    error: format!("fail: {:?}", e),
                }))
            }
            Ok(()) => {
                return Ok(tonic::Response::new(na::TerminatePodResp {
                    error: "".to_owned()
                }));
            }
        }
    }

    async fn node_config(
        &self,
        request: tonic::Request<na::NodeConfigReq>,
    ) -> SResult<tonic::Response<na::NodeConfigResp>, tonic::Status> {
        let req = request.into_inner();
        let node = match NodeFromString(&req.node) {
            Err(e) => {
                return Ok(tonic::Response::new(na::NodeConfigResp {
                    error: format!("fail: {:?}", e),
                }))
            }
            Ok(n) => {
                n
            }
        };
        
        match self.pmAgent.NodeConfigure(node).await {
            Err(e) => {
                return Ok(tonic::Response::new(na::NodeConfigResp {
                    error: format!("fail: {:?}", e),
                }))
            }
            Ok(()) => {
                return Ok(tonic::Response::new(na::NodeConfigResp {
                    error: "".to_owned()
                }))
            }
        }
    }
}