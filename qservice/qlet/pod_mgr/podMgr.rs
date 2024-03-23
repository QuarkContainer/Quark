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

use podMgr::node_register::NodeRegister;
use podMgr::qlet_store::QletStateService;
use qshare::k8s::ConfigMap;
use qshare::k8s::HostPathVolumeSource;
use qshare::k8s::Volume;
use qshare::node::ContainerPort;
use qshare::node::*;
use std::collections::BTreeMap;
use std::result::Result as SResult;
use tonic::transport::Server;

use podMgr::podmgr_agent::PmAgent;
use qshare::common::*;
use qshare::consts::ConfigMapFromString;
use qshare::consts::NodeFromString;
use qshare::consts::PodFromString;
use qshare::crictl;
use qshare::na;

use crate::pod_mgr::*;
use crate::QLET_CONFIG;

use super::cadvisor::provider::CadvisorInfoProvider;
use super::qnode::QuarkNode;
use super::{CADVISOR_PROVIDER, QLET_STORE, RUNTIME_MGR};

pub struct PodMgr {
    pub pmAgent: PmAgent,
}

impl PodMgr {
    pub async fn New() -> Result<Self> {
        CADVISOR_PROVIDER
            .set(CadvisorInfoProvider::New().await.unwrap())
            .unwrap();
        RUNTIME_MGR.set(RuntimeMgr::New(10).await.unwrap()).unwrap();
        IMAGE_MGR
            .set(ImageMgr::New(crictl::AuthConfig::default()).await.unwrap())
            .unwrap();
        QLET_STORE.set(QletStore::New().await.unwrap()).unwrap();

        let config = &NODE_CONFIG;
        let nodename = &QLET_CONFIG.nodeName;

        PmAgent::CleanPods(nodename).await?;

        let quarkNode = QuarkNode::NewQuarkNode(&QLET_CONFIG, &config)?;
        let pmAgent = PmAgent::New(&quarkNode)?;
        pmAgent.Start().await?;
        return Ok(Self { pmAgent: pmAgent });
    }

    pub fn CreateFuncPod(&self, req: na::CreateFuncPodReq) -> Result<IpAddress> {
        let mut labels = BTreeMap::new();
        for kv in req.labels {
            labels.insert(kv.key.clone(), kv.val.clone());
        }

        let mut annotations = BTreeMap::new();
        for kv in req.annotations {
            annotations.insert(kv.key.clone(), kv.val.clone());
        }

        let mut pod = PodDef {
            tenant: req.tenant.to_owned(),
            namespace: req.namespace.to_owned(),
            name: req.name.to_owned(),
            uid: uuid::Uuid::new_v4().to_string(),
            labels: labels,
            annotations: annotations,
            ..Default::default()
        };

        let mut volumes = Vec::new();
        let mut volumeMounts: Vec<VolumeMount> = Vec::new();

        for mount in &req.mounts {
            let volume = Volume {
                name: mount.host_path.clone(), // "/home/brad/rust/Quark/test".to_owned(),
                host_path: Some(HostPathVolumeSource {
                    path: mount.host_path.clone(), // "/home/brad/rust/Quark/test".to_owned(),
                    ..Default::default()
                }),
                ..Default::default()
            };

            volumes.push(volume);

            volumeMounts.push(VolumeMount {
                name: mount.host_path.clone(), // "/home/brad/rust/Quark/test".to_owned(),
                mount_path: mount.mount_path.clone(), // "/test".to_owned(),
                ..Default::default()
            });
        }

        pod.volumes = volumes;

        let mut containerPorts = Vec::new();
        for p in &req.ports {
            containerPorts.push(ContainerPort {
                container_port: p.container_port,
                host_port: Some(p.host_port),
                ..Default::default()
            })
        }

        let mut containerEnvs = BTreeMap::new();
        for env in req.envs {
            containerEnvs.insert(env.name.clone(), env.value.clone());
        }

        let container = ContainerDef {
            name: req.name.clone(),
            image: req.image.to_owned(),
            commands: req.commands,
            envs: containerEnvs,
            volume_mounts: volumeMounts,
            ports: containerPorts,
            ..Default::default()
        };

        pod.containers.push(container);

        let configMap = ConfigMap::default();

        let addr = self.pmAgent.CreatePod(&pod, &configMap)?;
        return Ok(addr);
    }

    pub fn CreatePod(&self, req: na::CreatePodReq) -> Result<()> {
        let pod = PodFromString(&req.pod)?;
        let configMap = ConfigMapFromString(&req.config_map)?;
        self.pmAgent.CreatePod(&pod, &configMap)?;
        return Ok(());
    }

    pub fn PodId(&self, tenant: &str, namespace: &str, name: &str) -> String {
        return format!("{}/{}/{}", tenant, namespace, name);
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
            error: "".to_owned(),
        }));
    }

    async fn get_pod(
        &self,
        request: tonic::Request<na::GetPodReq>,
    ) -> SResult<tonic::Response<na::GetPodResp>, tonic::Status> {
        let req = request.into_inner();
        let podId = &self.PodId(&req.tenant, &&req.namespace, &req.name);
        match NODEAGENT_STORE.GetPod(podId) {
            Ok((rev, pod)) => {
                error!("pod is {:?}", serde_json::to_string_pretty(&pod).unwrap());
                return Ok(tonic::Response::new(na::GetPodResp {
                    error: "".to_owned(),
                    pod: serde_json::to_string_pretty(&pod).unwrap(),
                    revision: rev,
                }));
            }
            Err(e) => {
                return Ok(tonic::Response::new(na::GetPodResp {
                    error: format!("fail: {:?}", e),
                    ..Default::default()
                }))
            }
        }
    }

    async fn terminate_pod(
        &self,
        request: tonic::Request<na::TerminatePodReq>,
    ) -> SResult<tonic::Response<na::TerminatePodResp>, tonic::Status> {
        let req = request.into_inner();
        let podId = &Self::PodId(&self, &req.tenant, &req.namespace, &req.name);
        match self.pmAgent.TerminatePod(podId) {
            Err(e) => {
                return Ok(tonic::Response::new(na::TerminatePodResp {
                    error: format!("fail: {:?}", e),
                }))
            }
            Ok(()) => {
                return Ok(tonic::Response::new(na::TerminatePodResp {
                    error: "".to_owned(),
                }));
            }
        }
    }

    async fn create_func_pod(
        &self,
        request: tonic::Request<na::CreateFuncPodReq>,
    ) -> SResult<tonic::Response<na::CreateFuncPodResp>, tonic::Status> {
        let req = request.into_inner();

        match self.CreateFuncPod(req) {
            Err(e) => {
                return Ok(tonic::Response::new(na::CreateFuncPodResp {
                    error: format!("fail: {:?}", e),
                    ipaddress: 0,
                }))
            }
            Ok(addr) => {
                return Ok(tonic::Response::new(na::CreateFuncPodResp {
                    error: "".to_owned(),
                    ipaddress: addr.0,
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
            Ok(n) => n,
        };

        match self.pmAgent.NodeConfigure(&node).await {
            Err(e) => {
                return Ok(tonic::Response::new(na::NodeConfigResp {
                    error: format!("fail: {:?}", e),
                }))
            }
            Ok(()) => {
                return Ok(tonic::Response::new(na::NodeConfigResp {
                    error: "".to_owned(),
                }))
            }
        }
    }
}

pub async fn PodMgrSvc() -> Result<()> {
    let podMgr = PodMgr::New().await?;

    let podMgrAddr = format!("0.0.0.0:{}", QLET_CONFIG.podMgrPort);

    let podMgrSvcFuture = Server::builder()
        .add_service(na::node_agent_service_server::NodeAgentServiceServer::new(
            podMgr,
        ))
        .serve(podMgrAddr.parse().unwrap());

    let nodeRegister = NodeRegister::New(
        &QLET_CONFIG.etcdAddresses,
        &QLET_CONFIG.nodeName,
        &QLET_CONFIG.nodeIp,
        QLET_CONFIG.podMgrPort,
        QLET_CONFIG.tsotSvcPort,
        QLET_CONFIG.stateSvcPort,
        &QLET_CONFIG.cidr,
    );

    let nodeRegisterFuture = nodeRegister.Process();

    let qletStateSvcFuture = QletStateService();

    info!("pod manager start ...");
    tokio::select! {
        _ = podMgrSvcFuture => {
            error!("podMgrSvcFuture finish");
        },
        _ = nodeRegisterFuture => {
            error!("nodeRegisterFuture finish");
        }
        _ = qletStateSvcFuture => {
            error!("qletStateSvcFuture finish");
        }
    }
    info!("pod manager finish ...");

    return Ok(());
}
