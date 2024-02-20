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
use std::time::Duration;

use tokio::sync::Semaphore;
//use qobjs::v1alpha2::{self as cri};

use crate::cri::client::*;
use qobjs::crictl::*;
use qobjs::runtime_types::*;
use qobjs::common::Result;
use qobjs::common::Error;

/*pub type ContainerState = i32;

pub const ContainerState_CONTAINER_CREATED: ContainerState = 0;
pub const ContainerState_CONTAINER_RUNNING: ContainerState = 1;
pub const ContainerState_CONTAINER_EXITED:  ContainerState = 2;
pub const ContainerState_CONTAINER_UNKNOWN: ContainerState = 3;*/

#[derive(Debug)]
pub struct RuntimeMgr {
pub runtimeService: CriClient,
    pub concurrency: Semaphore,
}

impl RuntimeMgr {
    pub async fn New(concurrency: usize) -> Result<Self> {
        let runtimeService = CriClient::Init().await?;
        return Ok(Self {
            runtimeService: runtimeService,
            concurrency: Semaphore::new(concurrency),
        })
    }

    // GetPodSandbox implements RuntimeService
    pub async fn GetPodSandbox(&self, podSandboxId: &str) -> Result<Option<PodSandbox>> {
        info!("get pod sandbox {}", podSandboxId);

        let filter = PodSandboxFilter {
            id: podSandboxId.to_string(),
            ..Default::default()
        };
        let sandboxes = self.runtimeService
            .ListPodSandbox(Some(filter))
            .await?;

        if sandboxes.len() > 0 {
            return Ok(Some(sandboxes[0].clone()))
        } 

        return Ok(None);
    }

    // Status implements cri.RuntimeService
    pub async fn GetRuntimeStatus(&self) -> Result<RuntimeStatus> {
        let resp = self
            .runtimeService
            .Status(false)
            .await?;
        
        if let Some(status) = resp.status {
            if status.conditions.len() >= 2 {
                return Ok(status)
            }
        }

        return Err(Error::CommonError("Failed to get runtime status: RuntimeReady or NetworkReady condition are not set".to_string()));
    }

    pub async fn CreateContainer(&self, 
        podSandboxId: &str, 
        containerConfig: Option<ContainerConfig>, 
        podSandboxConfig: Option<PodSandboxConfig>) -> Result<RuntimeContainer> {
        let _s = self.concurrency.acquire().await?;

        let containerId = self.runtimeService.CreateContainer(
            podSandboxId, 
            containerConfig.clone(), 
            podSandboxConfig).await?;
        
        let filter = ContainerFilter {
            id: containerId.clone(),
            pod_sandbox_id: podSandboxId.to_string(),
            ..Default::default()
        };
        let containers = self.runtimeService.ListContainers(Some(filter)).await?;

        if containers.len() == 1 {
            let container = RuntimeContainer {
                id: containerId.clone(),
                containerConfig: containerConfig,
                container: containers[0].clone(),
            };

            return Ok(container)
        } 

        return Err(Error::CommonError(format!("Do not get exact one container with id: {}", containerId)))
    }

   pub async fn CreateSandbox(&self, sandboxConfig: Option<PodSandboxConfig>, runtimeClassName: &str) -> Result<RuntimePod> {
        let _s = self.concurrency.acquire().await?;

        info!("Run pod sandbox SandboxConfig {:?}", &sandboxConfig);

        let podSandBoxId = self.runtimeService.RunPodSandbox(sandboxConfig.clone(), runtimeClassName).await?;

        let filter = PodSandboxFilter {
            id: podSandBoxId.clone(),
            ..Default::default()
        };
        let sandboxes = self.runtimeService.ListPodSandbox(Some(filter)).await?;
        if sandboxes.len() != 1 {
            return Err(Error::CommonError(format!("Do not get exact one pod sandbox with id: {}", &podSandBoxId)))
        }

        let mut pod = RuntimePod {
            id: podSandBoxId.clone(),
            IPs: Vec::new(),
            sandboxConfig: sandboxConfig,
            sandbox: Some(sandboxes[0].clone()),
        };

        let sandboxStatus = self.GetPodSandboxStatus(&podSandBoxId).await?;
        info!("Get pod sandbox status: {:?}", &sandboxStatus);

        if let Some(network) = sandboxStatus.network {
            pod.IPs.push(network.ip.clone());
            for ip in network.additional_ips {
                pod.IPs.push(ip.ip.clone());
            }
        }

        return Ok(pod)
    } 

    // GetImageLabel implements cri.RuntimeService
    pub async fn GetImageLabel(&self) -> Result<String> {
        unimplemented!();
    }

    // GetContainerStatus implements cri.RuntimeService
    pub async fn GetContainerStatus(&self, containerId: &str) -> Result<ContainerStatus> {
        let resp = self.runtimeService.ContainerStatus(containerId, false).await?;

        return Ok(resp.status.unwrap())
    }

    // GetPodStatus implements cri.RuntimeService
    pub async fn GetPodStatus(&self, podSandboxId: &str, containerIds: &[String]) -> Result<RuntimePodStatus> {
        info!("Get pod status include sandbox and container status PodSandboxID {}, ContainerIDs {:?}", podSandboxId, containerIds);
        let sandboxStatus = self.GetPodSandboxStatus(podSandboxId).await?;

        let mut podStatus = RuntimePodStatus {
            sandboxStatus: sandboxStatus,
            containerStatus: BTreeMap::new(),
        };
        for id in containerIds {
            let containerStatus = self.GetContainerStatus(id).await?;
            podStatus.containerStatus.insert(id.to_string(), containerStatus);
        }

        return Ok(podStatus)
    }

    // GetPods implements cri.RuntimeService
    pub async fn GetPods(&self) -> Result<Vec<RuntimePod>> {
        let mut podMap = BTreeMap::new();

        let resp = self.runtimeService.ListPodSandbox(None).await?;
        for v in resp {
            let mut pod = RuntimePod {
                id: v.id.clone(),
                IPs: Vec::new(),
                sandbox: Some(v),
                sandboxConfig: None,
            };

            let sandboxStatus = self.GetPodSandboxStatus(&pod.id).await?;
            if let Some(network) = sandboxStatus.network {
                pod.IPs.push(network.ip.clone());
                for ip in network.additional_ips {
                    pod.IPs.push(ip.ip.clone());
                }
            }
            podMap.insert(pod.id.clone(), pod);
        }

        let pods : Vec<_> = podMap.into_values().collect();
        return Ok(pods);
    }

    // StopContainer implements RuntimeService
    pub async fn StopContainer(&self, containerId: &str, timeout: Duration) -> Result<()> {
        let _s = self.concurrency.acquire().await;

        info!("Stop container ContainerID {}", containerId);

        self.runtimeService.StopContainer(containerId, timeout.as_secs() as i64).await?;
        return Ok(())
    }

    // StartContainer implements cri.RuntimeService
    pub async fn StartContainer(&self, containerId: &str) -> Result<()> {
        let _s = self.concurrency.acquire().await;

        info!("Start container ContainerID {}", containerId);
        self.runtimeService.StartContainer(containerId).await?;
        return Ok(())
    }

    // TerminateContainer implements cri.RuntimeService
    pub async fn TerminateContainer(&self, containerId: &str) -> Result<()> {
        let _s = self.concurrency.acquire().await;

        info!("Terminate container, stop immediately without gracePeriodContainerID {}", containerId);

        let status = self.GetContainerStatus(containerId).await?;

        if status.state == ContainerState::ContainerRunning as i32 {
            return self.runtimeService.StopContainer(containerId, 0).await;
        }

        return self.runtimeService.RemoveContainer(containerId).await;
    }

    // TerminatePod implements cri.RuntimeService
    pub async fn TerminatePod(&self, podSandboxId: &str, mut containerIds: Vec<String> ) -> Result<()> {
        let _s = self.concurrency.acquire().await;

        info!("Terminate pod PodSandboxID {}", podSandboxId);

        if containerIds.len() == 0 {
            let containers = self.GetPodContainers(podSandboxId).await?;
            for c in &containers {
                if c.state != ContainerState::ContainerExited as i32 {
                    info!("Terminate container in pod PodSandboxID {}, Container {:?}", podSandboxId, c);
                    self.runtimeService.StopContainer(&c.id, 0).await?;
                    containerIds.push(c.id.clone());
                }

                self.runtimeService.RemoveContainer(&c.id).await?;
            }
        }

        self.runtimeService.StopPodSandbox(podSandboxId).await?;
        self.runtimeService.RemovePodSandbox(podSandboxId).await?;
        return Ok(())
    }

    pub async fn GetPodSandboxStatus(&self, podSandboxID: &str) -> Result<PodSandboxStatus> {
        let resp = self.runtimeService.PodSandboxStatus(podSandboxID, false).await?;
        if resp.status.is_none() {
            return Err(Error::CommonError(format!("GetPodSandboxStatus for {} get none status", podSandboxID)));
        }

        return Ok(resp.status.unwrap())
    }

    pub async fn GetAllContainers(&self) -> Result<Vec<Container>> {
        info!("Get all containers in runtime");
        
        return self.runtimeService.ListContainers(None).await;
    }

    pub async fn GetPodContainers(&self, podSandboxId: &str) -> Result<Vec<Container>> {
        info!("Get all containers in sandbox PodSandboxID {}", podSandboxId);
        let filter = ContainerFilter {
            pod_sandbox_id: podSandboxId.to_string(),
            ..Default::default()
        };

        return self.runtimeService.ListContainers(Some(filter)).await;
    }

    pub async fn ExecCommand(&self, containerID: &str, cmd: Vec<String>, timeout: i64) -> Result<(Vec<u8>, Vec<u8>)> {
        let resp = self.runtimeService.ExecSync(containerID, cmd.to_vec(), timeout).await?;
        if resp.exit_code != 0 {
            let stderr = std::str::from_utf8(&resp.stderr)?;
            return Err(Error::CommonError(format!("command '{:?}' exited with {}: {}", cmd, resp.exit_code, stderr)))
        }

        return Ok((resp.stdout.to_vec(), resp.stderr.to_vec()))
    }
}

