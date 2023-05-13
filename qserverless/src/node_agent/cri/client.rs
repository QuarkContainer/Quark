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


use std::convert::TryFrom;

use tokio::net::UnixStream;
use tonic::transport::{Endpoint, Uri, Channel};
use tower::service_fn;

use qobjs::v1alpha2::*;
use qobjs::v1alpha2::runtime_service_client::RuntimeServiceClient;
use qobjs::v1alpha2::image_service_client::ImageServiceClient;

use qobjs::common::Result;
use qobjs::common::Error;

#[derive(Debug)]
pub struct CriClient {
    pub channel: Channel,
}

impl CriClient {
    pub async fn Init() -> Result<Self> {
        let path = "/run/containerd/containerd.sock";
        let channel = Endpoint::try_from("http://[::]")?
            .connect_with_connector(service_fn(move |_: Uri| UnixStream::connect(path)))
            .await?;
    
        return Ok(Self {
            channel: channel,
        })
    }

    pub fn RuntimeClient(&self) -> RuntimeServiceClient<Channel> {
        return RuntimeServiceClient::new(self.channel.clone())
    }

    pub fn ImageClient(&self) -> ImageServiceClient<Channel> {
        return ImageServiceClient::new(self.channel.clone())
    }

    // RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure
	// the sandbox is in ready state.
	pub async fn RunPodSandbox(&self, config: Option<PodSandboxConfig>, runtimeHandler: &str) -> Result<String> {
        let request = tonic::Request::new(RunPodSandboxRequest {
            config: config,
            runtime_handler: runtimeHandler.to_string(),
        });
        let response : RunPodSandboxResponse = self
            .RuntimeClient()
            .run_pod_sandbox(request)
            .await?
            .into_inner();
        
        return Ok(response.pod_sandbox_id)
    }

    // StopPodSandbox stops the sandbox. If there are any running containers in the
	// sandbox, they should be force terminated.
	pub async fn StopPodSandbox(&self, podSandboxID: &str) -> Result<()> {
        let request = tonic::Request::new(StopPodSandboxRequest {
            pod_sandbox_id: podSandboxID.to_string(),
        });
        let _response : StopPodSandboxResponse = self
            .RuntimeClient()
            .stop_pod_sandbox(request)
            .await?
            .into_inner();
        
        return Ok(())
    }

	// RemovePodSandbox removes the sandbox. If there are running containers in the
	// sandbox, they should be forcibly removed.
	pub async fn RemovePodSandbox(&self, podSandboxID: &str) -> Result<()> {
        let request = tonic::Request::new(RemovePodSandboxRequest {
            pod_sandbox_id: podSandboxID.to_string(),
        });
        let _response : RemovePodSandboxResponse = self
            .RuntimeClient()
            .remove_pod_sandbox(request)
            .await?
            .into_inner();
        
        return Ok(())
    }
	
    // PodSandboxStatus returns the Status of the PodSandbox.
	pub async fn PodSandboxStatus(&self, podSandboxID: &str, verbose: bool) -> Result<PodSandboxStatusResponse> {
        let request = tonic::Request::new(PodSandboxStatusRequest {
            pod_sandbox_id: podSandboxID.to_string(),
            verbose: verbose,
        });
        let response : PodSandboxStatusResponse = self
            .RuntimeClient()
            .pod_sandbox_status(request)
            .await?
            .into_inner();
        
        return Ok(response)
    }
	// ListPodSandbox returns a list of Sandbox.
	pub async fn ListPodSandbox(&self, filter: Option<PodSandboxFilter>) -> Result<Vec<PodSandbox>> {
        let request = tonic::Request::new(ListPodSandboxRequest {
            filter: filter,
        });
        let response : ListPodSandboxResponse = self
            .RuntimeClient()
            .list_pod_sandbox(request)
            .await?
            .into_inner();
        
        return Ok(response.items)
    }
	
    // PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
	pub async fn PortForward(&self, podSandboxId: &str,  ports: Vec<i32>) -> Result<PortForwardResponse> { 
        let request = tonic::Request::new(PortForwardRequest {
            pod_sandbox_id: podSandboxId.to_string(),
            port: ports,
        });
        let response : PortForwardResponse = self
            .RuntimeClient()
            .port_forward(request)
            .await?
            .into_inner();
        
        return Ok(response)
    }

    // CreateContainer creates a new container in specified PodSandbox.
    pub async fn CreateContainer(&self, podSandboxId: &str, config: Option<ContainerConfig>, sandboxConfig: Option<PodSandboxConfig>) -> Result<String> {
        let request = tonic::Request::new(CreateContainerRequest {
            pod_sandbox_id: podSandboxId.to_string(),
            config: config,
            sandbox_config: sandboxConfig,
        });
        let response : CreateContainerResponse = self
            .RuntimeClient()
            .create_container(request)
            .await?
            .into_inner();
        
        return Ok(response.container_id)
    }

	// StartContainer starts the container.
	pub async fn StartContainer(&self, containerID: &str) -> Result<()> {
        let request = tonic::Request::new(StartContainerRequest {
            container_id: containerID.to_string(),
        });
        let _response = self
            .RuntimeClient()
            .start_container(request)
            .await?
            .into_inner();
        
        return Ok(())
    }

	// StopContainer stops a running container with a grace period (i.e., timeout).
	pub async fn StopContainer(&self, containerID: &str, timeout: i64) -> Result<()> {
        let request = tonic::Request::new(StopContainerRequest {
            container_id: containerID.to_string(),
            timeout: timeout,
        });
        let _response = self
            .RuntimeClient()
            .stop_container(request)
            .await?
            .into_inner();
        
        return Ok(())
    }

	// RemoveContainer removes the container.
	pub async fn RemoveContainer(&self, containerID: &str) -> Result<()> {
        let request = tonic::Request::new(RemoveContainerRequest {
            container_id: containerID.to_string(),
        });
        let _response = self
            .RuntimeClient()
            .remove_container(request)
            .await?
            .into_inner();
        
        return Ok(())
    }

	// ListContainers lists all containers by filters.
	pub async fn ListContainers(&self, filter: Option<ContainerFilter>) -> Result<Vec<Container>> {
        let request = tonic::Request::new(ListContainersRequest {
            filter: filter,
        });
        
        let response  = self.RuntimeClient()
            .list_containers(request)
            .await?
            .into_inner();
        
        return Ok(response.containers)
    }

	// ContainerStatus returns the status of the container.
	pub async fn ContainerStatus(&self, containerID: &str, verbose: bool) -> Result<ContainerStatusResponse> {
        let request = tonic::Request::new(ContainerStatusRequest {
            container_id: containerID.to_string(),
            verbose: verbose
        });
        
        let response  = self.RuntimeClient()
            .container_status(request)
            .await?
            .into_inner();
        
        return Ok(response)
    }

	// UpdateContainerResources updates the cgroup resources for the container.
	pub async fn UpdateContainerResources(&self, containerID: &str, linux: Option<LinuxContainerResources>) -> Result<()> {
        let request = tonic::Request::new(UpdateContainerResourcesRequest {
            container_id: containerID.to_string(),
            linux: linux,
            windows: None,
            annotations: std::collections::HashMap::new(),
        });
        let _response = self
            .RuntimeClient()
            .update_container_resources(request)
            .await?
            .into_inner();
        
        return Ok(())
    }

	// ExecSync executes a command in the container, and returns the stdout output.
	// If command exits with a non-zero exit code, an error is returned.
	pub async fn ExecSync(&self, containerID: &str, cmd: Vec<String>, timeout: i64) -> Result<ExecSyncResponse> {
        let request = tonic::Request::new(ExecSyncRequest {
            container_id: containerID.to_string(),
            cmd: cmd,
            timeout: timeout
        });
        
        let response  = self.RuntimeClient()
            .exec_sync(request)
            .await?
            .into_inner();
        
        return Ok(response)
    }

	// Exec prepares a streaming endpoint to execute a command in the container, and returns the address.
	pub async fn Exec(&self, request: ExecRequest) -> Result<ExecResponse> {
        let response  = self.RuntimeClient()
            .exec(request)
            .await?
            .into_inner();
        
        return Ok(response)

    }

	// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
	pub async fn Attach(&self, request: AttachRequest) -> Result<AttachResponse> {
        let response  = self.RuntimeClient()
            .attach(request)
            .await?
            .into_inner();
        
        return Ok(response)
    }

	// ReopenContainerLog asks runtime to reopen the stdout/stderr log file
	// for the container. If it returns error, new container log file MUST NOT
	// be created.
	pub async fn ReopenContainerLog(&self, containerID: &str) -> Result<()> {
        let request = tonic::Request::new(ReopenContainerLogRequest {
            container_id: containerID.to_string(),
        });
        
        let _response  = self.RuntimeClient()
            .reopen_container_log(request)
            .await?
            .into_inner();
        
        return Ok(())
    }

    	// ContainerStats returns stats of the container. If the container does not
	// exist, the call returns an error.
	pub async fn ContainerStats(&self, containerID: &str) -> Result<Option<ContainerStats>> {
        let request = tonic::Request::new(ContainerStatsRequest {
            container_id: containerID.to_string(),
        });
        
        let response  = self.RuntimeClient()
            .container_stats(request)
            .await?
            .into_inner();
        
        return Ok(response.stats)
    } 

	// ListContainerStats returns stats of all running containers.
	pub async fn ListContainerStats(&self, filter: Option<ContainerStatsFilter>) -> Result<Vec<ContainerStats>> {
        let request = tonic::Request::new(ListContainerStatsRequest {
            filter: filter,
        });
        
        let response  = self.RuntimeClient()
            .list_container_stats(request)
            .await?
            .into_inner();
        
        return Ok(response.stats)
    }

	// PodSandboxStats returns stats of the pod. If the pod does not
	// exist, the call returns an error.
	pub async fn PodSandboxStats(&self, podSandboxID: &str) -> Result<PodSandboxStats> {
        let request = tonic::Request::new(PodSandboxStatsRequest {
            pod_sandbox_id: podSandboxID.to_string(),
        });
        
        let response  = self.RuntimeClient()
            .pod_sandbox_stats(request)
            .await?
            .into_inner();

        if response.stats.is_none() {
            return Err(Error::CommonError(format!("PodSandboxStats for {} get none stats", podSandboxID)));
        }
        
        return Ok(response.stats.unwrap())
    }
	// ListPodSandboxStats returns stats of all running pods.
	pub async fn ListPodSandboxStats(&self, filter: Option<PodSandboxStatsFilter>) -> Result<Vec<PodSandboxStats>> {
        let request = tonic::Request::new(ListPodSandboxStatsRequest {
            filter: filter,
        });
        
        let response  = self.RuntimeClient()
            .list_pod_sandbox_stats(request)
            .await?
            .into_inner();
        
        return Ok(response.stats)
    }

    // Version returns the runtime name, runtime version and runtime API version
	pub async fn Version(&self, apiVersion: &str) -> Result<VersionResponse> {
        let request = tonic::Request::new(VersionRequest {
            version: apiVersion.to_string(),
        });
        
        let response  = self.RuntimeClient()
            .version(request)
            .await?
            .into_inner();
        
        return Ok(response)
    } 

    // UpdateRuntimeConfig updates runtime configuration if specified
    pub async fn UpdateRuntimeConfig(&self, runtimeConfig: Option<RuntimeConfig>) -> Result<()> {
        let request = tonic::Request::new(UpdateRuntimeConfigRequest {
            runtime_config: runtimeConfig,
        });
        
        let _response  = self.RuntimeClient()
            .update_runtime_config(request)
            .await?
            .into_inner();
        
        return Ok(())
    }
	// Status returns the status of the runtime.
	pub async fn Status(&self, verbose: bool) -> Result<StatusResponse> {
        let request = tonic::Request::new(StatusRequest {
            verbose: verbose
        });
        
        let response  = self.RuntimeClient()
            .status(request)
            .await?
            .into_inner();
        
        return Ok(response)
    }

    // ListImages lists the existing images.
    pub async fn ListImages(&self, filter: Option<ImageFilter>) -> Result<Vec<Image>> {
        let request = tonic::Request::new(ListImagesRequest {
            filter: filter,
        });
        
        let response  = self.ImageClient()
            .list_images(request)
            .await?
            .into_inner();
        
        return Ok(response.images)
    }
	// ImageStatus returns the status of the image.
	pub async fn ImageStatus(&self, image: Option<ImageSpec>, verbose: bool) -> Result<ImageStatusResponse> {
        let request = tonic::Request::new(ImageStatusRequest {
            image: image,
            verbose: verbose,
        });
        
        let response  = self.ImageClient()
            .image_status(request)
            .await?
            .into_inner();
        
        return Ok(response)
    }

	// PullImage pulls an image with the authentication config.
	pub async fn PullImage(&self, image: Option<ImageSpec>, auth: Option<AuthConfig>, podSandboxConfig: Option<PodSandboxConfig>) -> Result<String> {
        let request = tonic::Request::new(PullImageRequest {
            image: image,
            auth: auth,
            sandbox_config: podSandboxConfig,
        });
        
        let response  = self.ImageClient()
            .pull_image(request)
            .await?
            .into_inner();
        
        return Ok(response.image_ref)
    }

	// RemoveImage removes the image.
	pub async fn RemoveImage(&self, image: Option<ImageSpec>) -> Result<()> {
        let request = tonic::Request::new(RemoveImageRequest {
            image: image,
        });
        
        let _response  = self.ImageClient()
            .remove_image(request)
            .await?
            .into_inner();
        
        return Ok(())
    }
	// ImageFsInfo returns information of the filesystem that is used to store images.
	pub async fn ImageFsInfo(&self) -> Result<Vec<FilesystemUsage>> {
        let request = tonic::Request::new(ImageFsInfoRequest {
        });
        
        let response  = self.ImageClient()
            .image_fs_info(request)
            .await?
            .into_inner();
        
        return Ok(response.image_filesystems)
    } 

}