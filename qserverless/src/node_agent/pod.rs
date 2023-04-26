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
use std::os::unix::prelude::PermissionsExt;
use std::path::Path;
use std::fs;
use std::fs::Permissions;
use std::sync::Mutex;

//use qobjs::core_types;
use k8s_openapi::api::core::v1 as k8s;
use tokio::sync::{mpsc, Notify};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use qobjs::runtime_types::{QuarkPod, QuarkContainer};
use qobjs::common::*;
use qobjs::k8s_util::*;
use qobjs::pb_gen::v1alpha2::{self as cri};
use qobjs::runtime_types::RuntimeContainer;
use qobjs::config::*;
use qobjs::runtime_types::*;

use crate::IMAGE_MGR;
use crate::RUNTIME_MGR;
use crate::pod_sandbox::AddPodSecurityContext;
use crate::runtime::k8s_helper::*;

use crate::nm_svc::NodeAgentMsg;
use crate::runtime::k8s_labels::*;
use crate::runtime::k8s_util::*;
use crate::container::*;

// The container runtime default profile should be used.
pub const SecurityProfile_RuntimeDefault: i32 = 0;
// Disable the feature for the sandbox or the container.
pub const SecurityProfile_Unconfined: i32 = 1;
// A pre-defined profile on the node should be used.
pub const SecurityProfile_Localhost: i32 = 2;

pub struct PodAgent {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub pod: QuarkPod,
    pub msgChannTx: mpsc::Sender<NodeAgentMsg>, 
    pub nodeConfig: NodeConfiguration,
    pub containers: BTreeMap<String, PodContainerAgent>,
}

impl PodAgent {
    pub async fn CreatePod(&mut self) -> Result<()> {
        let pod = self.pod.Pod();
        let podId = K8SUtil::PodId(&pod);
        info!("Create Pod Cgroup pod {}", &podId);
        /*
        // Create Cgroups for the pod and apply resource parameters
        klog.InfoS("Create Pod Cgroup", "pod", types.UniquePodName(a.pod))
        pcm := a.dependencies.QosManager
        if pcm.IsPodCgroupExist(pod) {
            // daemon pod can be recreated, allow cgroup exist for daemon pod cgroup
            if !a.pod.Daemon {
                pcm.UpdateQOSCgroups()
                return fmt.Errorf("cgroup already exist for pod %s, possibly previous cgroup not cleaned", types.UniquePodName(a.pod))
            }
        } else {
            err := pcm.CreatePodCgroup(pod)
            if err != nil {
                klog.ErrorS(err, "Failed to create pod cgroup ", "pod", types.UniquePodName(a.pod))
                return err
            }
            // call update qos to make sure cgroup manager internal state update to date
            pcm.UpdateQOSCgroups()
        }
        */
        info!("Make Pod data dirs pod {}", &podId);
        MakePodDataDir(&self.nodeConfig.RootPath, &pod)?;
        info!("Make Pod log dirs pod {}", &podId);
        MakePodLogDir(&self.nodeConfig.RootPath, &pod)?;
        /*
        // TODO, Try to attach and mount volumes into pod, mounted vol will be mounted into container later, do not support volume for now
        klog.InfoS("Prepare pod volumes", "pod", types.UniquePodName(a.pod))
        if err := a.dependencies.VolumeManager.WaitForAttachAndMount(pod); err != nil {
            klog.ErrorS(err, "Unable to attach or mount volumes for pod; skipping pod", "pod", types.UniquePodName(a.pod))
            return err
        }
         */
        info!("Pull pod secret pod {}", &podId);
        let pullSecrets = Vec::new();
        info!("Create pod sandbox {}", &podId);
        let runtimePod = self.CreatePodSandbox().await?;
        info!("Get pod sandbox {}, sandbox {:?}", &podId, &runtimePod);

        info!("Start pod init containers pod {}", &podId);
        for c in pod.spec.as_ref().unwrap().init_containers.as_ref().unwrap() {
            let runtimeContainer = self.CreateContainer(
                runtimePod.sandboxConfig.as_ref().unwrap(), 
                c, 
                &pullSecrets
            ).await?;
            let inner = QuarkContainerInner {
                state: RuntimeContainerState::Creating,
                initContainer: true,
                spec: c.clone(),
                runtimeContainer: runtimeContainer,
                containerStatus: None,
            };
            let container = QuarkContainer(Arc::new(Mutex::new(inner)));
            self.pod.lock().unwrap().containers.insert(c.name.clone(), container.clone());
            let containerAgent = PodContainerAgent::New(&self.msgChannTx, &self.pod, &container).await?;
            self.containers.insert(c.name.clone(), containerAgent);
        }

        self.pod.lock().unwrap().runtimePod = Arc::new(runtimePod);
        
        
        return Ok(())
    }


    pub async fn CreatePodSandbox(&self) -> Result<RuntimePod> {
        let pod = self.pod.Pod();
        let podsandboxConfig = self.GeneratePodSandboxConfig().await?;
        let podIp = K8SUtil::PodId(&pod);

        info!("Make pod log dir for pod {}", &podIp);
        fs::create_dir_all(&podsandboxConfig.log_directory)?;
        let perms = Permissions::from_mode(0o755);
        fs::set_permissions(&podsandboxConfig.log_directory, perms)?;

        let runtimehandler = &self.nodeConfig.RuntimeHandler;
        info!("Call runtime to create sandbox pod {} sandboxConfig is {:?}", &podIp, &podsandboxConfig);

        let runtimePod = RUNTIME_MGR.CreateSandbox(Some(podsandboxConfig), runtimehandler).await?;

        return Ok(runtimePod)
    }

    // generatePodSandboxConfig generates pod sandbox config from fornaxtypes.FornaxPod.
    pub async fn GeneratePodSandboxConfig(&self) -> Result<cri::PodSandboxConfig> {
        // fornax node will expect fornaxcore populate most of pod spec before send it
	    // it will not calulate hostname, all these staff
        let pod = self.pod.Pod();
        let podUID = pod.metadata.uid.as_deref().unwrap_or("").to_string();
        let mut podSandboxConfig = cri::PodSandboxConfig {
            metadata: Some(cri::PodSandboxMetadata {
                name: pod.metadata.name.as_deref().unwrap_or("").to_string(),
                namespace: pod.metadata.namespace.as_deref().unwrap_or("").to_string(),
                uid: podUID,
                ..Default::default()
            }),
            labels: NewPodLabels(&pod),
            annotations: NewPodAnnotations(&pod),
            ..Default::default()
        };

        let spec = pod.spec.as_ref().unwrap();

        podSandboxConfig.dns_config = Some(cri::DnsConfig::default());
        if !IsHostNetworkPod(&pod) && spec.hostname.is_some() {
            podSandboxConfig.hostname = spec.hostname.as_deref().unwrap_or("").to_string();
        }

        podSandboxConfig.log_directory = GetPodLogDir(
            DefaultPodLogsRootPath,
            pod.metadata.namespace.as_deref().unwrap_or(""), 
            pod.metadata.name.as_deref().unwrap_or(""),  
            pod.metadata.uid.as_deref().unwrap_or("")
        );

        let mut podMapping = Vec::new();
        for c in &spec.containers {
            for v in MakePortMappings(c) {
                podMapping.push(v);
            }
        }

        if podMapping.len() > 0 {
            podSandboxConfig.port_mappings = podMapping;
        }

        let lc = self.GeneratePodSandboxLinuxConfig()?;
        podSandboxConfig.linux = Some(lc);

        ApplySandboxResources(&self.nodeConfig, &pod, &mut podSandboxConfig)?;

        return Ok(podSandboxConfig)
    }

    pub fn GeneratePodSandboxLinuxConfig(&self) -> Result<cri::LinuxPodSandboxConfig> {
        let pod = self.pod.Pod();
        let mut lpsc = cri::LinuxPodSandboxConfig {
            cgroup_parent: "".to_owned(),
            security_context: Some(cri::LinuxSandboxSecurityContext {
                privileged: HasPrivilegedContainer(pod.spec.as_ref().unwrap()),
                seccomp: Some(cri::SecurityProfile {
                    profile_type: SecurityProfile_RuntimeDefault,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        AddPodSecurityContext(&pod, &mut lpsc);
        return Ok(lpsc)
    }

    pub async fn CreateContainer(
        &self, 
        podSandboxConfig: &cri::PodSandboxConfig,
        containerSpec: &k8s::Container,
        _pullSecrets: &Vec<k8s::Secret>,
    ) -> Result<RuntimeContainer> {
        info!("Pull image for container pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let pod = self.pod.Pod();

        let imageRef = IMAGE_MGR.PullImageForContainer(containerSpec, podSandboxConfig).await?;

        info!("Create container log dir pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let _logDir = BuildContainerLogsDirectory(&pod, &containerSpec.name)?;

        info!("Generate container runtime config pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let containerConfig = self.generateContainerConfig(containerSpec, &imageRef).await?;

        info!("Call runtime to create container pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let runtimeContainer = RUNTIME_MGR.CreateContainer(
            &self.pod.RuntimePod().sandbox.id.clone(), 
            Some(containerConfig), 
            Some(podSandboxConfig.clone()),
        ).await?;

        return Ok(runtimeContainer);
    }

    pub async fn Send(&self, msg: NodeAgentMsg) -> Result<()> {
        match self.msgChannTx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_) => {
                return Err(Error::CommonError(format!("PodAgent {} send message fail", self.pod.PodId())))
            }
        }
    }

    pub async fn generateContainerConfig(&self, container: &k8s::Container, imageRef: &cri::Image) -> Result<cri::ContainerConfig> {
        let pod = self.pod.Pod();
        BuildContainerLogsDirectory(&pod, &container.name)?;

        let containerLogsPath = ContainerLogFileName(&container.name, 0);
        let mut podIP = "".to_string();
        if self.pod.RuntimePod().IPs.len() > 0 {
            podIP = self.pod.RuntimePod().IPs[0].to_string();
        }

        let envs = MakeEnvironmentVariables(&pod, container, &k8s::ConfigMap::default(), &Vec::new(), &podIP, &self.pod.RuntimePod().IPs)?;
        
        let mut commands = Vec::new();
        if container.command.is_some() {
            for v in container.command.as_ref().unwrap() {
                let mut cmd = v.to_string();
                for e in &envs {
                    let oldv = format!("$({})", &e.name);
                    cmd = cmd.replace(&oldv, &e.value);
                }
                commands.push(cmd);
            }
        }

        let mut args = Vec::new();
        if container.args.is_some() {
            for v in container.args.as_ref().unwrap() {
                let mut arg = v.to_string();
                for e in &envs {
                    let oldv = format!("$({})", &e.name);
                    arg = arg.replace(&oldv, &e.value);
                }
                args.push(arg);
            }
        }
        
        let mut config = cri::ContainerConfig {
            metadata: Some(cri::ContainerMetadata {
                name: container.name.clone(),
                ..Default::default()
            }),
            image: Some(cri::ImageSpec {
                image: imageRef.id.clone(),
                ..Default::default()
            }),
            command: commands,
            args: args,
            working_dir: container.working_dir.as_deref().unwrap_or("").to_string(),
            labels: NewContainerLabels(container, &pod),
            annotations: NewContainerAnnotations(container, &pod, 0, &BTreeMap::new()),
            log_path: containerLogsPath,
            stdin: *container.stdin.as_ref().unwrap(),
            stdin_once: *container.stdin_once.as_ref().unwrap(),
            ..Default::default()
        };

        let uid = match &imageRef.uid {
            None => None,
            Some(v) => Some(v.value)
        };

        let username = imageRef.username.clone();
        generateLinuxContainerConfig(&self.nodeConfig, container, &pod, uid, &username, true);

        let mut criEnvs = Vec::with_capacity(envs.len());
        for env in &envs {
            criEnvs.push(cri::KeyValue {
                key: env.name.clone(),
                value: env.value.clone(),
            })
        }

        config.envs = criEnvs;

        return Ok(config)
    }

    pub async fn TerminateContainer(&self, container: &QuarkContainer) -> Result<()> {
        info!("Terminate container and remove it pod {} container {}", self.pod.PodId(), &container.lock().unwrap().spec.name.clone());
        
        RUNTIME_MGR.TerminateContainer(&container.lock().unwrap().runtimeContainer.id.clone()).await?;
        return Ok(())
    }
}

pub const DEFAULT_POD_LOGS_ROOT_PATH : &str = "/var/log/pods";

pub fn BuildContainerLogsDirectory(pod: &k8s::Pod, containerName: &str) -> Result<String> {
    let namespace = K8SUtil::Namespace(&pod.metadata);
    let name = K8SUtil::Name(&pod.metadata);
    let uid = K8SUtil::Uid(&pod.metadata);

    let podPath = Path::new(DEFAULT_POD_LOGS_ROOT_PATH)
        .join(namespace)
        .join(name)
        .join(uid);

    let containerpath = podPath
        .join(containerName)
        .into_os_string()
        .into_string()
        .unwrap();

    if !Path::new(&containerpath).exists() {
        fs::create_dir_all(&containerpath)?;
        let perms = Permissions::from_mode(0o755);
        fs::set_permissions(&containerpath, perms)?;
    }

    return Ok(containerpath)
}

pub fn BuildPodLogsDirectory(namespace: &str, podName: &str, uid: &str) -> Result<String> {
    //let podPath = format!("{}/{}/{}/{}", DEFAULT_POD_LOGS_ROOT_PATH, namespace, podName, uid);
    let podPath = Path::new(DEFAULT_POD_LOGS_ROOT_PATH)
        .join(namespace)
        .join(podName)
        .join(uid);

    if !podPath.exists() {
        fs::create_dir_all(&podPath)?;
        let perms = Permissions::from_mode(0o755);
        fs::set_permissions(&podPath, perms)?;
    }

    return Ok(podPath.into_os_string().into_string().unwrap());
}

pub fn ContainerLogFileName(containerName: &str, restartCount: usize) -> String {
    let path = Path::new(containerName).join(&format!("{}.log", restartCount));
    return path.into_os_string().into_string().unwrap();
} 
