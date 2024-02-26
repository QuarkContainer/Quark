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
use std::collections::HashMap;
use std::os::unix::prelude::PermissionsExt;
use std::path::Path;
use std::fs;
use std::fs::Permissions;
use std::sync::Mutex;
use std::time::Duration;
use core::ops::Deref;
use std::time::SystemTime;

use qshare::crictl::DnsConfig;
use qshare::node::*;
use tokio::sync::{mpsc, Notify};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use tokio::time;

use qshare::k8s;
            
use qshare::common::*;
use qshare::crictl;
use qshare::config::*;
use qshare::types::*;

use super::IMAGE_MGR;
use super::RUNTIME_MGR;

use super::pm_msg::*;
use super::qpod::*;
use super::qcontainer::*;
use super::runtime::k8s_helper::*;
use super::runtime::k8s_labels::*;
use super::runtime::k8s_util::*;
use super::container_agent::PodContainerAgent;

// The container runtime default profile should be used.
pub const SecurityProfile_RuntimeDefault: i32 = 0;
// Disable the feature for the sandbox or the container.
pub const SecurityProfile_Unconfined: i32 = 1;
// A pre-defined profile on the node should be used.
pub const SecurityProfile_Localhost: i32 = 2;

pub fn ValidatePodSpec(_pod: &k8s::Pod) -> Result<()> {
    return Ok(());
}

pub fn ValidateConfigMap(_configMap: &k8s::ConfigMap) -> Result<()> {
    return Ok(());
}

pub fn ValidateSecret(_secret: &k8s::Secret) -> Result<()> {
    return Ok(());
}

pub struct PodAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub pod: QuarkPod,
    pub supervisor: mpsc::Sender<NodeAgentMsg>, 
    pub agentChann: mpsc::Sender<NodeAgentMsg>,
    pub agentRx: Mutex<Option<mpsc::Receiver<NodeAgentMsg>>>,
    pub nodeConfig: NodeConfiguration,
    pub containers: Mutex<BTreeMap<String, PodContainerAgent>>,
}

pub enum PodType {
    Normal,
    Python(String)
}

#[derive(Clone)]
pub struct PodAgent(pub Arc<PodAgentInner>);

impl Deref for PodAgent {
    type Target = Arc<PodAgentInner>;

    fn deref(&self) -> &Arc<PodAgentInner> {
        &self.0
    }
}

impl PodAgent {
    pub fn New(supervisor: mpsc::Sender<NodeAgentMsg>, pod: &QuarkPod, nodeConfig: &NodeConfiguration) -> Self {
        let (tx, rx) = mpsc::channel::<NodeAgentMsg>(30);
        
        let inner = PodAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            pod: pod.clone(),
            supervisor: supervisor,
            agentChann: tx,
            agentRx: Mutex::new(Some(rx)),
            nodeConfig: nodeConfig.clone(),
            containers: Mutex::new(BTreeMap::new()),
        };

        return PodAgent(Arc::new(inner));
    }

    pub async fn CreatePod(&self) -> Result<()> {
        let pod = self.pod.Pod();
        let podId = self.pod.PodId();
        info!("Create Pod Cgroup pod {}", &podId);
        /*
        // Create Cgroups for the pod and apply resource parameters
        klog.InfoS("Create Pod Cgroup", QUARK_POD, types.UniquePodName(a.pod))
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
                klog.ErrorS(err, "Failed to create pod cgroup ", QUARK_POD, types.UniquePodName(a.pod))
                return err
            }
            // call update qos to make sure cgroup manager internal state update to date
            pcm.UpdateQOSCgroups()
        }
        */

        // let podType = match pod.read().unwrap().metadata.annotations.as_ref().unwrap().get(AnnotationFuncPodPackageType) {
        //     None => PodType::Normal,
        //     Some(t) => {
        //         if t == "python" {
        //             match pod.read().unwrap().metadata.annotations.as_ref().unwrap().get(AnnotationFuncPodPyPackageId) {
        //                 None => panic!("can't get annotation {} for python type package", AnnotationFuncPodPyPackageId),
        //                 Some(id) => PodType::Python(id.to_owned()),
        //             }
        //         } else {
        //             error!("get unknow package type {}", t);
        //             PodType::Normal
        //         }
        //     }
        // };

        info!("Make Pod data dirs pod {}", &podId);
        MakePodDataDir(&self.nodeConfig.RootPath, &pod.read().unwrap())?;
        info!("Make Pod log dirs pod {}", &podId);
        MakePodLogDir(&self.nodeConfig.RootPath, &pod.read().unwrap())?;
        /*
        // TODO, Try to attach and mount volumes into pod, mounted vol will be mounted into container later, do not support volume for now
        klog.InfoS("Prepare pod volumes", QUARK_POD, types.UniquePodName(a.pod))
        if err := a.dependencies.VolumeManager.WaitForAttachAndMount(pod); err != nil {
            klog.ErrorS(err, "Unable to attach or mount volumes for pod; skipping pod", QUARK_POD, types.UniquePodName(a.pod))
            return err
        }
         */

        let namespace = pod.read().unwrap().namespace.clone();

        info!("Create pod sandbox {}", &podId);
        let runtimePod = Arc::new(self.CreatePodSandbox().await?);
        info!("Get pod sandbox {}, sandbox {:?}", &podId, &runtimePod);

        self.pod.lock().unwrap().runtimePod = Some(runtimePod.clone());
        info!("Start pod init containers pod {}", &podId);
        let containers = pod.read().unwrap().init_containers.to_vec();
            for c in &containers {
                let runtimeContainer = self.CreateContainer(
                    runtimePod.sandboxConfig.as_ref().unwrap(), 
                    c, 
                    &namespace
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
                let containerAgent = PodContainerAgent::New(&self.agentChann, &self.pod, &container).await?;
                self.containers.lock().unwrap().insert(c.name.clone(), containerAgent.clone());
                containerAgent.Start().await?;
            }

        info!("Start pod containers pod {}", &podId);
        let containers = pod.read().unwrap().containers.to_vec();
        for c in &containers {
            let runtimeContainer = self.CreateContainer(
                runtimePod.sandboxConfig.as_ref().unwrap(), 
                c, 
                &namespace
            ).await?;
            let inner = QuarkContainerInner {
                state: RuntimeContainerState::Creating,
                initContainer: false,
                spec: c.clone(),
                runtimeContainer: runtimeContainer,
                containerStatus: None,
            };
            let container = QuarkContainer(Arc::new(Mutex::new(inner)));
            self.pod.lock().unwrap().containers.insert(c.name.clone(), container.clone());
            let containerAgent = PodContainerAgent::New(&self.agentChann, &self.pod, &container).await?;
            self.containers.lock().unwrap().insert(c.name.clone(), containerAgent.clone());
            containerAgent.Start().await?;
        }

        return Ok(())
    }

    pub const HouseKeepingPeriod : Duration = Duration::from_secs(5);

    pub async fn Stop(&self) -> Result<()> {
        self.stop.store(true, Ordering::SeqCst);
        let containers: Vec<_> = self.containers.lock().unwrap().values().cloned().collect();
        for c in &containers {
            c.Stop().await;
        }

        self.closeNotify.notify_waiters();
        return Ok(())
    }

    pub fn Start(&self) -> Result<()> {
        info!("Pod actor started {}", self.pod.PodId());
        let rx = self.agentRx.lock().unwrap().take().unwrap();
        let clone = self.clone();
        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return Ok(());
    }

    pub async fn Process(&self, mut rx: mpsc::Receiver<NodeAgentMsg>) -> Result<()> {
        let mut interval = time::interval(Self::HouseKeepingPeriod);
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                _ = interval.tick() => {
                    self.Send(NodeAgentMsg::HouseKeeping)?;
                }
                msg = rx.recv() => {
                    if let Some(msg) = msg {
                        match self.PodHandler(msg).await {
                            Ok(()) => (),
                            Err(e) => {
                                error!("PodHandler get failure {:?}", e);
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        
        return Ok(())
    }

    pub async fn PodHandler(&self, msg: NodeAgentMsg) -> Result<()> {
        let oldPodState = self.pod.PodState();

        let mut ret = match msg {
            NodeAgentMsg::PodCreate(_msg) => {
                self.Create().await
            }
            NodeAgentMsg::PodTerminate => {
                self.Terminate(false).await
            }
            NodeAgentMsg::PodContainerCreated(msg) => {
                self.OnPodContainerCreated(msg).await
            }
            NodeAgentMsg::PodContainerStarted(msg) => {
                self.OnPodContainerStarted(msg).await
            }
            NodeAgentMsg::PodContainerReady(msg) => {
                self.OnPodContainerReady(msg).await
            }
            NodeAgentMsg::PodContainerStopped(msg) => {
                self.OnPodContainerStopped(msg).await
            }
            NodeAgentMsg::PodContainerFailed(msg) => {
                self.OnPodContainerFailed(msg).await
            }
            NodeAgentMsg::HouseKeeping => {
                //self.PodHouseKeeping().await
                Ok(())
            }

            _ => Ok(())
        };

        self.pod.SetPodStatus(None);
        if ret.is_ok() {
            if self.pod.PodState() == PodState::Terminated {
                ret = self.Cleanup().await;
                self.pod.SetPodStatus(None);
            }
        }
        
        if self.pod.PodState() != PodState::Deleted  
            && (oldPodState != self.pod.PodState() || self.pod.PodState() == PodState::Cleanup) {
            self.Notify(NodeAgentMsg::PodStatusChange(
                PodStatusChange {
                    pod: self.pod.clone(),
                }
            ));
        }

        return ret;
    }

    pub async fn PodHouseKeeping(&self) -> Result<()> {
        let podState = self.pod.PodState();
        info!("House keeping  pod {} podState {:?}", self.pod.PodId(), &podState);

        if podState == PodState::Terminated || podState == PodState::Running {
            return Ok(())
        }

        let runtimePod = self.pod.RuntimePod();

        if podState == PodState::Terminating || podState == PodState::Failed {
            self.Terminate(true).await?;
        } else if runtimePod.is_none() {
            self.pod.SetPodState(PodState::Terminated);
        } else if runtimePod.is_some() && runtimePod.as_ref().unwrap().sandbox.is_none() {
            let sandbox = RUNTIME_MGR.get().unwrap().GetPodSandbox(&runtimePod.as_ref().unwrap().id).await?;
            match sandbox {
                None => self.pod.SetPodState(PodState::Terminated),
                Some(_ps) => {
                    self.Terminate(true).await?;
                }
            }
        } 

        return Ok(())
        
    }

    pub async fn OnPodContainerCreated(&self, msg: PodContainerCreated) -> Result<()> {
        let pod = &msg.pod;
        let container = &msg.container;

        info!("pod Container created pod {} container {}", pod.PodId(), container.ContainerName());
        let podState = pod.PodState();
        if podState == PodState::Terminating || podState == PodState::Terminated {
            info!("Pod Container created after when pod is in terminating state pod {} container {}", pod.PodId(), container.ContainerName());
            self.TerminateContainer(container).await?;
        }
        
        // TODO, update pod cpu, memory resource usage
	    return Ok(())
    }

    pub async fn OnPodContainerStarted(&self, msg: PodContainerStarted) -> Result<()> {
        let pod = &msg.pod;
        let container = &msg.container;

        info!("pod Container started pod {} container {}", pod.PodId(), container.ContainerName());
        
        // TODO, update pod cpu, memory resource usage
	    return Ok(())
    }

    pub async fn OnPodContainerFailed(&self, msg: PodContainerFailed) -> Result<()> {
        let pod = &msg.pod;
        let container = &msg.container;

        info!("OnPodContainerFailed1: pod Container Failed pod {} container {}", pod.PodId(), container.ContainerName());
        
        // TODO, update pod cpu, memory resource usage
	    return self.HandlePodContainerExit(pod, container).await;
    }

    pub async fn OnPodContainerStopped(&self, msg: PodContainerStopped) -> Result<()> {
        let pod = &msg.pod;
        let container = &msg.container;

        info!("pod Container stopped pod {} container {}", pod.PodId(), container.ContainerName());
        
        // TODO, release cpu, memory resource stat usage
	    return self.HandlePodContainerExit(pod, container).await;
    }

    // when a container report it's ready, set pod to running state if all container are ready and init containers exit normally
    pub async fn OnPodContainerReady(&self, msg: PodContainerReady) -> Result<()> {
        let pod = &msg.pod;
        let container = &msg.container;

        info!("pod Container is ready pod {} container {}", pod.PodId(), container.ContainerName());
        let mut allContainerReady = true;
        let containers : Vec<_> = self.pod.lock().unwrap().containers.values().cloned().collect();
        for c in containers {
            if c.InitContainer() {
                allContainerReady = allContainerReady && c.ContainerExit();
            } else {
                allContainerReady = allContainerReady && c.ContainerRunning();
            }
        }

        if allContainerReady {
            pod.SetPodState(PodState::Running);
        }
        
        return Ok(())
    }

    pub async fn HandlePodContainerExit(&self, _pod: &QuarkPod, container: &QuarkContainer) -> Result<()> {
        let containerName = container.ContainerName();
        let removed = self.containers.lock().unwrap().remove(&containerName);
        if let Some(container) = removed {
            container.Close()?;
        }
        if container.InitContainer() {
            if container.ContainerExitNormal() {
                // init container is expected to run to end
            } else if container.ContainerExitAbnormal() {
                // init container failed, terminate pod
                return self.Terminate(true).await;
            }
        } else {
            return self.Terminate(true).await;
        }

        return Ok(())
    }
    
    pub async fn Cleanup(&self) -> Result<()> {
        info!("Cleanup pod {}", self.pod.PodId());
        self.CleanupPod().await?;
        self.pod.SetPodState(PodState::Cleanup);
        return Ok(())
    }

    pub async fn Create(&self) -> Result<()> {
        info!("Creating pod {}", self.pod.PodId());
        if self.pod.PodInTerminating() {
            return Err(Error::CommonError(format!("Pod {} is being terminated or already terminated", self.pod.PodId())));
        }

        if self.pod.PodCreated() {
            return Err(Error::CommonError(format!("Pod {} is already created, NodeMgr is in sync", self.pod.PodId())));
        }

        self.pod.SetPodState(PodState::Creating);
        self.CreatePod().await?;

        self.pod.SetPodState(PodState::Created);

        return Ok(())
    }

    // terminate evacute session if there are live sessions, and terminate pod containers,
    // containers are notified to exit itself, and use onPodContainerStopped call back to get container status,
    // and recall this method to check if all container are finished, and finally set pod to terminted state
    // termiante method do no-op if pod state is already in terminating state unless forece retry is true
    pub async fn Terminate(&self, forceTerminatePod: bool) -> Result<()> {
        info!("Stopping container and terminating pod {} state {:?} force {}", 
            self.pod.PodId(), self.pod.PodState(), forceTerminatePod);

        let pod = self.pod.Pod();
        let hasDeleted = pod.read().unwrap().deletion_timestamp.is_some();
        let utc = SystemTime::now();
        if !hasDeleted {
            pod.write().unwrap().deletion_timestamp = Some(utc);
        }

        if self.pod.PodState() == PodState::Terminated || self.pod.PodState() == PodState::Cleanup {
            return Ok(())
        }

        self.pod.SetPodState(PodState::Terminating);
        let mut gracefulPeriodSeconds = 0;
        if let Some(period) = pod.read().unwrap().deletion_grace_period_seconds {
            gracefulPeriodSeconds = period;
        }

        self.TerminatePod(Duration::from_secs(gracefulPeriodSeconds as _), forceTerminatePod).await?;
        self.pod.SetPodState(PodState::Terminated);
        
        return Ok(())
    }

    pub const DefaultStopContainerGracePeriod : Duration = Duration::from_secs(30);

    pub async fn TerminatePod(&self, gracefulPeriod: Duration, force: bool) -> Result<()> {
        let podState = self.pod.PodState();
        if podState != PodState::Terminating && podState != PodState::Failed {
            return Err(Error::CommonError(format!("Pod {} is in not terminatable state {:?}", self.pod.PodId(), podState)));
        }

        let mut gracefulPeriod = gracefulPeriod;
        
        let containers : Vec<PodContainerAgent> = self.containers.lock().unwrap().values().cloned().collect();
        for c in &containers {
            let status = c.container.lock().unwrap().containerStatus.clone();
            if let Some(status) = status {
                if !ContainerExit(&Some(status)) && !force {
                    if gracefulPeriod.is_zero() {
                        gracefulPeriod = Self::DefaultStopContainerGracePeriod;
                    }

                    c.SendMsg(NodeAgentMsg::PodContainerStopping(
                        PodContainerStopping {
                            pod: self.pod.clone(),
                            container: c.container.clone(),
                            gracePeriod: gracefulPeriod.clone(),
                        }
                    ));

                    let mut pollingInterval = time::interval(time::Duration::from_secs(1));
                    let mut timeoutInternval = time::interval(gracefulPeriod);
                    loop {
                        tokio::select! {
                            _ = pollingInterval.tick() => {
                                let status = c.container.lock().unwrap().containerStatus.clone();
                                if status.is_none() || ContainerExit(&status) {
                                    break;
                                }
                            }
                            _ = timeoutInternval.tick() => {
                                break;
                            }
                        }
                    }
                }

                let qc = c.container.clone();
                self.TerminateContainer(&qc).await?;
            }
        }

        return Ok(());
    } 

    pub async fn CleanupPod(&self) -> Result<()> {
        info!("Remove Pod sandbox pod {}", self.pod.PodId());
        let runtimePod = self.pod.RuntimePod();

        if let Some(runtimePod) = runtimePod {
            if runtimePod.sandbox.is_some() {
                self.RemovePodSandbox(&runtimePod.sandbox.as_ref().unwrap().id, &runtimePod.sandboxConfig.as_ref().unwrap()).await?;
            }
        }
        
        /*
        	// TODO, Try to unmount volumes into pod, mounted vol will be detached by volumemanager if volume not required anymore
            klog.InfoS("Unmount Pod volume", QUARK_POD, types.UniquePodName(a.pod))
            if err := a.dependencies.VolumeManager.UnmountPodVolume(pod); err != nil {
                klog.ErrorS(err, "Unable to unmount volumes for pod", QUARK_POD, types.UniquePodName(a.pod))
                return err
	        }
         */
        
        // Remove data directories for the pod
	    info!("Remove Pod Data dirs pod {}", self.pod.PodId());
        let pod = self.pod.Pod();
        CleanupPodDataDirs(&self.nodeConfig.RootPath, &pod.read().unwrap())?;

        info!("Remove Pod log dirs pod {}", self.pod.PodId());
        CleanupPodLogDir(&self.nodeConfig.RootPath, &pod.read().unwrap())?;
            
        /*
        	// remove cgroups for the pod and apply resource parameters
            klog.InfoS("Remove Pod Cgroup", QUARK_POD, types.UniquePodName(a.pod))
            pcm := a.dependencies.QosManager
            if pcm.IsPodCgroupExist(pod) {
                err := pcm.DeletePodCgroup(pod)
                if err != nil {
                    return fmt.Errorf("Cgroup deletion failed for pod %v, err %v", pod.UID, err)
                }
                // call update qos to make sure cgroup manager internal state update to date
                pcm.UpdateQOSCgroups()
            } else {
                // call update qos to make sure cgroup manager internal state update to date
                pcm.UpdateQOSCgroups()
            }
        */
        
        return Ok(())
    }

    pub async fn RemovePodSandbox(&self, podSandboxId: &str, podSandboxConfig: &crictl::PodSandboxConfig) -> Result<()> {
        RUNTIME_MGR.get().unwrap().TerminatePod(podSandboxId, Vec::new()).await?;
        fs::remove_dir_all(&podSandboxConfig.log_directory)?;
        return Ok(())
    }

    pub fn Notify(&self, msg: NodeAgentMsg) {
        self.supervisor.try_send(msg).unwrap();
    } 

    pub fn NotifyContainer(&self, containerName: &str, msg: NodeAgentMsg) -> Result<()> {
        let container = match self.containers.lock().unwrap().get(containerName) {
            None => {
                return Err(Error::CommonError(format!("Container with name {} not found", containerName)));
            }
            Some(c) => c.clone()
        };

        container.SendMsg(msg);
        return Ok(())
    }

    pub async fn CreatePodSandbox(&self) -> Result<RuntimePod> {
        let podsandboxConfig = self.GeneratePodSandboxConfig()?;
        let podIp = self.pod.PodId();

        info!("Make pod log dir for pod {} path is {}", &podIp, &podsandboxConfig.log_directory);
        fs::create_dir_all(&podsandboxConfig.log_directory)?;
        let perms = Permissions::from_mode(0o755);
        fs::set_permissions(&podsandboxConfig.log_directory, perms)?;

        let runtime = self.pod.Pod().read().unwrap().runtime_class_name.clone();
        let runtimehandler = match runtime {
            None => self.nodeConfig.RuntimeHandler.clone(),
            Some(runtime) => runtime.clone(),
        };
        
        info!("Call runtime to create sandbox runtimehandler is {} pod {} sandboxConfig is {:?}", &runtimehandler, &podIp, &podsandboxConfig);

        let runtimePod = RUNTIME_MGR.get().unwrap().CreateSandbox(Some(podsandboxConfig), &runtimehandler).await?;

        return Ok(runtimePod)
    }

    // generatePodSandboxConfig generates pod sandbox config .
    pub fn GeneratePodSandboxConfig(&self) -> Result<crictl::PodSandboxConfig> {
        // nodeagent will expect nodemgr populate most of pod spec before send it
	    // it will not calulate hostname, all these staff
        let pod = self.pod.Pod();
        let pod = pod.read().unwrap();

        let namespaceSearch = format!("{}.svc.cluster.local", &pod.namespace);
        
        let podUID = pod.uid.clone();
        let mut podSandboxConfig = crictl::PodSandboxConfig {
            metadata: Some(crictl::PodSandboxMetadata {
                name: pod.name.clone(),
                namespace: pod.namespace.clone(),
                uid: podUID,
                ..Default::default()
            }),
            labels: NewPodLabels(&pod),
            annotations: NewPodAnnotations(&pod),
            dns_config: Some(DnsConfig {
                servers: vec!["127.0.0.53".to_string()],
                searches: vec![namespaceSearch],
                options: vec!["ndots:2".to_string(), "edns0".to_string()]
            }),
            ..Default::default()
        };

        if !IsHostNetworkPod(&pod) {
            podSandboxConfig.hostname = pod.host_name.clone();
        }

        podSandboxConfig.log_directory = GetPodLogDir(
            DefaultPodLogsRootPath,
            &pod.namespace, 
            &pod.name,  
            &pod.uid,
        );

        let mut podMapping = Vec::new();
        for c in &pod.containers {
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

    pub fn GeneratePodSandboxLinuxConfig(&self) -> Result<crictl::LinuxPodSandboxConfig> {
        // let pod = self.pod.Pod();
        // let pod = pod.read().unwrap();
        let lpsc = crictl::LinuxPodSandboxConfig {
            cgroup_parent: "".to_owned(),
            security_context: None,
            // security_context: Some(crictl::LinuxSandboxSecurityContext {
            //     privileged: HasPrivilegedContainer(pod.spec.as_ref().unwrap()),
            //     seccomp: Some(crictl::SecurityProfile {
            //         profile_type: SecurityProfile_RuntimeDefault,
            //         ..Default::default()
            //     }),
            //     ..Default::default()
            // }),
            ..Default::default()
        };

        //AddPodSecurityContext(&pod, &mut lpsc);
        return Ok(lpsc)
    }

    pub async fn CreateContainer(
        &self, 
        podSandboxConfig: &crictl::PodSandboxConfig,
        containerSpec: &ContainerDef,
        namespace: &str
    ) -> Result<RuntimeContainer> {
        info!("Pull image for container pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let pod = self.pod.Pod();
        let imageRef = IMAGE_MGR.get().unwrap().PullImageForContainer(&containerSpec.image, podSandboxConfig).await?;

        info!("Create container log dir pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let _logDir = BuildContainerLogsDirectory(&pod.read().unwrap(), &containerSpec.name)?;

        info!("Generate container runtime config pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let containerConfig = self.generateContainerConfig(containerSpec, &imageRef, namespace).await?;

        info!("Call runtime to create container pod {} container {}", self.pod.PodId(), &containerSpec.name);
        let runtimeContainer = RUNTIME_MGR.get().unwrap().CreateContainer(
            &self.pod.RuntimePod().as_ref().unwrap().sandbox.as_ref().unwrap().id.clone(), 
            Some(containerConfig), 
            Some(podSandboxConfig.clone()),
        ).await?;

        return Ok(runtimeContainer);
    }

    pub fn Send(&self, msg: NodeAgentMsg) -> Result<()> {
        match self.agentChann.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_) => {
                return Err(Error::CommonError(format!("PodAgent {} send message fail", self.pod.PodId())))
            }
        }
    }

    pub async fn generateContainerConfig(&self, container: &ContainerDef, imageRef: &crictl::Image, namespace: &str) -> Result<crictl::ContainerConfig> {
        let pod = self.pod.Pod();
        BuildContainerLogsDirectory(&pod.read().unwrap(), &container.name)?;

        let containerLogsPath = ContainerLogFileName(&container.name, 0);
        // let mut podIP = "".to_string();
        // if self.pod.RuntimePod().as_ref().unwrap().IPs.len() > 0 {
        //     podIP = self.pod.RuntimePod().as_ref().unwrap().IPs[0].to_string();
        // }

        let envs = MakeEnvironmentVariables(container)?;
        
        let mut commands = Vec::new();
        for v in &container.commands {
            let mut cmd = v.to_string();
            for e in &envs {
                let oldv = format!("$({})", &e.name);
                cmd = cmd.replace(&oldv, &e.value);
            }
            commands.push(cmd);
        }

        let mut args = Vec::new();
        for v in &container.args {
            let mut arg = v.to_string();
            for e in &envs {
                let oldv = format!("$({})", &e.name);
                arg = arg.replace(&oldv, &e.value);
            }
            args.push(arg);
        }
        
        let mut config = crictl::ContainerConfig {
            metadata: Some(crictl::ContainerMetadata {
                name: container.name.clone(),
                ..Default::default()
            }),
            image: Some(crictl::ImageSpec {
                image: imageRef.id.clone(),
                ..Default::default()
            }),
            command: commands,
            args: args,
            working_dir: container.working_dir.clone(),
            labels: NewContainerLabels(container, &pod),
            annotations: NewContainerAnnotations(container, &pod, 0, &BTreeMap::new()),
            //devices:
            mounts: MakeMounts(&pod, container, namespace).await?,
            log_path: containerLogsPath,
            stdin: container.stdin,
            stdin_once: container.stdin_once,
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
            criEnvs.push(crictl::KeyValue {
                key: env.name.clone(),
                value: env.value.clone(),
            })
        }

        config.envs = criEnvs;

        return Ok(config)
    }

    pub async fn TerminateContainer(&self, container: &QuarkContainer) -> Result<()> {
        info!("Terminate container and remove it pod {} container {}", self.pod.PodId(), &container.lock().unwrap().spec.name.clone());
        
        let id = container.lock().unwrap().runtimeContainer.id.clone();
        RUNTIME_MGR.get().unwrap().TerminateContainer(&id).await?;
        return Ok(())
    }
}

pub const DEFAULT_POD_LOGS_ROOT_PATH : &str = "/var/log/pods";

pub fn BuildContainerLogsDirectory(pod: &PodDef, containerName: &str) -> Result<String> {
    let namespace = &pod.namespace;
    let name = &pod.name;
    let uid = &pod.uid;

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


// determinePodSandboxIP determines the IP addresses of the given pod sandbox.
pub fn DeterminePodSandboxIPs(podNamespace: &str, podName: &str, podSandbox: &crictl::PodSandboxStatus) -> Vec<String> {
    let mut podIps = Vec::new();
    match &podSandbox.network {
        None => {
            info!("Pod Sandbox status doesn't have network information, cannot report IPs pod namespace {} name {}", podNamespace, podName);
            return podIps;
        }
        Some(network) => {
            // pick primary IP
            if network.ip.len() != 0 {
                podIps.push(network.ip.clone());
            }

            for ip in &network.additional_ips {
                podIps.push(ip.ip.clone());
            }
        }
    }

    return podIps;
}

pub fn AddPodSecurityContext(pod: &mut PodDef, lpsc: &mut crictl::LinuxPodSandboxConfig) {
    let mut sysctls = HashMap::new();
    if pod.security_context.is_none() {
        pod.security_context = Some(k8s::PodSecurityContext::default());
    }
    if let Some(sc) = &pod.security_context {
        if let Some(ctls) = &sc.sysctls {
            for c in ctls {
                sysctls.insert(c.name.clone(), c.value.clone());
            }
        }

        lpsc.sysctls = sysctls;

        if let Some(run_as_user) = sc.run_as_user {
            lpsc.security_context.as_mut().unwrap().run_as_user = Some(crictl::Int64Value {
                value: run_as_user,
            });
        }

        if let Some(run_as_group) = sc.run_as_group {
            lpsc.security_context.as_mut().unwrap().run_as_group = Some(crictl::Int64Value {
                value: run_as_group,
            });
        }

        lpsc.security_context.as_mut().unwrap().namespace_options = Some(NamespacesForPod(pod));

        if let Some(fsgroup) = sc.fs_group {
            lpsc.security_context.as_mut().unwrap().supplemental_groups.push(fsgroup);
        }

        if let Some(fsgroups) = &sc.supplemental_groups {
            for sg in fsgroups {
                lpsc.security_context.as_mut().unwrap().supplemental_groups.push(*sg);
            }
        }

        if let Some(opts) = &sc.se_linux_options {
            lpsc.security_context.as_mut().unwrap().selinux_options = Some(crictl::SeLinuxOption {
                user: opts.user.as_deref().unwrap_or("").to_string(),
                role: opts.role.as_deref().unwrap_or("").to_string(),
                r#type: opts.type_.as_deref().unwrap_or("").to_string(),
                level: opts.level.as_deref().unwrap_or("").to_string(),

            })
        }
    }

}
