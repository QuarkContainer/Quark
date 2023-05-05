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

use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use tokio::sync::mpsc;
use tokio::time;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use tokio::sync::Notify;
use core::ops::Deref;
use core::time::Duration;

//use qobjs::core_types::LifecycleHandler;
use k8s_openapi::api::core::v1 as k8s;
use qobjs::pb_gen::v1alpha2 as cri;
use qobjs::runtime_types::*;
use qobjs::common::*;
use crate::RUNTIME_MGR;
use crate::nm_svc::*;

pub struct PodContainerAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,
    pub started: AtomicBool,
    pub pod: QuarkPod,
    pub container: QuarkContainer,

    pub supervisor: mpsc::Sender<NodeAgentMsg>,
    pub agentChann: mpsc::Sender<NodeAgentMsg>,
}

#[derive(Clone)]
pub struct PodContainerAgent(Arc<PodContainerAgentInner>);

impl Deref for PodContainerAgent {
    type Target = Arc<PodContainerAgentInner>;

    fn deref(&self) -> &Arc<PodContainerAgentInner> {
        &self.0
    }
}

unsafe impl Send for PodContainerAgent{}

impl PodContainerAgent {
    pub async fn New(supervisor: &mpsc::Sender<NodeAgentMsg>, pod: &QuarkPod, container: &QuarkContainer) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<NodeAgentMsg>(30);
        
        let inner = PodContainerAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            started: AtomicBool::new(false),
            pod: pod.clone(),
            container: container.clone(),
            supervisor: supervisor.clone(),
            agentChann: tx,
        };
        let agent = Self(Arc::new(inner));

        let agent1 = agent.clone();
        tokio::spawn(async move {
            agent1.Process(rx).await.unwrap();
        });


        return Ok(agent);
    }

    pub async fn Start(&self) -> Result<()> {
        let hasStatus: bool = self.container.lock().unwrap().containerStatus.is_some();

        if !hasStatus {
            self.container.SetState(RuntimeContainerState::Created);
            self.Notify(NodeAgentMsg::PodContainerCreated (
                PodContainerCreated {
                    pod: self.pod.clone(),
                    container: self.container.clone(),
                }
            )).await;

            self.StartContainer().await?;
        } else {
            self.started.store(true, Ordering::SeqCst);
            
        }

        return Ok(())
    }

    pub async fn CreateContainer() -> Result<()> {
        return Ok(())
    }

    pub fn SendMsg(&self, msg: NodeAgentMsg) {
        self.agentChann.try_send(msg).unwrap();
    } 

    pub async fn Process(&self, mut rx: mpsc::Receiver<NodeAgentMsg>) -> Result<()> {
        let mut interval = time::interval(time::Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                _ = interval.tick() => {
                    if !self.started.load(Ordering::Relaxed) {
                        continue;
                    }
                    
                    let containerId = self.container.lock().unwrap().runtimeContainer.id.clone();
                    let status = crate::RUNTIME_MGR.get().unwrap().GetContainerStatus(&containerId).await?;
                    self.OnContainerProbeResult(&status).await;
                }
                msg = rx.recv() => {
                    if let Some(msg) = msg {
                        self.ContainerHandler(msg).await?;
                    } else {
                        break;
                    }
                }
            }
        }
        
        return Ok(())
    }

    pub async fn ContainerHandler(&self, msg: NodeAgentMsg) -> Result<()> {
        match &msg {
            NodeAgentMsg::PodContainerStopping(msg) => {
                self.StopContainer(msg.gracePeriod).await?;
            }
            NodeAgentMsg::PodContainerStarting(_) => {
                self.StartContainer().await?;
            }
            _ => {
                panic!("ContainerHandler fail with unexpected msg {:?}", msg);
            }
        }

        return Ok(())
    }

    pub fn ContainerName(&self) -> String {
        return self.container.ContainerName();
    }

    pub async fn OnContainerProbeResult(&self, result: &cri::ContainerStatus) {
        self.container.lock().unwrap().containerStatus = Some(result.clone());
        let result = Some(result.clone());
        let initContainer = self.container.lock().unwrap().initContainer;
        if ContainerExit(&result) && !initContainer {
            info!("Container exit pod {} container {} exit code {} finished at {:?}", 
                self.pod.PodId(), self.container.ContainerName(), result.as_ref().unwrap().exit_code, result.as_ref().unwrap().finished_at);
            self.OnContainerFailed().await;
        }

        let hasReadinessProbe = self.container.lock().unwrap().spec.readiness_probe.is_some();
        if !hasReadinessProbe && ContainerRunning(&result) {
            if self.container.State() == RuntimeContainerState::Started {
                self.OnContainerReady().await;
            }
        }
    }

    pub async fn OnContainerFailed(&self) {
        info!("Container failed pod {} containerName {}", self.pod.PodId(), self.container.ContainerName());
        self.Notify(NodeAgentMsg::PodContainerFailed(
            PodContainerFailed {
                pod: self.pod.clone(),
                container: self.container.clone(),
            }
        )).await;
    }

    pub async fn OnContainerStarted(&self) -> Result<()> {
        if self.container.State() == RuntimeContainerState::Created {
            self.container.SetState(RuntimeContainerState::Started);
            self.Notify(NodeAgentMsg::PodContainerStarted(
                PodContainerStarted {
                    pod: self.pod.clone(),
                    container: self.container.clone(),
                }
            )).await;
        }

        if self.InStoppingProcess() {
            return Ok(())
        }

        let initContainer = self.container.lock().unwrap().initContainer;
        if !initContainer {
            // todo: start liveness and readyness probe

            info!("Run post start lifecycle handler pod {} containerName {}", self.pod.PodId(), self.container.ContainerName());
            let lifecycle = self.container.lock().unwrap().spec.lifecycle.clone();
            if let Some(lifecycle) = lifecycle {
                if let Some(postStart) = lifecycle.post_start {   
                    self.RunLifecycleHandler(&self.pod, &self.container, &postStart).await?;
                }
            }
        }

        return Ok(())
    }

    pub async fn OnContainerReady(&self) {
        if !self.InStoppingProcess() {
            self.container.SetState(RuntimeContainerState::Running);
            info!("Container ready pod {} containerName {}", self.pod.PodId(), self.container.ContainerName());
            self.Notify(NodeAgentMsg::PodContainerReady(
                PodContainerReady {
                    pod: self.pod.clone(),
                    container: self.container.clone(),
                }
            )).await;
        }
    }

    pub fn InStoppingProcess(&self) -> bool {
        let state = self.container.State();
        return state == RuntimeContainerState::Stopping 
            || state == RuntimeContainerState::Stopped
            || state == RuntimeContainerState::Terminated
            || state == RuntimeContainerState::Terminating;
     
    }

    pub async fn StartContainer(&self) -> Result<()> {
        info!("Start container pod {} container {}", self.pod.PodId(), self.container.lock().unwrap().spec.name);
        let containerId = self.container.lock().unwrap().runtimeContainer.id.clone();
        RUNTIME_MGR.get().unwrap().StartContainer(&containerId).await?;

        // todo: starup probe
        // no startup probe, assume it started, run container post started hook
        self.OnContainerStarted().await?;
        self.started.store(true, Ordering::SeqCst);
        return Ok(())
    }

    pub async fn StopContainer(&self, dur: Duration) -> Result<()> {
        if self.container.State() == RuntimeContainerState::Running {
            let lifecycle = self.container.lock().unwrap().spec.lifecycle.clone();
            if let Some(lifecycle) = lifecycle {
                if let Some(handler) = &lifecycle.pre_stop {
                    info!("Running pre stop lifecycle handler pod {} container {}", self.pod.PodId(), self.container.lock().unwrap().spec.name);
                    let future = self.RunHttpHandler(&self.pod, &self.container, &handler);
                    tokio::select! {
                        ret = future => {
                            match ret {
                                Ok(_) => (),
                                Err(e) => {
                                    info!("Pre stop lifecycle handler failed pod {} container {} error {:?}", 
                                        self.pod.PodId(), self.container.lock().unwrap().spec.name, e);
                    
                                }
                            }
                        }
                        _ = tokio::time::sleep(dur) => {

                        }
                    }
                }
            }
        }

        self.container.SetState(RuntimeContainerState::Stopping);

        self.closeNotify.notify_one();

        self.StopRuntimeContainer(dur).await?;
        self.container.SetState(RuntimeContainerState::Stopped);

        self.Notify(NodeAgentMsg::PodContainerStopped(
            PodContainerStopped {
                pod: self.pod.clone(),
                container: self.container.clone(),
            }
        )).await;

        return Ok(())
    }

    pub async fn StopRuntimeContainer(&self, dur: Duration) -> Result<()> {
        let containerId = self.container.lock().unwrap().runtimeContainer.id.clone();
        let status = RUNTIME_MGR.get().unwrap().GetContainerStatus(&containerId).await?;

        self.container.lock().unwrap().containerStatus = Some(status.clone());
        if ContainerExit(&Some(status.clone())) {
            return Ok(())
        }

        RUNTIME_MGR.get().unwrap().StopContainer(&containerId, dur).await?;
        let status = RUNTIME_MGR.get().unwrap().GetContainerStatus(&containerId).await?;
        self.container.lock().unwrap().containerStatus = Some(status);
        return Ok(())
    }

    pub fn Send(&self, msg: NodeAgentMsg) -> Result<()> {
        match self.agentChann.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(_) => {
                return Err(Error::CommonError(format!("PodContainerAgent {} send message fail", self.container.lock().unwrap().spec.name)))
            }
        }
    }

    pub async fn Notify(&self, msg: NodeAgentMsg) {
        self.supervisor.try_send(msg).unwrap();
    }


    pub async fn Stop(&self) {
        self.closeNotify.notify_one();
    }

    pub async fn RunLifecycleHandler(&self, pod: &QuarkPod, container: &QuarkContainer, handle: &k8s::LifecycleHandler) -> Result<String> {
        if let Some(exec) = &handle.exec {
            let containerid = container.lock().unwrap().runtimeContainer.id.clone();
            match crate::RUNTIME_MGR.get().unwrap().ExecCommand(&containerid, exec.command.as_ref().unwrap().to_vec(), 0).await {
                Err(e) => return Err(e),
                Ok((stdout, _stderr)) => {
                    let stdout = std::str::from_utf8(&stdout)?;
                    return Ok(stdout.to_string());
                }
            }
        }

        if let Some(_) = &handle.http_get {
            let msg = self.RunHttpHandler(pod, container, handle).await?;
            return Ok(msg);
        }

        return Err(Error::CommonError("Cannot run lifecycle handler as handler is unknown".to_string()));
    }

    pub async fn RunHttpHandler(&self, pod: &QuarkPod, container: &QuarkContainer, handle: &k8s::LifecycleHandler) -> Result<String> {
        let httpGet = handle.http_get.as_ref().unwrap();
        let host = match &httpGet.host {
            None => {
                if pod.RuntimePod().as_ref().unwrap().IPs.len() == 0 {
                    return Err(Error::CommonError(format!("failed to find container ip")));
                }
                pod.RuntimePod().as_ref().unwrap().IPs[0].clone()
            }
            Some(h) => h.to_string(),
        };

        let port = match &httpGet.port {
            IntOrString::Int(v) => *v,
            IntOrString::String(s) => {
                if s.len() == 0 {
                    80
                } else {
                    ResolvePort(&s, &container.lock().unwrap().spec)?
                }
            }
        };

        let url = format!("http://{}:{}/{:?}", host, port, &httpGet.path.as_ref().unwrap());
        let resp = reqwest::get(url).await?.text().await?;
        return Ok(resp);
    }
}

pub fn ResolvePort(portStr: &str, container: &k8s::Container) -> Result<i32> {
    match portStr.parse::<i32>() {
        Ok(p) => return Ok(p),
        Err(_) => (),
    };

    let ports = match &container.ports {
        None => return Err(Error::CommonError(format!("couldn't find port: {:?} in {:?}", portStr, container))),
        Some(p) => p
    };

    for portSpec in ports {
        if Some(portStr.to_string()) == portSpec.name {
            return Ok(portSpec.container_port)
        }
    }

    return Err(Error::CommonError(format!("couldn't find port: {:?} in {:?}", portStr, container)));
}