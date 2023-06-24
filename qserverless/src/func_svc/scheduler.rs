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

use std::ops::Add;
use std::ops::Sub;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use core::ops::Deref;
use tokio::sync::Notify;
use tokio::sync::mpsc;

use qobjs::cacher_client::CacherClient;
use qobjs::k8s;
use qobjs::types::*;
use qobjs::common::*;

use crate::package::Package;

#[derive(Debug, Clone, Copy)]
pub struct Resource {
    pub mem: u64,
    pub cpu: u64,
}

impl Default for Resource {
    fn default() -> Self {
        return Self {
            mem: 1024 * 1024,
            cpu: 1000,
        }
    }
}

impl Add for Resource {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            mem: self.mem + other.mem,
            cpu: self.cpu + other.cpu,
        }
    }
}

impl Sub for Resource {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        //error!("Resource::sub {:?} - {:?}", &self, &other);
        Self {
            mem: self.mem - other.mem,
            cpu: self.cpu - other.cpu,
        }
    }
}

impl Resource {
    pub fn Fullfil(&self, req: &Self) -> bool {
        return req.mem <= self.mem && req.cpu <= self.cpu;
    }
}

pub enum SchedulerMsg {
    CreatePod(CreatePod),
    TerminatePod(TerminatePod),
}

#[derive(Debug)]
pub struct CreatePod {
    pub podName: String,
    pub package: Package,
}

#[derive(Debug)]
pub struct TerminatePod {
    pub namespace: String,
    pub podName: String,
}


#[derive(Debug)]
pub struct SchedulerInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub agentChann: mpsc::Sender<SchedulerMsg>,
    pub nodeMgrAddr: String
}

#[derive(Debug, Clone)]
pub struct Scheduler(Arc<SchedulerInner>);

impl Deref for Scheduler {
    type Target = Arc<SchedulerInner>;

    fn deref(&self) -> &Arc<SchedulerInner> {
        &self.0
    }
}

impl Scheduler {
    pub fn New(nodeMgrAddr: &str) -> Self {
        let nodeMgrAddr = &format!("http://{}", nodeMgrAddr);
        let (tx, rx) = mpsc::channel(30);
        let inner = SchedulerInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            agentChann: tx,
            nodeMgrAddr: nodeMgrAddr.to_string()
        };
        let ret = Scheduler(Arc::new(inner));
        let clone = ret.clone();
        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return ret;
    }

    pub fn Close(&self){
        self.closeNotify.notify_waiters();
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn SchedulePod(&self, podName: &str, package: &Package) -> Result<()> {
        let msg = CreatePod {
            podName: podName.to_string(),
            package: package.clone(),
        };
        match self.agentChann.try_send(SchedulerMsg::CreatePod(msg)) {
            Ok(()) => return Ok(()),
            Err(_) => return Err(Error::MpscSendFail),
        }
    }

    pub fn TerminatePod(&self, namespace: &str, podName: &str) -> Result<()> {
        let msg = TerminatePod {
            namespace: namespace.to_owned(),
            podName: podName.to_owned(),
        };
        match self.agentChann.try_send(SchedulerMsg::TerminatePod(msg)) {
            Ok(()) => return Ok(()),
            Err(_) => return Err(Error::MpscSendFail),
        }
    }

    pub async fn CreatePod(&self, client: &CacherClient, podName: &str, package: &Package) -> Result<()> {
        let mut pod = k8s::Pod::default();

        pod.metadata.namespace = Some(package.Namespace());
        pod.metadata.name = Some(podName.to_owned());
        let mut annotations = package.lock().unwrap().Annotations();
        annotations.insert(AnnotationFuncPodPackageName.to_owned(), package.Name());
        pod.metadata.annotations = Some(annotations);
        
        pod.spec = Some(package.PodSpec());
        for container in &mut pod.spec.as_mut().unwrap().containers {
            if container.env.is_none() {
                container.env = Some(Vec::new());
            }

            container.env.as_mut().unwrap().push(k8s::EnvVar {
                name: EnvVarNodeMgrPodId.to_string(),
                value: Some(podName.to_string()),
                ..Default::default()
            });

            container.env.as_mut().unwrap().push(k8s::EnvVar {
                name: EnvVarNodeMgrNamespace.to_string(),
                value: Some(package.Namespace()),
                ..Default::default()
            });

            container.env.as_mut().unwrap().push(k8s::EnvVar {
                name: EnvVarNodeMgrPackageId.to_string(),
                value: Some(package.Name()),
                ..Default::default()
            });

            container.env.as_mut().unwrap().push(k8s::EnvVar {
                name: EnvVarNodeAgentAddr.to_string(),
                value: Some(DefaultNodeAgentAddr.to_string()),
                ..Default::default()
            });
        }

        let podStr = serde_json::to_string(&pod)?;
        let dataObj = DataObject::NewFromK8sObj(QUARK_POD, &pod.metadata, podStr);
        client.Create(QUARK_POD, dataObj.Obj()).await?;
        return Ok(())
    }

    pub async fn Process(&self, rx: mpsc::Receiver<SchedulerMsg>) -> Result<()> {
        let mut rx = rx;
        let closeNotify = self.closeNotify.clone();
        let mut cacheClient;
        loop {
            loop {
                 match CacherClient::New(self.nodeMgrAddr.clone().into()).await {
                    Ok(c) => {
                        cacheClient = c;
                        break;
                    }
                    Err(e) => {
                        error!("Scheduler can't connect to node manager with error {:?}", e);
                    } 
                }

                let duration = Duration::from_secs(1);
                tokio::time::sleep(duration).await;
            }

            loop {
                tokio::select! {
                    _ = closeNotify.notified() => {
                        return Ok(());
                    }
                    msg = rx.recv() => {
                        let msg = match msg {
                            None => {
                                info!("schedule finish");
                                return Ok(());
                            }
                            Some(msg) => msg,
                        };

                        match msg {
                            SchedulerMsg::CreatePod(msg) => {
                                match self.CreatePod(&cacheClient, &msg.podName, &msg.package).await {
                                    Ok(()) => (),
                                    Err(_) => break,
                                }
                            }
                            SchedulerMsg::TerminatePod(msg) => {
                                match cacheClient.Delete(QUARK_POD, &msg.namespace, &msg.podName).await {
                                    Ok(_) => (),
                                    Err(_) => break,
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
