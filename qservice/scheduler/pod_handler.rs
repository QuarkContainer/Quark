// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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

use std::ops::Deref;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use qshare::node::PodState;
use tokio::sync::mpsc;
use tokio::sync::Notify;
use tokio::time;

use qshare::metastore::data_obj::EventType;
use qshare::na;
use qshare::na::Env;
use qshare::na::Kv;
use qshare::obj_mgr::func_mgr::*;

use qshare::common::*;
use qshare::metastore::data_obj::DeltaEvent;

use crate::OBJ_REPO;

#[derive(Debug, Clone)]
pub enum PodHandlerMsg {
    AskFuncPod(na::AskFuncPodReq),
    DisableFuncPod(na::DisableFuncPodReq),
}

#[derive(Debug)]
pub struct PodHandlerInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub fpChann: mpsc::Sender<DeltaEvent>,
    pub msgChann: mpsc::Sender<PodHandlerMsg>,

    pub rxChann: Mutex<Option<mpsc::Receiver<DeltaEvent>>>,
    pub msgRxChann: Mutex<Option<mpsc::Receiver<PodHandlerMsg>>>,

    pub nextWorkId: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct PodHandler(pub Arc<PodHandlerInner>);

impl Deref for PodHandler {
    type Target = Arc<PodHandlerInner>;

    fn deref(&self) -> &Arc<PodHandlerInner> {
        &self.0
    }
}

impl PodHandler {
    pub fn New() -> Self {
        let (tx, rx) = mpsc::channel::<DeltaEvent>(1000);
        let (mtx, mrx) = mpsc::channel::<PodHandlerMsg>(1000);

        let inner = PodHandlerInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            fpChann: tx,
            msgChann: mtx,

            rxChann: Mutex::new(Some(rx)),
            msgRxChann: Mutex::new(Some(mrx)),
            nextWorkId: AtomicU64::new(1),
        };

        let handler = Self(Arc::new(inner));
        return handler;
    }

    // wait for the cluster funcpod state fully updated
    pub const WAITING_PERIOD: Duration = Duration::from_secs(5);

    pub fn EnqEvent(&self, event: &DeltaEvent) -> Result<()> {
        self.fpChann.try_send(event.clone()).unwrap();
        return Ok(());
    }

    pub fn EnqMsg(&self, msg: &PodHandlerMsg) {
        self.msgChann.try_send(msg.clone()).unwrap();
    }

    pub async fn Process(&self) -> Result<()> {
        let mut rx = self.rxChann.lock().unwrap().take().unwrap();
        let mut mrx = self.msgRxChann.lock().unwrap().take().unwrap();
        let mut interval = time::interval(Self::WAITING_PERIOD);

        let _ = interval.tick().await;

        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                event = rx.recv() => {
                    if let Some(event) = event {
                        let obj = event.obj.clone();
                        match &event.type_ {
                            EventType::Added => {
                                match &obj.kind as &str {
                                    FuncPackageSpec::KEY => {
                                        let spec = FuncPackageSpec::FromDataObject(obj)?;
                                        self.ProcessAddFuncPackage(&spec).await?;
                                    }
                                    _ => {
                                    }
                                }
                            }
                            EventType::Modified => {}
                            EventType::Deleted => {
                                match &obj.kind as &str {
                                    FuncPackageSpec::KEY => {
                                        let spec = FuncPackageSpec::FromDataObject(obj)?;
                                        self.ProcessRemoveFuncPackage(&spec).await?;
                                    }
                                    _ => {
                                    }
                                }
                            }
                            _o => {
                                return Err(Error::CommonError(format!(
                                    "PodHandler::ProcessDeltaEvent {:?}",
                                    event
                                )));
                            }
                        }
                    } else {
                        break;
                    }
                }
                m = mrx.recv() => {
                    if let Some(msg) = m {
                        match msg {
                            PodHandlerMsg::AskFuncPod(m) => {
                                let pods = OBJ_REPO
                                    .get()
                                    .unwrap()
                                    .GetFuncPods(&m.tenant, &m.namespace, &m.funcname)?;

                                let mut needNewWorker = true;
                                for pod in &pods {
                                    if pod.state == PodState::MemHibernated || pod.state == PodState::DiskHibernated {
                                        self.WakeupWorker(&m.tenant, &m.namespace, &m.funcname, &pod.id).await?;
                                        needNewWorker = false;
                                    }
                                }

                                if needNewWorker {
                                    let fp = OBJ_REPO
                                    .get()
                                    .unwrap()
                                    .GetFuncPackage(&m.tenant, &m.namespace, &m.funcname)?;
                                    let _ip = self.StartWorker(&fp.spec).await?;
                                }
                            }
                            PodHandlerMsg::DisableFuncPod(m) => {
                                self.HibernateWorker(&m.tenant, &m.namespace, &m.funcname, &m.id).await?;
                            }
                        }
                    }else {
                        break;
                    }
                }
            }
        }

        return Ok(());
    }

    pub async fn ProcessAddFuncPackage(&self, spec: &FuncPackageSpec) -> Result<()> {
        let pods =
            OBJ_REPO
                .get()
                .unwrap()
                .GetFuncPods(&spec.tenant, &spec.namespace, &spec.funcname)?;

        if pods.len() >= 1 {
            return Ok(());
        }

        let _ip = self.StartWorker(&spec).await?;

        return Ok(());
    }

    pub async fn ProcessRemoveFuncPackage(&self, spec: &FuncPackageSpec) -> Result<()> {
        let pods =
            OBJ_REPO
                .get()
                .unwrap()
                .GetFuncPods(&spec.tenant, &spec.namespace, &spec.funcname)?;

        if pods.len() == 0 {
            return Ok(());
        }

        for pod in &pods {
            match self
                .StopWorker(&pod.tenant, &pod.namespace, &pod.funcname, &pod.id)
                .await
            {
                Ok(()) => (),
                Err(e) => {
                    error!(
                        "fail to stopper func worker {:?} with error {:#?}",
                        pod.PodKey(),
                        e
                    );
                }
            }
        }

        return Ok(());
    }

    pub fn NextWorkerId(&self) -> u64 {
        return self.nextWorkId.fetch_add(1, Ordering::AcqRel);
    }

    pub async fn StartWorker(&self, spec: &FuncPackageSpec) -> Result<IpAddress> {
        let tenant = &spec.tenant;
        let namespace = &spec.namespace;
        let funcname = &spec.funcname;
        let id = self.NextWorkerId();

        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let mounts = vec![na::Mount {
            host_path: "/home/brad/rust/Quark/test".to_owned(),
            mount_path: "/test".to_owned(),
        }];

        let commands = spec.commands.clone();
        let mut envs = Vec::new();

        for e in &spec.envs {
            envs.push(Env {
                name: e.0.clone(),
                value: e.1.clone(),
            })
        }

        let mut annotations = Vec::new();
        annotations.push(Kv {
            key: FUNCPOD_TYPE.to_owned(),
            val: FUNCPOD_PROMPT.to_owned(),
        });

        annotations.push(Kv {
            key: FUNCPOD_FUNCNAME.to_owned(),
            val: funcname.to_owned(),
        });

        let request = tonic::Request::new(na::CreateFuncPodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcname: funcname.to_owned(),
            id: format!("{id}"),
            image: spec.image.clone(),
            labels: Vec::new(),
            annotations: annotations,
            commands: commands,
            envs: envs,
            mounts: mounts,
            ports: Vec::new(),
        });

        let response = client.create_func_pod(request).await?;
        let resp = response.into_inner();

        if resp.error.is_empty() {
            let addr = IpAddress(resp.ipaddress);
            return Ok(addr);
        }

        return Err(Error::CommonError(format!(
            "create pod fail with error {}",
            resp.error
        )));
    }

    pub async fn HibernateWorker(
        &self,
        tenant: &str,
        namespace: &str,
        funcname: &str,
        id: &str,
    ) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::HibernatePodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcname: funcname.to_owned(),
            id: id.to_owned(),
            hibernate_type: 1,
        });
        let response = client.hibernate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {} {}",
                namespace, funcname, id, resp.error
            );
        }

        return Ok(());
    }

    pub async fn WakeupWorker(
        &self,
        tenant: &str,
        namespace: &str,
        funcname: &str,
        id: &str,
    ) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::WakeupPodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcname: funcname.to_owned(),
            id: id.to_owned(),
            hibernate_type: 1,
        });
        let response = client.wakeup_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {} {}",
                namespace, funcname, id, resp.error
            );
        }

        return Ok(());
    }

    pub async fn StopWorker(
        &self,
        tenant: &str,
        namespace: &str,
        funcname: &str,
        id: &str,
    ) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::TerminatePodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcname: funcname.to_owned(),
            id: id.to_owned(),
        });
        let response = client.terminate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to stop worker {} {} {} {} {}",
                tenant, namespace, funcname, id, resp.error
            );
        }

        return Ok(());
    }

    // pub async fn GetNewPodNode(&self, fp: &FuncPackageSpec) -> Result<Option> {}
}
