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
use std::time::Duration;

use qshare::metastore::data_obj::EventType;
use qshare::na;
use qshare::na::Env;
use qshare::na::Kv;
use qshare::obj_mgr::func_mgr::*;
use tokio::sync::mpsc;
use tokio::sync::Notify;

use qshare::common::*;
use qshare::metastore::data_obj::DeltaEvent;
use tokio::time;

use crate::OBJ_REPO;

pub struct PodHandlerInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub fpChann: mpsc::Sender<DeltaEvent>,
    pub nextWorkId: AtomicU64,
}

#[derive(Clone)]
pub struct PodHandler(pub Arc<PodHandlerInner>);

impl Deref for PodHandler {
    type Target = Arc<PodHandlerInner>;

    fn deref(&self) -> &Arc<PodHandlerInner> {
        &self.0
    }
}

impl PodHandler {
    pub async fn New() -> Self {
        let (tx, rx) = mpsc::channel::<DeltaEvent>(1000);

        let inner = PodHandlerInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            fpChann: tx,
            nextWorkId: AtomicU64::new(1),
        };

        let handler = Self(Arc::new(inner));

        let clone = handler.clone();

        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return handler;
    }

    // wait for the cluster funcpod state fully updated
    pub const WAITING_PERIOD: Duration = Duration::from_secs(5);

    pub async fn Process(&self, mut rx: mpsc::Receiver<DeltaEvent>) -> Result<()> {
        let mut interval = time::interval(Self::WAITING_PERIOD);

        let _ = interval.tick().await;

        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                msg = rx.recv() => {
                    if let Some(event) = msg {
                        let obj = event.obj.clone();
                        assert!(&obj.kind == FuncPackageSpec::KEY);
                        match &event.type_ {
                            EventType::Added => {}
                            EventType::Modified => {}
                            EventType::Deleted => {}
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
            }
        }

        return Ok(());
    }

    pub async fn ProcessAddFuncPackage(&self, fp: &FuncPackage) -> Result<()> {
        let spec = &fp.spec;
        let pods =
            OBJ_REPO
                .get()
                .unwrap()
                .GetFuncPods(&spec.tenant, &spec.namespace, &spec.name)?;

        if pods.len() >= 1 {
            return Ok(());
        }

        let _ip = self.StartWorker(&fp).await?;

        return Ok(());
    }

    pub async fn ProcessRemoveFuncPackage(&self, fp: &FuncPackage) -> Result<()> {
        let spec = &fp.spec;
        let pods =
            OBJ_REPO
                .get()
                .unwrap()
                .GetFuncPods(&spec.tenant, &spec.namespace, &spec.name)?;

        if pods.len() >= 1 {
            return Ok(());
        }

        let _ip = self.StartWorker(&fp).await?;

        return Ok(());
    }

    pub fn NextWorkerId(&self) -> u64 {
        return self.nextWorkId.fetch_add(1, Ordering::AcqRel);
    }

    pub async fn StartWorker(&self, funcPackage: &FuncPackage) -> Result<IpAddress> {
        let tenant = &funcPackage.spec.tenant;
        let namespace = &funcPackage.spec.namespace;
        let funcName = &funcPackage.spec.name;
        let id = self.NextWorkerId();
        let workerName = format!("{}_{}", funcName, id);

        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let mounts = vec![na::Mount {
            host_path: "/home/brad/rust/Quark/test".to_owned(),
            mount_path: "/test".to_owned(),
        }];

        let commands = funcPackage.spec.commands.clone();
        let mut envs = Vec::new();

        for e in &funcPackage.spec.envs {
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
            val: funcName.to_owned(),
        });

        let request = tonic::Request::new(na::CreateFuncPodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            name: workerName.to_owned(),
            image: funcPackage.spec.image.clone(),
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
        workerName: &str,
    ) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::HibernatePodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            name: workerName.to_owned(),
            hibernate_type: 1,
        });
        let response = client.hibernate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {}",
                namespace, workerName, resp.error
            );
        }

        return Ok(());
    }

    pub async fn WakeupWorker(
        &self,
        tenant: &str,
        namespace: &str,
        workerName: &str,
    ) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::WakeupPodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            name: workerName.to_owned(),
            hibernate_type: 1,
        });
        let response = client.wakeup_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {}",
                namespace, workerName, resp.error
            );
        }

        return Ok(());
    }

    pub async fn StopWorker(&self, tenant: &str, namespace: &str, workerName: &str) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::TerminatePodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            name: workerName.to_owned(),
        });
        let response = client.terminate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to stop worker {} {} {} {}",
                tenant, namespace, workerName, resp.error
            );
        }

        return Ok(());
    }

    // pub async fn GetNewPodNode(&self, fp: &FuncPackageSpec) -> Result<Option> {}
}
