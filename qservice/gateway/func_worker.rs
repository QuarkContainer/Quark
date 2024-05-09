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
// limitations under the Licens

use core::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::response::Response;
use qshare::metastore::data_obj::{DeltaEvent, EventType};
use qshare::node::{PodDef, PodState};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Notify};
use tokio::sync::{oneshot, Mutex as TMutex};
use tokio::time::Duration;

use http_body_util::Empty;
use hyper::body::{Bytes, Incoming};
use hyper::client::conn::http1::SendRequest;
use hyper::Request;
use hyper::StatusCode;
use hyper_util::rt::TokioIo;

use qshare::common::*;
use qshare::na::{self, Env, Kv};

use crate::func_agent_mgr::{FuncAgent, WorkerUpdate};
use crate::TSOT_CLIENT;
use qshare::obj_mgr::func_mgr::*;

pub const DEFAULT_PARALLEL_LEVEL: usize = 1;
pub const LIVENESS_URL: &str = "http://127.0.0.1/liveness";
pub const READINESS_URL: &str = "http://127.0.0.1/readiness";
pub const FUNCCALL_URL: &str = "http://127.0.0.1/funccall";
pub const RESPONSE_LIMIT: usize = 4 * 1024 * 1024; // 4MB
pub const WORKER_PORT: u16 = 80;

#[derive(Debug)]
pub struct FuncWorkerInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub workerId: String,
    pub workerName: String,

    pub tenant: String,
    pub namespace: String,
    pub funcname: String,
    pub id: String,
    pub ipAddr: IpAddress,
    pub port: u16,
    pub parallelLevel: usize,
    pub keepaliveTime: u64,
    pub ongoingReqCnt: AtomicUsize,

    pub reqQueue: mpsc::Sender<FuncClientReq>,
    pub finishQueue: mpsc::Sender<()>,
    pub eventChann: mpsc::Sender<DeltaEvent>,
    pub funcClientCnt: AtomicUsize,
    pub funcAgent: FuncAgent,

    pub state: Mutex<FuncWorkerState>,

    pub connPool: TMutex<Vec<QHttpCallClient>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncWorkerState {
    // the new pod's state is not ready
    Probation,
    // the worker is ready to process any requests and no running request
    Idle,
    // the worker is transition from idle to probation,
    // i.e. idle timeout and notice scheduler to disable the pod but the pod state is still in ready
    Disabling,
    // the worker is processing a request
    Processing,
    //Hibernated,
}

#[derive(Debug, Clone)]
pub struct FuncWorker(Arc<FuncWorkerInner>);

impl Deref for FuncWorker {
    type Target = Arc<FuncWorkerInner>;

    fn deref(&self) -> &Arc<FuncWorkerInner> {
        &self.0
    }
}

impl FuncWorker {
    pub async fn New(
        id: &str,
        tenant: &str,
        namespace: &str,
        funcname: &str,
        addr: IpAddress,
        parallelLeve: usize,
        keepaliveTime: u64,
        funcAgent: &FuncAgent,
    ) -> Result<Self> {
        let workerName = format!("{}", id);
        let (tx, rx) = mpsc::channel::<FuncClientReq>(parallelLeve);
        let (finishTx, finishRx) = mpsc::channel::<()>(parallelLeve);
        let (etx, erx) = mpsc::channel(30);

        let inner = FuncWorkerInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            workerId: id.to_owned(),
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcname: funcname.to_owned(),
            id: format!("{id}"),
            workerName: workerName.clone(),
            ipAddr: addr,
            port: WORKER_PORT,
            parallelLevel: parallelLeve,
            keepaliveTime,
            ongoingReqCnt: AtomicUsize::new(0),

            reqQueue: tx,
            finishQueue: finishTx,
            eventChann: etx,
            funcClientCnt: AtomicUsize::new(0),
            funcAgent: funcAgent.clone(),
            state: Mutex::new(FuncWorkerState::Probation),
            connPool: TMutex::new(Vec::new()),
        };

        let worker = Self(Arc::new(inner));

        let clone = worker.clone();
        tokio::spawn(async move {
            clone.Process(rx, erx, finishRx).await.unwrap();
        });

        return Ok(worker);
    }

    pub fn State(&self) -> FuncWorkerState {
        return self.state.lock().unwrap().clone();
    }

    pub fn SetState(&self, state: FuncWorkerState) {
        *self.state.lock().unwrap() = state;
    }

    pub async fn DisableWorker(&self) -> Result<()> {
        let mut schedClient =
            na::scheduler_service_client::SchedulerServiceClient::connect("http://127.0.0.1:9008")
                .await?;

        let req = na::DisableFuncPodReq {
            tenant: self.tenant.clone(),
            namespace: self.namespace.clone(),
            funcname: self.funcname.clone(),
            id: self.id.clone(),
        };

        let request = tonic::Request::new(req);
        let response = schedClient.disable_func_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() == 0 {
            return Ok(());
        }

        return Err(Error::CommonError(format!(
            "DisableWorker pod fail with error {}",
            resp.error
        )));
    }

    pub async fn Close(&self) {
        let closeNotify = self.closeNotify.clone();
        closeNotify.notify_one();
    }

    pub fn AvailableSlot(&self) -> usize {
        let state = self.State();
        if state == FuncWorkerState::Idle || state == FuncWorkerState::Processing {
            return self.parallelLevel - self.ongoingReqCnt.load(Ordering::SeqCst);
        } else {
            return 0;
        }
    }

    pub fn AssignReq(&self, req: FuncClientReq) {
        self.reqQueue.try_send(req).unwrap();
    }

    pub fn EnqEvent(&self, event: DeltaEvent) {
        self.eventChann.try_send(event).unwrap();
    }

    pub async fn Process(
        &self,
        reqQueueRx: mpsc::Receiver<FuncClientReq>,
        eventQueueRx: mpsc::Receiver<DeltaEvent>,
        idleClientRx: mpsc::Receiver<()>,
    ) -> Result<()> {
        let mut reqQueueRx = reqQueueRx;
        let mut eventQueueRx = eventQueueRx;
        let mut idleClientRx = idleClientRx;

        loop {
            let state = self.State();
            match state {
                FuncWorkerState::Probation => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        deltaEvent = eventQueueRx.recv() => {
                            if let Some(event) = deltaEvent {

                                let obj = event.obj.clone();
                                assert!(&obj.kind == PodDef::KEY);
                                let podDef = PodDef::FromDataObject(obj)?;
                                if event.type_ == EventType::Added || event.type_ == EventType::Modified {
                                    if podDef.state == PodState::Running {
                                        self.SetState(FuncWorkerState::Idle);
                                        self.WaitForPod().await?;
                                        self.funcAgent
                                            .SendWorkerStatusUpdate(WorkerUpdate::Ready(self.clone()));

                                    } else {
                                        self.SetState(FuncWorkerState::Probation);
                                    }
                                } else if event.type_ == EventType::Deleted {
                                    self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
                                    break;
                                } else {
                                    unreachable!("FuncWorker::Process eventQueueRx get unexpected event {:?}", event.type_);
                                }

                            } else {
                                unreachable!("FuncWorker::Process eventQueueRx closed");
                            }
                        }
                    }
                }
                FuncWorkerState::Disabling => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            // we clean all the waiting request
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        deltaEvent = eventQueueRx.recv() => {
                            if let Some(event) = deltaEvent {
                                let obj = event.obj.clone();
                                assert!(&obj.kind == PodDef::KEY);
                                let podDef = PodDef::FromDataObject(obj)?;
                                if event.type_ == EventType::Added || event.type_ == EventType::Modified {
                                    if podDef.state == PodState::Running {
                                        self.SetState(FuncWorkerState::Idle);
                                    } else {
                                        self.SetState(FuncWorkerState::Probation);
                                    }
                                } else if event.type_ == EventType::Deleted {
                                    self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
                                    break;
                                } else {
                                    unreachable!("FuncWorker::Process eventQueueRx get unexpected event {:?}", event.type_);
                                }

                            } else {
                                unreachable!("FuncWorker::Process eventQueueRx closed");
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_secs(self.keepaliveTime)) => {
                            self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::IdleTimeout(self.clone()));
                            // self.HibernateWorker().await?;
                            self.SetState(FuncWorkerState::Disabling);
                        }
                    }
                }
                FuncWorkerState::Idle => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            // we clean all the waiting request
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        deltaEvent = eventQueueRx.recv() => {
                            if let Some(event) = deltaEvent {
                                let obj = event.obj.clone();
                                assert!(&obj.kind == PodDef::KEY);
                                let podDef = PodDef::FromDataObject(obj)?;
                                if event.type_ == EventType::Added || event.type_ == EventType::Modified {
                                    if podDef.state == PodState::Running {
                                        self.SetState(FuncWorkerState::Idle);
                                    } else {
                                        self.SetState(FuncWorkerState::Probation);
                                    }
                                } else if event.type_ == EventType::Deleted {
                                    self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
                                    break;
                                } else {
                                    unreachable!("FuncWorker::Process eventQueueRx get unexpected event {:?}", event.type_);
                                }
                            } else {
                                unreachable!("FuncWorker::Process eventQueueRx closed");
                            }
                        }
                        _e = self.ProbeLiveness() => {
                            self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
                            break;
                        }
                        req = reqQueueRx.recv() => {
                            match req {
                                None => {
                                    return Ok(())
                                }
                                Some(req) => {
                                    self.SetState(FuncWorkerState::Processing);
                                    self.ongoingReqCnt.fetch_add(1, Ordering::SeqCst);
                                    let client = self.NewHttpCallClient().await?;
                                    req.Send(client);
                                }
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_secs(self.keepaliveTime)) => {
                            self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::IdleTimeout(self.clone()));
                            // self.HibernateWorker().await?;
                            self.SetState(FuncWorkerState::Disabling);
                        }

                    }
                }
                FuncWorkerState::Processing => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        _e = self.ProbeLiveness() => {
                            self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));

                            break;
                        }
                        req = reqQueueRx.recv() => {
                            match req {
                                None => {
                                    return Ok(())
                                }
                                Some(req) => {
                                    self.ongoingReqCnt.fetch_add(1, Ordering::SeqCst);
                                    let client = self.NewHttpCallClient().await?;
                                    req.Send(client);
                                }
                            }
                        }
                        worker = idleClientRx.recv() => {
                            match worker {
                                None => {
                                    return Ok(())
                                }
                                Some(()) => {
                                    let cnt = self.ongoingReqCnt.fetch_sub(1, Ordering::SeqCst);
                                    if cnt == 1 {
                                        self.SetState(FuncWorkerState::Idle);
                                    }
                                    self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::RequestDone(self.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }

    pub async fn NewHttpCallClient(&self) -> Result<QHttpCallClient> {
        let stream = self.ConnectPod().await?;
        let client = QHttpCallClient::New(self.finishQueue.clone(), stream).await?;
        return Ok(client);
    }

    pub async fn ReturnHttpCallClient(&self, client: QHttpCallClient) -> Result<()> {
        self.connPool.lock().await.push(client);
        return Ok(());
    }

    pub async fn GetHttpCallClient(&self) -> Result<QHttpCallClient> {
        match self.connPool.lock().await.pop() {
            None => return self.NewHttpCallClient().await,
            Some(client) => return Ok(client),
        }
    }

    pub async fn WaitForPod(&self) -> Result<()> {
        self.WaitforLiveness().await?;
        self.WaitforReadiness().await?;
        return Ok(());
    }

    pub async fn ProbeLiveness(&self) -> Result<()> {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;

            match self.HttpPing(LIVENESS_URL).await {
                Err(e) => {
                    return Err(e);
                }
                Ok(_s) => (),
            }
        }
    }

    pub async fn HttpPing(&self, url: &str) -> Result<()> {
        let url = url.parse::<hyper::Uri>()?;

        let authority = url.authority().unwrap().clone();

        let stream = self.ConnectPod().await?;
        let mut client = QHttpClient::New(stream).await?;

        // Create an HTTP request with an empty body and a HOST header
        let req = Request::builder()
            .uri(url.path())
            .header(hyper::header::HOST, authority.as_str())
            //.header(hyper::header::CONNECTION, "keep-alive, Keep-Alive")
            .body(Empty::<Bytes>::new())?;

        // Await the response...
        let res = client.Send(req).await?;

        match res.status() {
            StatusCode::OK => return Ok(()),
            _ => {
                return Err(Error::CommonError(format!("HttpPing fail")));
            }
        }
    }

    pub async fn WaitforLiveness(&self) -> Result<()> {
        for _ in 0..100 {
            match self.HttpPing(LIVENESS_URL).await {
                Err(_) => (),
                Ok(s) => {
                    return Ok(s);
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!(
            "FuncWorker::ConnectPingPod timeout"
        )));
    }

    pub async fn WaitforReadiness(&self) -> Result<()> {
        for _ in 0..100 {
            match self.HttpPing(READINESS_URL).await {
                Err(_) => (),
                Ok(s) => {
                    return Ok(s);
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!(
            "FuncWorker::ConnectPingPod timeout"
        )));
    }

    pub async fn ConnectPod(&self) -> Result<TcpStream> {
        tokio::time::sleep(Duration::from_millis(100)).await;

        for _ in 0..100 {
            match self.TryConnectPod(WORKER_PORT).await {
                Err(_) => (),
                Ok(s) => return Ok(s),
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!(
            "FuncWorker::ConnectPingPod timeout"
        )));
    }

    pub fn PodNamespace(&self) -> String {
        return format!("{}/{}", &self.tenant, &self.namespace);
    }

    pub async fn TryConnectPod(&self, port: u16) -> Result<TcpStream> {
        let addr = self.ipAddr.AsBytes();
        let stream = TSOT_CLIENT
            .get()
            .unwrap()
            .Connect(&self.tenant, &self.namespace, addr, port)
            .await?;
        return Ok(stream);
    }

    pub async fn StartWorker(
        tenant: &str,
        namespace: &str,
        funcname: &str,
        id: &str,
        funcPackage: &FuncPackage,
    ) -> Result<IpAddress> {
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
            val: funcname.to_owned(),
        });

        let request = tonic::Request::new(na::CreateFuncPodReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcname: funcname.to_owned(),
            id: id.to_owned(),
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

    pub async fn StopWorker(&self) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::TerminatePodReq {
            tenant: self.tenant.clone(),
            namespace: self.namespace.clone(),
            funcname: self.funcname.clone(),
            id: self.id.clone(),
        });
        let response = client.terminate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to stop worker {} {} {}",
                self.namespace, self.funcname, resp.error
            );
        }

        return Ok(());
    }

    pub async fn HibernateWorker(&self) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::HibernatePodReq {
            tenant: self.tenant.clone(),
            namespace: self.namespace.clone(),
            funcname: self.funcname.clone(),
            id: self.id.clone(),
            hibernate_type: 1,
        });
        let response = client.hibernate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {}",
                self.namespace, self.funcname, resp.error
            );
        }

        return Ok(());
    }

    pub async fn WakeupWorker(&self) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::WakeupPodReq {
            tenant: self.tenant.clone(),
            namespace: self.namespace.clone(),
            funcname: self.funcname.clone(),
            id: self.id.clone(),
            hibernate_type: 1,
        });
        let response = client.wakeup_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {}",
                self.namespace, self.funcname, resp.error
            );
        }

        return Ok(());
    }
}

#[derive(Debug)]
pub struct HttpResponse {
    pub status: StatusCode,
    pub response: String,
}

#[derive(Debug)]
pub struct FuncClientReq {
    pub reqId: u64,
    pub tenant: String,
    pub namespace: String,
    pub funcName: String,
    pub tx: oneshot::Sender<QHttpCallClient>,
}

impl FuncClientReq {
    pub fn Send(self, client: QHttpCallClient) {
        match self.tx.send(client) {
            Ok(()) => (),
            Err(_) => (),
        }
    }
}

#[derive(Debug)]
pub struct QHttpClient {
    sender: SendRequest<Empty<Bytes>>,
}

impl QHttpClient {
    pub async fn New(stream: TcpStream) -> Result<Self> {
        let io = TokioIo::new(stream);
        let (sender, conn) = hyper::client::conn::http1::handshake(io).await?;
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                error!("Error in connection: {}", e);
            }
        });
        return Ok(Self { sender: sender });
    }

    pub async fn Send(&mut self, req: Request<Empty<Bytes>>) -> Result<Response<Incoming>> {
        tokio::select! {
            res = self.sender.send_request(req) => {
                match res {
                    Err(e) => return Err(Error::CommonError(format!("Error in connection: {}", e))),
                    Ok(r) => return Ok(r)
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct QHttpCallClient {
    pub finishQueue: mpsc::Sender<()>,
    sender: SendRequest<axum::body::Body>,
}

impl Drop for QHttpCallClient {
    fn drop(&mut self) {
        self.finishQueue.try_send(()).ok();
    }
}

impl QHttpCallClient {
    pub async fn New(finishQueue: mpsc::Sender<()>, stream: TcpStream) -> Result<Self> {
        let io = TokioIo::new(stream);
        let (sender, conn) = hyper::client::conn::http1::handshake(io).await?;
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                error!("Error in connection: {}", e);
            }
        });
        return Ok(Self {
            finishQueue: finishQueue,
            sender: sender,
        });
    }

    pub async fn Send(&mut self, req: Request<axum::body::Body>) -> Result<Response<Incoming>> {
        tokio::select! {
            res = self.sender.send_request(req) => {
                match res {
                    Err(e) => return Err(Error::CommonError(format!("Error in connection: {}", e))),
                    Ok(r) => return Ok(r)
                }
            }
        }
    }
}
