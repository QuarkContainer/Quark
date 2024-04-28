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
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::response::Response;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Notify};
use tokio::sync::{oneshot, Mutex as TMutex};
use tokio::time::Duration;

use http_body_util::{BodyExt, Empty};
use hyper::body::{Bytes, Incoming};
use hyper::client::conn::http1::SendRequest;
use hyper::Request;
use hyper::StatusCode;
use hyper_util::rt::TokioIo;

use qshare::common::*;
use qshare::na::{self, Env, Kv};

use crate::{PromptReq, FUNCPOD_FUNCNAME, FUNCPOD_PROMPT, FUNCPOD_TYPE, TSOT_CLIENT};
use qshare::obj_mgr::func_mgr::FuncPackage;

lazy_static::lazy_static! {
    pub static ref FUNCAGENT_MGR: FuncAgentMgr = FuncAgentMgr::default();
}

#[derive(Debug, Default)]
pub struct FuncAgentMgrInner {
    pub agents: BTreeMap<String, FuncAgent>,
}

#[derive(Debug, Default, Clone)]
pub struct FuncAgentMgr(Arc<TMutex<FuncAgentMgrInner>>);

impl Deref for FuncAgentMgr {
    type Target = Arc<TMutex<FuncAgentMgrInner>>;

    fn deref(&self) -> &Arc<TMutex<FuncAgentMgrInner>> {
        &self.0
    }
}

impl FuncAgentMgr {
    pub async fn Call(&self, funcPackage: &FuncPackage, req: PromptReq) -> HttpResponse {
        let agent = {
            let key = funcPackage.spec.Key();
            let mut inner = self.lock().await;
            match inner.agents.get(&key) {
                Some(agent) => agent.clone(),
                None => {
                    let agent = FuncAgent::New(funcPackage).await;
                    inner.agents.insert(key, agent.clone());
                    agent
                }
            }
        };

        let (tx, rx) = oneshot::channel();
        agent.EnqReq(req, tx);
        let resp = rx.await.unwrap();
        return resp;
    }
}

#[derive(Debug)]
pub enum WorkerState {
    Creating,
    Working,
    Idle,
    Evicating,
    Fail,
    Killing,
}

#[derive(Debug)]
pub enum WorkerUpdate {
    Ready(FuncWorker), // parallel level
    WorkerFail(FuncWorker),
    RequestDone(FuncWorker),
    IdleTimeout(FuncWorker),
}

#[derive(Debug)]
pub struct FuncAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub tenant: String,
    pub namespace: String,
    pub funcName: String,
    pub funcPackge: FuncPackage,

    pub waitingReqs: VecDeque<FuncReq>,
    pub reqQueueTx: mpsc::Sender<FuncReq>,
    pub workerStateUpdateTx: mpsc::Sender<WorkerUpdate>,
    pub availableSlot: usize,
    pub startingSlot: usize,
    pub workers: BTreeMap<u64, FuncWorker>,
    pub nextWorkerId: u64,
    pub nextReqId: u64,
}

impl FuncAgentInner {
    pub fn AvailableWorker(&self) -> Option<FuncWorker> {
        for (_, worker) in &self.workers {
            if worker.AvailableSlot() > 0 {
                return Some(worker.clone());
            }
        }

        return None;
    }

    pub fn RemoveWorker(&mut self, worker: &FuncWorker) -> Result<()> {
        self.workers.remove(&worker.workerId);
        return Ok(());
    }

    pub fn NextWorkerId(&mut self) -> u64 {
        self.nextWorkerId += 1;
        return self.nextWorkerId;
    }

    pub fn AssignReq(&mut self, req: FuncReq) {
        for (_, worker) in &self.workers {
            if worker.AvailableSlot() > 0 {
                worker.AssignReq(req);
                self.availableSlot -= 1;
                break;
            }
        }
    }

    pub fn NextReqId(&mut self) -> u64 {
        self.nextReqId += 1;
        return self.nextReqId;
    }
}

#[derive(Debug, Clone)]
pub struct FuncAgent(Arc<Mutex<FuncAgentInner>>);

impl Deref for FuncAgent {
    type Target = Arc<Mutex<FuncAgentInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncAgentInner>> {
        &self.0
    }
}

impl FuncAgent {
    pub async fn New(funcPackage: &FuncPackage) -> Self {
        let (rtx, rrx) = mpsc::channel(30);
        let (wtx, wrx) = mpsc::channel(30);
        let inner = FuncAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            tenant: funcPackage.spec.tenant.clone(),
            namespace: funcPackage.spec.namespace.clone(),
            funcName: funcPackage.spec.name.to_owned(),
            funcPackge: funcPackage.clone(),
            waitingReqs: VecDeque::new(),
            reqQueueTx: rtx,
            workerStateUpdateTx: wtx,
            availableSlot: 0,
            startingSlot: 0,
            workers: BTreeMap::new(),
            nextWorkerId: 0,
            nextReqId: 0,
        };

        let ret = Self(Arc::new(Mutex::new(inner)));

        let clone = ret.clone();
        tokio::spawn(async move {
            clone.Process(rrx, wrx).await.unwrap();
        });

        return ret;
    }

    pub fn EnqReq(&self, req: PromptReq, tx: oneshot::Sender<HttpResponse>) {
        let funcReq = FuncReq {
            reqId: self.lock().unwrap().NextReqId(),
            tenant: req.tenant,
            namespace: req.namespace,
            funcName: req.func,
            request: req.prompt,
            tx: tx,
        };
        self.lock().unwrap().reqQueueTx.try_send(funcReq).unwrap();
    }

    pub async fn Close(&self) {
        let closeNotify = self.lock().unwrap().closeNotify.clone();
        closeNotify.notify_one();
    }

    pub async fn Process(
        &self,
        reqQueueRx: mpsc::Receiver<FuncReq>,
        workerStateUpdateRx: mpsc::Receiver<WorkerUpdate>,
    ) -> Result<()> {
        let mut reqQueueRx = reqQueueRx;
        let mut workerStateUpdateRx = workerStateUpdateRx;

        let closeNotify = self.lock().unwrap().closeNotify.clone();

        loop {
            tokio::select! {
                _ = closeNotify.notified() => {
                    self.lock().unwrap().stop.store(false, Ordering::SeqCst);
                    break;
                }
                workReq = reqQueueRx.recv() => {
                    if let Some(req) = workReq {
                        self.ProcessReq(req).await;
                    } else {
                        unreachable!("FuncAgent::Process reqQueueRx closed");
                    }
                }
                stateUpdate = workerStateUpdateRx.recv() => {
                    if let Some(update) = stateUpdate {
                        match update {
                            WorkerUpdate::Ready(worker) => {
                                let slot = worker.AvailableSlot();
                                self.IncrSlot(slot);
                                self.lock().unwrap().startingSlot -= slot;
                                self.TryProcessOneReq();
                            }
                            WorkerUpdate::RequestDone(_) => {
                                self.IncrSlot(1);
                                self.TryProcessOneReq();
                            }
                            WorkerUpdate::WorkerFail(worker) => {
                                let slot = worker.AvailableSlot();
                                self.DecrSlot(slot);
                            }
                            WorkerUpdate::IdleTimeout(worker) => {
                                let slot = worker.AvailableSlot();
                                self.DecrSlot(slot);
                                self.lock().unwrap().RemoveWorker(&worker)?;
                                worker.Close().await;
                            }
                        }
                    } else {
                        unreachable!("FuncAgent::Process reqQueueRx closed");
                    }
                }
            }
        }

        return Ok(());
    }

    pub fn SendWorkerStatusUpdate(&self, update: WorkerUpdate) {
        let statusUpdateTx = self.lock().unwrap().workerStateUpdateTx.clone();
        statusUpdateTx.try_send(update).unwrap();
    }

    pub fn IncrSlot(&self, cnt: usize) {
        self.lock().unwrap().availableSlot += cnt;
    }

    pub fn DecrSlot(&self, cnt: usize) {
        self.lock().unwrap().availableSlot -= cnt;
    }

    pub fn TryProcessOneReq(&self) {
        let mut inner = self.lock().unwrap();

        if inner.availableSlot == 0 {
            return;
        }

        match inner.waitingReqs.pop_front() {
            None => return,
            Some(req) => {
                for (_, worker) in &inner.workers {
                    if worker.AvailableSlot() > 0 {
                        worker.AssignReq(req);
                        inner.availableSlot -= 1;
                        break;
                    }
                }
            }
        }
    }

    pub async fn ProcessReq(&self, req: FuncReq) {
        if self.lock().unwrap().availableSlot == 0 {
            let mut needNewWorker = false;
            {
                let mut inner = self.lock().unwrap();
                inner.waitingReqs.push_back(req);

                if inner.waitingReqs.len() > inner.startingSlot {
                    needNewWorker = true;
                    inner.startingSlot += DEFAULT_PARALLEL_LEVEL;
                }
            }

            let tenant = self.lock().unwrap().tenant.clone();
            let namespace = self.lock().unwrap().namespace.clone();
            let funcName = self.lock().unwrap().funcName.clone();
            if needNewWorker {
                let id = self.lock().unwrap().NextWorkerId();
                let keepaliveTime = self
                    .lock()
                    .unwrap()
                    .funcPackge
                    .spec
                    .keepalivePolicy
                    .keepaliveTime;
                match FuncWorker::New(
                    id,
                    &tenant,
                    &namespace,
                    &funcName,
                    DEFAULT_PARALLEL_LEVEL,
                    keepaliveTime,
                    self,
                )
                .await
                {
                    Err(e) => {
                        self.lock().unwrap().startingSlot -= DEFAULT_PARALLEL_LEVEL;
                        error!(
                            "FuncAgent::ProcessReq new funcworker fail with error {:?}",
                            e
                        );
                    }
                    Ok(worker) => {
                        self.lock().unwrap().workers.insert(id, worker);
                    }
                };
            }
        } else {
            self.lock().unwrap().AssignReq(req);
        }
    }
}

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

    pub workerId: u64,
    pub workerName: String,

    pub tenant: String,
    pub namespace: String,
    pub funcName: String,
    pub ipAddr: IpAddress,
    pub port: u16,
    pub parallelLevel: usize,
    pub keepaliveTime: u64,
    pub ongoingReqCnt: AtomicUsize,

    pub reqQueue: mpsc::Sender<FuncReq>,
    pub idleFuncClientQueue: mpsc::Sender<FuncWorkerClient>,
    pub idleFuncClients: Mutex<Vec<FuncWorkerClient>>,
    pub funcClientCnt: AtomicUsize,
    pub funcAgent: FuncAgent,

    pub connPool: TMutex<Vec<QHttpCallClient>>,
}

#[derive(Debug)]
pub enum FuncWorkerState {
    Idle,
    Processing,
    Hibernated,
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
        id: u64,
        tenant: &str,
        namespace: &str,
        funcName: &str,
        parallelLeve: usize,
        keepaliveTime: u64,
        funcAgent: &FuncAgent,
    ) -> Result<Self> {
        let funcPackage = funcAgent.lock().unwrap().funcPackge.clone();

        let workerName = format!("{}_{}", funcName, id);
        let addr =
            Self::StartWorker(tenant, namespace, funcName, &workerName, &funcPackage).await?;
        let (tx, rx) = mpsc::channel::<FuncReq>(parallelLeve);
        let (workerTx, workerRx) = mpsc::channel::<FuncWorkerClient>(parallelLeve);

        let inner = FuncWorkerInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            workerId: id,
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcName: funcName.to_owned(),
            workerName: workerName.clone(),
            ipAddr: addr,
            port: WORKER_PORT,
            parallelLevel: parallelLeve,
            keepaliveTime,
            ongoingReqCnt: AtomicUsize::new(0),

            reqQueue: tx,
            idleFuncClientQueue: workerTx,
            idleFuncClients: Mutex::new(Vec::new()),
            funcClientCnt: AtomicUsize::new(0),
            funcAgent: funcAgent.clone(),
            connPool: TMutex::new(Vec::new()),
        };

        let worker = Self(Arc::new(inner));

        let clone = worker.clone();
        tokio::spawn(async move {
            clone.Process(rx, workerRx).await.unwrap();
        });

        return Ok(worker);
    }

    pub async fn Close(&self) {
        let closeNotify = self.closeNotify.clone();
        closeNotify.notify_one();
    }

    pub fn AvailableSlot(&self) -> usize {
        return self.parallelLevel - self.ongoingReqCnt.load(Ordering::SeqCst);
    }

    pub fn NewFuncWorkerClient(&self) -> FuncWorkerClient {
        self.funcClientCnt.fetch_add(1, Ordering::SeqCst);
        let client = FuncWorkerClient::New(
            &self.tenant,
            &self.namespace,
            &self.funcName,
            &self.ipAddr,
            self.port,
            &self,
        );
        return client;
    }

    pub fn AssignReq(&self, req: FuncReq) {
        self.reqQueue.try_send(req).unwrap();
    }

    pub fn RequestDone(&self, client: &FuncWorkerClient) {
        // ignore failure
        self.idleFuncClientQueue.try_send(client.clone()).ok();
    }

    pub async fn Process(
        &self,
        reqQueueRx: mpsc::Receiver<FuncReq>,
        idleClientRx: mpsc::Receiver<FuncWorkerClient>,
    ) -> Result<()> {
        self.WaitForPod().await?;
        self.funcAgent
            .SendWorkerStatusUpdate(WorkerUpdate::Ready(self.clone()));

        let mut reqQueueRx = reqQueueRx;
        let mut idleClientRx = idleClientRx;
        let mut state = FuncWorkerState::Idle;

        loop {
            match state {
                FuncWorkerState::Idle => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            // we clean all the waiting request
                            self.DrainReqs(reqQueueRx).await;
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        e = self.ProbeLiveness() => {
                            self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
                            error!("FuncWorker::Process ProbeLiveness fail with error {:?}", e);
                            break;
                        }
                        req = reqQueueRx.recv() => {
                            match req {
                                None => {
                                    return Ok(())
                                }
                                Some(req) => {
                                    state = FuncWorkerState::Processing;
                                    self.ongoingReqCnt.fetch_add(1, Ordering::SeqCst);
                                    let workerClient = match self.idleFuncClients.lock().unwrap().pop() {
                                        None => {
                                            assert!(self.funcClientCnt.load(Ordering::SeqCst) < self.parallelLevel);
                                            let client = self.NewFuncWorkerClient();
                                            client
                                        }
                                        Some(client) => client,
                                    };
                                    workerClient.SendReq(req);
                                }
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_secs(self.keepaliveTime)) => {
                            // self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::IdleTimeout(self.clone()));
                            self.HibernateWorker().await?;
                            state = FuncWorkerState::Hibernated;
                        }

                    }
                }
                FuncWorkerState::Hibernated => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            // we clean all the waiting request
                            self.DrainReqs(reqQueueRx).await;
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        req = reqQueueRx.recv() => {
                            match req {
                                None => {
                                    return Ok(())
                                }
                                Some(req) => {
                                    state = FuncWorkerState::Processing;
                                    self.WakeupWorker().await?;
                                    self.WaitForPod().await?;
                                    self.ongoingReqCnt.fetch_add(1, Ordering::SeqCst);
                                    let workerClient = match self.idleFuncClients.lock().unwrap().pop() {
                                        None => {
                                            assert!(self.funcClientCnt.load(Ordering::SeqCst) < self.parallelLevel);
                                            let client = self.NewFuncWorkerClient();
                                            client
                                        }
                                        Some(client) => client,
                                    };
                                    workerClient.SendReq(req);
                                }
                            }
                        }
                    }
                }
                FuncWorkerState::Processing => {
                    tokio::select! {
                        _ = self.closeNotify.notified() => {
                            self.stop.store(false, Ordering::SeqCst);
                            self.DrainReqs(reqQueueRx).await;
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        e = self.ProbeLiveness() => {
                            self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
                            error!("FuncWorker::Process ProbeLiveness fail with error {:?}", e);
                            break;
                        }
                        req = reqQueueRx.recv() => {
                            match req {
                                None => {
                                    return Ok(())
                                }
                                Some(req) => {
                                    self.ongoingReqCnt.fetch_add(1, Ordering::SeqCst);
                                    let workerClient = match self.idleFuncClients.lock().unwrap().pop() {
                                        None => {
                                            assert!(self.funcClientCnt.load(Ordering::SeqCst) < self.parallelLevel);
                                            let client = self.NewFuncWorkerClient();
                                            client
                                        }
                                        Some(client) => client,
                                    };
                                    workerClient.SendReq(req);
                                }
                            }
                        }
                        worker = idleClientRx.recv() => {
                            match worker {
                                None => {
                                    return Ok(())
                                }
                                Some(worker) => {
                                    let cnt = self.ongoingReqCnt.fetch_sub(1, Ordering::SeqCst);
                                    if cnt == 1 {
                                        state = FuncWorkerState::Idle;
                                    }
                                    self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::RequestDone(self.clone()));
                                    self.idleFuncClients.lock().unwrap().push(worker);
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }

    pub async fn DrainReqs(&self, reqQueueRx: mpsc::Receiver<FuncReq>) {
        let mut reqQueueRx = reqQueueRx;
        loop {
            let req = reqQueueRx.try_recv();
            match req {
                Err(_e) => {
                    break;
                }
                Ok(req) => {
                    let response = HttpResponse {
                        status: StatusCode::BAD_GATEWAY,
                        response: format!("Service Unavaiable {}", self.workerId),
                    };

                    // accept the failure.
                    // todo: do we need to handle failure?
                    req.tx.send(response).ok();
                }
            }
        }
    }

    pub async fn NewHttpCallClient(&self) -> Result<QHttpCallClient> {
        let stream = self.ConnectPod().await?;
        let client = QHttpCallClient::New(stream).await?;
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
        funcName: &str,
        workerName: &str,
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

    pub async fn StopWorker(&self) -> Result<()> {
        let mut client =
            na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
                .await?;

        let request = tonic::Request::new(na::TerminatePodReq {
            tenant: self.tenant.clone(),
            namespace: self.namespace.clone(),
            name: self.workerName.clone(),
        });
        let response = client.terminate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to stop worker {} {} {}",
                self.namespace, self.funcName, resp.error
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
            name: self.workerName.clone(),
            hibernate_type: 1,
        });
        let response = client.hibernate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {}",
                self.namespace, self.funcName, resp.error
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
            name: self.workerName.clone(),
            hibernate_type: 1,
        });
        let response = client.wakeup_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!(
                "Fail to Hibernate worker {} {} {}",
                self.namespace, self.funcName, resp.error
            );
        }

        return Ok(());
    }
}

#[derive(Debug)]
pub struct FuncWorkerClientInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub tenant: String,
    pub namespace: String,
    pub funcName: String,
    pub ipAddr: IpAddress,
    pub port: u16,

    pub reqQueue: mpsc::Sender<FuncReq>,
    pub funcWorker: FuncWorker,
}

#[derive(Debug, Clone)]
pub struct FuncWorkerClient(Arc<FuncWorkerClientInner>);

impl Deref for FuncWorkerClient {
    type Target = Arc<FuncWorkerClientInner>;

    fn deref(&self) -> &Arc<FuncWorkerClientInner> {
        &self.0
    }
}

impl FuncWorkerClient {
    pub fn New(
        tenant: &str,
        namespace: &str,
        funcName: &str,
        ipAddr: &IpAddress,
        port: u16,
        funcWorker: &FuncWorker,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<FuncReq>(1);
        let inner = FuncWorkerClientInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcName: funcName.to_owned(),
            ipAddr: ipAddr.clone(),
            port: port,

            reqQueue: tx,
            funcWorker: funcWorker.clone(),
        };

        let client = Self(Arc::new(inner));

        let clone = client.clone();
        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return client;
    }

    pub fn SendReq(&self, req: FuncReq) {
        self.reqQueue.try_send(req).unwrap();
    }

    pub async fn TryConnectPod(&self) -> Result<QHttpCallClient> {
        let addr = self.ipAddr.AsBytes();
        let stream = TSOT_CLIENT
            .get()
            .unwrap()
            .Connect(&self.tenant, &self.namespace, addr, self.port)
            .await?;
        let client: QHttpCallClient = QHttpCallClient::New(stream).await?;

        return Ok(client);
    }

    pub async fn Process(&self, reqQueueRx: mpsc::Receiver<FuncReq>) -> Result<()> {
        let mut reqQueueRx = reqQueueRx;

        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    return Ok(())
                }
                req = reqQueueRx.recv() => {
                    match req {
                        None => {
                            // the Funcworker close the req channel
                            return Ok(())
                        }
                        Some(req) => {
                            let mut client = self.TryConnectPod().await?;
                            self.HttpCall(&mut client, req).await?;
                            self.funcWorker.RequestDone(self);
                        }
                    }
                }
            }
        }
    }

    pub async fn HttpCall(&self, client: &mut QHttpCallClient, req: FuncReq) -> Result<()> {
        let promptReq = PromptReq {
            tenant: req.tenant.clone(),
            namespace: req.namespace.clone(),
            func: req.funcName.clone(),
            prompt: req.request.clone(),
        };

        let body = serde_json::to_string(&promptReq)?;

        let url = FUNCCALL_URL.parse::<hyper::Uri>()?;

        let authority = url.authority().unwrap().clone();

        let httpReq = Request::post(url.path())
            .header(hyper::header::HOST, authority.as_str())
            .header(hyper::header::CONTENT_TYPE, "application/json")
            .body(body)?;

        match client.Send(httpReq).await {
            Err(e) => {
                let resp = HttpResponse {
                    status: StatusCode::BAD_REQUEST,
                    response: format!("service fail with error {:?}", e),
                };
                req.tx.send(resp).unwrap();
                return Ok(());
            }
            Ok(mut res) => {
                let mut output = String::new();
                while let Some(next) = res.frame().await {
                    match next {
                        Err(e) => {
                            let resp = HttpResponse {
                                status: StatusCode::BAD_REQUEST,
                                response: format!("service fail with error {:?}", e),
                            };
                            req.tx.send(resp).unwrap();
                            return Ok(());
                        }
                        Ok(frame) => {
                            let chunk = frame.data_ref().unwrap().to_vec();
                            let str = String::from_utf8(chunk).unwrap();
                            output = output + &str;
                        }
                    }
                }

                let resp = HttpResponse {
                    status: res.status(),
                    response: output,
                };
                req.tx.send(resp).unwrap();
                return Ok(());
            }
        }
    }
}

#[derive(Debug)]
pub struct HttpResponse {
    pub status: StatusCode,
    pub response: String,
}

#[derive(Debug)]
pub struct FuncReq {
    pub reqId: u64,
    pub tenant: String,
    pub namespace: String,
    pub funcName: String,
    pub request: String,
    pub tx: oneshot::Sender<HttpResponse>,
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
    sender: SendRequest<String>,
}

impl QHttpCallClient {
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

    pub async fn Send(&mut self, req: Request<String>) -> Result<Response<Incoming>> {
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
