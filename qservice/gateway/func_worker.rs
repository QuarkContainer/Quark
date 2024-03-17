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

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use core::ops::Deref;

use axum::response::Response;
use tokio::net::TcpStream;
use tokio::sync::{oneshot, Mutex as TMutex};
use tokio::sync::{mpsc, Notify};
use tokio::time::Duration;

use hyper::client::conn::http1::SendRequest;
use hyper::StatusCode;
use hyper::Request;
use hyper::body::{Bytes, Incoming};
use hyper_util::rt::TokioIo;
use http_body_util::{BodyExt, Empty};

use qshare::common::*;
use qshare::na::{self, Env};

use crate::func_mgr::FuncPackage;
use crate::{PromptReq, TSOT_CLIENT};

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

    pub namespace: String,
    pub funcName: String,
    pub funcPackge: FuncPackage,

    pub waitingReqs: VecDeque<FuncReq>,
    pub reqQueueTx: mpsc::Sender<FuncReq>,
    pub workerStateUpdateTx: mpsc::Sender<WorkerUpdate>,
    pub priQueue: Vec<FuncWorker>,
    pub availableSlot: usize,
    pub startingSlot: usize,
    pub workers: BTreeMap<u64, FuncWorker>,
    pub nextWorkerId: u64,
    pub nextReqId: u64,
}

impl FuncAgentInner {
    pub fn AvailableWorker(&self) -> Option<FuncWorker> {
        for worker in &self.priQueue {
            if worker.AvailableSlot() > 0 {
                return Some(worker.clone());
            }
        }

        return None;
    }

    pub fn RemoveWorker(&mut self, worker: &FuncWorker) -> Result<()> {
        for i in 0..self.priQueue.len() {
            if self.priQueue[i].workerId == worker.workerId {
                self.priQueue.remove(i);
                return Ok(())
            }
        }

        return Err(Error::NotExist(format!("FuncAgentInner RemoveWorker {:?}", worker)));
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
            namespace: funcPackage.spec.namespace.clone(),
            funcName: funcPackage.spec.name.to_owned(),
            funcPackge: funcPackage.clone(),
            waitingReqs: VecDeque::new(),
            reqQueueTx: rtx,
            workerStateUpdateTx: wtx,
            priQueue: Vec::new(),
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
            namespace: req.namespace,
            funcName: req.func,
            request: req.prompt,
            tx: tx,
        };
        self.lock().unwrap().reqQueueTx.try_send(funcReq).unwrap();
    } 

    pub async fn Process(
        &self, 
        reqQueueRx: mpsc::Receiver<FuncReq>,
        workerStateUpdateRx: mpsc::Receiver<WorkerUpdate>
    ) -> Result<()>{
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
                                self.lock().unwrap().priQueue.push(worker);
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
                            WorkerUpdate::IdleTimeout(_worker) => {
                                unimplemented!()
                            }
                        }
                    } else {
                        unreachable!("FuncAgent::Process reqQueueRx closed");
                    }
                }
            }
        }

        return Ok(())
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
            return
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

            let namespace = self.lock().unwrap().namespace.clone();
            let funcName = self.lock().unwrap().funcName.clone();
            if needNewWorker {
                let id = self.lock().unwrap().NextWorkerId();
                match FuncWorker::New(id, &namespace, &funcName, DEFAULT_PARALLEL_LEVEL, self).await {
                    Err(e) => {
                        self.lock().unwrap().startingSlot -= DEFAULT_PARALLEL_LEVEL;
                        error!("FuncAgent::ProcessReq new funcworker fail with error {:?}", e);
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
    pub namespace: String,
    pub funcName: String,
    pub ipAddr: IpAddress,
    pub port: u16,
    pub parallelLevel: usize,
    pub ongoingReqCnt: AtomicUsize,

    pub reqQueue: mpsc::Sender<FuncReq>,
    pub funcAgent: FuncAgent,

    pub connPool: TMutex<Vec<QHttpCallClient>>,
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
        namespace: &str,
        funcName: &str,
        parallelLeve: usize,
        funcAgent: &FuncAgent
    ) -> Result<Self> {
        let funcPackage = funcAgent.lock().unwrap().funcPackge.clone();
        let addr = Self::StartWorker(namespace, funcName, &funcPackage).await?;
        let (tx, rx) = mpsc::channel::<FuncReq>(parallelLeve);
        
        let inner = FuncWorkerInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            workerId: id,
            namespace: namespace.to_owned(),
            funcName: funcName.to_owned(),
            ipAddr: addr,
            port: WORKER_PORT,
            parallelLevel: parallelLeve,
            ongoingReqCnt: AtomicUsize::new(0),
            reqQueue: tx,
            funcAgent: funcAgent.clone(),
            connPool: TMutex::new(Vec::new()),
        };

        let ret = Self(Arc::new(inner));

        let clone = ret.clone();
        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return Ok(ret);
    }

    pub fn AvailableSlot(&self) -> usize {
        return self.parallelLevel - self.ongoingReqCnt.load(Ordering::SeqCst);
    }

    pub fn AssignReq(&self, req: FuncReq) {
        self.reqQueue.try_send(req).unwrap();
    }

    pub async fn Process(&self, reqQueueRx: mpsc::Receiver<FuncReq>) -> Result<()> {
        let mut client = self.WaitForPod().await?;

        self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::Ready(self.clone()));

        let mut reqQueueRx = reqQueueRx;
        
        loop {
            tokio::select! {
                e = self.ProbeLiveness(&mut client) => {
                    self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
                    error!("FuncWorker::Process ProbeLiveness fail with error {:?}", e);
                    break;
                }
                req = reqQueueRx.recv() => {
                    match req {
                        None => {
                            // the FuncAgent close the req channel, the worker needs to stop 
                            self.StopWorker().await?;
                            return Ok(())
                        }
                        Some(req) => {
                            self.ongoingReqCnt.fetch_add(1, Ordering::SeqCst);
                            self.HttpCall(req).await?;
                            self.ongoingReqCnt.fetch_sub(1, Ordering::SeqCst);
                            self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::RequestDone(self.clone()));
                        }
                    }
                }
            }
        }

        return Ok(())
    }

    pub async fn NewHttpCallClient(&self) -> Result<QHttpCallClient> {
        let stream = self.ConnectPod().await?;
        let client = QHttpCallClient::New(stream).await?;
        return Ok(client)
    }

    pub async fn ReturnHttpCallClient(&self, client: QHttpCallClient) -> Result<()> {
        self.connPool.lock().await.push(client);
        return Ok(())
    }

    pub async fn GetHttpCallClient(&self) -> Result<QHttpCallClient> {
        match self.connPool.lock().await.pop() {
            None => return self.NewHttpCallClient().await,
            Some(client) => return Ok(client)
        }
    }

    pub async fn HttpCall(&self, req: FuncReq) -> Result<()> {
        let mut client = self.GetHttpCallClient().await?;

        let promptReq = PromptReq {
            namespace: req.namespace.clone(),
            func: req.funcName.clone(),
            prompt: req.request.clone(),
        };

        let body = serde_json::to_string(&promptReq)?;
        
        let httpReq = Request::post(FUNCCALL_URL)
            .header("Content-Type", "application/json")
            .body(body)?;

        
        match client.Send(httpReq).await {
            Err(e) => {
                let resp = HttpResponse {
                    status: StatusCode::BAD_REQUEST,
                    response: format!("service fail with error {:?}", e),
                };
                req.tx.send(resp).unwrap();
                return Ok(())
            },
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
                            return Ok(())
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
                    response: output
                };
                req.tx.send(resp).unwrap();
                return Ok(())
            }
        }
    }

    pub async fn WaitForPod(&self) -> Result<QHttpClient> {
        let stream = self.ConnectPod().await?;
        
        let mut client = QHttpClient::New(stream).await?;
        self.WaitforLiveness(&mut client).await?;
        
        self.WaitforReadiness(&mut client).await?;    
        return Ok(client)
    }

    pub async fn ProbeLiveness(&self, client: &mut QHttpClient) -> Result<()> {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            match self.HttpPing(client, LIVENESS_URL).await {
                Err(e) => return Err(e),
                Ok(_s) => ()
            }
        }
    }

    pub async fn HttpPing(&self, client: &mut QHttpClient, url: &str) -> Result<()> {
        let url = url.parse::<hyper::Uri>()?;

        let authority = url.authority().unwrap().clone();

        // Create an HTTP request with an empty body and a HOST header
        let req = Request::builder()
            .uri(url)
            .header(hyper::header::HOST, authority.as_str())
            .body(Empty::<Bytes>::new())?;

        // Await the response...
        let res = client.Send(req).await?;
        
        match res.status() {
            StatusCode::OK =>return Ok(()),
            _ => {
                return Err(Error::CommonError(format!("HttpPing fail")));
            }
        }

    }

    pub async fn WaitforLiveness(&self, client: &mut QHttpClient) -> Result<()> {
        for _ in 0..100 {
            match self.HttpPing(client, LIVENESS_URL).await {
                Err(_) => (),
                Ok(s) => return Ok(s),
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!("FuncWorker::ConnectPingPod timeout")));
    }

    pub async fn WaitforReadiness(&self, client: &mut QHttpClient) -> Result<()> {
        for _ in 0..100 {
            match self.HttpPing(client, READINESS_URL).await {
                Err(_) => (),
                Ok(s) => return Ok(s),
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!("FuncWorker::ConnectPingPod timeout")));
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

        return Err(Error::CommonError(format!("FuncWorker::ConnectPingPod timeout")));
    }

    pub async fn TryConnectPod(&self, port: u16) -> Result<TcpStream> {
        let addr = self.ipAddr.AsBytes();
        let stream = TSOT_CLIENT.get().unwrap().Connect(&self.namespace, addr, port).await?;
        return Ok(stream)
    }

    pub async fn StartWorker(namespace: &str, name: &str, funcPackage: &FuncPackage) -> Result<IpAddress> {
        let mut client = na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888").await?;
        
        let mounts = vec![
            na::Mount {
                host_path: "/home/brad/rust/Quark/test".to_owned(),
                mount_path: "/test".to_owned(),
            }
        ];

        let commands = funcPackage.spec.commands.clone();
        let mut envs = Vec::new();

        for e in &funcPackage.spec.envs {
            envs.push(Env {
                name: e.0.clone(),
                value: e.1.clone(),
            })
        }
        let _envs = funcPackage.spec.envs.clone();

        let request = tonic::Request::new(na::CreateFuncPodReq {
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            image: "ubuntu".into(),
            commands: commands,
            envs: envs,
            mounts: mounts,
            ports: Vec::new(),
        });

        let response = client.create_func_pod(request).await?;
        let resp = response.into_inner();
    
        if resp.error.is_empty() {
            let addr = IpAddress(resp.ipaddress);
            return Ok(addr)
        }
        
        return Err(Error::CommonError(format!("create pod fail with error {}", resp.error)));
    }

    pub async fn StopWorker(&self) -> Result<()> {
        let mut client = na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888").await?;
    
        let request = tonic::Request::new(na::TerminatePodReq {
            namespace: "ns1".into(),
            name: "name1".into()
        });
        let response = client.terminate_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() != 0 {
            error!("Fail to stop worker {} {}", self.namespace, self.funcName);
        }

        return Ok(())
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
    pub namespace: String,
    pub funcName: String,
    pub request: String,
    pub tx: oneshot::Sender<HttpResponse>
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
        return Ok(Self {
            sender: sender,
        })
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
        return Ok(Self {
            sender: sender,
        })
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
