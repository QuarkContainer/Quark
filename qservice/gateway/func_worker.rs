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

use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio::sync::{mpsc, Notify};
use tokio::time::Duration;

use hyper::client::conn::http1::SendRequest;
use hyper::StatusCode;
use hyper::Request;
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use http_body_util::{BodyExt, Empty};

use qshare::common::*;
use qshare::na;

use crate::TSOT_CLIENT;

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
    pub waitingReqs: VecDeque<FuncReq>,
    pub reqQueueTx: mpsc::Sender<FuncReq>,
    pub workerStateUpdateTx: mpsc::Sender<WorkerUpdate>,
    pub priQueue: Vec<FuncWorker>,
    pub availableSlot: usize,
    pub startingSlot: usize,
    pub workers: BTreeMap<u64, FuncWorker>,
    pub nextWorkerId: u64,
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
    pub async fn New(namespace: &str, funcName: &str) -> Self {
        let (rtx, rrx) = mpsc::channel(30);
        let (wtx, wrx) = mpsc::channel(30);
        let inner = FuncAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            namespace: namespace.to_owned(),
            funcName: funcName.to_owned(),
            waitingReqs: VecDeque::new(),
            reqQueueTx: rtx,
            workerStateUpdateTx: wtx,
            priQueue: Vec::new(),
            availableSlot: 0,
            startingSlot: 0,
            workers: BTreeMap::new(),
            nextWorkerId: 0,
        };

        let ret = Self(Arc::new(Mutex::new(inner)));

        let clone = ret.clone();
        tokio::spawn(async move {
            clone.Process(rrx, wrx).await.unwrap();
        });

        return ret;
    }

    pub fn EnqReq(&self, req: FuncReq) {
        self.lock().unwrap().reqQueueTx.try_send(req).unwrap();
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
            self.lock().unwrap().waitingReqs.push_back(req);

            let mut needNewWorker = false;
            if self.lock().unwrap().waitingReqs.len() > self.lock().unwrap().startingSlot {
                needNewWorker = false;
                self.lock().unwrap().startingSlot += DEFAULT_PARALLEL_LEVEL;
            }

            let namespace = self.lock().unwrap().namespace.clone();
            let funcName = self.lock().unwrap().funcName.clone();
            if needNewWorker {
                let id = self.lock().unwrap().NextWorkerId();
                match FuncWorker::New(id, &namespace, &funcName, DEFAULT_PARALLEL_LEVEL, self).await {
                    Err(e) => {
                        error!("new worker fail with error {:?}", e);
                        self.lock().unwrap().startingSlot -= DEFAULT_PARALLEL_LEVEL;
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
pub const READINESS_PROBE_PORT: u16 = 5555;
pub const LIVENESS_URL: &str = "http://liveness.qactor.io/liveness";
pub const READINESS_URL: &str = "http://liveness.qactor.io/readiness";
pub const FUNCCALL_URL: &str = "http://liveness.qactor.io/funccall";
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
        let addr = Self::StartWorker(namespace, funcName).await?;
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
            funcAgent: funcAgent.clone()
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
        let mut livenessSender = self.WaitForPod().await?;

        self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::Ready(self.clone()));

        let mut reqQueueRx = reqQueueRx;

        loop {
            tokio::select! {
                e = self.ProbeLiveness(&mut livenessSender) => {
                    error!("FuncWorker fail with error {:?}", e);
                    self.funcAgent.SendWorkerStatusUpdate(WorkerUpdate::WorkerFail(self.clone()));
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

    pub async fn HttpCall(&self, req: FuncReq) -> Result<()> {
        let stream = self.ConnectPingPod().await?;
        let io = TokioIo::new(stream);
        let (mut sender, _conn) = hyper::client::conn::http1::handshake(io).await?;

        let httpReq = Request::post(FUNCCALL_URL)
            .body(req.request.clone())?;

    
        match sender.send_request(httpReq).await {
            Err(e) => {
                let resp = HttpResponse {
                    status: StatusCode::BAD_REQUEST,
                    response: format!("service fail with error {:?}", e),
                };
                req.tx.send(resp).unwrap();
                return Ok(())
            },
            Ok(res) => {
                let mut output = String::new();
                while let Some(next) = output.frame().await {
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

    pub async fn WaitForPod(&self) -> Result<SendRequest<Empty<Bytes>>> {
        let stream = self.ConnectPingPod().await?;
        let io = TokioIo::new(stream);
        let (mut sender, _conn) = hyper::client::conn::http1::handshake(io).await?;

        self.WaitforLiveness(&mut sender).await?;

        self.WaitforReadiness(&mut sender).await?;

        return Ok(sender)
    }

    pub async fn ProbeLiveness(&self, sender: &mut SendRequest<Empty<Bytes>>) -> Result<()> {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            match self.HttpPing(sender, LIVENESS_URL).await {
                Err(e) => return Err(e),
                Ok(_s) => ()
            }
        }
    }

    pub async fn HttpPing(&self, sender: &mut SendRequest<Empty<Bytes>>, url: &str) -> Result<()> {
        let url = url.parse::<hyper::Uri>()?;

        let authority = url.authority().unwrap().clone();

        // Create an HTTP request with an empty body and a HOST header
        let req = Request::builder()
            .uri(url)
            .header(hyper::header::HOST, authority.as_str())
            .body(Empty::<Bytes>::new())?;

        // Await the response...
        let res = sender.send_request(req).await?;

        match res.status() {
            StatusCode::OK =>return Ok(()),
            _ => {
                return Err(Error::CommonError(format!("HttpPing fail")));
            }
        }

    }

    pub async fn WaitforLiveness(&self, sender: &mut SendRequest<Empty<Bytes>>) -> Result<()> {
        for _ in 0..100 {
            match self.HttpPing(sender, LIVENESS_URL).await {
                Err(_) => (),
                Ok(s) => return Ok(s),
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!("FuncWorker::ConnectPingPod timeout")));
    }

    pub async fn WaitforReadiness(&self, sender: &mut SendRequest<Empty<Bytes>>) -> Result<()> {
        for _ in 0..100 {
            match self.HttpPing(sender, READINESS_URL).await {
                Err(_) => (),
                Ok(s) => return Ok(s),
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!("FuncWorker::ConnectPingPod timeout")));
    }


    pub async fn ConnectPingPod(&self) -> Result<TcpStream> {
        tokio::time::sleep(Duration::from_millis(100)).await;

        for _ in 0..100 {
            match self.ConnectPod(READINESS_PROBE_PORT).await {
                Err(_) => (),
                Ok(s) => return Ok(s),
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        return Err(Error::CommonError(format!("FuncWorker::ConnectPingPod timeout")));
    }

    pub async fn ConnectPod(&self, port: u16) -> Result<TcpStream> {
        let addr = self.ipAddr.AsBytes();
        let stream = TSOT_CLIENT.get().unwrap().Connect(&self.namespace, addr, port).await?;
        return Ok(stream)
    }

    pub async fn StartWorker(namespace: &str, name: &str) -> Result<IpAddress> {
        let mut client = na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888").await?;
        
        let request = tonic::Request::new(na::CreateFuncPodReq {
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            image: "ubuntu".into(),
            commands: Vec::new(),
            envs: Vec::new(),
            mounts: Vec::new(),
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