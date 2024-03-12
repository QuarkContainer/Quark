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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use core::ops::Deref;

use tokio::sync::oneshot;
use tokio::sync::{mpsc, Notify};

use hyper::StatusCode;

use qshare::common::*;

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
    RequestDone(u64), // reqId
    WorkerFail(FuncWorker),
    IdleTimeout(FuncWorker),
}

#[derive(Debug)]
pub struct WorkerStateUpdate {
    pub workerId: u64,
    pub update: WorkerUpdate
}

#[derive(Debug)]
pub struct FuncAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub parallelLevel: usize,

    pub waitingReqs: VecDeque<FuncReq>,
    pub reqQueueTx: mpsc::Sender<FuncReq>,
    pub workerStateUpdateTx: mpsc::Sender<WorkerStateUpdate>,
    pub priQueue: Vec<FuncWorker>,
    pub availableSlot: usize,
    pub workers: BTreeMap<String, FuncWorker>
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
    pub async fn Process(
        &self, 
        reqQueueRx: mpsc::Receiver<FuncReq>,
        workerStateUpdateRx: mpsc::Receiver<WorkerStateUpdate>
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
                        self.ProcessReq(req);
                    } else {
                        unreachable!("FuncAgent::Process reqQueueRx closed");
                    }
                }
                stateUpdate = workerStateUpdateRx.recv() => {
                    if let Some(update) = stateUpdate {
                        match update.update {
                            WorkerUpdate::Ready(worker) => {
                                self.AddSlot(worker.AvailableSlot());
                                self.lock().unwrap().priQueue.push(worker);
                            }
                            WorkerUpdate::RequestDone(_id) => {
                                self.AddSlot(1);
                                self.TryProcessOneReq();
                            }
                            WorkerUpdate::WorkerFail(worker) => {
                                let slot = worker.AvailableSlot();
                                self.RemoveSlot(slot);
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


    pub fn AddSlot(&self, cnt: usize) {
        self.lock().unwrap().availableSlot += cnt;
    }

    pub fn RemoveSlot(&self, cnt: usize) {
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

    pub fn ProcessReq(&self, req: FuncReq) {
        let mut inner = self.lock().unwrap();

        if inner.availableSlot == 0 {
            inner.waitingReqs.push_back(req);
        } else {
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

#[derive(Debug)]
pub struct FuncWorkerInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub workerId: u64,
    pub namespace: String,
    pub funcName: String,
    pub ipAddr: [u8; 4],
    pub port: u16,
    pub parallelLevel: usize,
    pub ongoingReqCnt: usize,

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
    pub fn AvailableSlot(&self) -> usize {
        unimplemented!()
    }

    pub fn AssignReq(&self, req: FuncReq) {
        self.reqQueue.try_send(req).unwrap();
    }

    // pub fn Process(&self, reqQueueRx: mpsc::Receiver<FuncReq>) -> Result<()> {
    //     unimplemented!()
    // }

    // pub async fn ReadinessPing(&self, namespace: &str, ip: &[u8; 4], port: u16) -> Result<()> {
    //     unimplemented!()
    // }
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
    pub tx: oneshot::Sender<HttpResponse>
}