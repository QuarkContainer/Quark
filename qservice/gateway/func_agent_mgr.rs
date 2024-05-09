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

use core::ops::Deref;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use qshare::metastore::data_obj::{DeltaEvent, EventType};
use qshare::node::PodDef;
use tokio::sync::oneshot;
use tokio::sync::{mpsc, Notify};

use qshare::{common::*, na};

use crate::func_worker::*;
use crate::OBJ_REPO;
use qshare::obj_mgr::func_mgr::*;

lazy_static::lazy_static! {
    pub static ref FUNCAGENT_MGR: FuncAgentMgr = FuncAgentMgr::default();
}

#[derive(Debug, Default)]
pub struct FuncAgentMgrInner {
    pub agents: BTreeMap<String, FuncAgent>,
}

#[derive(Debug, Default, Clone)]
pub struct FuncAgentMgr(Arc<Mutex<FuncAgentMgrInner>>);

impl Deref for FuncAgentMgr {
    type Target = Arc<Mutex<FuncAgentMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncAgentMgrInner>> {
        &self.0
    }
}

impl FuncAgentMgr {
    pub async fn GetClient(
        &self,
        tenant: &str,
        namespace: &str,
        funcname: &str,
    ) -> Result<QHttpCallClient> {
        let funcPackage = OBJ_REPO
            .get()
            .unwrap()
            .GetFuncPackage(tenant, namespace, funcname)?;

        let agent = {
            let key = funcPackage.spec.Key();
            let mut inner = self.lock().unwrap();
            match inner.agents.get(&key) {
                Some(agent) => agent.clone(),
                None => {
                    let agent = FuncAgent::New(&funcPackage);
                    inner.agents.insert(key, agent.clone());
                    agent
                }
            }
        };

        let (tx, rx) = oneshot::channel();
        agent.EnqReq(tenant, namespace, funcname, tx);
        let client = match rx.await {
            Err(_) => {
                return Err(Error::CommonError(format!("funcworker fail ...")));
            }
            Ok(client) => client,
        };
        return Ok(client);
    }

    pub fn FuncPodEventHandler(&self, event: DeltaEvent) -> Result<()> {
        let obj = event.obj.clone();
        let podDef = PodDef::FromDataObject(obj)?;
        let agent = {
            let key = podDef.FuncPackageKey();

            let mut inner = self.lock().unwrap();
            match inner.agents.get(&key) {
                Some(agent) => agent.clone(),
                None => {
                    let funcPackage = OBJ_REPO.get().unwrap().GetFuncPackage(
                        &podDef.tenant,
                        &podDef.namespace,
                        &podDef.funcname,
                    )?;
                    let agent = FuncAgent::New(&funcPackage);
                    inner.agents.insert(key, agent.clone());
                    agent
                }
            }
        };

        agent.EnqEvent(event);
        return Ok(());
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

    pub waitingReqs: VecDeque<FuncClientReq>,
    pub reqQueueTx: mpsc::Sender<FuncClientReq>,
    pub eventChann: mpsc::Sender<DeltaEvent>,
    pub workerStateUpdateTx: mpsc::Sender<WorkerUpdate>,
    pub availableSlot: usize,
    pub startingSlot: usize,
    pub workers: BTreeMap<String, FuncWorker>,
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

    pub fn Key(&self) -> String {
        return format!("{}/{}/{}", &self.tenant, &self.namespace, &self.funcName);
    }

    pub fn RemoveWorker(&mut self, worker: &FuncWorker) -> Result<()> {
        self.workers.remove(&worker.workerId);
        return Ok(());
    }

    pub fn NextWorkerId(&mut self) -> u64 {
        self.nextWorkerId += 1;
        return self.nextWorkerId;
    }

    pub fn AssignReq(&mut self, req: FuncClientReq) {
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
    pub fn New(funcPackage: &FuncPackage) -> Self {
        let (rtx, rrx) = mpsc::channel(30);
        let (etx, erx) = mpsc::channel(30);
        let (wtx, wrx) = mpsc::channel(30);
        let inner = FuncAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            tenant: funcPackage.spec.tenant.clone(),
            namespace: funcPackage.spec.namespace.clone(),
            funcName: funcPackage.spec.funcname.to_owned(),
            funcPackge: funcPackage.clone(),
            waitingReqs: VecDeque::new(),
            reqQueueTx: rtx,
            eventChann: etx,
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
            clone.Process(rrx, erx, wrx).await.unwrap();
        });

        return ret;
    }

    pub fn EnqReq(
        &self,
        tenant: &str,
        namespace: &str,
        funcname: &str,
        tx: oneshot::Sender<QHttpCallClient>,
    ) {
        let funcReq = FuncClientReq {
            reqId: self.lock().unwrap().NextReqId(),
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            funcName: funcname.to_owned(),
            tx: tx,
        };
        self.lock().unwrap().reqQueueTx.try_send(funcReq).unwrap();
    }

    pub fn EnqEvent(&self, event: DeltaEvent) {
        self.lock().unwrap().eventChann.try_send(event).unwrap();
    }

    pub async fn Close(&self) {
        let closeNotify = self.lock().unwrap().closeNotify.clone();
        closeNotify.notify_one();
    }

    pub async fn AskFuncPod(&self) -> Result<()> {
        let mut schedClient =
            na::scheduler_service_client::SchedulerServiceClient::connect("http://127.0.0.1:9008")
                .await?;

        let tenant = self.lock().unwrap().tenant.clone();
        let namespace = self.lock().unwrap().namespace.clone();
        let funcname = self.lock().unwrap().funcName.clone();
        let req = na::AskFuncPodReq {
            tenant: tenant,
            namespace: namespace,
            funcname: funcname,
        };

        let request = tonic::Request::new(req);
        let response = schedClient.ask_func_pod(request).await?;
        let resp = response.into_inner();
        if resp.error.len() == 0 {
            return Ok(());
        }

        return Err(Error::CommonError(format!(
            "AskFuncPod fail with error {}",
            resp.error
        )));
    }

    pub async fn Process(
        &self,
        reqQueueRx: mpsc::Receiver<FuncClientReq>,
        eventQueueRx: mpsc::Receiver<DeltaEvent>,
        workerStateUpdateRx: mpsc::Receiver<WorkerUpdate>,
    ) -> Result<()> {
        let mut eventQueueRx = eventQueueRx;
        let mut reqQueueRx = reqQueueRx;
        let mut workerStateUpdateRx = workerStateUpdateRx;

        let closeNotify = self.lock().unwrap().closeNotify.clone();

        loop {
            // let key = self.lock().unwrap().Key();
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
                deltaEvent = eventQueueRx.recv() => {
                    if let Some(event) = deltaEvent {
                        self.ProcessDeltaEvent(event).await?;
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
                                // self.lock().unwrap().startingSlot -= slot;
                                self.TryProcessOneReq();
                            }
                            WorkerUpdate::RequestDone(_) => {
                                self.IncrSlot(1);
                                self.TryProcessOneReq();
                            }
                            WorkerUpdate::WorkerFail(worker) => {
                                let slot = worker.parallelLevel;
                                self.DecrSlot(slot);
                                self.lock().unwrap().RemoveWorker(&worker)?;
                                worker.Close().await;
                            }
                            WorkerUpdate::IdleTimeout(worker) => {
                                let slot = worker.parallelLevel;
                                self.DecrSlot(slot);
                                worker.DisableWorker().await?;
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

    pub async fn ProcessDeltaEvent(&self, event: DeltaEvent) -> Result<()> {
        let obj = event.obj.clone();
        assert!(&obj.kind == PodDef::KEY);
        match &event.type_ {
            EventType::Added => {
                self.ProcessAddFuncPod(event).await?;
                return Ok(());
            }
            EventType::Modified => {
                let podDef = PodDef::FromDataObject(obj)?;

                match self.lock().unwrap().workers.get(&podDef.id) {
                    None => {
                        return Err(Error::CommonError(format!(
                            "FuncAgent::ProcessDeltaEvent get unknown pod {:#?}",
                            podDef
                        )));
                    }
                    Some(worker) => {
                        worker.EnqEvent(event);
                    }
                }
            }
            EventType::Deleted => {
                let podDef = PodDef::FromDataObject(obj)?;
                match self.lock().unwrap().workers.remove(&podDef.id) {
                    None => {
                        return Err(Error::CommonError(format!(
                            "FuncAgent::ProcessDeltaEvent get unknown pod {:#?}",
                            podDef
                        )));
                    }
                    Some(worker) => {
                        worker.EnqEvent(event);
                    }
                }
            }
            _o => {
                return Err(Error::CommonError(format!(
                    "NamespaceMgr::ProcessDeltaEvent {:?}",
                    event
                )));
            }
        }

        return Ok(());
    }

    pub async fn ProcessAddFuncPod(&self, event: DeltaEvent) -> Result<()> {
        let obj = event.obj.clone();
        let podDef = PodDef::FromDataObject(obj)?;
        let fp = OBJ_REPO.get().unwrap().GetFuncPackage(
            &podDef.tenant,
            &podDef.namespace,
            &podDef.funcname,
        )?;

        self.lock().unwrap().startingSlot += DEFAULT_PARALLEL_LEVEL;

        let keepaliveTime = fp.spec.keepalivePolicy.keepaliveTime;
        match FuncWorker::New(
            &podDef.id,
            &podDef.tenant,
            &podDef.namespace,
            &podDef.funcname,
            IpAddress(podDef.ipAddr),
            DEFAULT_PARALLEL_LEVEL,
            keepaliveTime,
            self,
        )
        .await
        {
            Err(e) => {
                error!(
                    "FuncAgent::ProcessReq new funcworker fail with error {:?}",
                    e
                );
            }
            Ok(worker) => {
                self.lock()
                    .unwrap()
                    .workers
                    .insert(podDef.id.clone(), worker.clone());
                worker.EnqEvent(event);
            }
        };

        self.lock().unwrap().startingSlot -= DEFAULT_PARALLEL_LEVEL;

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
            None => {
                return;
            }
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

    pub async fn ProcessReq(&self, req: FuncClientReq) {
        // let key = self.lock().unwrap().Key();
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
            if needNewWorker {
                // request scheduler to start a new pod
                match self.AskFuncPod().await {
                    Err(e) => {
                        error!(
                            "FuncAgent::ProcessReq new funcworker fail with error {:?}",
                            e
                        );
                    }
                    Ok(()) => (),
                }
                self.lock().unwrap().startingSlot -= DEFAULT_PARALLEL_LEVEL;
            }
        } else {
            self.lock().unwrap().AssignReq(req);
        }
    }
}
