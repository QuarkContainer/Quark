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

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::SystemTime;
use std::result::Result as SResult;
use core::ops::Deref;
use tokio::sync::mpsc;

use qobjs::common::*;
use qobjs::func;

use crate::FUNC_NODE_MGR;
use crate::SCHEDULER;
use crate::func_node::FuncNode;
use crate::scheduler::Resource;
use crate::task_queue::*;
use crate::func_call::*;
use crate::func_pod::*;
use crate::package::*;

#[derive(Debug)]
pub struct Func {
    pub namespace: String,
    pub package: String,
    pub name: String,
}

#[derive(Debug)]
pub struct FuncInstance {
    pub id: String,
    pub func: Func,
    pub parameter: String,
    pub runningPod: Option<FuncPod>,
}

#[derive(Debug, Default)]
pub struct FuncSvcInner {
    pub queue: TaskQueue,
    
    pub packages: BTreeMap<PackageId, Package>,
    pub pods: BTreeMap<FuncCallId, FuncPod>,
    pub funcInstances: BTreeMap<u64, Func>,

    // assume there is no duplicate schedule time
    pub keepalivePods: BTreeMap<SystemTime, FuncPod>,

    // from packageId to Pods
    pub idlePods: BTreeMap<PackageId, Vec<FuncPod>>,
    
    // the task queues are waiting resource to create new pod
    pub waitResourceQueue: TaskQueues,

    pub freeingResource: Resource,
    pub freeResource: Resource,
    pub totalResource: Resource,
}

impl FuncSvcInner {
    pub fn New() -> Self {
        return Self {
            totalResource: Resource { 
                mem: 1024 * 1014 * 1014, 
                cpu: 100 * 1024, 
            },
            freeResource: Resource { 
                mem: 1024 * 1014 * 1014, 
                cpu: 100 * 1024, 
            },
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
pub struct FuncSvc(Mutex<FuncSvcInner>);

impl Deref for FuncSvc {
    type Target = Mutex<FuncSvcInner>;

    fn deref(&self) -> &Mutex<FuncSvcInner> {
        &self.0
    }
}

impl FuncSvc {
    pub fn New() -> Self {
        let inner = FuncSvcInner::New();
        return Self(Mutex::new(inner));
    }

    pub async fn OnNodeRegister(
        &self, 
        req: func::FuncAgentRegisterReq,
        stream: tonic::Streaming<func::FuncSvcMsg>,
        tx: mpsc::Sender<SResult<func::FuncSvcMsg, tonic::Status>>
    ) -> Result<()> {
        let nodeId = req.node_id.clone();
        
        let node = match FUNC_NODE_MGR.Get(&nodeId) {
            Err(_) => {
                let node = FuncNode::New(&nodeId);
                FUNC_NODE_MGR.Insert(&nodeId, &node);
                node
            }
            Ok(n) => n.clone()
        };

        node.CreateProcessor(req, stream, tx).await?;
        return Ok(())
    }
}
impl FuncSvcInner {
    // when a pod exiting process finish, free the occupied resources and schedule creating new pod
    pub fn OnPodExit(&mut self, pod: &FuncPod) -> Result<()> {
        let package = pod.package.clone().unwrap();
    
        self.freeResource = self.freeResource + package.ReqResource();
        if pod.IsExiting() {
            self.freeingResource = self.freeingResource - package.ReqResource();
        }
        
        return self.TryCreatePod();
    } 

    // when recieve a new func call 
    pub fn OnNewFuncCall(&mut self, funcCall: &FuncCall) -> Result<()> {
        let package = funcCall.Package();
        
        let ret = package.lock().unwrap().OnNewFuncCall(funcCall)?;
        match ret {
            None => return Ok(()),
            Some(t) => {
                self.waitResourceQueue.Enq(&t);
            }
        }

        self.TryCreatePod()?;
        return self.CleanKeepalivePods();
    }

    pub fn NeedEvictPod(&self, package: &Package) -> bool {
        let pri = package.TopPriority();
        // if it has batch task, if the system free resource is less than threshold, evict it
        if pri > START_BATCHTASK_PRI
            && self.NeedEvictTask(&package.ReqResource()) {
                return true;
            }
        // it if it has interactive task, compare it with global waiting task, 
        return pri > self.waitResourceQueue.TopPriority() ;
    }

    pub fn GetPackage(&self, packageId: &PackageId) -> Result<Package> {
        match self.packages.get(packageId) {
            None => return Err(Error::NotExist),
            Some(p) => return Ok(p.clone()),
        };
    }

    // it is called when there is a Pod finished a task
    pub fn OnFreePod(&mut self, pod: &FuncPod) -> Result<()> {
        let package = pod.package.clone().unwrap();

        if self.NeedEvictPod(&package) {
            self.EvictPod(pod, &package.ReqResource())?;
            return Ok(());
        }
        
        let (keepalive, task) = package.lock().unwrap().OnFreePod(pod)?;
        match task {
            None => (),
            Some(t) => {
                self.waitResourceQueue.Remove(&t);
            }
        }

        if keepalive {
            self.PushKeepalivePod(pod)?;
        }

        return Ok(());
    }

/*    // it is called when there is a new Pod created
   pub fn OnCreatedPod(&mut self, pod: &FuncPod) -> Result<()> {
        let package = pod.package.clone();

        if self.NeedEvictPod(&package) {
            self.EvictPod(pod, &package.ReqResource())?;
            return Ok(());
        }

        let keepalive = package.lock().unwrap().OnCreatedPod(pod)?;

        if keepalive {
            self.PushKeepalivePod(pod)?;
        }
        return Ok(());
    }
 */
    pub fn PushKeepalivePod(&mut self, pod: &FuncPod) -> Result<()> {
        let time = pod.KeepaliveTime()?;
        self.keepalivePods.insert(time, pod.clone());
        return Ok(())
    }

    pub fn CleanKeepalivePods(&mut self) -> Result<()> {
        let mut removePods = Vec::new();
        let mut resource = self.FreeingResource();
        if resource.mem * 100 / self.totalResource.mem > KEEP_BATCHTASK_THRESHOLD {
            return Ok(());
        }

        for (_time, pod) in &self.keepalivePods {
            let package = pod.package.clone().unwrap();
            removePods.push((pod.clone(), package.ReqResource()));
            resource = resource + package.ReqResource();
            package.lock().unwrap().RemoveKeepalivePod(pod)?;
            if resource.mem * 100 / self.totalResource.mem > KEEP_BATCHTASK_THRESHOLD {
                break;
            }
        }

        for (p, resource) in removePods {
            self.keepalivePods.remove(&p.KeepaliveTime()?);
            self.EvictPod(&p, &resource)?;
            
        }

        return Ok(())
    }

    pub fn TryCreatePod(&mut self) -> Result<()> {
        loop {
            let task = match self.waitResourceQueue.Peek() {
                None => return Ok(()),
                Some(t) => t,
            };

            if !self.freeResource.Fullfil(&task.ReqResource()) {
                return Ok(()); // no enough resource
            }

            let package = task.Package();
            if task.Priority() >= START_BATCHTASK_PRI && self.NeedEvictTask(&package.ReqResource()) {
                // only batch task left and the memory usage has exceed the threshold
                return Ok(())
            }

            self.CreatePod(&package)?;
            self.waitResourceQueue.Remove(&task);
        }
    }

    pub fn CreatePod(&mut self, package: &Package) -> Result<()> {
        assert!(self.freeResource.Fullfil(&package.ReqResource()));
        self.freeResource = self.freeResource - package.ReqResource();
        
        let podName = uuid::Uuid::new_v4().to_string();
        SCHEDULER.Schedule(&podName, package)?;

        return Ok(())
    }

    pub fn FreeingResource(&self) -> Resource {
        return self.freeResource + self.freeingResource;
    }

    pub fn NeedEvictTask(&self, extraResource: &Resource) -> bool {
        let freemem = self.FreeingResource().mem + extraResource.mem;
        return freemem * 100 / self.totalResource.mem < KEEP_BATCHTASK_THRESHOLD;
    }

    pub fn EvictPod(&mut self, pod: &FuncPod, freeResource: &Resource) -> Result<()> {
        *pod.state.lock().unwrap() = FuncPodState::Exiting;
        self.freeingResource = self.freeingResource + *freeResource;
        unimplemented!();
    }
}