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
use core::ops::Deref;

use qobjs::common::*;

use crate::scheduler::Resource;
use crate::task_queue::*;
use crate::func_context::*;
use crate::func_pod::*;
use crate::func_node::*;
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

pub struct FuncSvcInner {
    pub queue: TaskQueue,
    
    pub nodes: BTreeMap<String, FuncNode>,
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

pub struct FuncSvc(Mutex<FuncSvcInner>);

impl Deref for FuncSvc {
    type Target = Mutex<FuncSvcInner>;

    fn deref(&self) -> &Mutex<FuncSvcInner> {
        &self.0
    }
}

impl FuncSvcInner {
    // when a pod exiting process finish, free the occupied resources and schedule creating new pod
    pub fn OnPodExit(&mut self, pod: &FuncPod) -> Result<()> {
        let package = pod.package.clone();
    
        self.freeResource = self.freeResource + package.ReqResource();
        self.freeingResource = self.freeingResource - package.ReqResource();

        return self.TryCreatePod();
    } 

    // when recieve a new task 
    pub fn OnNewTask(&mut self, task: &TaskItem) -> Result<()> {
        let package = task.Package();
        
        let ret = package.lock().unwrap().OnNewTask(task)?;
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
        if pri > START_BATCHTASK_PRI
            && self.NeedEvictTask(&package.ReqResource()) {
                return true;
            }
        return package.TopPriority() > self.waitResourceQueue.TopPriority() ;
    }

    pub fn GetPackage(&self, packageId: &PackageId) -> Result<Package> {
        match self.packages.get(packageId) {
            None => return Err(Error::NotExist),
            Some(p) => return Ok(p.clone()),
        };
    }

    // it is called when there is a Pod finished a task
    pub fn OnFreePod(&mut self, pod: &FuncPod) -> Result<()> {
        let package = pod.package.clone();

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

    // it is called when there is a new Pod created
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
            let package = pod.package.clone();
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
            if task.priority >= START_BATCHTASK_PRI && self.NeedEvictTask(&package.ReqResource()) {
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
        unimplemented!();
    }

    pub fn FreeingResource(&self) -> Resource {
        return self.freeResource + self.freeingResource;
    }

    pub fn NeedEvictTask(&self, extraResource: &Resource) -> bool {
        let freemem = self.FreeingResource().mem + extraResource.mem;
        return freemem * 100 / self.totalResource.mem < KEEP_BATCHTASK_THRESHOLD;
    }

    pub fn EvictPod(&mut self, _pod: &FuncPod, freeResource: &Resource) -> Result<()> {
        self.freeingResource = self.freeingResource + *freeResource;
        unimplemented!();
    }
}