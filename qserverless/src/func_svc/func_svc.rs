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
use std::sync::Arc;
use std::time::SystemTime;
use core::ops::Deref;

use qobjs::common::*;
use qobjs::k8s;

use crate::scheduler::Resource;
use crate::task_queue::*;
use crate::func_context::*;

#[derive(Debug)]
pub struct PackageInner  {
    pub namespace: String,
    pub package: String,

    pub spec: k8s::PodSpec,

    pub reqResource: Resource,

    // the pod count which runing task
    pub runningPodCnt: usize,
    // the pod count which in creating state
    pub creatingPodCnt: usize,

    // total waiting tasks
    pub waitingQueue: PackageTaskQueue,
    pub keepalivePods: BTreeMap<SystemTime, FuncPod>,
}

impl PackageInner {
    pub fn PopPod(&mut self) -> Option<FuncPod> {
        let mut iter = self.keepalivePods.iter().rev();
        let (time, pod) = match iter.next() {
            None => return None,
            Some((time, pod)) => (*time, pod.clone()),
        };

        drop(iter);

        self.keepalivePods.remove(&time);
        return Some(pod);
    }

    pub fn RemoveKeepalivePod(&mut self, pod: &FuncPod) {
        self.keepalivePods.remove(&pod.IdleTime());
    }

    pub fn PushPod(&mut self, pod: &FuncPod) {
        let time = *pod.lastIdleTime.lock().unwrap();
        self.keepalivePods.insert(time, pod.clone());
    }

    // when there is a new task, return task which needs creating new pod
    pub fn OnNewTask(&mut self, task: &TaskItem) -> Result<Option<TaskItem>> {
        match self.PopPod() {
            Some(pod) => {
                assert!(self.waitingQueue.waitingTask == 0);
                pod.RunTask(task)?;
                self.runningPodCnt += 1;
                return Ok(None)
            }
            None => {
                self.waitingQueue.Push(task);
                return Ok(self.waitingQueue.GetWaitingTask(self.creatingPodCnt));
            }
        }
    }

    // when a new Pod is created, return whether the pod has been keepalived
    pub fn OnCreatedPod(&mut self, pod: &FuncPod) -> Result<bool> {
        match self.waitingQueue.Pop() {
            None => {
                pod.SetIdle();
                self.PushPod(pod);
                return Ok(true)
            }
            Some(t) => {
                pod.RunTask(&t)?;
                self.runningPodCnt += 1;
                return Ok(false)
            }
        }
    }

    // when a pod finish processing last task, return the task which needs removed from global task Queue
    pub fn OnFreePod(&mut self, pod: &FuncPod) -> Result<(bool, Option<TaskItem>)> {
        match self.waitingQueue.Pop() {
            None => {
                pod.SetIdle();
                self.PushPod(pod);
                return Ok((true, None))
            }
            Some(t) => {
                pod.RunTask(&t)?;
                let removeTask = self.waitingQueue.GetWaitingTask(self.creatingPodCnt);
                return Ok((false, removeTask));
            }
        }
    }

    pub fn TopPriority(&self) -> usize {
        return self.waitingQueue.TopPriority();
    }
}

#[derive(Debug, Clone)]
pub struct Package(Arc<Mutex<PackageInner>>);

impl Deref for Package {
    type Target = Arc<Mutex<PackageInner>>;

    fn deref(&self) -> &Arc<Mutex<PackageInner>> {
        &self.0
    }
}

impl Package {
    pub fn ReqResource(&self) -> Resource {
        return self.lock().unwrap().reqResource;
    }

    pub fn AddFuncPod(&self, _pod: &FuncPod) -> Result<()> {
        unimplemented!();
    }

    pub fn PopPod(&self) -> Option<FuncPod> {
        unimplemented!();
    }

    pub fn TopPriority(&self) -> usize {
        return self.lock().unwrap().TopPriority();
    }

    // need create pod for the top priority, if pri is 10, no need create pod
    /*pub fn TopPriNeedCreatingPod(&self) -> usize {
        let inner = self.lock().unwrap();
        return inner.waitingQueue.PriAfterNTask(inner.creatingPod as usize);
    }*/

   
}

#[derive(Debug)]
pub struct Func {
    pub namespace: String,
    pub package: String,
    pub name: String,
}

#[derive(Debug)]
pub struct FuncPodInner {
    pub id: String,
    pub packageId: PackageId,
    pub node: RunNode,
    pub lastIdleTime: Mutex<SystemTime>,
}

#[derive(Debug, Clone)]
pub struct FuncPod(Arc<FuncPodInner>);

impl Deref for FuncPod {
    type Target = Arc<FuncPodInner>;

    fn deref(&self) -> &Arc<FuncPodInner> {
        &self.0
    }
}

impl FuncPod {
    pub fn RunTask(&self, _task: &TaskItem) -> Result<()> {
        unimplemented!();
    }

    pub fn SetIdle(&self) -> SystemTime {
        let curr = SystemTime::now();
        *self.lastIdleTime.lock().unwrap() = curr;
        return curr;
    }

    pub fn IdleTime(&self) -> SystemTime {
        return *self.lastIdleTime.lock().unwrap();
    }
}

#[derive(Debug)]
pub struct FuncInstance {
    pub id: String,
    pub func: Func,
    pub parameter: String,
    pub runningPod: Option<FuncPod>,
}

#[derive(Debug, Clone)]
pub struct RunNode {
    pub conn: Option<NodeAgentConnection>
}

#[derive(Debug, Clone)]
pub struct NodeAgentConnection {

}

pub struct FuncSvcInner {
    pub queue: TaskQueue,
    
    pub nodes: BTreeMap<String, RunNode>,
    pub packages: BTreeMap<PackageId, Package>,
    pub pods: BTreeMap<FuncId, FuncPod>,
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
        {
            let package = match self.packages.get_mut(&pod.packageId) {
                None => return Err(Error::NotExist),
                Some(p) => p,
            };
    
            self.freeResource = self.freeResource + package.ReqResource();
            self.freeingResource = self.freeingResource - package.ReqResource();
    
        }

        return self.TryCreatePod();
    } 

    // when recieve a new task 
    pub fn OnNewTask(&mut self, task: &TaskItem) -> Result<()> {
        let packageId = task.PackageId();
        let package = self.GetPackage(&packageId)?;
        
        let ret = package.lock().unwrap().OnNewTask(task)?;
        match ret {
            None => return Ok(()),
            Some(t) => {
                self.waitResourceQueue.Enq(&t);
            }
        }

        return self.TryCreatePod();
    }

    pub fn NeedKillPod(&self, package: &Package) -> bool {
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
        let package = self.GetPackage(&pod.packageId)?;

        if self.NeedKillPod(&package) {
            self.EvicatePod(pod, &package.ReqResource())?;
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
        let package = self.GetPackage(&pod.packageId)?;

        if self.NeedKillPod(&package) {
            self.EvicatePod(pod, &package.ReqResource())?;
            return Ok(());
        }

        let keepalive = package.lock().unwrap().OnCreatedPod(pod)?;

        if keepalive {
            self.PushKeepalivePod(pod)?;
        }
        return Ok(());
    }

    pub fn PushKeepalivePod(&mut self, pod: &FuncPod) -> Result<()> {
        let time = pod.IdleTime();
        self.keepalivePods.insert(time, pod.clone());
        return Ok(())
    }

    pub fn CleanKeepalivePod(&mut self) -> Result<()> {
        let mut removePods = Vec::new();
        let mut resource = self.FreeingResource();
        if resource.mem * 100 / self.totalResource.mem > KEEP_BATCHTASK_THRESHOLD {
            return Ok(());
        }

        for (_time, pod) in &self.keepalivePods {
            let package = self.GetPackage(&pod.packageId)?;
            removePods.push((pod.clone(), package.ReqResource()));
            resource = resource + package.ReqResource();
            package.lock().unwrap().RemoveKeepalivePod(pod);
            if resource.mem * 100 / self.totalResource.mem > KEEP_BATCHTASK_THRESHOLD {
                break;
            }
        }

        for (p, resource) in removePods {
            self.keepalivePods.remove(&p.IdleTime());
            self.EvicatePod(&p, &resource)?;
            
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

            let package = self.GetPackage(&task.PackageId())?;
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

    // get next idle pod to free
    pub fn NextIdlePod(&self) -> Option<FuncPod> {
        unimplemented!()
    }

    pub fn KeepalivePod(&self, _pod: &FuncPod) -> Result<()> {
        unimplemented!();
    }

    pub fn FreeingResource(&self) -> Resource {
        return self.freeResource + self.freeingResource;
    }

    pub fn NeedEvictTask(&self, extraResource: &Resource) -> bool {
        let freemem = self.FreeingResource().mem + extraResource.mem;
        return freemem * 100 / self.totalResource.mem < KEEP_BATCHTASK_THRESHOLD;
    }

    pub fn EvicatePod(&mut self, _pod: &FuncPod, freeResource: &Resource) -> Result<()> {
        self.freeingResource = self.freeingResource + *freeResource;
        unimplemented!();
    }
}