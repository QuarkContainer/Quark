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
    pub pods: Vec<FuncPod>,
}

impl PackageInner {
    // when there is a new task, return task which needs creating new pod
    pub fn OnNewTask(&mut self, task: &TaskItem) -> Result<Option<TaskItem>> {
        match self.pods.pop() {
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

    // when a new Pod is created, return 
    pub fn OnCreatedPod(&mut self, pod: &FuncPod) -> Result<()> {
        match self.waitingQueue.Pop() {
            None => {
                self.pods.push(pod.clone());
                return Ok(())
            }
            Some(t) => {
                pod.RunTask(&t)?;
                self.runningPodCnt += 1;
                return Ok(())
            }
        }
    }

    // when a pod finish processing last task, return the task which needs removed from global task Queue
    pub fn OnFreePod(&mut self, pod: &FuncPod) -> Result<Option<TaskItem>> {
        match self.waitingQueue.Pop() {
            None => {
                self.pods.push(pod.clone());
                return Ok(None)
            }
            Some(t) => {
                pod.RunTask(&t)?;
                let removeTask = self.waitingQueue.GetWaitingTask(self.creatingPodCnt);
                return Ok(removeTask);
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

#[derive(Debug, Clone)]
pub struct FuncPod {
    pub id: String,
    pub packageId: PackageId,
    pub node: RunNode,
    pub lastSchedueTime: SystemTime,
}

impl FuncPod {
    pub fn RunTask(&self, _task: &TaskItem) -> Result<()> {
        unimplemented!();
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
    pub lastSchedulePod: BTreeMap<SystemTime, FuncPod>,

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
                return Ok(());
            }
        }
    }

    // try schedule a task in waiting queue, if find a waiting task and success, return true, otherwise false
    /*pub fn TryScheduleTask(&self) -> Result<bool> {
        let task = match self.taskQueues.Peek() {
            None => return Ok(false),
            Some(t) => t,
        };

        let packageId = task.PackageId();
        let package = self.GetPackage(&packageId)?;

        // if it is batch task and the system are in Evicting state, enque task.
        if task.priority >= START_BATCHTASK_PRI && self.NeedEvictTask(&package.ReqResource()) { // it is a batch task
            return Ok(false);
        }

        if self.freeResource.Fullfil(&package.ReqResource()) {
            self.freeResource = self.freeResource - package.ReqResource();
            // try to get a idle pod of the package
            match package.PopPod() {
                None => (),
                Some(p) => {
                    // if there is free pod, that means there is no active task waiting, just run the new one
                    p.RunTask(&task)?;
                    return Ok(true)
                }
            };


        }
        
    }*/

    pub fn NeedKillPod(&self, package: &Package) -> bool {
        return package.TopPriority() > self.waitResourceQueue.TopPriority();
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

        package.lock().unwrap().OnFreePod(pod)?;
        return Ok(());
    }

    // it is called when there is a new Pod created
   pub fn OnCreatedPod(&mut self, pod: &FuncPod) -> Result<()> {
        let package = self.GetPackage(&pod.packageId)?;

        if self.NeedKillPod(&package) {
            self.EvicatePod(pod, &package.ReqResource())?;
            return Ok(());
        }

        package.lock().unwrap().OnCreatedPod(pod)?;
        return Ok(());
    }

    pub fn EnqTask(&mut self, _task: &TaskItem) -> Result<()> {


        unimplemented!();
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