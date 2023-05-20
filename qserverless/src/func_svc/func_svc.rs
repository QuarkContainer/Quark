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

    pub runningPod: u64,
    pub idlePod: u64,
    pub waitingTask: u64,

    pub waitingQueue: TaskQueues,
    pub pods: Vec<FuncPod>,
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
}

#[derive(Debug)]
pub struct Func {
    pub namespace: String,
    pub package: String,
    pub name: String,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct RunNode {
    pub conn: Option<NodeAgentConnection>
}

#[derive(Debug)]
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
    
    pub taskQueues: TaskQueues,

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
    /* resource manager
    1. priority design
       a. 1 ~ 5 interactive
       b. 5 ~ 10 batch
    2. System state (no prempty) expect memory consumption, actual memory consumption
       a. Normal: expect mem < 80 any task can be scheduled
       b. Restrict: expect mem > 80, batch job will be hold and start 
       //c. Interacive queue: all batch task pods has been evited. All interactive task will scheudle one by one
       
    3. Triggers
        1. When there is pod exit
            a. Update system state
            b. try to schedule new task, if so create new pod
        2. When there is task finished  
            a. check whether is pending task
            b. if the pending task can run in current pod, schedule it
            c. if not, evicat the current pod
        3. When there is new task comming
            a. put in the task queue
            b. try to schedule new task
        4. after schedule a new pod
            a. Update expected memory
            b. if enter expected memory > 90%, start pod evict until expected memory less than 90%  
        5. When there is new pod ready
            a. Update actual memory
            b. 
    4. schedule new task
        1. Peek top priority task
        2. if the task 


    1. When the free memory/cpu is less than 10%, start to evict idle pod or stop schedule low pri task
    2. When there is pod exiting event, check whether there is waiting task to fit in memory
    3. when there is task finish, 
        a. check whether system is in evicting state

    */

    pub fn OnPodExit(&mut self, pod: &FuncPod) -> Result<()> {
        {
            let package = match self.packages.get_mut(&pod.packageId) {
                None => return Err(Error::NotExist),
                Some(p) => p,
            };
    
            self.freeResource = self.freeResource + package.ReqResource();
            self.freeingResource = self.freeingResource - package.ReqResource();
        }


        let task = match self.taskQueues.Peek() {
            None => return Ok(()), // no task are waiting resource
            Some(t) => t,
        };


        let packageId = task.PackageId();
        let package = self.GetPackage(&packageId)?;

        if task.priority >= START_BATCHTASK_PRI && self.NeedEvictTask(&package.ReqResource()) { // it is a batch task
            return Ok(());
        }

        return self.TryCreatePod(&package);
    } 

    pub fn OnNewTask(&mut self, task: &TaskItem) -> Result<()> {
        let packageId = task.PackageId();
        let package = match self.packages.get(&packageId) {
            None => return Err(Error::NotExist),
            Some(p) => p.clone(),
        };
        
        if task.priority >= START_BATCHTASK_PRI && self.NeedEvictTask(&package.ReqResource()) { // it is a batch task
            return self.EnqTask(task);
        }


        match package.PopPod() {
            None => (),
            Some(p) => {
                // if there is free pod, that means there is no active task waiting, just run the new one
                return p.RunTask(task);
            }
        };

        self.EnqTask(task)?;

        return self.TryCreatePod(&package);
    }

    pub fn GetPackage(&self, packageId: &PackageId) -> Result<Package> {
        match self.packages.get(packageId) {
            None => return Err(Error::NotExist),
            Some(p) => return Ok(p.clone()),
        };
    }

    // it is called when there is a new Pod ready for there is a Pod finished a task
    pub fn OnFreePod(&mut self, pod: &FuncPod) -> Result<()> {
        let package = self.GetPackage(&pod.packageId)?;

        let podTask = match package.lock().unwrap().waitingQueue.Peek() {
            None => {
                if self.NeedEvictTask(&Resource::default()) {
                    // need free resource for other task
                    return self.EvicatePod(pod, &package.ReqResource());
                }
                
                return self.KeepalivePod(pod);
            }
            Some(t) => t,
        };

        let reqResourceForHigherPri = self.taskQueues.ReqResourceForHigherPri(podTask.priority);

        if !self.FreeingResource().Fullfil(&reqResourceForHigherPri) // need more resource to run lower pri task
            || (podTask.priority >= START_BATCHTASK_PRI && self.NeedEvictTask(&Resource::default()))  {
            return self.EvicatePod(pod, &package.ReqResource());
        } else {
            return pod.RunTask(&podTask)
        }
    }

    pub fn EnqTask(&mut self, _task: &TaskItem) -> Result<()> {
        unimplemented!();
    }

    pub fn TryCreatePod(&mut self, package: &Package) -> Result<()> {
        if !self.freeResource.Fullfil(&package.ReqResource()) {
            //error!("no enough resource to schedule a interactive task");
            return Ok(()) // no enough resource to new the pod
        };

        self.freeResource = self.freeResource - package.ReqResource();

        return self.CreatePod(package);
    }

    pub fn CreatePod(&mut self, _package: &Package) -> Result<()> {
        unimplemented!()
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