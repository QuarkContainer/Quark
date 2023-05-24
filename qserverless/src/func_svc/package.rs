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
use core::ops::Deref;
use std::time::SystemTime;

use qobjs::k8s;
use qobjs::common::*;

use crate::func_call::FuncCall;
use crate::task_queue::*;
use crate::scheduler::*;
use crate::func_pod::*;

#[derive(Debug, Clone)]
pub struct PackageId {
    pub namespace: String,
    pub packageName: String,
}

impl ToString for PackageId {
    fn to_string(&self) -> String {
        return format!("{}/{}", &self.namespace, &self.packageName);
    }
}

impl PackageId {
    pub fn New(packetIdStr: &str) -> Result<Self> {
        let strs : Vec<&str> = packetIdStr.splitn(2, "/").collect();
        if strs.len() != 2 {
            return Err(Error::CommonError(format!("invalid PackageId str {:?}", packetIdStr)));
        }

        return Ok(Self {
            namespace: strs[0].to_string(),
            packageName: strs[1].to_string(),
        })
    }
}

impl Ord for PackageId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.namespace == other.namespace {
            return other.packageName.cmp(&self.packageName);
        }

        return other.namespace.cmp(&other.namespace);
    }
}

impl PartialOrd for PackageId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PackageId {
    fn eq(&self, other: &Self) -> bool {
        return self.namespace == other.namespace && self.packageName == other.packageName;
    }
}

impl Eq for PackageId {}

#[derive(Debug)]
pub struct PackageInner  {
    pub namespace: String,
    pub name: String,

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
    pub fn PopKeepalivePod(&mut self) -> Option<FuncPod> {
        let mut iter = self.keepalivePods.iter().rev();
        let (time, pod) = match iter.next() {
            None => return None,
            Some((time, pod)) => (*time, pod.clone()),
        };

        drop(iter);

        self.keepalivePods.remove(&time);
        return Some(pod);
    }

    pub fn RemoveKeepalivePod(&mut self, pod: &FuncPod) -> Result<()> {
        self.keepalivePods.remove(&pod.KeepaliveTime()?);
        return Ok(())
    }

    pub fn PushKeepalivePod(&mut self, pod: &FuncPod) -> Result<()> {
        let time = pod.KeepaliveTime()?;
        self.keepalivePods.insert(time, pod.clone());
        return Ok(())
    }

    // when there is a new task, return task which needs creating new pod
    pub fn OnNewTask(&mut self, task: &FuncCall) -> Result<Option<FuncCall>> {
        match self.PopKeepalivePod() {
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
                pod.SetKeepalive();
                self.PushKeepalivePod(pod)?;
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
    pub fn OnFreePod(&mut self, pod: &FuncPod) -> Result<(bool, Option<FuncCall>)> {
        match self.waitingQueue.Pop() {
            None => {
                pod.SetKeepalive();
                self.PushKeepalivePod(pod)?;
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
    pub fn PackageId(&self) -> PackageId {
        let inner = self.lock().unwrap();
        return PackageId { 
            namespace: inner.namespace.clone(), 
            packageName: inner.name.clone()
        }
    }

    pub fn Name(&self) -> String {
        return self.lock().unwrap().name.clone();
    }

    pub fn Namespace(&self) -> String {
        return self.lock().unwrap().namespace.clone();
    }

    pub fn Spec(&self) -> k8s::PodSpec {
        return self.lock().unwrap().spec.clone();
    }

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

pub struct PackageMgr {
    pub packages: Mutex<BTreeMap<PackageId, Package>>,
}

impl PackageMgr {
    pub fn New() -> Self {
        return Self {
            packages: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn Get(&self, packageId: &PackageId) -> Option<Package> {
        return self.packages.lock().unwrap().get(packageId).cloned();
    }
}