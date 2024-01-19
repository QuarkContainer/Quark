// Copyright (c) 2023 Quark Container Authors / 2018 The gVisor Authors.
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

use once_cell::sync::OnceCell;
use uuid::Uuid;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::SystemTime;

use qshare::common::*;
use qshare::crictl;
use qshare::k8s;
use qshare::k8s_util::*;

use super::qnode::QuarkNode;
use super::runtime::runtime::RuntimeMgr;
use super::runtime::image_mgr::ImageMgr;
use super::cadvisor::provider::CadvisorInfoProvider;
use super::qpod::*;
use super::qcontainer::*;


pub struct PodMgr {
    pub cadviceProvider: CadvisorInfoProvider,
    pub runtimeMgr: RuntimeMgr,
    pub imageMgr: ImageMgr,

    pub node: QuarkNode,
    
}

impl PodMgr {
    // pub async fn New() -> Result<Self> {
    //     let cadviceProvider = CadvisorInfoProvider::New().await?;
    //     let runtimeMgr = RuntimeMgr::New(10).await?;
    //     let imageMgr = ImageMgr::New(crictl::AuthConfig::default()).await?;

    //     return Ok(Self {
    //         cadviceProvider: cadviceProvider,
    //         runtimeMgr: runtimeMgr,
    //         imageMgr: imageMgr
    //     });
    // }

    pub fn BuildAQuarkPod(&self, state: PodState, pod: &k8s::Pod, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<QuarkPod> {
        ValidatePodSpec(pod)?;

        let mut pod = pod.clone();
        if pod.metadata.uid.is_none() {
            pod.metadata.uid = Some(Uuid::new_v4().to_string());
        }

        if pod.metadata.annotations.is_none() {
            pod.metadata.annotations = Some(BTreeMap::new());
        }
        
        let inner = QuarkPodInner {
            id: K8SUtil::PodId(&pod),
            podState: state,
            isDaemon: isDaemon,
            pod: Arc::new(RwLock::new(pod)),
            configMap: configMap.clone(),
            runtimePod: None,
            containers: BTreeMap::new(),
            lastTransitionTime: SystemTime::now(), 
        };

        let annotationsNone = inner.pod.read().unwrap().metadata.annotations.is_none();
        if annotationsNone {
            inner.pod.write().unwrap().metadata.annotations = Some(BTreeMap::new());
        }

        //let nodeId = K8SUtil::NodeId(&(*self.node.node.lock().unwrap()));
        //inner.pod.write().unwrap().metadata.annotations.as_mut().unwrap().insert(AnnotationNodeMgrNode.to_string(), nodeId);

        let qpod = QuarkPod(Arc::new(Mutex::new(inner)));
        let k8snode = self.node.node.lock().unwrap().clone();
        qpod.SetPodStatus(Some(&k8snode));
        
        return Ok(qpod);
    }

    pub fn StartPodAgent(&self, qpod: &QuarkPod) -> Result<PodAgent> {
        qpod.Pod().write().unwrap().status.as_mut().unwrap().host_ip = Some(self.node.node.lock().unwrap().status.as_ref().unwrap().addresses.as_ref().unwrap()[0].address.clone());
        qpod.Pod().write().unwrap().metadata.annotations.as_mut().unwrap().insert(AnnotationNodeMgrNode.to_string(), K8SUtil::NodeId(&(*self.node.node.lock().unwrap())));

        let podAgent = PodAgent::New(self.agentChann.clone(), qpod, &self.node.nodeConfig);
        let podId = qpod.lock().unwrap().id.clone();
        self.node.pods.lock().unwrap().insert(podId.clone(), qpod.clone());
        self.pods.lock().unwrap().insert(podId, podAgent.clone());
        return Ok(podAgent);
    }

    pub fn CreatePodAgent(&self, pod: &k8s::Pod, configMap: &Option<k8s::ConfigMap>, isDaemon: bool) -> Result<PodAgent> {
        let qpod = self.BuildAQuarkPod(PodState::Creating, pod, configMap, isDaemon)?;

        let podAgent = self.StartPodAgent(&qpod)?;
        return Ok(podAgent);
    }

    // pub async fn CreatePod(&self, pod: &k8s::Pod, configMap: &k8s::ConfigMap) -> Result<()> {
    // }
}