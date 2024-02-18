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

use qshare::node::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::Mutex;
use core::ops::Deref;

use qshare::crictl;
use qshare::types::*;

// ContainersReady indicates whether all containers in the pod are ready.
pub const ContainersReady: &str = "ContainersReady";

// ContainerStatus represents the cri status of a container and nodeagent status.
#[derive(Serialize, Deserialize, Debug)]
pub struct RuntimeContainerStatus {
    pub status: crictl::ContainerStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QuarkContainerInner {
    pub state: RuntimeContainerState,
    pub initContainer: bool,
    pub spec: ContainerDef,
    pub runtimeContainer: RuntimeContainer,
    pub containerStatus: Option<crictl::ContainerStatus>,
}

#[derive(Debug, Clone)]
pub struct QuarkContainer(pub Arc<Mutex<QuarkContainerInner>>);

impl Deref for QuarkContainer {
    type Target = Arc<Mutex<QuarkContainerInner>>;

    fn deref(&self) -> &Arc<Mutex<QuarkContainerInner>> {
        &self.0
    }
}

impl QuarkContainer {
    pub fn State(&self) -> RuntimeContainerState {
        return self.lock().unwrap().state
    }

    pub fn SetState(&self, state: RuntimeContainerState) {
        self.lock().unwrap().state = state;
    }

    pub fn ContainerName(&self) -> String {
        return self.lock().unwrap().spec.name.clone();
    }

    pub fn InitContainer(&self) -> bool {
        return self.lock().unwrap().initContainer;
    }

    pub fn ContainerExit(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerExit(&status); 
    }

    pub fn ContainerExitNormal(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerExitNormal(&status); 
    }

    pub fn ContainerExitAbnormal(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerExitAbnormal(&status); 
    }

    pub fn ContainerStatusUnknown(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerStatusUnknown(&status); 
    }

    pub fn ContainerRunning(&self) -> bool {
        let status = self.lock().unwrap().containerStatus.clone();
        return ContainerRunning(&status); 
    }
}


pub fn ContainerExit(status: &Option<crictl::ContainerStatus>) -> bool {
    if let Some(status) = status {
        return status.finished_at != 0 && status.state == crictl::ContainerState::ContainerExited as i32;
    }

    return false;
}

pub fn ContainerExitNormal(status: &Option<crictl::ContainerStatus>) -> bool {
    return ContainerExit(status) && status.as_ref().unwrap().exit_code == 0;
}

pub fn ContainerExitAbnormal(status: &Option<crictl::ContainerStatus>) -> bool {
    return ContainerExit(status) && status.as_ref().unwrap().exit_code != 0;
}

pub fn ContainerStatusUnknown(status: &Option<crictl::ContainerStatus>) -> bool {
    if let Some(status) = status {
        return status.state == crictl::ContainerState::ContainerUnknown as i32;
    }

    return false;
}

pub fn ContainerRunning(status: &Option<crictl::ContainerStatus>) -> bool {
    if let Some(status) = status {
        return status.state == crictl::ContainerState::ContainerRunning as i32;
    }

    return false;
}
