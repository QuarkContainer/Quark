// Copyright (c) 2021 Quark Container Authors
//
// Licensed un&der the Apache License, Version 2.0 (the "License");
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

use crate::common::*;
use spin::Mutex;
use std::{collections::HashMap, collections::HashSet, str::FromStr};

pub struct CtrlInfo {
    // ingresses: port number --> Ingress
    pub ingresses: Mutex<HashMap<u16, Ingress>>,

    // rdma_ingresses: port number --> RdmaIngress
    pub rdma_ingresses: Mutex<HashMap<u16, RdmaIngress>>,

    pub fds: Mutex<HashMap<i32, FdType>>,

    pub isCMConnected: Mutex<bool>,

    pub epoll_fd: Mutex<i32>,
}

impl Default for CtrlInfo {
    fn default() -> CtrlInfo {
        let ingresses: HashMap<u16, Ingress> = HashMap::new();
        let rdma_ingresses: HashMap<u16, RdmaIngress> = HashMap::new();
        let fds: HashMap<i32, FdType> = HashMap::new();
        CtrlInfo {
            ingresses: Mutex::new(ingresses),
            rdma_ingresses: Mutex::new(rdma_ingresses),
            fds: Mutex::new(fds),
            isCMConnected: Mutex::new(false),
            epoll_fd: Mutex::new(0),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Ingress {
    pub name: String,
    pub service: String,
    pub portNumber: u16,
    pub resource_version: i32,
}

#[derive(Default, Debug, Clone)]
pub struct RdmaIngress {
    pub portNumber: u16,
    pub service: String,
    pub targetPortNumber: u16,
    pub resource_version: i32,
}

impl CtrlInfo {
    pub fn isCMConnected_set(&self, value: bool) {
        let mut isCMConnected = self.isCMConnected.lock();
        *isCMConnected = value;
    }

    pub fn isCMConnected_get(&self) -> bool {
        self.isCMConnected.lock().clone()
    }

    pub fn fds_insert(&self, key: i32, value: FdType) {
        self.fds.lock().insert(key, value);
    }

    pub fn fds_get(&self, key: &i32) -> Option<FdType> {
        match self.fds.lock().get(key) {
            Some(fdType) => {
                return Some(fdType.clone());
            },
            None=>{
                return None;
            }
        }
    }

    pub fn epoll_fd_set(&self, value: i32) {
        *self.epoll_fd.lock() = value;
    }

    pub fn epoll_fd_get(&self) -> i32 {
        self.epoll_fd.lock().clone()
    }
}
