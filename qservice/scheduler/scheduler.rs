// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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
// limitations under

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(deprecated)]

use qshare::common::*;

pub mod obj_repo;

#[tokio::main]
async fn main() -> Result<()> {
    log4rs::init_file(
        "/etc/quark/scheduler_logging_config.yaml",
        Default::default(),
    )
    .unwrap();

    return Ok(());
}

pub enum PodState {
    Creating,
    Created,
    Running,
    Evacuating,
    Terminating,
    MemHibernating,
    MemHibernated,
    DiskHibernating,
    DiskHibernated,
    Waking,
}

pub struct Scheduler {}

impl Scheduler {
    // need one more pod for the funcpackage to service request
    pub fn ScaleOut(&self, _fpKey: &str) -> Result<()> {
        unimplemented!()
    }

    // ok to scale in one pod
    pub fn ScaleIn(&self, _fpKey: &str) -> Result<()> {
        unimplemented!()
    }

    pub fn DemissionPod(&self, _podid: &str) -> Result<()> {
        unimplemented!()
    }
}
