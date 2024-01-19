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

pub mod cadvisor;
pub mod cri;
pub mod runtime;
pub mod qcontainer;
pub mod qpod;
pub mod qnode;
pub mod container_agent;
pub mod pod_agent;
pub mod pm_msg;

// pub mod podMgr;

use once_cell::sync::OnceCell;

use runtime::runtime::*;
use runtime::image_mgr::*;
use cadvisor::provider::CadvisorInfoProvider;

pub static RUNTIME_MGR: OnceCell<RuntimeMgr> = OnceCell::new();
pub static IMAGE_MGR: OnceCell<ImageMgr> = OnceCell::new();
pub static CADVISOR_PROVIDER: OnceCell<CadvisorInfoProvider> = OnceCell::new();