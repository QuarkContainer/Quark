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
pub mod nodeagent_store;
pub mod pm_msg;
pub mod node_status;
pub mod podmgr_agent;
pub mod podMgr;
pub mod pod_sandbox;
pub mod cidr;
pub mod namespaceMgr;
pub mod node_register;
pub mod node_mgr;

use std::collections::BTreeMap;
use std::sync::Arc;

use once_cell::sync::OnceCell;

use qshare::config::NodeAgentConfig;
use qshare::config::NodeConfiguration;
use qshare::config::SYSTEM_CONFIGS;
use qshare::config::SystemConfig;
use runtime::runtime::*;
use runtime::image_mgr::*;
use cadvisor::provider::CadvisorInfoProvider;
use nodeagent_store::NodeAgentStore;
use tokio::sync::Notify;

use self::namespaceMgr::NamespaceMgr;

lazy_static::lazy_static! {
    pub static ref NODEAGENT_STORE: NodeAgentStore = {
        NodeAgentStore::New().unwrap()
    };

    pub static ref NODE_READY_NOTIFY : Arc<Notify> = Arc::new(Notify::new());

    pub static ref NODEAGENT_CONFIG: NodeAgentConfig = {
        let systemConfigs: BTreeMap<String, SystemConfig> = serde_json::from_str(SYSTEM_CONFIGS).unwrap();

        let configName = ConfigName();

        let systemConfig = match systemConfigs.get(&configName) {
            None => panic!("there is no system config named {}", configName),
            Some(c) => c,
        };
    
        systemConfig.nodeAgentConfig.clone()
    };

    pub static ref NAMESPACE_MGR : NamespaceMgr = NamespaceMgr::New();

    pub static ref NODE_CONFIG : NodeConfiguration = {
        NodeConfiguration::Default().unwrap()
    };
}

pub fn ConfigName() -> String {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        args[1].clone()
    } else {
        "product".to_string()
    }
}

pub static RUNTIME_MGR: OnceCell<RuntimeMgr> = OnceCell::new();
pub static IMAGE_MGR: OnceCell<ImageMgr> = OnceCell::new();
pub static CADVISOR_PROVIDER: OnceCell<CadvisorInfoProvider> = OnceCell::new();