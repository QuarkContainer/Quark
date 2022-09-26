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
// limitations under the License.

use crate::common::*;
use super::constants::*;
use crate::rdma_ctrlconn::*;
use crate::RDMA_CTLINFO;
use svc_client::quark_cm_service_client::QuarkCmServiceClient;
use svc_client::MaxResourceVersionMessage;
use svc_client::ConfigMapMessage;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::time::*;
use tonic::Request;

pub mod svc_client {
    tonic::include_proto!("quarkcmsvc");
}

#[derive(Debug)]
pub struct ConfigMapInformer {
    pub max_resource_version: i32,
}

impl ConfigMapInformer {
    pub fn new() -> ConfigMapInformer {
        ConfigMapInformer{
            max_resource_version: 0,
        }
    }
}

impl ConfigMapInformer {
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;

        let ref configMaps_message = client.list_config_map(()).await?.into_inner().config_maps;
        if configMaps_message.len() > 0 {
            for configMap_message in configMaps_message {
                self.handle(configMap_message);
            }
        }

        loop {
            match self.run_watch().await {
                Ok(_) => {}
                Err(e) => {
                    println!("ConfigMap watch error: {:?}", e);
                }
            }

            // println!("ConfigMapInformer sleeps 1 second for next watch session.");
            sleep(Duration::from_secs(1)).await;
        }        
    }

    async fn run_watch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Start configMap run_watch. max_resource_version: {}", self.max_resource_version);
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;
        let mut configMap_stream = client
            .watch_config_map(Request::new(MaxResourceVersionMessage {
                max_resource_version: self.max_resource_version,
                // max_resource_version: 0,
            }))
            .await?
            .into_inner();

        while let Some(configMap_message) = configMap_stream.message().await? {
            self.handle(&configMap_message);
        }
        Ok(())
    }

    fn handle(&mut self, configMap_message: &ConfigMapMessage) {
        let name = &configMap_message.name;
        let mut configMaps_map = RDMA_CTLINFO.configMaps.lock();
        if configMap_message.event_type == EVENT_TYPE_SET {
            let configMap = ConfigMap {
                name: name.clone(),
                value: configMap_message.value.clone(),
                resource_version: configMap_message.resource_version,
                index: AtomicUsize::new(0),
            };
            configMaps_map.insert(name.clone(), configMap);
            if configMap_message.resource_version > self.max_resource_version {
                self.max_resource_version = configMap_message.resource_version;
            }
        } else if configMap_message.event_type == EVENT_TYPE_DELETE {
            if configMaps_map.contains_key(name) {
                if configMaps_map[&name.clone()].resource_version < configMap_message.resource_version {
                    configMaps_map.remove(name);
                }
            }
        }
        if configMap_message.resource_version > self.max_resource_version {
            self.max_resource_version = configMap_message.resource_version;
        }
        println!("Handled ConfigMap: {:?}", configMap_message);
        println!("Debug: configMaps_map len:{} {:?}", configMaps_map.len(), configMaps_map);
    }
}
