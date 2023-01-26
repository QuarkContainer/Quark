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
use svc_client::EndpointsMessage;
use std::collections::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::time::*;
use tonic::Request;

pub mod svc_client {
    tonic::include_proto!("quarkcmsvc");
}

#[derive(Debug)]
pub struct EndpointsInformer {
    pub max_resource_version: i32,
}

impl EndpointsInformer {
    pub fn new() -> EndpointsInformer {
        EndpointsInformer{
            max_resource_version: 0,
        }
    }
}

impl EndpointsInformer {
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;

        let ref endpointses_message = client.list_endpoints(()).await?.into_inner().endpointses;
        if endpointses_message.len() > 0 {
            for endpoints_message in endpointses_message {
                self.handle(endpoints_message);
            }
        }

        loop {
            match self.run_watch().await {
                Ok(_) => {}
                Err(e) => {
                    println!("Endpoints watch error: {:?}", e);
                }
            }

            // println!("EndpointsInformer sleeps 1 second for next watch session.");
            sleep(Duration::from_secs(1)).await;
        }        
    }

    async fn run_watch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Start endpoints run_watch. max_resource_version: {}", self.max_resource_version);
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;
        let mut endpoints_stream = client
            .watch_endpoints(Request::new(MaxResourceVersionMessage {
                max_resource_version: self.max_resource_version,
                // max_resource_version: 0,
            }))
            .await?
            .into_inner();

        while let Some(endpoints_message) = endpoints_stream.message().await? {
            self.handle(&endpoints_message);
        }
        Ok(())
    }

    fn handle(&mut self, endpoints_message: &EndpointsMessage) {
        let name = &endpoints_message.name;
        let mut endpointses_map = RDMA_CTLINFO.endpointses.lock();
        if endpoints_message.event_type == EVENT_TYPE_SET {
            let mut ip_with_ports = Vec::new();
            for ipWithPortStr in &endpoints_message.ip_with_ports {
                if !ipWithPortStr.contains(":") {
                    continue;
                }
                let splitted = ipWithPortStr.split(":").collect::<Vec<_>>();
                ip_with_ports.push(IpWithPort {
                    ip: splitted[0].to_string().parse::<u32>().unwrap().to_be(),
                    port: Port {
                        protocal: splitted[1].to_string(),
                        port: splitted[2].to_string().parse::<u16>().unwrap().to_be(),
                    }
                });
            }

            let endpoints = Endpoints {
                name: name.clone(),
                ip_with_ports: ip_with_ports,
                resource_version: endpoints_message.resource_version,
                index: HashMap::from([(PROTOCOL_TCP.into(), AtomicUsize::new(0)), (PROTOCOL_UDP.into(), AtomicUsize::new(0))]),
            };
            endpointses_map.insert(name.clone(), endpoints);
            if endpoints_message.resource_version > self.max_resource_version {
                self.max_resource_version = endpoints_message.resource_version;
            }
        } else if endpoints_message.event_type == EVENT_TYPE_DELETE {
            if endpointses_map.contains_key(name) {
                if endpointses_map[&name.clone()].resource_version < endpoints_message.resource_version {
                    endpointses_map.remove(name);
                }
            }
        }
        if endpoints_message.resource_version > self.max_resource_version {
            self.max_resource_version = endpoints_message.resource_version;
        }
        println!("Handled Endpoints: {:?}", endpoints_message);
        println!("Debug: endpointses_map len:{} {:?}", endpointses_map.len(), endpointses_map);
    }
}
