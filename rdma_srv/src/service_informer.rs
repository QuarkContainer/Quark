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
use svc_client::ServiceMessage;
use std::collections::HashSet;
use tokio::time::*;
use tonic::Request;

pub mod svc_client {
    tonic::include_proto!("quarkcmsvc");
}

#[derive(Debug)]
pub struct ServiceInformer {
    pub max_resource_version: i32,
}

impl ServiceInformer {
    pub fn new() -> ServiceInformer {
        ServiceInformer{
            max_resource_version: 0,
        }
    }
}

impl ServiceInformer {
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;

        let ref services_message = client.list_service(()).await?.into_inner().services;
        if services_message.len() > 0 {
            for service_message in services_message {
                self.handle(service_message);
            }
        }

        loop {
            match self.run_watch().await {
                Ok(_) => {}
                Err(e) => {
                    println!("Service watch error: {:?}", e);
                }
            }

            // println!("ServiceInformer sleeps 1 second for next watch session.");
            sleep(Duration::from_secs(1)).await;
        }        
    }

    async fn run_watch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Start service run_watch. max_resource_version: {}", self.max_resource_version);
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;
        let mut service_stream = client
            .watch_service(Request::new(MaxResourceVersionMessage {
                max_resource_version: self.max_resource_version,
                // max_resource_version: 0,
            }))
            .await?
            .into_inner();

        while let Some(service_message) = service_stream.message().await? {
            self.handle(&service_message);
        }
        Ok(())
    }

    fn handle(&mut self, service_message: &ServiceMessage) {
        let name = &service_message.name;
        let ip = &service_message.cluster_ip;
        let mut services_map = RDMA_CTLINFO.services.lock();        
        if service_message.event_type == EVENT_TYPE_SET {
            let mut ports = HashSet::new();
            for portStr in &service_message.ports {
                let splitted = portStr.split(":").collect::<Vec<_>>();
                ports.insert(Port {
                    protocal: splitted[0].to_string(),
                    port: splitted[1].to_string().parse::<i32>().unwrap(),
                });
            }

            let service = Service {
                name: name.clone(),
                cluster_ip: ip.clone(),
                ports : ports,
                resource_version: service_message.resource_version,
            };
            services_map.insert(ip.clone(), service);
            if service_message.resource_version > self.max_resource_version {
                self.max_resource_version = service_message.resource_version;
            }
        } else if service_message.event_type == EVENT_TYPE_DELETE {
            if services_map.contains_key(ip) {
                if services_map[ip].resource_version < service_message.resource_version {
                    services_map.remove(ip);
                }
            }
        }
        if service_message.resource_version > self.max_resource_version {
            self.max_resource_version = service_message.resource_version;
        }
        println!("Handled Service: {:?}", service_message);
        println!("Debug: services_map len:{} {:?}", services_map.len(), services_map);
    }
}
