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

use crate::common::*;
use super::constants::*;
use crate::rdma_ctrlconn::*;
use crate::RDMA_CTLINFO;
use std::io::Error;
use std::{env, mem, ptr, thread, time};
use svc_client::quark_cm_service_client::QuarkCmServiceClient;
use svc_client::MaxResourceVersionMessage;
use svc_client::IngressMessage;
use tokio::time::*;
use tonic::Request;

pub mod svc_client {
    tonic::include_proto!("quarkcmsvc");
}

#[derive(Debug)]
pub struct IngressInformer {
    pub max_resource_version: i32,
}

impl IngressInformer {
    pub fn new() -> IngressInformer {
        IngressInformer{
            max_resource_version: 0,
        }
    }
}

impl IngressInformer {
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;
        RDMA_CTLINFO.isCMConnected_set(true);

        let ref ingresses_message = client.list_ingress(()).await?.into_inner().ingresses;
        if ingresses_message.len() > 0 {
            for ingress_message in ingresses_message {
                self.handle(ingress_message)?;
            }
        }

        loop {
            match self.run_watch().await {
                Ok(_) => {}
                Err(e) => {
                    println!("Ingress watch error: {:?}", e);
                }
            }

            // println!("IngressInformer sleeps 1 second for next watch session.");
            sleep(Duration::from_secs(1)).await;
        }        
    }

    async fn run_watch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Start ingress run_watch. max_resource_version: {}", self.max_resource_version);
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;
        let mut ingress_stream = client
            .watch_ingress(Request::new(MaxResourceVersionMessage {
                max_resource_version: self.max_resource_version,
                // max_resource_version: 0,
            }))
            .await?
            .into_inner();

        while let Some(ingress_message) = ingress_stream.message().await? {
            self.handle(&ingress_message)?;
        }
        Ok(())
    }

    fn handle(&mut self, ingress_message: &IngressMessage) -> Result<(), Box<dyn std::error::Error>> {
        println!("Start to handle Ingress: {:?}", ingress_message);
        let potNumber = ingress_message.port_number as u16;
        let mut ingresses_map = RDMA_CTLINFO.ingresses.lock();
        if ingress_message.event_type == EVENT_TYPE_SET {
            let ingress = Ingress {
                name: ingress_message.name.clone(),
                service: ingress_message.service.clone(),
                portNumber: potNumber,
                resource_version: ingress_message.resource_version,
            };
            ingresses_map.insert(potNumber, ingress);
            if ingress_message.resource_version > self.max_resource_version {
                self.max_resource_version = ingress_message.resource_version;
            }

            let server_fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
            RDMA_CTLINFO.fds_insert(server_fd, FdType::TCPSocketServer(potNumber));
            unblock_fd(server_fd);
            let epoll_fd = RDMA_CTLINFO.epoll_fd_get();
            epoll_add(epoll_fd, server_fd, read_write_event(server_fd as u64))?;

            unsafe {
                let serv_addr: libc::sockaddr_in = libc::sockaddr_in {
                    sin_family: libc::AF_INET as u16,
                    sin_port: potNumber.to_be(),
                    sin_addr: libc::in_addr {
                        s_addr: u32::from_be_bytes([0, 0, 0, 0]).to_be(),
                    },
                    sin_zero: mem::zeroed(),
                };
        
                let result = libc::bind(
                    server_fd,
                    &serv_addr as *const libc::sockaddr_in as *const libc::sockaddr,
                    mem::size_of_val(&serv_addr) as u32,
                );
                if result < 0 {
                    libc::close(server_fd);
                    panic!("last OS error: {:?}", Error::last_os_error());
                }
                libc::listen(server_fd, 128);
            }

        } else if ingress_message.event_type == EVENT_TYPE_DELETE {
            if ingresses_map.contains_key(&potNumber) {
                if ingresses_map[&potNumber].resource_version < ingress_message.resource_version {
                    ingresses_map.remove(&potNumber);
                }
            }
        }
        if ingress_message.resource_version > self.max_resource_version {
            self.max_resource_version = ingress_message.resource_version;
        }
        println!("Handled Ingress: {:?}", ingress_message);
        println!("Debug: ingresses_map len:{} {:?}", ingresses_map.len(), ingresses_map);

        Ok(())
    }
}
