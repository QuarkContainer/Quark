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
use svc_client::NodeMessage;
use tokio::time::*;
use tonic::Request;

pub mod svc_client {
    tonic::include_proto!("quarkcmsvc");
}

#[derive(Debug)]
pub struct NodeInformer {
    pub max_resource_version: i32,
}

impl NodeInformer {
    pub fn new() -> NodeInformer {
        NodeInformer{
            max_resource_version: 0,
        }
    }
}

impl NodeInformer {
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;

        let ref nodes_message = client.list_node(()).await?.into_inner().nodes;
        if nodes_message.len() > 0 {
            for node_message in nodes_message {
                self.handle(node_message);
            }
        }

        loop {
            match self.run_watch().await {
                Ok(_) => {}
                Err(e) => {
                    println!("Node watch error: {:?}", e);
                }
            }

            // println!("NodeInformer sleeps 1 second for next watch session.");
            sleep(Duration::from_secs(1)).await;
        }        
    }

    async fn run_watch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // todo Hong: shall share client. And be careful to handle when connection is broken and need reconnect. Watch pod, node, service shall using the same client
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;
        let mut node_stream = client
            .watch_node(Request::new(MaxResourceVersionMessage {
                max_resource_version: self.max_resource_version,
                // max_resource_version: 0,
            }))
            .await?
            .into_inner();

        while let Some(node_message) = node_stream.message().await? {
            self.handle(&node_message);
        }
        Ok(())
    }

    fn handle(&mut self, node_message: &NodeMessage) {
        let ip = node_message.ip;
        let mut nodes_map = RDMA_CTLINFO.nodes.lock();
        let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
        let epoll_fd = RDMA_CTLINFO.epoll_fd_get();
        if node_message.event_type == EVENT_TYPE_SET {
            let node = Node {
                hostname: node_message.hostname.clone(),
                ipAddr: ip,
                subnet: node_message.subnet,
                netmask: node_message.net_mask,
                timestamp: node_message.creation_timestamp,
                resource_version: node_message.resource_version,
            };
            nodes_map.insert(ip, node);
            if node_message.resource_version > self.max_resource_version {
                self.max_resource_version = node_message.resource_version;
            }
            RDMA_CTLINFO.fds_insert(fd, Srv_FdType::NodeEventFd(NodeEvent{
                is_delete: false,
                ip: ip,
            }));
            unblock_fd(fd);
            match epoll_add(epoll_fd, fd, read_write_event(fd as u64)) {
                Ok(_) => {}
                Err(e) => {
                    println!("epoll_add error for node: {:?}", e);
                }
            }            
        } else if node_message.event_type == EVENT_TYPE_DELETE {
            if nodes_map.contains_key(&ip) {
                if nodes_map[&ip].resource_version < node_message.resource_version {
                    nodes_map.remove(&ip);
                    RDMA_CTLINFO.fds_insert(fd, Srv_FdType::NodeEventFd(NodeEvent{
                        is_delete: true,
                        ip: ip,
                    }));
                    unblock_fd(fd);
                    match epoll_add(epoll_fd, fd, read_write_event(fd as u64)) {
                        Ok(_) => {}
                        Err(e) => {
                            println!("epoll_add error for node: {:?}", e);
                        }
                    }
                }
            }
        }
        if node_message.resource_version > self.max_resource_version {
            self.max_resource_version = node_message.resource_version;
        }
        println!("Handled Node: {:?}", node_message);
        println!("Debug: nodes_map len:{} {:?}", nodes_map.len(), nodes_map);
    }
}
