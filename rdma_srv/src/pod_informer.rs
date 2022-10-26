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

use super::constants::*;
use crate::common::*;
use crate::rdma_agent;
use crate::rdma_agent::RDMAAgent;
use crate::rdma_ctrlconn::*;
use crate::rdma_srv::*;
use crate::RDMA_CTLINFO;
use crate::RDMA_SRV;
use svc_client::quark_cm_service_client::QuarkCmServiceClient;
use svc_client::MaxResourceVersionMessage;
use svc_client::PodMessage;
use tokio::time::*;
use tonic::Request;

pub mod svc_client {
    tonic::include_proto!("quarkcmsvc");
}

#[derive(Debug)]
pub struct PodInformer {
    pub max_resource_version: i32,
}

impl PodInformer {
    pub fn new() -> PodInformer {
        PodInformer {
            max_resource_version: 0,
        }
    }
}

impl PodInformer {
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;

        let ref pods_message = client.list_pod(()).await?.into_inner().pods;
        if pods_message.len() > 0 {
            for pod_message in pods_message {
                self.handle(pod_message);
            }
        }

        loop {
            match self.run_watch().await {
                Ok(_) => {}
                Err(e) => {
                    println!("Pod watch error: {:?}", e);
                }
            }

            // println!("PodInformer sleeps 1 second for next watch session.");
            sleep(Duration::from_secs(1)).await;
        }
    }

    async fn run_watch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "Start pod run_watch. max_resource_version: {}",
            self.max_resource_version
        );
        let mut client = QuarkCmServiceClient::connect(GRPC_SERVER_ADDRESS).await?;
        let mut pod_stream = client
            .watch_pod(Request::new(MaxResourceVersionMessage {
                max_resource_version: self.max_resource_version,
                // max_resource_version: 0,
            }))
            .await?
            .into_inner();

        while let Some(pod_message) = pod_stream.message().await? {
            self.handle(&pod_message);
        }
        Ok(())
    }

    fn handle(&mut self, pod_message: &PodMessage) {
        let ip = pod_message.ip;
        let mut pods_map = RDMA_CTLINFO.pods.lock();
        let mut podIdToVpcIpAddr_map = RDMA_CTLINFO.podIdToVpcIpAddr.lock();
        let mut vpcIpAddrTopodId_map = RDMA_CTLINFO.vpcIpAddrToPodIdMappings.lock();
        if pod_message.event_type == EVENT_TYPE_SET {
            let pod = Pod {
                key: pod_message.key.clone(),
                vpcId: 1, // TODO: vpcId hard coded for now!
                ip: ip,
                node_name: pod_message.node_name.clone(),
                container_id: pod_message.container_id.clone(),
                resource_version: pod_message.resource_version,
            };
            let vpcIpAddr = VpcIpAddr {
                vpcId: pod.vpcId,
                ipAddr: pod.ip,
            };
            podIdToVpcIpAddr_map.insert(
                pod.container_id.clone(),
                vpcIpAddr,
            );
            vpcIpAddrTopodId_map.insert(
                vpcIpAddr,
                pod.container_id.clone(),
            );
            let mut needInsert = false;
            let mut agent = RDMAAgent::NewDummyAgent();
            match RDMA_SRV.vpcIpAddrToAgents.lock().get(&VpcIpAddr {
                vpcId: 1,
                ipAddr: pod.ip,
            }) {
                Some(_agent) => {}
                None => {
                    match RDMA_SRV
                        .podIdToAgents
                        .lock()
                        .get(pod.container_id.as_bytes())
                    {
                        Some(rdmaAgent) => {
                            *rdmaAgent.ipAddr.lock() = vpcIpAddr.ipAddr;
                            *rdmaAgent.vpcId.lock() = vpcIpAddr.vpcId;
                            needInsert = true;
                            agent = rdmaAgent.clone();
                        }
                        None => {}
                    }
                }
            }
            if needInsert {
                RDMA_SRV.vpcIpAddrToAgents.lock().insert(
                    vpcIpAddr,
                    agent,
                );
            }

            pods_map.insert(pod.key.clone(), pod);
            if pod_message.resource_version > self.max_resource_version {
                self.max_resource_version = pod_message.resource_version;
            }
        } else if pod_message.event_type == EVENT_TYPE_DELETE {
            let vpcIpAddr = VpcIpAddr {
                vpcId: 1,
                ipAddr: ip,
            };
            vpcIpAddrTopodId_map.remove(&vpcIpAddr);
            RDMA_SRV.vpcIpAddrToAgents.lock().remove(&vpcIpAddr);
            if pods_map.contains_key(&pod_message.key) {
                if pods_map[&pod_message.key].resource_version < pod_message.resource_version {
                    let podId = &pods_map[&pod_message.key].container_id;
                    podIdToVpcIpAddr_map.remove(podId);
                    pods_map.remove(&pod_message.key);
                }
            }
        }
        if pod_message.resource_version > self.max_resource_version {
            self.max_resource_version = pod_message.resource_version;
        }
        println!("Handled Pod: {:?}", pod_message);
        println!("Debug: pods_map len:{} {:?}", pods_map.len(), pods_map);
        println!(
            "Debug: containerids_map len:{} {:?}",
            podIdToVpcIpAddr_map.len(),
            podIdToVpcIpAddr_map
        );
        println!(
            "Debug: ip_to_podId_map len:{} {:?}",
            vpcIpAddrTopodId_map.len(),
            vpcIpAddrTopodId_map
        );
    }
}
