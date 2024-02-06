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

use cni_plugin::error::CniError;
use tonic::Request;

use cni_plugin::*;
use qshare::config::TSOT_CNI_PORT;
use qshare::tsot_cni;
use qshare::common::*;

#[macro_use]
extern crate log;
extern crate simple_logging;

pub async fn get_pod_sandbox_addr(namespace: &str, pod_uid: &str, pod_name: &str, container_id: &str) -> Result<tsot_cni::GetPodSandboxAddrResp> {
    let addr = format!("http://127.0.0.1:{}", TSOT_CNI_PORT);
    error!("get_pod_sandbox_addr 1 {}", &addr);
    let mut client = tsot_cni::tsot_cni_service_client::TsotCniServiceClient::connect(addr).await?;
    let req = Request::new(tsot_cni::GetPodSandboxAddrReq {
        namespace: namespace.to_owned(),
        pod_uid: pod_uid.to_owned(),
        pod_name: pod_name.to_owned(),
        container_id: container_id.to_owned(),
    });

    let response = client.get_pod_sandbox_addr(req).await?;
    error!("get_pod_sandbox_addr 3 {:?}", &response);
    let resp = response.into_inner();

    return Ok(resp);
}

pub async fn remove_pod_sandbox(namespace: &str, pod_uid: &str, pod_name: &str, container_id: &str) -> Result<tsot_cni::RemovePodSandboxResp> {
    let addr = format!("http://127.0.0.1:{}", TSOT_CNI_PORT);
    let mut client = tsot_cni::tsot_cni_service_client::TsotCniServiceClient::connect(addr).await?;
    let req = Request::new(tsot_cni::RemovePodSandboxReq{
        namespace: namespace.to_owned(),
        pod_uid: pod_uid.to_owned(), 
        pod_name: pod_name.to_owned(),
        container_id: container_id.to_owned(),
    });

    let response = client.remove_pod_sandbox(req).await?;
    let resp = response.into_inner();

    return Ok(resp);
}

#[tokio::main]
async fn main() {
    log4rs::init_file("/etc/quark/cni_logging_config.yaml", Default::default()).unwrap();

    let mut namespace = "".to_owned();
    let mut podname = "".to_owned();
    let mut pod_uid = "".to_owned();

    match std::env::var("CNI_ARGS") {
        Ok(args) => {
            let splitted = args.split(";");
            for item in splitted {
                let keyval = item.split("=").collect::<Vec<&str>>();
                assert!(keyval.len() == 2);
                match keyval[0] {
                    "K8S_POD_NAMESPACE" => {
                        namespace = keyval[1].to_owned();
                    }
                    "K8S_POD_NAME" => {
                        podname = keyval[1].to_owned();
                    }
                    "K8S_POD_UID" => {
                        pod_uid = keyval[1].to_owned();
                    }
                    _ => ()
                }
            }
        }
        _ => ()
    }

    error!("namespace is {namespace}, podname is {podname}, pod_uid is {pod_uid}");

    let input = Cni::load().into_inputs().unwrap();
    error!("plugins is {:#?}", &input);

    let container_id = input.container_id.clone();

    let cni_version = input.config.cni_version.clone(); // for error
	
    match input.command {
        Command::Add => {
            let addr = match get_pod_sandbox_addr(&namespace, &pod_uid, &podname, &container_id).await {
                Err(e) => {
                    error!("get_pod_sandbox_addr error  1 {:?}", e);
                    let err = CniError::Generic(format!("get pod address fail with {:?}", e));
                    reply::reply(err.into_reply(cni_version))
                }
                Ok(resp) => {
                    if resp.error.len() == 0 {
                        resp.ip_addr
                    } else {
                        let err = CniError::Generic(format!("get pod address fail with {:?}", resp.error));
                        error!("get_pod_sandbox_addr error  2 {:?}", &err);
                        reply::reply(err.into_reply(cni_version))
                    }
                }
            };

            let a = (addr >> 24) as u8;
            let b = (addr >> 16) as u8;
            let c = (addr >> 8) as u8;
            let d = (addr >> 0) as u8;
            let str = format!("{a}.{b}.{c}.{d}");

            let res = reply::SuccessReply {
                cni_version,
                interfaces: Default::default(),
                ips: vec![
                    reply::Ip {
                        address: str.parse().unwrap(), // "10.1.0.4".parse().unwrap(),
                        gateway: None,
                        interface: None
                    }
                ],
                // ips: Default::default(),
                routes: Default::default(),
                dns: Default::default(),
                specific: Default::default(),
            };
        
            reply::reply(res);
        }
        Command::Del => {
            match remove_pod_sandbox(&namespace, &pod_uid, &podname, &container_id).await {
                Err(e) => {
                    let _err = CniError::Generic(format!("remove_pod_sandbox fail with {:?}", e));
                    // ignore any fail to avoid block pods killing
                    // reply::reply(err.into_reply(cni_version))
                }
                Ok(resp) => {
                    if resp.error.len() != 0 {
                        let _err = CniError::Generic(format!("remove_pod_sandbox fail with {:?}", resp.error));
                        // ignore any fail to avoid block pods killing
                        // reply::reply(err.into_reply(cni_version))
                    }
                }
            };

            let res = reply::SuccessReply {
                cni_version,
                interfaces: Default::default(),
                ips: Default::default(),
                routes: Default::default(),
                dns: Default::default(),
                specific: Default::default(),
            };
        
            reply::reply(res);
        }
        cmd => {
            let err = CniError::Generic(format!("unsupported command {:?}", cmd));
            reply::reply(err.into_reply(cni_version))
        }
    }
}