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

#![allow(non_snake_case)]

use std::env;

use qshare::common::IpAddress;
use qshare::na;
use qshare::na::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Vec::new();
    for arg in env::args() {
        args.push(arg);
    }

    let cmd = if args.len() < 2 { "new" } else { &args[1] };

    match cmd {
        "new" => {
            return NewPod().await;
        }
        "remove" => {
            return RemovePod().await;
        }
        "get" => {
            return GetPod().await;
        }
        _ => {
            panic!("doesn't support the command {:?}", cmd);
        }
    }
}

async fn RemovePod() -> Result<(), Box<dyn std::error::Error>> {
    let mut client =
        na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
            .await?;

    let request = tonic::Request::new(TerminatePodReq {
        tenant: "t1".into(),
        namespace: "ns1".into(),
        name: "name1".into(),
    });
    let response = client.terminate_pod(request).await?;

    println!("RESPONSE={:?}", response);

    Ok(())
}

async fn GetPod() -> Result<(), Box<dyn std::error::Error>> {
    let mut client =
        na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
            .await?;

    let request = tonic::Request::new(GetPodReq {
        tenant: "t1".into(),
        namespace: "ns1".into(),
        name: "name1".into(),
    });
    let response = client.get_pod(request).await?;

    let resp = response.into_inner();

    println!("revision={:?}", resp.revision);
    println!("pod=");
    let lines = resp.pod.split("\n");
    for l in lines {
        println!("{}", l);
    }

    // let json = serde_json::json!(&resp.pod);
    // print!("resp1={}", serde_json::to_string_pretty(&json).unwrap()); // &'\n'.to_string()));
    println!("err={:?}", resp.error);

    Ok(())
}

async fn NewPod() -> Result<(), Box<dyn std::error::Error>> {
    let mut client =
        na::node_agent_service_client::NodeAgentServiceClient::connect("http://127.0.0.1:8888")
            .await?;

    let args: Vec<String> = std::env::args().collect();
    let cmd = if args.len() > 2 {
        args[2].clone()
    } else {
        "default".to_owned()
    };

    let mut commands = match &cmd[..] {
        "rserver" => vec!["/test/rust/dns/target/debug/server".to_owned()],
        "rclient" => vec!["/test/rust/dns/target/debug/client".to_owned()],
        "server" => vec!["/test/rust/dns/target/debug/server".to_owned()],
        "client" => vec!["/test/rust/dns/target/debug/client".to_owned()],
        "dns" => vec!["/test/rust/dns/target/debug/dns".to_owned()],
        "cat" => vec!["/usr/bin/cat".to_owned(), "/etc/resolv.conf".to_owned()],
        "ping" => vec!["ping".to_owned(), "ping".to_owned()],
        _ => vec!["/usr/bin/sleep".to_owned(), "1".to_owned()],
    };

    for i in 3..args.len() {
        commands.push(args[i].to_owned());
    }

    // let commands = vec![
    //     "/usr/bin/echo".to_owned(),
    //     "asdf >".to_owned(),
    //     "/test/a.txt".to_owned()
    // ];

    // let commands = vec![
    //     "/usr/bin/cp".to_owned(),
    //     "/test/a.txt".to_owned(),
    //     "/test/b.txt".to_owned()
    // ];

    // let commands = vec![
    //     "/usr/bin/rm".to_owned(),
    //     "/test/a.txt".to_owned()
    // ];

    // let commands = vec![
    //     "/usr/bin/touch".to_owned(),
    //     "/test/a.txt".to_owned()
    // ];

    // let commands = vec![
    //     "/test/unixsocket/client/target/debug/client".to_owned(),
    //     "200".to_owned()
    // ];

    // let commands = vec![
    //     "/test/c/server".to_owned()
    // ];

    // let commands = vec![
    //     "/usr/bin/sleep".to_owned(),
    //     "200".to_owned()
    // ];

    let envs = vec![na::Env {
        name: "testenv".to_owned(),
        value: "testenv123".to_owned(),
    }];

    let ports = vec![na::ContainerPort {
        host_port: 1234,
        container_port: 1234,
    }];

    let mounts = vec![na::Mount {
        host_path: "/home/huawei/cchen/".to_owned(),
        mount_path: "/cchen".to_owned(),
    }];

    let request: tonic::Request<CreateFuncPodReq> = tonic::Request::new(CreateFuncPodReq {
        tenant: "t1".into(),
        namespace: "ns1".into(),
        name: cmd.into(), //"name1".into(),
        image: "ubuntu".into(),
        labels: Vec::new(),
        annotations: Vec::new(),
        commands: commands,
        envs: envs,
        mounts: mounts,
        ports: ports,
    });

    let response = client.create_func_pod(request).await?;

    let resp: CreateFuncPodResp = response.into_inner();
    let addr = IpAddress(resp.ipaddress);
    println!("Ipaddr is {:?} RESPONSE={:?}", addr.AsBytes(), resp);

    Ok(())
}
