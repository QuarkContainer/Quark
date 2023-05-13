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

#![allow(dead_code)]
#![allow(non_snake_case)]
#[allow(non_camel_case_types)]

use qobjs::cacher_client::CacherClient;
use qobjs::selection_predicate::*;
use qobjs::k8s;

pub mod service_directory {
    tonic::include_proto!("service_directory"); // The string specified here must match the proto package name
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cacheClient = CacherClient::New("http://127.0.0.1:8890".into()).await.unwrap();

    println!("nodelist is {:?}", cacheClient.List("node", "", &ListOption::default()).await.unwrap());

    let mut nodeWs = cacheClient
        .Watch("node", "", &ListOption::default())
        .await.unwrap();

    let mut podWs = cacheClient
        .Watch("pod", "", &ListOption::default())
        .await.unwrap();

    loop {
        tokio::select! {
            //event = nodeWs.Next() => println!("node event is {:#?}", event),
            event = podWs.Next() => {
                let event = event.unwrap().unwrap();
                let podStr = &event.obj.data;
                let pod : k8s::Pod = serde_json::from_str(podStr)?;
                println!("pod event is {:?}/{:#?}", event.type_, pod.status)
            }
        }
    }

    //return Ok(());
}
