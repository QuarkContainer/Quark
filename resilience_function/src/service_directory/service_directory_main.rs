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

//#[macro_use]
extern crate lazy_static;

pub mod etcd_store;
pub mod shared;
pub mod selector;
pub mod validation;

pub mod service_directory {
    tonic::include_proto!("service_directory"); // The string specified here must match the proto package name
}

use tonic::{transport::Server, Request, Response, Status};
use service_directory::service_directory_service_server::{ServiceDirectoryService, ServiceDirectoryServiceServer};
use service_directory::*;
use selector::*;

use crate::etcd_store::*;
use crate::shared::common::Result as QResult;

#[derive(Default)]
pub struct ServiceDirectoryImpl {}

pub const KEY_PREFIX : &str = "Quark";

#[tonic::async_trait]
impl ServiceDirectoryService for ServiceDirectoryImpl {

    // This is to verify the grpc server is working.
    // 1. go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
    // 2. Launch the grpc server
    // 3. grpcurl -plaintext -proto resilience_function/proto/service_directory.proto -d '{"client_name": "a client"}' [::]:50071 service_directory.ServiceDirectoryService/TestPing
    async fn test_ping(
        &self,
        request: Request<TestRequestMessage>,
    ) -> Result<Response<TestResponseMessage>, Status> {
        println!("Request from {:?}", request.remote_addr());

        let response = TestResponseMessage {
            server_name: "Server".to_owned(),
        };
        Ok(Response::new(response))
    }

    async fn put(
        &self,
        request: Request<PutRequestMessage>,
    ) -> Result<Response<PutResponseMessage>, Status> {
        println!("Request from {:?}", request.remote_addr());

        let response = PutResponseMessage {
            revision: 1,
        };
        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> QResult<()> {
    //EtcdStoreTest().await?;
    //println!("test 1");
    SelectorTest();
    Ok(())
}

pub fn SelectorTest() {
    let selector = Parse("foo in (a), a=1");
    println!("selector is {:?}", selector);
}

async fn gRpcServer() -> QResult<()> {
    let addr = "[::1]:50071".parse().unwrap();    
    let service_directory_server = ServiceDirectoryImpl::default();

    println!("service_resilience server listening on {}", addr);

    Server::builder()
        .add_service(ServiceDirectoryServiceServer::new(service_directory_server))
        .serve(addr)
        .await?;

    Ok(())
}

async fn EtcdStoreTest() -> QResult<()> {
    let mut store = EtcdStore::New("localhost:2379").await?;

    let val = "test";
    let obj = Object {
        key: Some(ObjectKey {
            kind: "test_kind".into(),
            namespace: "test_namespace".into(),
            name: "test_name".into(),
        }),
        meta: Some(ObjectMeta { labels: Vec::new(), annotations: Vec::new() }),
        attribute: None,
        val: val.as_bytes().to_vec(),
    };

    store.Create("testkey", &DataObject(obj)).await?;
    let obj = store.Get("testkey", 0).await?;
    println!("obj is {:?}", obj);

    store.Delete("testkey", obj.unwrap().0.attribute.unwrap().reversion).await?;
    return Ok(())
}