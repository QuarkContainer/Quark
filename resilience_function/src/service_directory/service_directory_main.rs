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
#[macro_use]
extern crate serde_derive;

pub mod etcd_store;
pub mod shared;
pub mod selector;
pub mod validation;
pub mod selection_predicate;
pub mod types;
pub mod watch;
pub mod etcd_client;

pub mod service_directory {
    tonic::include_proto!("service_directory"); // The string specified here must match the proto package name
}

use tonic::{transport::Server, Request, Response, Status};
use service_directory::service_directory_service_server::{ServiceDirectoryService, ServiceDirectoryServiceServer};
use service_directory::*;

use crate::etcd_store::*;
use crate::selection_predicate::ListOption;
use crate::shared::common::Result as QResult;
use crate::selection_predicate::*;

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
    EtcdTest1().await?;
    //println!("test 1");
    //SelectorTest();
    Ok(())
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
    let mut store = EtcdStore::New("localhost:2379", false).await?;

    let val = "test";
    let obj = Object {
        kind: "test_kind".into(),
        namespace: "test_namespace".into(),
        name: "test_name".into(),
        labels: Vec::new(), 
        annotations: Vec::new(),
        val: val.to_string(),
    };

    store.Clear("testkey").await?;
    

    store.Create("testkey/abc", &obj.into()).await?;
    let obj = store.Get("testkey/abc", 0).await?;
    println!("obj is {:?}", obj);

    let objs = store.List("testkey/", &ListOption::default()).await?;

    println!("objs is {:?}", objs);
    store.Delete("testkey/abc", obj.unwrap().lock().reversion).await?;
    return Ok(())
}

pub fn ComputePodKey(obj: &DataObject) -> String {
    return format!("/pods/{}/{}", &obj.Namespace(), &obj.Name());
}

// SeedMultiLevelData creates a set of keys with a multi-level structure, returning a resourceVersion
// from before any were created along with the full set of objects that were persisted
async fn SeedMultiLevelData(store: &mut EtcdStore) -> QResult<(i64, Vec<DataObject>)> {
    // Setup storage with the following structure:
    //  /
    //   - first/
    //  |         - bar
    //  |
    //   - second/
    //  |         - bar
    //  |         - foo
    //  |
    //   - third/
    //  |         - barfoo
    //  |         - foo
    let barFirst = DataObject::NewPod("first", "bar", "", "")?;
    let barSecond = DataObject::NewPod("second", "bar", "", "")?;
    let fooSecond = DataObject::NewPod("second", "foo", "", "")?;
    let barfooThird = DataObject::NewPod("third", "barfoo", "", "")?;
    let fooThird = DataObject::NewPod("third", "foo", "", "")?;

    store.Clear("pods").await?;

    struct Test {
        key: String,
        obj: DataObject,
    }

    let mut tests = [
        Test {
            key: ComputePodKey(&barFirst),
            obj: barFirst,
        },
        Test {
            key: ComputePodKey(&barSecond),
            obj: barSecond,
        },
        Test {
            key: ComputePodKey(&fooSecond),
            obj: fooSecond,
        },
        Test {
            key: ComputePodKey(&barfooThird),
            obj: barfooThird,
        },
        Test {
            key: ComputePodKey(&fooThird),
            obj: fooThird,
        },
    ];

    let initRv = store.Clear("pods").await?;

    for t in &mut tests {
        let obj = t.obj.lock().obj.clone();
        let rev = store.Create(&t.key, &obj).await?;
        t.obj.lock().metadata.reversion = rev;
    }

    let mut pods = Vec::new();
    for t in tests {
        pods.push(t.obj);
    }

    return Ok((initRv, pods))
}

pub async fn EtcdTest1() -> QResult<()> {
    let mut store = EtcdStore::New("localhost:2379", true).await?;

    let (_, preset) = SeedMultiLevelData(&mut store).await?;
    
    let listOptions = ListOption {
        revision: 0,
        revisionMatch: RevisionMatch::Exact,
        predicate: SelectionPredicate { limit:2, ..Default::default() },
    };

    let list = store.List("/pods/second", &listOptions).await?;
    assert!(list.continue_.is_some()==false);

    assert!(list.objs.len()==2, "objs is {:?}", list);
    for i in 0..list.objs.len() {
        assert!(preset[i+1] == list.objs[i], 
            "expect {:#?}, actual {:#?}", preset[i+1], &list.objs[i]);
    }

    return Ok(())
}