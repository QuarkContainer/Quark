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


use tonic::{transport::Server, Request, Response, Status};
use qobjs::service_directory::service_directory_service_server::{ServiceDirectoryService, ServiceDirectoryServiceServer};
use qobjs::service_directory::*;

use qobjs::common::Result as QResult;

#[derive(Default)]
pub struct ServiceDirectoryImpl {}

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

pub async fn gRpcServer() -> QResult<()> {
    let addr = "[::1]:50071".parse().unwrap();    
    let service_directory_server = ServiceDirectoryImpl::default();

    info!("service_resilience server listening on {}", addr);

    Server::builder()
        .add_service(ServiceDirectoryServiceServer::new(service_directory_server))
        .serve(addr)
        .await?;

    Ok(())
}

