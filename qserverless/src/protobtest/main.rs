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

//use tonic::{transport::Server, Request, Response, Status};
use service_directory::service_directory_service_client::ServiceDirectoryServiceClient;
use service_directory::*;

pub mod service_directory {
    tonic::include_proto!("service_directory"); // The string specified here must match the proto package name
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ServiceDirectoryServiceClient::connect("http://[::1]:50071").await?;

    let request = tonic::Request::new(PutRequestMessage {
        object_type: "test".into(),
        obj: Some(Object {
            kind: "test_kind".into(),
            namespace: "test_namespace".into(),
            name: "test_name".into(),
            labels: Vec::new(),
            annotations: Vec::new(),
            val: "test".into(),
        }),
    });
    let response = client.put(request).await?;

    println!("RESPONSE={:?}", response);

    Ok(())
}
