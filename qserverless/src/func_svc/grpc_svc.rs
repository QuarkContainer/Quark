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

use std::result::Result as SResult;

use tokio_stream::wrappers::ReceiverStream;

use qobjs::func;
use qobjs::common::*;

#[derive(Debug, Clone)]
pub struct FuncGrpcSvc {}

#[tonic::async_trait]
impl func::func_svc_service_server::FuncSvcService for FuncGrpcSvc {
    type StreamProcessStream = ReceiverStream<SResult<func::FuncSvcMsg, tonic::Status>>;
    
    async fn stream_process(
        &self,
        request: tonic::Request<tonic::Streaming<func::FuncSvcMsg>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        let mut stream = request.into_inner();
        let v = stream.message().await.unwrap().unwrap();
        println!("FuncSvcService message {:?}", v);
        unimplemented!();
    }
}

pub async fn GrpcService() -> Result<()> {
    use tonic::transport::Server;
    use qobjs::func::func_svc_service_server::FuncSvcServiceServer;

    let svc = FuncGrpcSvc{};

    let qmetaFuture = Server::builder()
        .add_service(FuncSvcServiceServer::new(svc.clone()))
        .serve("127.0.0.1:8891".parse().unwrap());

    info!("nodemgr start ...");
    tokio::select! {
        _ = qmetaFuture => {}
    }

    Ok(())
}