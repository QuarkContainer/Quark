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

use qobjs::types::FUNCSVC_ADDR;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use qobjs::func;
use qobjs::common::*;
use tonic::Status;

use crate::func_svc::FuncSvc;

#[derive(Debug, Clone)]
pub struct FuncGrpcSvc {}

#[tonic::async_trait]
impl func::func_svc_service_server::FuncSvcService for FuncSvc {
    type StreamProcessStream = ReceiverStream<SResult<func::FuncSvcMsg, tonic::Status>>;
    
    async fn stream_process(
        &self,
        request: tonic::Request<tonic::Streaming<func::FuncSvcMsg>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        let mut stream = request.into_inner();
        let v = stream.message().await.unwrap().unwrap();

        let registeMsg = match v.event_body {
            None => {
                error!("empty event_body");
                return Err(tonic::Status::aborted("empty event_body"));
            }
            Some(body) => {
                match body {
                    func::func_svc_msg::EventBody::FuncAgentRegisterReq(req) => req,
                    x => {
                        return Err(tonic::Status::aborted(format!("expect FuncAgentRegisterReq but get {:?}", x)));
                    }
                }
            }
        };

        let (tx, rx) = mpsc::channel(30);
        match self.OnNodeRegister(registeMsg, stream, tx).await {
            Ok(_) => (),
            Err(e) => {
                return Err(Status::aborted(format!("get error {:?}", e)));
            }
        }
        return Ok(tonic::Response::new(ReceiverStream::new(rx)));
    }
}

pub async fn FuncSvcGrpcService() -> Result<()> {
    use tonic::transport::Server;
    use qobjs::func::func_svc_service_server::FuncSvcServiceServer;

    let svc = FuncSvc::default();

    let funcSvcFuture = Server::builder()
        .add_service(FuncSvcServiceServer::new(svc))
        .serve(FUNCSVC_ADDR.parse().unwrap());

    info!("func service start ...");
    tokio::select! {
        _ = funcSvcFuture => {}
    }

    Ok(())
}