// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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

use qshare::common::*;
use qshare::na;
use tonic::transport::Server;

use crate::pod_handler::PodHandlerMsg;
use crate::POD_HANDLER;

pub struct SchedulerSvc {}

#[tonic::async_trait]
impl na::scheduler_service_server::SchedulerService for SchedulerSvc {
    async fn ask_func_pod(
        &self,
        request: tonic::Request<na::AskFuncPodReq>,
    ) -> SResult<tonic::Response<na::AskFuncPodResp>, tonic::Status> {
        let msg: na::AskFuncPodReq = request.into_inner();
        POD_HANDLER
            .get()
            .unwrap()
            .EnqMsg(&PodHandlerMsg::AskFuncPod(msg));
        return Ok(tonic::Response::new(na::AskFuncPodResp {
            error: "".to_owned(),
        }));
    }

    async fn disable_func_pod(
        &self,
        request: tonic::Request<na::DisableFuncPodReq>,
    ) -> SResult<tonic::Response<na::DisableFuncPodResp>, tonic::Status> {
        let msg = request.into_inner();
        POD_HANDLER
            .get()
            .unwrap()
            .EnqMsg(&PodHandlerMsg::DisableFuncPod(msg));
        return Ok(tonic::Response::new(na::DisableFuncPodResp {
            error: "".to_owned(),
        }));
    }
}

pub async fn RunSchedulerSvc() -> Result<()> {
    let svc = SchedulerSvc {};

    let svcAddr = format!("0.0.0.0:{}", 9008);

    let svcfuture = Server::builder()
        .add_service(na::scheduler_service_server::SchedulerServiceServer::new(
            svc,
        ))
        .serve(svcAddr.parse().unwrap());
    svcfuture.await?;
    return Ok(());
}
