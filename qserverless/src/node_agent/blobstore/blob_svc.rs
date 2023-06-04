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
use qobjs::func::BlobSvcMsg;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use qobjs::func;
use qobjs::common::*;
use super::blob_session::BlobSession;

pub struct BlogSvc {
    pub session: BlobSession,
}

impl BlogSvc {
    pub async fn Process(
        &self,
        stream: tonic::Streaming<func::BlobSvcMsg>,
        agentTx: mpsc::Sender<SResult<func::BlobSvcMsg, tonic::Status>>
    ) -> Result<()> {
        let mut stream = stream;

        loop {
            let msg = match stream.message().await {
                Err(e) => {
                    error!("BlogSvc lose connection with error {:?}", e);
                    break;
                }
                Ok(m) =>  {
                    match m {
                        None => {
                            error!("BlogSvc lose connection with none message");
                            break;
                        }
                        Some(m) => m
                    }
                }
            };

            self.ProcessMsg(msg, &agentTx).await?;
        }

        return Ok(())
    }

    pub async fn ProcessMsg(
        &self, 
        msg: BlobSvcMsg, 
        agentTx: &mpsc::Sender<SResult<func::BlobSvcMsg, tonic::Status>>
    ) -> Result<()> {
        let body = msg.event_body.unwrap();
        let resp = match body {
            func::blob_svc_msg::EventBody::BlobCreateReq(msg) => {
                match self.session.Create(&msg.namespace, &msg.name) {
                    Ok(id) => {
                        let resp = func::BlobCreateResp {
                            msg_id: msg.msg_id,
                            id: id,
                            error: String::new()
                        };
                        func::BlobSvcMsg {
                            event_body: Some(func::blob_svc_msg::EventBody::BlobCreateResp(resp))
                        }
                    }
                    Err(e) => {
                        let resp = func::BlobCreateResp {
                            msg_id: msg.msg_id,
                            id: 0,
                            error: format!("{:?}", e),
                        };
                        func::BlobSvcMsg {
                            event_body: Some(func::blob_svc_msg::EventBody::BlobCreateResp(resp))
                        }
                    }
                }
            }
            func::blob_svc_msg::EventBody::BlobReadReq(msg) => {
                let mut buf = Vec::with_capacity(msg.len as usize);
                buf.resize(msg.len as usize, 0u8);

                match self.session.Read(msg.id, msg.len) {
                    Ok(buf) => {
                        let resp = func::BlobReadResp {
                            msg_id: msg.msg_id,
                            data: buf,
                            error: String::new()
                        };
                        func::BlobSvcMsg {
                            event_body: Some(func::blob_svc_msg::EventBody::BlobReadResp(resp))
                        }
                    }
                    Err(e) => {
                        let resp = func::BlobReadResp {
                            msg_id: msg.msg_id,
                            data: Vec::new(),
                            error: format!("{:?}", e),
                        };
                        func::BlobSvcMsg {
                            event_body: Some(func::blob_svc_msg::EventBody::BlobReadResp(resp))
                        }
                    }
                }
            }
            
            _ => return Ok(())
        };
        
        match agentTx.send(Ok(resp)).await {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }
}

#[tonic::async_trait]
impl func::blob_service_server::BlobService for BlogSvc {
    type StreamProcessStream = ReceiverStream<SResult<func::BlobSvcMsg, tonic::Status>>;
    
    async fn stream_process(
        &self,
        _request: tonic::Request<tonic::Streaming<func::BlobSvcMsg>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        unimplemented!();
    }
}