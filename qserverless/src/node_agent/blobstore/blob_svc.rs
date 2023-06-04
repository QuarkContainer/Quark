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
use qobjs::func::BlobSvcReq;
use qobjs::utility::SystemTimeProto;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use qobjs::func;
use qobjs::common::*;
use super::blob_session::BlobSession;

pub struct BlogSvc {
    pub blobSession: BlobSession,
}

impl BlogSvc {
    pub async fn Process(
        &self,
        stream: tonic::Streaming<func::BlobSvcReq>,
        agentTx: mpsc::Sender<SResult<func::BlobSvcResp, tonic::Status>>
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
        msg: BlobSvcReq, 
        agentTx: &mpsc::Sender<SResult<func::BlobSvcResp, tonic::Status>>
    ) -> Result<()> {
        let body = msg.event_body.unwrap();
        let msgId = msg.msg_id;
        let resp = match body {
            func::blob_svc_req::EventBody::BlobOpenReq(msg) => {
                match self.blobSession.Open(&msg.namespace, &msg.name) {
                    Ok((id, b)) => {
                        let inner = b.lock().unwrap();
                        let resp = func::BlobOpenResp {
                            msg_id: msg.msg_id,
                            id: id,
                            namespace: inner.namespace.clone(),
                            name: inner.name.clone(),
                            size: inner.size as u64,
                            checksum: inner.checksum.clone(),
                            create_time: Some(SystemTimeProto::FromSystemTime(inner.createTime).ToTimeStamp()),
                            last_access_time: Some(SystemTimeProto::FromSystemTime(inner.lastAccessTime).ToTimeStamp()),
                            error: String::new(),
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobOpenResp(resp))
                        }
                    }
                    Err(e) => {
                        let resp = func::BlobOpenResp {
                            msg_id: msg.msg_id,
                            error: format!("{:?}", e),
                            ..Default::default()
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobOpenResp(resp))
                        }
                    }
                }
            }
            func::blob_svc_req::EventBody::BlobReadReq(msg) => {
                let mut buf = Vec::with_capacity(msg.len as usize);
                buf.resize(msg.len as usize, 0u8);

                match self.blobSession.Read(msg.id, msg.len) {
                    Ok(buf) => {
                        let resp = func::BlobReadResp {
                            msg_id: msg.msg_id,
                            data: buf,
                            error: String::new()
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobReadResp(resp))
                        }
                    }
                    Err(e) => {
                        let resp = func::BlobReadResp {
                            msg_id: msg.msg_id,
                            data: Vec::new(),
                            error: format!("{:?}", e),
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobReadResp(resp))
                        }
                    }
                }
            }
            func::blob_svc_req::EventBody::BlobSeekReq(msg) => {
                match self.blobSession.Seek(msg.id, msg.seek_type, msg.pos) {
                    Ok(offset) => {
                        let resp = func::BlobSeekResp {
                            msg_id: msg.msg_id,
                            offset: offset,
                            error: String::new()
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobSeekResp(resp))
                        }
                    }
                    Err(e) => {
                        let resp = func::BlobSeekResp {
                            msg_id: msg.msg_id,
                            offset: 0,
                            error: format!("{:?}", e),
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobSeekResp(resp))
                        }
                    }
                }
            }
            func::blob_svc_req::EventBody::BlobCloseReq(msg) => {
                match self.blobSession.Close(msg.id) {
                    Ok(()) => {
                        let resp = func::BlobCloseResp {
                            msg_id: msg.msg_id,
                            error: String::new()
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobCloseResp(resp))
                        }
                    }
                    Err(e) => {
                        let resp = func::BlobCloseResp {
                            msg_id: msg.msg_id,
                            error: format!("{:?}", e),
                        };
                        func::BlobSvcResp {
                            msg_id: msgId,
                            event_body: Some(func::blob_svc_resp::EventBody::BlobCloseResp(resp))
                        }
                    }
                }
            }
        };
        
        match agentTx.send(Ok(resp)).await {
            Ok(()) => return Ok(()),
            Err(_e) => return Err(Error::CommonError(format!("send fail ...")))
        };
    }
}

#[tonic::async_trait]
impl func::blob_service_server::BlobService for BlogSvc {
    type StreamProcessStream = ReceiverStream<SResult<func::BlobSvcResp, tonic::Status>>;
    
    async fn stream_process(
        &self,
        _request: tonic::Request<tonic::Streaming<func::BlobSvcReq>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        unimplemented!();
    }
}