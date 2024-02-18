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

use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::RwLock;
use std::result::Result as SResult;

use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status, Request};
use tokio::sync::mpsc;

use crate::qmeta;
use crate::metastore::data_obj::*;
use crate::metastore::selection_predicate::*;
use crate::metastore::selector::*;
use crate::metastore::cache_store::CacheStore;
use crate::metastore::cache_store::ChannelRev;

#[derive(Debug, Default, Clone)]
pub struct SvcDir(Arc<RwLock<SvcDirInner>>);

impl Deref for SvcDir {
    type Target = Arc<RwLock<SvcDirInner>>;

    fn deref(&self) -> &Arc<RwLock<SvcDirInner>> {
        &self.0
    }
}

impl SvcDir {
    pub fn GetCacher(&self, objType: &str) -> Option<CacheStore> {
        return match self.read().unwrap().map.get(objType) {
            None => None,
            Some(c) => Some(c.clone()),
        };
    }
}

#[derive(Debug, Default)]
pub struct SvcDirInner {
    pub map: BTreeMap<String, CacheStore>,
    pub channelRev: ChannelRev,
    pub version: String,
}


#[tonic::async_trait]
impl qmeta::q_meta_service_server::QMetaService for SvcDir {
    async fn version(
        &self,
        request: Request<qmeta::VersionRequestMessage>,
    ) -> SResult<Response<qmeta::VersionResponseMessage>, Status> {
        error!("Request from {:?}", request.remote_addr());

        let response = qmeta::VersionResponseMessage {
            version: self.read().unwrap().version.clone(),
        };
        Ok(Response::new(response))
    }

    async fn get(
        &self,
        request: Request<qmeta::GetRequestMessage>,
    ) -> SResult<Response<qmeta::GetResponseMessage>, Status> {
        let req = request.get_ref();
        let cacher = match self.GetCacher(&req.obj_type) {
            None => {
                return Ok(Response::new(qmeta::GetResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    obj: None,
                }))
            }
            Some(c) => c,
        };

        match cacher.Get(&req.namespace, &req.name, req.revision).await {
            Err(e) => {
                return Ok(Response::new(qmeta::GetResponseMessage {
                    error: format!("Fail: {:?}", e),
                    obj: None,
                }))
            }
            Ok(o) => {
                return Ok(Response::new(qmeta::GetResponseMessage {
                    error: "".into(),
                    obj: match o {
                        None => None,
                        Some(o) => Some(o.Obj()),
                    },
                }))
            }
        }
    }

    async fn list(
        &self,
        request: Request<qmeta::ListRequestMessage>,
    ) -> SResult<Response<qmeta::ListResponseMessage>, Status> {
        let req = request.get_ref();
        let cacher = match self.GetCacher(&req.obj_type) {
            None => {
                return Ok(Response::new(qmeta::ListResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                    objs: Vec::new(),
                }))
            }
            Some(c) => c,
        };

        let labelSelector = match Selector::Parse(&req.label_selector) {
            Err(e) => {
                return Ok(Response::new(qmeta::ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };
        let fieldSelector = match Selector::Parse(&req.field_selector) {
            Err(e) => {
                return Ok(Response::new(qmeta::ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };

        let opts = ListOption {
            revision: req.revision,
            revisionMatch: RevisionMatch::Exact,
            predicate: SelectionPredicate {
                label: labelSelector,
                field: fieldSelector,
                limit: 00,
                continue_: None,
            },
        };

        match cacher.List(&req.namespace, &opts).await {
            Err(e) => {
                return Ok(Response::new(qmeta::ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(resp) => {
                let mut objs = Vec::new();
                for o in resp.objs {
                    objs.push(o.Obj());
                }
                return Ok(Response::new(qmeta::ListResponseMessage {
                    error: "".into(),
                    revision: resp.revision,
                    objs: objs,
                }));
            }
        }
    }

    type WatchStream = std::pin::Pin<Box<dyn futures::Stream<Item = SResult<qmeta::WEvent, Status>> + Send>>;

    async fn watch(
        &self,
        request: Request<qmeta::WatchRequestMessage>,
    ) -> SResult<Response<Self::WatchStream>, Status> {
        let (tx, rx) = mpsc::channel(200);
        let stream = ReceiverStream::new(rx);
        
        let svcDir = self.clone();
        tokio::spawn(async move {
            let req = request.get_ref();
            let cacher = match svcDir.GetCacher(&req.obj_type) {
                None => {
                    tx.send(Err(Status::invalid_argument(&format!(
                        "doesn't support obj type {}",
                        &req.obj_type
                    ))))
                    .await
                    .ok();
                    return;
                }
                Some(c) => c,
            };

            let labelSelector = match Selector::Parse(&req.label_selector) {
                Err(e) => {
                    tx.send(Err(Status::invalid_argument(&format!("Fail: {:?}", e))))
                        .await
                        .ok();
                    
                    return;
                }
                Ok(s) => s,
            };
            let fieldSelector = match Selector::Parse(&req.field_selector) {
                Err(e) => {
                    tx.send(Err(Status::invalid_argument(&format!("Fail: {:?}", e))))
                        .await
                        .ok();
                    return;
                }
                Ok(s) => s,
            };

            let predicate = SelectionPredicate {
                label: labelSelector,
                field: fieldSelector,
                limit: 00,
                continue_: None,
            };

            match cacher.Watch(&req.namespace, req.revision, predicate) {
                Err(e) => {
                    tx.send(Err(Status::invalid_argument(&format!("Fail: {:?}", e))))
                        .await
                        .ok();
                    return;
                }
                Ok(mut w) => loop {
                    let event = w.stream.recv().await;
                    match event {
                        None => return,
                        Some(event) => {
                            let eventType = match event.type_ {
                                EventType::None => 0,
                                EventType::Added => 1,
                                EventType::Modified => 2,
                                EventType::Deleted => 3,
                                EventType::Error(s) => {
                                    tx.send(Err(Status::invalid_argument(&format!(
                                        "Fail: {:?}",
                                        s
                                    ))))
                                    .await
                                    .ok();
                                    return;
                                }
                            };
                            
                            let we = qmeta::WEvent {
                                event_type: eventType,
                                obj: Some(event.obj.Obj()),
                            };
                            match tx.send(Ok(we)).await {
                                Ok(()) => (),
                                Err(e) => {
                                    tx.send(Err(Status::invalid_argument(&format!(
                                        "Fail: {:?}",
                                        e
                                    ))))
                                    .await
                                    .ok();
                                    return;
                                }
                            }
                        }
                    }
                },
            }
        });

        return Ok(Response::new(Box::pin(stream) as Self::WatchStream));
    }
    
    async fn read_obj(
        &self,
        _request: tonic::Request<qmeta::ReadObjReq>,
    ) -> SResult<tonic::Response<qmeta::ReadObjResp>, tonic::Status>{
        unimplemented!()
        // let req = request.get_ref();
        // match self.objectMgr.ReadObject(&req.namespace, &req.name).await {
        //     Err(e) => {
        //         return Ok(Response::new(qmeta::ReadObjResp {
        //             error: format!("Fail: {:?}", e),
        //             ..Default::default()
        //         }))
        //     }
        //     Ok(data) => {
        //         return Ok(Response::new(qmeta::ReadObjResp {
        //             data: data,
        //             ..Default::default()
        //         }))
        //     }
        // }
    }
    
    async fn list_obj(
        &self,
        _request: tonic::Request<qmeta::ListObjReq>,
    ) -> SResult<tonic::Response<qmeta::ListObjResp>, tonic::Status>{
        unimplemented!()
        // let req = request.get_ref();
        // match self.objectMgr.ListObjects(&req.namespace, &req.prefix).await {
        //     Err(e) => {
        //         return Ok(Response::new(qmeta::ListObjResp {
        //             error: format!("Fail: {:?}", e),
        //             ..Default::default()
        //         }))
        //     }
        //     Ok(data) => {
        //         let mut objs = Vec::new();
        //         for o in data {
        //             objs.push(qmeta::ObjMeta {
        //                 name: o.name,
        //                 size: o.size,
        //             });
        //         }
        //         return Ok(Response::new(qmeta::ListObjResp {
        //             objs: objs,
        //             ..Default::default()
        //         }))
        //     }
        // }
    }
}
