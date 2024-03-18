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
use std::sync::Arc;

use qshare::etcd::etcd_store::EtcdStore;
use qshare::metastore::cache_store::CacheStore;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status, Request};
use tokio::sync::mpsc;

use qshare::common::*;
use qshare::metastore::svc_dir::SvcDir;
use qshare::qmeta;
use qshare::metastore::data_obj::*;
use qshare::metastore::selection_predicate::*;
use qshare::metastore::selector::*;

use crate::QletAggrStore::QletAggrStore;

lazy_static::lazy_static! {
    pub static ref ETCD_OBJECTS: Vec<&'static str> = vec!["node_info", "namespace_info", "funcpackage"];
}

pub const VERSION: &str = "0.1";

#[derive(Debug, Default, Clone)]
pub struct StateSvc {
    pub svcDir: SvcDir,
}

impl StateSvc {
    pub async fn EtcdInit(&self, etcdAddr: &str) -> Result<()> {
        let store = EtcdStore::New(etcdAddr, true).await?;
        let channelRev = self.svcDir.read().unwrap().channelRev.clone();
        for i in 0..ETCD_OBJECTS.len() {
            let t = ETCD_OBJECTS[i];
            let c = CacheStore::New(Some(Arc::new(store.clone())), t, 0, &channelRev).await?;
            self.svcDir.write().unwrap().map.insert(t.to_string(), c);
        }
        
        return Ok(());
    }
}

#[tonic::async_trait]
impl qmeta::q_meta_service_server::QMetaService for StateSvc {
    async fn version(
        &self,
        request: Request<qmeta::VersionRequestMessage>,
    ) -> SResult<Response<qmeta::VersionResponseMessage>, Status> {
        error!("Request from {:?}", request.remote_addr());

        let response = qmeta::VersionResponseMessage {
            version: VERSION.to_string(),
        };
        Ok(Response::new(response))
    }

    async fn get(
        &self,
        request: Request<qmeta::GetRequestMessage>,
    ) -> SResult<Response<qmeta::GetResponseMessage>, Status> {
        let req = request.get_ref();
        let cacher = match self.svcDir.GetCacher(&req.obj_type) {
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
        let cacher = match self.svcDir.GetCacher(&req.obj_type) {
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
        
        let svcDir = self.svcDir.clone();
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

pub async fn StateService() -> Result<()> {
    use tonic::transport::Server;
    use qshare::qmeta::q_meta_service_server::QMetaServiceServer;

    let stateSvc = StateSvc::default();
    stateSvc.EtcdInit("localhost:2379").await?;

    let qletAggrStore = QletAggrStore::New(&stateSvc.svcDir.ChannelRev()).await?;

    stateSvc.svcDir.AddCacher(qletAggrStore.NodeStore());
    stateSvc.svcDir.AddCacher(qletAggrStore.PodStore());
    let qletAggrStoreFuture = qletAggrStore.Process();

    let stateSvcFuture = Server::builder()
        .add_service(QMetaServiceServer::new(stateSvc))
        .serve(STATESVC_ADDR.parse().unwrap());

    info!("state service start ...");
    tokio::select! {
        _ = stateSvcFuture => {
            info!("stateSvcFuture finish...");
        }
        ret = qletAggrStoreFuture => {
            info!("qletAggrStoreFuture finish... {:?}", ret);
        }
    }

    Ok(())
}