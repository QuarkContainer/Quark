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

use futures::Stream;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};

use qobjs::selection_predicate::{ListOption, SelectionPredicate};
use qobjs::selector::Selector;
use qobjs::types::{DataObject, EventType};

use qobjs::common::Result as QResult;
use qobjs::service_directory::service_directory_service_server::{
    ServiceDirectoryService, ServiceDirectoryServiceServer,
};
use qobjs::service_directory::*;

use crate::svc_dir::*;

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
        error!("Request from {:?}", request.remote_addr());

        let response = TestResponseMessage {
            server_name: "Server".to_owned(),
        };
        Ok(Response::new(response))
    }

    async fn put(
        &self,
        request: Request<PutRequestMessage>,
    ) -> Result<Response<PutResponseMessage>, Status> {
        error!("Request from {:?}", request.remote_addr());

        let response = PutResponseMessage { revision: 1 };
        Ok(Response::new(response))
    }

    async fn create(
        &self,
        request: Request<CreateRequestMessage>,
    ) -> Result<Response<CreateResponseMessage>, Status> {
        //info!("create Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(CreateResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                }))
            }
            Some(c) => c,
        };

        match &req.obj {
            None => {
                return Ok(Response::new(CreateResponseMessage {
                    error: format!("Invalid input: Empty obj"),
                    revision: 0,
                }))
            }
            Some(o) => {
                let dataObj = o.into();
                match cacher.Create(&dataObj).await {
                    Err(e) => {
                        return Ok(Response::new(CreateResponseMessage {
                            error: format!("Fail: {:?}", e),
                            revision: 0,
                        }))
                    }
                    Ok(obj) => {
                        return Ok(Response::new(CreateResponseMessage {
                            error: "".into(),
                            revision: obj.Revision(),
                        }))
                    }
                }
            }
        }
    }

    async fn get(
        &self,
        request: Request<GetRequestMessage>,
    ) -> Result<Response<GetResponseMessage>, Status> {
        //info!("get Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(GetResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    obj: None,
                }))
            }
            Some(c) => c,
        };

        match cacher.Get(&req.namespace, &req.name, req.revision).await {
            Err(e) => {
                return Ok(Response::new(GetResponseMessage {
                    error: format!("Fail: {:?}", e),
                    obj: None,
                }))
            }
            Ok(o) => {
                return Ok(Response::new(GetResponseMessage {
                    error: "".into(),
                    obj: match o {
                        None => None,
                        Some(o) => Some(o.Obj()),
                    },
                }))
            }
        }
    }

    async fn delete(
        &self,
        request: Request<DeleteRequestMessage>,
    ) -> Result<Response<DeleteResponseMessage>, Status> {
        //info!("Delete Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(DeleteResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                }))
            }
            Some(c) => c,
        };

        match cacher.Delete(&req.namespace, &req.name).await {
            Err(e) => {
                return Ok(Response::new(DeleteResponseMessage {
                    error: format!("Fail: {:?}", e),
                    revision: 0,
                }))
            }
            Ok(rev) => {
                return Ok(Response::new(DeleteResponseMessage {
                    error: "".into(),
                    revision: rev,
                }))
            }
        }
    }

    async fn update(
        &self,
        request: Request<UpdateRequestMessage>,
    ) -> Result<Response<UpdateResponseMessage>, Status> {
        //info!("create Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(UpdateResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                }))
            }
            Some(c) => c,
        };

        match &req.obj {
            None => {
                return Ok(Response::new(UpdateResponseMessage {
                    error: format!("Invalid input: Empty obj"),
                    revision: 0,
                }))
            }
            Some(o) => {
                let dataObj: DataObject = o.into();
                match cacher.Update(&dataObj).await {
                    Err(e) => {
                        return Ok(Response::new(UpdateResponseMessage {
                            error: format!("Fail: {:?}", e),
                            revision: 0,
                        }))
                    }
                    Ok(obj) => {
                        return Ok(Response::new(UpdateResponseMessage {
                            error: "".into(),
                            revision: obj.Revision(),
                        }))
                    }
                }
            }
        }
    }

    async fn list(
        &self,
        request: Request<ListRequestMessage>,
    ) -> Result<Response<ListResponseMessage>, Status> {
        //info!("create Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    ..Default::default()
                }))
            }
            Some(c) => c,
        };

        let labelSelector = match Selector::Parse(&req.label_selector) {
            Err(e) => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };
        let fieldSelector = match Selector::Parse(&req.field_selector) {
            Err(e) => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };

        let opts = ListOption {
            revision: req.revision,
            revisionMatch: qobjs::selection_predicate::RevisionMatch::Exact,
            predicate: SelectionPredicate {
                label: labelSelector,
                field: fieldSelector,
                limit: 00,
                continue_: None,
            },
        };

        match cacher.List(&req.namespace, &opts).await {
            Err(e) => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(resp) => {
                let mut objs = Vec::new();
                for o in resp.objs {
                    objs.push(o.Obj());
                }
                return Ok(Response::new(ListResponseMessage {
                    error: "".into(),
                    revision: resp.revision,
                    objs: objs,
                }));
            }
        }
    }

    type WatchStream = Pin<Box<dyn Stream<Item = Result<WEvent, Status>> + Send>>;

    async fn watch(
        &self,
        request: Request<WatchRequestMessage>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        //info!("watch Request {:#?}", &request);

        let (tx, rx) = mpsc::channel(200);
        let stream = ReceiverStream::new(rx);

        tokio::spawn(async move {
            let req = request.get_ref();
            let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
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
                            let we = WEvent {
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
}

type WatchStream = Pin<Box<dyn Stream<Item = Result<WEvent, Status>> + Send>>;

pub async fn gRpcServer() -> QResult<()> {
    let addr = "[::1]:50071".parse().unwrap();
    let service_directory_server = ServiceDirectoryImpl::default();

    info!("service_resilience server listening on {}", addr);

    SVC_DIR.write().await.Init("localhost:2379").await?;

    Server::builder()
        .add_service(ServiceDirectoryServiceServer::new(service_directory_server))
        .serve(addr)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::etcd_store::*;
    use qobjs::common::Error as QError;
    use qobjs::store::ThreadSafeStore;
    use qobjs::{
        cacher_client::*, informer::EventHandler, informer_factory::InformerFactory,
        types::DeltaEvent,
    };
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc::unbounded_channel;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::sync::mpsc::UnboundedSender;
    use tokio::sync::Mutex as TMutex;

    async fn gRPCTest() -> QResult<()> {
        let mut client = CacherClientInner::New("http://[::1]:50071".into()).await?;

        let obj = DataObject::NewPod("namespace1", "name1", "", "")?;
        let rev = client.Create("pod", obj.Obj()).await?;
        let obj = obj.CopyWithRev(rev);

        let mut ws = client
            .Watch("pod", "namespace1", &ListOption::default())
            .await?;
        let event = ws.Next().await?;
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Added);
        assert!(event.obj == obj);

        let objx = DataObject::NewPod("namespace1", "name2", "", "")?;
        let rev = client.Create("pod", objx.Obj()).await?;
        let objx = objx.CopyWithRev(rev);

        let event = ws.Next().await?;
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Added);
        assert!(event.obj == objx);

        let obj1 = client.Get("pod", "namespace1", "name1", 0).await?;
        assert!(obj1.is_some());
        let obj1 = obj1.unwrap();
        assert!(
            obj.clone() == obj1,
            "expect is {:#?}, actual is {:#?}",
            obj,
            obj1
        );

        let objs = client.List("pod", "", &ListOption::default()).await?;
        assert!(objs.objs.len() == 2);
        assert!(
            obj.clone() == objs.objs[0],
            "expect is {:#?}, actual is {:#?}",
            obj,
            objs.objs[0]
        );
        assert!(
            objx.clone() == objs.objs[1],
            "expect is {:#?}, actual is {:#?}",
            objx,
            objs.objs[1]
        );

        let objs = client
            .List("pod", "namespace1", &ListOption::default())
            .await?;
        assert!(objs.objs.len() == 2);
        assert!(
            obj.clone() == objs.objs[0],
            "expect is {:#?}, actual is {:#?}",
            obj,
            objs.objs[0]
        );
        assert!(
            objx.clone() == objs.objs[1],
            "expect is {:#?}, actual is {:#?}",
            objx,
            objs.objs[1]
        );

        let obj2 = DataObject::NewPod("namespace1", "name1", "xxx", "")?;
        let rev = client.Update("pod", &obj2).await?;
        let obj2 = obj2.CopyWithRev(rev);

        let event = ws.Next().await?;
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Modified, "event is {:#?}", event);
        assert!(
            event.obj == obj2,
            "event is {:#?}, expect is {:#?}",
            event,
            obj2
        );

        let obj3 = client.Get("pod", "namespace1", "name1", 0).await?;
        assert!(obj3.is_some());
        let obj3 = obj3.unwrap();
        assert!(obj2 == obj3);

        let rev = client.Delete("pod", "namespace1", "name1").await.unwrap();
        let obj2 = obj2.CopyWithRev(rev);

        let event = ws.Next().await?;
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Deleted, "event is {:#?}", event);
        assert!(
            event.obj == obj2,
            "event is {:#?}, expect is {:#?}",
            event,
            obj2
        );

        let obj4 = client.Get("pod", "namespace1", "name1", 0).await?;
        assert!(obj4.is_none());

        let objs = client.List("pod", "", &ListOption::default()).await?;
        assert!(objs.objs.len() == 1);
        assert!(
            objx.clone() == objs.objs[0],
            "expect is {:#?}, actual is {:#?}",
            objx,
            objs.objs[1]
        );

        return Ok(());
    }

    async fn gRPCSrvTest() -> QResult<()> {
        let mut store = EtcdStore::New("localhost:2379", true).await?;
        let _initRv = store.Clear("pod").await?;
        tokio::spawn(async move {
            // Process each socket concurrently.
            gRpcServer().await.unwrap();
        });

        tokio::time::sleep(Duration::from_secs(1)).await;
        gRPCTest().await?;

        return Ok(());
    }

    //#[test]
    fn gRPCTestSync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(gRPCSrvTest()).unwrap();
    }

    #[derive(Debug)]
    pub struct InformerHandler {
        pub tx: UnboundedSender<DeltaEvent>,
        pub rx: TMutex<UnboundedReceiver<DeltaEvent>>,
    }

    impl EventHandler for InformerHandler {
        fn handle(&self, _store: &ThreadSafeStore, event: &DeltaEvent) {
            self.tx.send(event.clone()).ok();
        }
    }

    impl InformerHandler {
        pub fn New() -> Self {
            let (tx, rx) = unbounded_channel();
            return Self {
                tx: tx,
                rx: TMutex::new(rx),
            };
        }

        pub async fn Read(&self) -> Option<DeltaEvent> {
            return self.rx.lock().await.recv().await;
        }

        pub async fn Pop(&self) -> QResult<Option<DeltaEvent>> {
            tokio::select! {
                d = self.Read() => return Ok(d),
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    return Err(QError::Timeout);
                }
            }
        }
    }

    async fn InformerTest() -> QResult<()> {
        let mut client = CacherClientInner::New("http://[::1]:50071".into()).await?;

        error!("InformerTest 1");
        let factory = InformerFactory::New("http://[::1]:50071", "").await?;
        factory.AddInformer("pod", &ListOption::default()).await?;
        let informer = factory.GetInformer("pod").await?;
        let handler1 = Arc::new(InformerHandler::New());
        let _id1 = informer.AddEventHandler(handler1.clone()).await?;

        let obj = DataObject::NewPod("namespace1", "name1", "", "")?;
        let rev = client.Create("pod", obj.Obj()).await?;
        let obj = obj.CopyWithRev(rev);

        let handler2 = Arc::new(InformerHandler::New());
        let id2 = informer.AddEventHandler(handler2.clone()).await?;

        let store = informer.read().await.store.clone();
        let event = handler1.Pop().await.unwrap();
        assert!(
            event
                == Some(DeltaEvent {
                    type_: EventType::Added,
                    inInitialList: false,
                    obj: obj.clone(),
                    oldObj: None,
                }),
            "event is {:?}/{:?}",
            event,
            &store
        );

        let event = handler2.Pop().await.unwrap();
        assert!(
            event
                == Some(DeltaEvent {
                    type_: EventType::Added,
                    inInitialList: true,
                    obj: obj.clone(),
                    oldObj: None,
                }),
            "event is {:#?}/{:?}",
            event,
            &store
        );

        let objx = DataObject::NewPod("namespace1", "name2", "", "")?;
        let rev = client.Create("pod", objx.Obj()).await?;
        let objx = objx.CopyWithRev(rev);

        let event = handler1.Pop().await.unwrap();
        assert!(
            event
                == Some(DeltaEvent {
                    type_: EventType::Added,
                    inInitialList: false,
                    obj: objx.clone(),
                    oldObj: None,
                }),
            "event is {:?}/{:?}",
            event,
            &store
        );

        let event = handler2.Pop().await.unwrap();
        assert!(
            event
                == Some(DeltaEvent {
                    type_: EventType::Added,
                    inInitialList: false,
                    obj: objx.clone(),
                    oldObj: None,
                }),
            "event is {:?}/{:?}",
            event,
            &store
        );

        informer.RemoveEventHandler(id2).await;

        let obj2 = DataObject::NewPod("namespace1", "name1", "xxx", "")?;
        let rev = client.Update("pod", &obj2).await?;
        let obj2 = obj2.CopyWithRev(rev);

        let event = handler1.Pop().await.unwrap();
        assert!(
            event
                == Some(DeltaEvent {
                    type_: EventType::Modified,
                    inInitialList: false,
                    obj: obj2.clone(),
                    oldObj: Some(obj.clone()),
                }),
            "event is {:?}/{:?}",
            event,
            &store
        );

        match handler2.Pop().await {
            Err(QError::Timeout) => (),
            _ => panic!("handler2 get data after disable"),
        };

        let rev = client.Delete("pod", "namespace1", "name1").await.unwrap();
        let obj2 = obj2.CopyWithRev(rev);

        let event = handler1.Pop().await.unwrap();
        assert!(
            event
                == Some(DeltaEvent {
                    type_: EventType::Deleted,
                    inInitialList: false,
                    obj: obj2.clone(),
                    oldObj: None,
                }),
            "event is {:#?}/{:#?}",
            event,
            &store
        );

        factory.Close().await?;

        return Ok(());
    }

    async fn gRPCSrvTest1() -> QResult<()> {
        let mut store = EtcdStore::New("localhost:2379", true).await?;
        let _initRv = store.Clear("pod").await?;
        tokio::spawn(async move {
            // Process each socket concurrently.
            gRpcServer().await.unwrap();
        });

        tokio::time::sleep(Duration::from_secs(1)).await;
        InformerTest().await?;

        return Ok(());
    }

    #[test]
    fn InformerTestSync() {
        use log::LevelFilter;

        simple_logging::log_to_file("/var/log/quark/service_diretory.log", LevelFilter::Info)
            .unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(gRPCSrvTest1()).unwrap();
    }
}
