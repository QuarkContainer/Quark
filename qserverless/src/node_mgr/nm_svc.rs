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

use std::{collections::BTreeMap, sync::Mutex};

use qobjs::object_mgr::ObjectbMgrTrait;
use qobjs::object_mgr::SqlObjectMgr;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status, Request};
use tokio::sync::mpsc;
use core::ops::Deref;
use std::sync::Arc;
use std::result::Result as SResult;

use qobjs::k8s;

use qobjs::nm as nm_svc;
use qobjs::common::Result;
use qobjs::selection_predicate::*;
use qobjs::selector::*;
use qobjs::types::*;
use qobjs::qmeta as qmeta;

use crate::VERSION;
use crate::na_client::*;
use crate::etcd::etcd_svc::EtcdSvc;
use crate::SVC_DIR;

#[derive(Debug)]
pub struct NodeMgrSvcInner {
    pub clients: Mutex<BTreeMap<String, NodeAgentClient>>,
    pub etcdSvc: EtcdSvc,
    pub objectMgr: SqlObjectMgr,
}

// NodeMgrSvc direct connect to NodeAgent
#[derive(Debug, Clone)]
pub struct NodeMgrSvc(Arc<NodeMgrSvcInner>);

impl Deref for NodeMgrSvc {
    type Target = Arc<NodeMgrSvcInner>;

    fn deref(&self) -> &Arc<NodeMgrSvcInner> {
        &self.0
    }
}

impl NodeMgrSvc {
    pub async fn New(ObjectDbAddr: &str, ) -> Result<Self> {
        let objectMgr = SqlObjectMgr::New(ObjectDbAddr).await?;
        let inner = NodeMgrSvcInner {
            clients: Mutex::new(BTreeMap::new()),
            etcdSvc: EtcdSvc::default(),
            objectMgr,
        };

        return Ok(Self(Arc::new(inner)));
    }

    pub fn NodeAgent(&self, nodeId: &str) -> Option<NodeAgentClient> {
        return self.clients.lock().unwrap().get(nodeId).cloned();
    }
}

#[tonic::async_trait]
impl nm_svc::node_agent_service_server::NodeAgentService for NodeMgrSvc {
    type StreamProcessStream = ReceiverStream<SResult<nm_svc::NodeAgentReq, Status>>;
    async fn stream_process(
        &self,
        request: tonic::Request<tonic::Streaming<nm_svc::NodeAgentRespMsg>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        let stream = request.into_inner();
        let (tx, rx) = mpsc::channel(30);
        let client = NodeAgentClient::New(self, stream, tx);
        tokio::spawn(async move {
            client.Process().await;
        });
        return Ok(Response::new(ReceiverStream::new(rx)));
    }
}

#[tonic::async_trait]
impl qmeta::q_meta_service_server::QMetaService for NodeMgrSvc {
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

    async fn create(
        &self,
        request: Request<qmeta::CreateRequestMessage>,
    ) -> SResult<Response<qmeta::CreateResponseMessage>, Status> {
        let req = request.get_ref();
        let objType = &req.obj_type;
        if objType == QUARK_POD ||  objType == QUARK_NODE {
            match &req.obj {
                None => {
                    let response = qmeta::CreateResponseMessage {
                        error: "NodeMgr get pod create request with zero size obj".to_owned(),
                        ..Default::default()
                    };
                    return Ok(Response::new(response));
                }
                Some(obj) => {
                    let mut pod: k8s::Pod = match serde_json::from_str(&obj.data) {
                        Ok(pod) => pod, 
                        Err(e) => {
                            let response = qmeta::CreateResponseMessage {
                                error: format!("NodeMgr get pod create request with pod json deserilize error {:?}", e),
                                ..Default::default()
                            };
                            return Ok(Response::new(response));
                        }
                    };
                    /*if pod.metadata.annotations.is_none() {
                        let response = qmeta::CreateResponseMessage {
                            error: format!("NodeMgr get pod create request with pod with annotations"),
                            ..Default::default()
                        };
                        return Ok(Response::new(response));
                    }
                    let node = match pod.metadata.annotations.as_ref().unwrap().get(AnnotationNodeMgrNode) {
                        None => {
                            return Ok(Response::new(qmeta::CreateResponseMessage {
                                error: format!("create pod fail with error empty {} annotation", AnnotationNodeMgrNode),
                                ..Default::default()
                            }))
                        },
                        Some(s) => s,
                    };*/
                    let node = match crate::NM_CACHE.get().unwrap().SchedulePod(&pod) {
                        None => {
                            return Ok(Response::new(qmeta::CreateResponseMessage {
                                error: format!("create pod fail as no resource"),
                                ..Default::default()
                            }))
                        },
                        Some(node) => node,
                    };

                    if pod.metadata.annotations.is_none() {
                        pod.metadata.annotations = Some(BTreeMap::new());
                    }

                    error!("nm create pod {:?} on node {}", &pod.metadata.name, &node);

                    pod.metadata.annotations.as_mut().unwrap().insert(AnnotationNodeMgrNode.to_owned(), node.clone());
                    match crate::NM_CACHE.get().unwrap().CreatePod(&node, &pod, &k8s::ConfigMap::default()).await {
                        Err(e) => {
                            return Ok(Response::new(qmeta::CreateResponseMessage {
                                error: format!("create pod fail with error {:?}", e),
                                ..Default::default()
                            }))
                        }
                        Ok(()) =>  {
                            return Ok(Response::new(qmeta::CreateResponseMessage {
                                error: String::new(),
                                ..Default::default()
                            }))
                        }
                    }
                }
            }
        }

        return self.etcdSvc.create(request).await;
    }

    async fn get(
        &self,
        request: Request<qmeta::GetRequestMessage>,
    ) -> SResult<Response<qmeta::GetResponseMessage>, Status> {
        let req = request.get_ref();
        let objType = &req.obj_type;
        if objType == QUARK_POD ||  objType == QUARK_NODE {
            let cacher = match crate::NM_CACHE.get().unwrap().GetCacher(&req.obj_type) {
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
        
        return self.etcdSvc.get(request).await;
    }

    async fn delete(
        &self,
        request: Request<qmeta::DeleteRequestMessage>,
    ) -> SResult<Response<qmeta::DeleteResponseMessage>, Status> {
        let req = request.get_ref();
        let objType = &req.obj_type;
        if objType == QUARK_POD { // ||  objType == QUARK_NODE {
            let podId = format!("{}/{}", &req.namespace, &req.name);
            match crate::NM_CACHE.get().unwrap().TerminatePod(&podId).await {
                Err(e) => {
                    return Ok(Response::new(qmeta::DeleteResponseMessage {
                        error: format!("create pod fail with error {:?}", e),
                        ..Default::default()
                    }));
                }
                Ok(()) =>  {
                    return Ok(Response::new(qmeta::DeleteResponseMessage {
                        error: String::new(),
                        ..Default::default()
                    }));
                }
            }
        }

        return self.etcdSvc.delete(request).await;
    }

    async fn update(
        &self,
        request: Request<qmeta::UpdateRequestMessage>,
    ) -> SResult<Response<qmeta::UpdateResponseMessage>, Status> {
        let req = request.get_ref();
        let objType = &req.obj_type;
        if objType == QUARK_POD ||  objType == QUARK_NODE {
            let response = qmeta::UpdateResponseMessage {
                error: "NodeMgr doesn't support update pod".to_owned(),
                ..Default::default()
            };
            return Ok(Response::new(response));
        }

        return self.etcdSvc.update(request).await;
    }

    async fn list(
        &self,
        request: Request<qmeta::ListRequestMessage>,
    ) -> SResult<Response<qmeta::ListResponseMessage>, Status> {
        let req = request.get_ref();
        let objType = &req.obj_type;
            if objType == QUARK_POD ||  objType == QUARK_NODE {
            let cacher = match crate::NM_CACHE.get().unwrap().GetCacher(&req.obj_type) {
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

        return self.etcdSvc.list(request).await;
    }

    type WatchStream = std::pin::Pin<Box<dyn futures::Stream<Item = SResult<qmeta::WEvent, Status>> + Send>>;

    async fn watch(
        &self,
        request: Request<qmeta::WatchRequestMessage>,
    ) -> SResult<Response<Self::WatchStream>, Status> {
        let (tx, rx) = mpsc::channel(200);
        let stream = ReceiverStream::new(rx);
        
        tokio::spawn(async move {
            let req = request.get_ref();
            let objType = &req.obj_type;
            let cacher = if objType == QUARK_POD ||  objType == QUARK_NODE {
                match crate::NM_CACHE.get().unwrap().GetCacher(&req.obj_type) {
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
                }
            } else {
                match SVC_DIR.GetCacher(&req.obj_type).await {
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
                }
            }
            ;

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

    async fn put_obj(
        &self,
        request: tonic::Request<qmeta::PutObjReq>,
    ) -> SResult<tonic::Response<qmeta::PutObjResp>, tonic::Status> {
        let req = request.get_ref();
        match self.objectMgr.PutObject(&req.namespace, &req.name, &req.data).await {
            Err(e) => {
                return Ok(Response::new(qmeta::PutObjResp {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(()) => {
                return Ok(Response::new(qmeta::PutObjResp {
                    ..Default::default()
                }))
            }
        }
    }

    async fn delete_obj(
        &self,
        request: tonic::Request<qmeta::DeleteObjReq>,
    ) -> SResult<tonic::Response<qmeta::DeleteObjResp>, tonic::Status>{
        let req = request.get_ref();
        match self.objectMgr.DeleteObject(&req.namespace, &req.name).await {
            Err(e) => {
                return Ok(Response::new(qmeta::DeleteObjResp {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(()) => {
                return Ok(Response::new(qmeta::DeleteObjResp {
                    ..Default::default()
                }))
            }
        }
    }
    
    async fn read_obj(
        &self,
        request: tonic::Request<qmeta::ReadObjReq>,
    ) -> SResult<tonic::Response<qmeta::ReadObjResp>, tonic::Status>{
        let req = request.get_ref();
        match self.objectMgr.ReadObject(&req.namespace, &req.name).await {
            Err(e) => {
                return Ok(Response::new(qmeta::ReadObjResp {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(data) => {
                return Ok(Response::new(qmeta::ReadObjResp {
                    data: data,
                    ..Default::default()
                }))
            }
        }
    }
    
    async fn list_obj(
        &self,
        request: tonic::Request<qmeta::ListObjReq>,
    ) -> SResult<tonic::Response<qmeta::ListObjResp>, tonic::Status>{
        let req = request.get_ref();
        match self.objectMgr.ListObjects(&req.namespace, &req.prefix).await {
            Err(e) => {
                return Ok(Response::new(qmeta::ListObjResp {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(data) => {
                let mut objs = Vec::new();
                for o in data {
                    objs.push(qmeta::ObjMeta {
                        name: o.name,
                        size: o.size,
                    });
                }
                return Ok(Response::new(qmeta::ListObjResp {
                    objs: objs,
                    ..Default::default()
                }))
            }
        }
    }

    async fn read_func_log(
        &self,
        request: tonic::Request<qmeta::ReadFuncLogReq>,
    ) -> SResult<tonic::Response<qmeta::ReadFuncLogResp>, tonic::Status> {
        let _req = request.get_ref();
        unimplemented!();
    }
}

pub async fn GrpcService() -> Result<()> {
    use tonic::transport::Server;
    use qobjs::qmeta::q_meta_service_server::QMetaServiceServer;

    let svc = NodeMgrSvc::New(OBJECTDB_ADDR).await?;

    let qmetaFuture = Server::builder()
        .add_service(QMetaServiceServer::new(svc.clone()))
        .serve(QMETASVC_ADDR.parse().unwrap());

    let nodeAgentSvc = qobjs::nm::node_agent_service_server::NodeAgentServiceServer::new(svc.clone());
    let naFuture = Server::builder()
        .add_service(nodeAgentSvc)
        .serve(NODEMGRSVC_ADDR.parse().unwrap());

    info!("nodemgr start ...");
    tokio::select! {
        _ = qmetaFuture => {}
        _ = naFuture => {}
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use qobjs::cacher_client::CacherClient;
    use qobjs::informer_factory::InformerFactory;
    use qobjs::informer::*;

    #[actix_rt::test]
    async fn NMTest() {
        let qmetaSvcAddr = &format!("http://{}", QMETASVC_ADDR);
        let cacheClient = CacherClient::New(qmetaSvcAddr.into()).await.unwrap();

        let list = cacheClient.List(QUARK_POD, "default", &ListOption::default()).await.unwrap();
        println!("list1 is {:?}", list);

        //let client = NodeMgrClient::New(qmetaSvcAddr.into()).await.unwrap();
        let podstr = r#"
        {
            "apiVersion":"v1",
            "kind":"Pod",
            "metadata":{
                "name":"nginx",
                "namespace": "default"
            },
            "spec":{
                "hostNetwork": true,
                "containers":[
                    {
                        "name":"nginx",
                        "image":"nginx:alpine",
                        "ports":[
                            {
                                "containerPort": 80,
                                "hostIP": "192.168.0.22",
                                "hostPort": 88
                            }
                        ]
                    }
                ]
            }
        }"#;

        let mut pod : k8s::Pod = serde_json::from_str(podstr).unwrap();
        let mut annotations = BTreeMap::new();
        annotations.insert(AnnotationNodeMgrNode.to_string(), "qserverless.quarksoft.io/brad-desktop".to_string());
        pod.metadata.annotations = Some(annotations);
        let podStr = serde_json::to_string(&pod).unwrap();
        let dataObj = DataObject::NewFromK8sObj(QUARK_POD, &pod.metadata, podStr);

        cacheClient.Create(QUARK_POD, dataObj.Obj()).await.unwrap();

        std::thread::sleep(std::time::Duration::from_secs(5));

        let list = cacheClient.List(QUARK_POD, "default", &ListOption::default()).await.unwrap();
        println!("list2 is {:?}", list);
        std::thread::sleep(std::time::Duration::from_secs(5));
        let list = cacheClient.List(QUARK_POD, "default", &ListOption::default()).await.unwrap();
        println!("list3 is {:?}", list);

        cacheClient.Delete(QUARK_POD, &dataObj.Namespace(), &dataObj.Name()).await.unwrap();
        let list = cacheClient.List(QUARK_POD, "default", &ListOption::default()).await.unwrap();
        println!("list4 is {:?}", list);
        assert!(false);
    
    }

    #[actix_rt::test]
    async fn EtcdList() {
        let qmetaSvcAddr = &format!("http://{}", QMETASVC_ADDR);
        let client = CacherClient::New(qmetaSvcAddr.into()).await.unwrap();
        println!("list is {:#?}", client.List("pod", "", &ListOption::default()).await.unwrap());
        assert!(false);
    }

    #[actix_rt::test]
    async fn EtcdTest() {
        let qmetaSvcAddr = &format!("http://{}", QMETASVC_ADDR);
        let client = CacherClient::New(qmetaSvcAddr.into()).await.unwrap();
        
        let obj = DataObject::NewPod("namespace1", "name1").unwrap();
        let rev = client.Create("pod", obj.Obj()).await.unwrap();
        let obj = obj.CopyWithRev(rev);

        let mut ws = client
            .Watch("pod", "namespace1", &ListOption::default())
            .await.unwrap();
        let event = ws.Next().await.unwrap();
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Added);
        assert!(event.obj == obj);

        let objx = DataObject::NewPod("namespace1", "name2").unwrap();
        let rev = client.Create("pod", objx.Obj()).await.unwrap();
        let objx = objx.CopyWithRev(rev);

        let event = ws.Next().await.unwrap();
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Added);
        assert!(event.obj == objx);

        let obj1 = client.Get("pod", "namespace1", "name1", 0).await.unwrap();
        assert!(obj1.is_some());
        let obj1 = obj1.unwrap();
        assert!(
            obj.clone() == obj1,
            "expect is {:#?}, actual is {:#?}",
            obj,
            obj1
        );

        let objs = client.List("pod", "", &ListOption::default()).await.unwrap();
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
            .await.unwrap();
        assert!(objs.objs.len() == 2, "list is {:?}", &objs);
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

        let obj2 = DataObject::NewPod("namespace1", "name1").unwrap();
        let rev = client.Update("pod", &obj2).await.unwrap();
        let obj2 = obj2.CopyWithRev(rev);

        let event = ws.Next().await.unwrap();
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Modified, "event is {:#?}", event);
        assert!(
            event.obj == obj2,
            "event is {:#?}, expect is {:#?}",
            event,
            obj2
        );

        let obj3 = client.Get("pod", "namespace1", "name1", 0).await.unwrap();
        assert!(obj3.is_some());
        let obj3 = obj3.unwrap();
        assert!(obj2 == obj3);

        let rev = client.Delete("pod", "namespace1", "name1").await.unwrap();
        let obj2 = obj2.CopyWithRev(rev);

        let event = ws.Next().await.unwrap();
        assert!(event.is_some());
        let event = event.unwrap();
        assert!(event.type_ == EventType::Deleted, "event is {:#?}", event);
        assert!(
            event.obj == obj2,
            "event is {:#?}, expect is {:#?}",
            event,
            obj2
        );

        let obj4 = client.Get("pod", "namespace1", "name1", 0).await.unwrap();
        assert!(obj4.is_none());

        let objs = client.List("pod", "", &ListOption::default()).await.unwrap();
        assert!(objs.objs.len() == 1);
        assert!(
            objx.clone() == objs.objs[0],
            "expect is {:#?}, actual is {:#?}",
            objx,
            objs.objs[1]
        );
    }

    use qobjs::store::ThreadSafeStore;
    use tokio::sync::mpsc::unbounded_channel;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::sync::mpsc::UnboundedSender;
    use tokio::sync::Mutex as TMutex;
    use std::time::Duration;
    use qobjs::common::*;

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

        pub async fn Pop(&self) -> Result<Option<DeltaEvent>> {
            tokio::select! {
                d = self.Read() => return Ok(d),
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    return Err(Error::Timeout);
                }
            }
        }
    }

    #[actix_rt::test]
    async fn InformerTest() {
        let client = CacherClient::New(QMETASVC_ADDR.into()).await.unwrap();
        
        error!("InformerTest 1");
        let factory = InformerFactory::New(QMETASVC_ADDR, "").await.unwrap();
        factory.AddInformer("pod", &ListOption::default()).await.unwrap();
        let informer = factory.GetInformer("pod").await.unwrap();
        let handler1 = Arc::new(InformerHandler::New());
        let _id1 = informer.AddEventHandler(handler1.clone()).await.unwrap();

        let obj = DataObject::NewPod("namespace1", "name1").unwrap();
        let rev = client.Create("pod", obj.Obj()).await.unwrap();
        let obj = obj.CopyWithRev(rev);

        let handler2 = Arc::new(InformerHandler::New());
        let id2 = informer.AddEventHandler(handler2.clone()).await.unwrap();

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

        let objx = DataObject::NewPod("namespace1", "name2").unwrap();
        let rev = client.Create("pod", objx.Obj()).await.unwrap();
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

        let obj2 = DataObject::NewPod("namespace1", "name1").unwrap();
        let rev = client.Update("pod", &obj2).await.unwrap();
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
            Err(Error::Timeout) => (),
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

        factory.Close().await.unwrap();
    }
}