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

use core::ops::Deref;
use std::sync::Arc;

use tokio::sync::Mutex as TMutex;
use tonic::transport::Channel;
use tonic::Request;
use tonic::Streaming;

// use crate::object_mgr::ObjectMeta;
use super::data_obj::*;
use super::selection_predicate::ListOption;
use crate::common::*;
use crate::qmeta::q_meta_service_client::QMetaServiceClient;
use crate::qmeta::*;

#[derive(Debug)]
pub struct ObjectMeta {
    pub name: String,
    pub size: i32,
}

#[derive(Debug, Clone)]
pub struct CacherClient(Arc<TMutex<CacherClientInner>>);

impl Deref for CacherClient {
    type Target = Arc<TMutex<CacherClientInner>>;

    fn deref(&self) -> &Arc<TMutex<CacherClientInner>> {
        &self.0
    }
}

impl CacherClient {
    pub async fn New(qmetaSvcAddr: String) -> Result<Self> {
        let inner = CacherClientInner::New(qmetaSvcAddr).await?;
        return Ok(Self(Arc::new(TMutex::new(inner))));
    }

    pub async fn Get(
        &self,
        objType: &str,
        tenant: &str,
        namespace: &str,
        name: &str,
        revision: i64,
    ) -> Result<Option<DataObject>> {
        let mut inner = self.lock().await;
        return inner.Get(objType, tenant, namespace, name, revision).await;
    }

    pub async fn List(
        &self,
        objType: &str,
        tenant: &str,
        namespace: &str,
        opts: &ListOption,
    ) -> Result<DataObjList> {
        let mut inner = self.lock().await;
        return inner.List(objType, tenant, namespace, opts).await;
    }

    pub async fn Watch(
        &self,
        objType: &str,
        tenant: &str,
        namespace: &str,
        opts: &ListOption,
    ) -> Result<WatchStream> {
        let mut inner = self.lock().await;
        return inner.Watch(objType, tenant, namespace, opts).await;
    }

    pub async fn ReadObject(&self, tenant: &str, namespace: &str, name: &str) -> Result<Vec<u8>> {
        let mut inner = self.lock().await;
        return inner.ReadObject(tenant, namespace, name).await;
    }

    pub async fn ListObjects(
        &self,
        tenant: &str,
        namespace: &str,
        prefix: &str,
    ) -> Result<Vec<ObjectMeta>> {
        let mut inner = self.lock().await;
        return inner.ListObjects(tenant, namespace, prefix).await;
    }
}

#[derive(Debug)]
pub struct CacherClientInner {
    pub client: QMetaServiceClient<Channel>,
}

impl CacherClientInner {
    pub async fn New(qmetaSvcAddr: String) -> Result<Self> {
        let client = QMetaServiceClient::connect(qmetaSvcAddr).await?;
        return Ok(Self { client: client });
    }

    pub async fn Get(
        &mut self,
        objType: &str,
        tenant: &str,
        namespace: &str,
        name: &str,
        revision: i64,
    ) -> Result<Option<DataObject>> {
        let req = GetRequestMessage {
            obj_type: objType.to_owned(),
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            revision: revision,
        };

        let mut response = self.client.get(Request::new(req)).await?;
        let resp = response.get_mut();
        if resp.error.len() == 0 {
            match resp.obj.take() {
                None => return Ok(None),
                Some(o) => return Ok(Some(DataObject::NewFromObj(&o))),
            }
        }

        return Err(Error::CommonError(resp.error.clone()));
    }

    pub async fn List(
        &mut self,
        objType: &str,
        tenant: &str,
        namespace: &str,
        opts: &ListOption,
    ) -> Result<DataObjList> {
        let req = ListRequestMessage {
            obj_type: objType.to_owned(),
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            revision: opts.revision,
            label_selector: opts.predicate.label.String(),
            field_selector: opts.predicate.field.String(),
        };

        let response = self.client.list(Request::new(req)).await?;
        let resp = response.into_inner();
        if resp.error.len() == 0 {
            let mut objs = Vec::new();
            for dataO in &resp.objs {
                let obj: DataObject = DataObject::NewFromObj(dataO);
                objs.push(obj)
            }

            let dol = DataObjList {
                objs: objs,
                revision: resp.revision,
                ..Default::default()
            };

            return Ok(dol);
        }

        return Err(Error::CommonError(resp.error.clone()));
    }

    pub async fn Watch(
        &mut self,
        objType: &str,
        tenant: &str,
        namespace: &str,
        opts: &ListOption,
    ) -> Result<WatchStream> {
        let req = WatchRequestMessage {
            obj_type: objType.to_owned(),
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            revision: opts.revision,
            label_selector: opts.predicate.label.String(),
            field_selector: opts.predicate.field.String(),
        };

        let response = self.client.watch(Request::new(req)).await?;
        let resp = response.into_inner();
        return Ok(WatchStream { stream: resp });
    }

    async fn ReadObject(&mut self, tenant: &str, namespace: &str, name: &str) -> Result<Vec<u8>> {
        let req = ReadObjReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            name: name.to_owned(),
        };

        let response = self.client.read_obj(Request::new(req)).await?;
        let resp = response.into_inner();
        if resp.error.len() == 0 {
            return Ok(resp.data);
        }

        return Err(Error::CommonError(resp.error.to_owned()));
    }

    async fn ListObjects(
        &mut self,
        tenant: &str,
        namespace: &str,
        prefix: &str,
    ) -> Result<Vec<ObjectMeta>> {
        let req = ListObjReq {
            tenant: tenant.to_owned(),
            namespace: namespace.to_owned(),
            prefix: prefix.to_owned(),
        };

        let response = self.client.list_obj(Request::new(req)).await?;
        let resp = response.into_inner();
        if resp.error.len() == 0 {
            let mut objs = Vec::new();
            for o in resp.objs {
                objs.push(ObjectMeta {
                    name: o.name,
                    size: o.size,
                })
            }
            return Ok(objs);
        }

        return Err(Error::CommonError(resp.error.to_owned()));
    }
}

pub struct WatchStream {
    stream: Streaming<WEvent>,
}

impl WatchStream {
    pub async fn Next(&mut self) -> Result<Option<WatchEvent>> {
        let event = self.stream.message().await?;
        match event {
            None => return Ok(None),
            Some(e) => {
                let eventType = match e.event_type {
                    0 => EventType::None,
                    1 => EventType::Added,
                    2 => EventType::Modified,
                    3 => EventType::Deleted,
                    _ => {
                        return Err(Error::CommonError(format!(
                            "invalid watch response type {}",
                            e.event_type
                        )))
                    }
                };

                if e.obj.is_none() {
                    return Err(Error::CommonError(format!(
                        "invalid watch response ob is none"
                    )));
                }

                let watchEvent = WatchEvent {
                    type_: eventType,
                    obj: DataObject::NewFromObj(&e.obj.unwrap()),
                };

                return Ok(Some(watchEvent));
            }
        }
    }
}
