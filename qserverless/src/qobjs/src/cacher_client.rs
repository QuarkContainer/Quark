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

use std::sync::Arc;
use core::ops::Deref;

use tokio::sync::Mutex as TMutex;
use tonic::Streaming;
use tonic::transport::Channel;
use tonic::Request;

use crate::selection_predicate::ListOption;
use crate::qmeta::q_meta_service_client::QMetaServiceClient;
use crate::qmeta::*;
use crate::common::*;
use crate::types::DataObjList;
use crate::types::DataObject;
use crate::types::EventType;
use crate::types::WatchEvent;

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

    pub async fn Create(&self, objType: &str, obj: Obj) -> Result<i64> {
        let mut inner = self.lock().await;
        return inner.Create(objType, obj).await;
    }

    pub async fn Get(&self, objType: &str, namespace: &str, name: &str, revision: i64) -> Result<Option<DataObject>> {
        let mut inner = self.lock().await;
        return inner.Get(objType, namespace, name, revision).await;
    }

    pub async fn Delete(&self, objType: &str, namespace: &str, name: &str) -> Result<i64> {
        let mut inner = self.lock().await;
        return inner.Delete(objType, namespace, name).await;
    }

    pub async fn Update(&self, objType: &str, obj: &DataObject) -> Result<i64> {
        let mut inner = self.lock().await;
        return inner.Update(objType, obj).await;
    }

    pub async fn List(&self, objType: &str, namespace: &str, opts: &ListOption) -> Result<DataObjList> {
        let mut inner = self.lock().await;
        return inner.List(objType, namespace, opts).await;
    }

    pub async fn Watch(&self, objType: &str, namespace: &str, opts: &ListOption) -> Result<WatchStream> {
        let mut inner = self.lock().await;
        return inner.Watch(objType, namespace, opts).await;
    }
}

#[derive(Debug)]
pub struct CacherClientInner {
    pub client: QMetaServiceClient<Channel>,
}

impl CacherClientInner {
    pub async fn New(qmetaSvcAddr: String) -> Result<Self> {
        let client = QMetaServiceClient::connect(qmetaSvcAddr).await?;
        return Ok(Self {
            client: client,
        })
    }

    pub async fn Create(&mut self, objType: &str, obj: Obj) -> Result<i64> {
        let req = CreateRequestMessage {
            obj_type: objType.to_string(),
            obj: Some(obj),
        };

        let response = self.client.create(Request::new(req)).await?;
        let resp = response.get_ref();
        if resp.error.len() == 0 {
            return Ok(resp.revision)
        }
        
        return Err(Error::CommonError(resp.error.clone()))
    }

    pub async fn Get(&mut self, objType: &str, namespace: &str, name: &str, revision: i64) -> Result<Option<DataObject>> {
        let req = GetRequestMessage {
            obj_type: objType.to_string(),
            namespace: namespace.to_string(),
            name: name.to_string(),
            revision: revision,
        };

        let mut response = self.client.get(Request::new(req)).await?;
        let resp = response.get_mut();
        if resp.error.len() == 0 {
            match resp.obj.take() {
                None => return Ok(None),
                Some(o) => {
                    return Ok(Some(DataObject::NewFromObj(&o)))
                }
            }
        }
        
        return Err(Error::CommonError(resp.error.clone()))
    }

    pub async fn Delete(&mut self, objType: &str, namespace: &str, name: &str) -> Result<i64> {
        let req = DeleteRequestMessage {
            obj_type: objType.to_string(),
            namespace: namespace.to_string(),
            name: name.to_string(),
        };

        let mut response = self.client.delete(Request::new(req)).await?;
        let resp = response.get_mut();
        if resp.error.len() == 0 {
            return Ok(resp.revision)
        }
        
        return Err(Error::CommonError(resp.error.clone()))
    }

    pub async fn Update(&mut self, objType: &str, obj: &DataObject) -> Result<i64> {
        let req = UpdateRequestMessage {
            obj_type: objType.to_string(),
            obj: Some(obj.Obj()),
        };

        let response = self.client.update(Request::new(req)).await?;
        let resp = response.get_ref();
        if resp.error.len() == 0 {
            return Ok(resp.revision)
        }
        
        return Err(Error::CommonError(resp.error.clone()))
    }

    pub async fn List(&mut self, objType: &str, namespace: &str, opts: &ListOption) -> Result<DataObjList> {
        let req = ListRequestMessage {
            obj_type: objType.to_string(),
            namespace: namespace.to_string(),
            revision: opts.revision,
            label_selector: opts.predicate.label.String(),
            field_selector: opts.predicate.field.String(),
        };

        let response = self.client.list(Request::new(req)).await?;
        let resp = response.into_inner();
        if resp.error.len() == 0 {
            let mut objs = Vec::new();
            for dataO in &resp.objs {
                let obj : DataObject = DataObject::NewFromObj(dataO);
                objs.push(obj)
            }

            let dol = DataObjList {
                objs: objs,
                revision: resp.revision,
                ..Default::default()
            };

            return Ok(dol)
        }
        
        return Err(Error::CommonError(resp.error.clone()))
    }

    pub async fn Watch(&mut self, objType: &str, namespace: &str, opts: &ListOption) -> Result<WatchStream> {
        let req = WatchRequestMessage {
            obj_type: objType.to_string(),
            namespace: namespace.to_string(),
            revision: opts.revision,
            label_selector: opts.predicate.label.String(),
            field_selector: opts.predicate.field.String(),
        };

        let response = self.client.watch(Request::new(req)).await?;
        let resp = response.into_inner();
        return Ok(WatchStream { stream: resp })
    }
}

pub struct WatchStream {
    stream: Streaming<WEvent>
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
                    _ => return Err(Error::CommonError(format!("invalid watch response type {}", e.event_type))),
                };

                if e.obj.is_none() {
                    return Err(Error::CommonError(format!("invalid watch response ob is none")));
                }

                let watchEvent = WatchEvent {
                    type_: eventType,
                    obj: DataObject::NewFromObj(&e.obj.unwrap()),
                };

                return Ok(Some(watchEvent))
            }
        }
    }
}