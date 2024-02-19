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

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

use crate::common::*;
use crate::metastore::informer::Informer;
use crate::metastore::cache_store::CacheStore;

use super::cacher_client::CacherClient;
use super::data_obj::{DeltaEvent, EventType};
use super::informer::EventHandler;
use super::selection_predicate::ListOption;
use super::store::ThreadSafeStore;

#[derive(Debug)]
pub struct AggregateClientInner {
    pub closeNotify: Arc<Notify>,
    pub closed: AtomicBool,

    pub aggregateCacher: CacheStore,
    
    pub objType: String,
    pub namespace: String,
}

#[derive(Debug, Clone)]
pub struct AggregateClient(Arc<AggregateClientInner>);

impl Deref for AggregateClient {
    type Target = Arc<AggregateClientInner>;

    fn deref(&self) -> &Arc<AggregateClientInner> {
        &self.0
    }
}

impl EventHandler for AggregateClient {
    fn handle(&self, _store: &ThreadSafeStore, event: &DeltaEvent) {
        match event.type_ {
            EventType::Added => {
                self.aggregateCacher.Add(&event.obj).unwrap();
            }
            EventType::Modified => {
                self.aggregateCacher.Update(&event.obj).unwrap();
            }
            EventType::Deleted => {
                self.aggregateCacher.Remove(&event.obj).unwrap();
            }
            _ => {
                error!("AggregateClient::handle get unexpect event {:#?}", event);
            }
        }
    }
    
}

impl AggregateClient {
    pub fn New(aggregateCacher: &CacheStore, objType: &str, namespace: &str) -> Result<Self> {
        let inner = AggregateClientInner {
            closeNotify: Arc::new(Notify::new()),
            closed: AtomicBool::new(false),

            aggregateCacher: aggregateCacher.clone(),
            
            objType: objType.to_owned(),
            namespace: namespace.to_owned(),
        };

        return Ok(Self(Arc::new(inner)));
    }

    pub fn Close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }

    pub async fn Process(&self, client: &CacherClient, listNotify: Arc<Notify>) -> Result<()> {
        let informer = Informer::New(client, &self.objType, &self.namespace, &ListOption::default()).await?;
        informer.AddEventHandler(Arc::new(self.clone())).await?;
        loop {
            tokio::select! { 
                _ = self.closeNotify.notified() => {
                    return Ok(())
                }
                ret = informer.InitList() => {
                    match ret {
                        Err(e) => {
                            error!("AggregateClient initlist fail with error {:?} {:#?}", e, self);

                            // wait for network connection ready
                            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                            continue;
                        }
                        Ok(()) => break,
                    }
                }
            }
        }

        listNotify.notify_waiters();

        loop {
            tokio::select! { 
                _ = self.closeNotify.notified() => {
                    if self.closed.swap(true, Ordering::SeqCst) == false {
                        let store = informer.read().await.store.clone();
                        let lock = store.read();
                        for (_k, obj) in &lock.map {
                            self.aggregateCacher.Remove(obj)?;
                        }
                    }
                    break;
                }
                ret = informer.WatchUpdate() => {
                    error!("AggregateClient WatchUpdate finish with result {:?}", ret);

                }
            }

            // wait for network connection ready
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        
        return Ok(())
    }
}

