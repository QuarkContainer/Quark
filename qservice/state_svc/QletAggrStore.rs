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

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use qshare::metastore::data_obj::{DeltaEvent, EventType, STATESVC_ADDR};
use qshare::metastore::informer::{EventHandler, Informer};
use qshare::metastore::selection_predicate::ListOption;
use qshare::metastore::store::ThreadSafeStore;
use qshare::types::NodeInfo;
use tokio::sync::Notify;
use core::ops::Deref;

use qshare::common::*;
use qshare::metastore::cache_store::{CacheStore, ChannelRev};
use qshare::metastore::aggregate_client::AggregateClient;

#[derive(Debug)]
pub struct QletAggrStoreInner {
    pub agents: BTreeMap<String, QletAgent>,
    pub podStore: CacheStore,
    pub nodeStore: CacheStore,
}

#[derive(Debug, Clone)]
pub struct QletAggrStore(Arc<Mutex<QletAggrStoreInner>>);

impl Deref for QletAggrStore {
    type Target = Arc<Mutex<QletAggrStoreInner>>;

    fn deref(&self) -> &Arc<Mutex<QletAggrStoreInner>> {
        &self.0
    }
}

impl QletAggrStore {
    pub async fn New(channelRev: &ChannelRev) -> Result<Self> {
        let inner = QletAggrStoreInner {
            agents: BTreeMap::new(),
            podStore: CacheStore::New(None, "pod", 0, channelRev).await?,
            nodeStore: CacheStore::New(None, "node", 0, channelRev).await?,
        };

        return Ok(Self(Arc::new(Mutex::new(inner))));
    }

    pub fn PodStore(&self) -> CacheStore {
        return self.lock().unwrap().podStore.clone();
    }

    pub fn NodeStore(&self) -> CacheStore {
        return self.lock().unwrap().nodeStore.clone();
    }

    pub async fn Process(&self) -> Result<()> {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let addr = format!("http://{}", STATESVC_ADDR);
        let informer = Informer::New(vec![addr], "node_info", "", "", &ListOption::default()).await?;
        
        informer.AddEventHandler(Arc::new(self.clone())).await?;
        let notify = Arc::new(Notify::new());
        informer.Process(notify).await?;
        return Ok(())
    }

    pub fn AddQletAgent(&self, nodeName: &str, svcAddr: &str) -> Result<()> {
        let podStore = self.PodStore();
        let nodeStore = self.NodeStore();
        
        let agent = QletAgent::New(nodeName, svcAddr, &nodeStore, &podStore)?;

        self.lock().unwrap().agents.insert(nodeName.to_owned(), agent.clone());
        tokio::spawn(async move {
            agent.Process().await.unwrap();
        });

        return Ok(())
    }

    pub fn RemoveQletAgent(&self, nodeName: &str) -> Result<()> {
        let agent = match self.lock().unwrap().agents.remove(nodeName) {
            Some(a) => a,
            None => {
                return Err(Error::NotExist(format!("QletAggrStore::RemoveQletAgent {}", nodeName)));
            }
        };

        agent.Close();
        return Ok(())
    }
}

impl EventHandler for QletAggrStore {
    fn handle(&self, _store: &ThreadSafeStore, event: &DeltaEvent) {
        match &event.type_ {
            EventType::Added => {
                let obj = &event.obj;
                let nodeInfo : NodeInfo = serde_json::from_str(&obj.data)
                    .expect(&format!("NodeMgr::handle deserialize fail for {}", &obj.data));

                let stateSvcPort : u16 = nodeInfo.stateSvcPort;
                let stateSvcAddr = format!("http://{}:{}", &nodeInfo.nodeIp, stateSvcPort);
                self.AddQletAgent(&nodeInfo.nodeName, &stateSvcAddr).unwrap();
            }
            EventType::Deleted => {
                let obj = event.oldObj.as_ref().unwrap();
                let nodeInfo : NodeInfo = serde_json::from_str(&obj.data)
                    .expect(&format!("NodeMgr::handle deserialize fail for {}", &obj.data));
                
                self.RemoveQletAgent(&nodeInfo.nodeName).unwrap();
            }
            _ => ()
        }
    }
}

#[derive(Debug)]
pub struct QletAgentInner {
    pub closeNotify: Arc<Notify>,
    pub closed: AtomicBool,

    pub nodeName: String,
    pub svcAddr: String,
    pub nodeClient: AggregateClient,
    pub podClient: AggregateClient,
}

#[derive(Debug, Clone)]
pub struct QletAgent(Arc<QletAgentInner>);

impl Deref for QletAgent {
    type Target = Arc<QletAgentInner>;

    fn deref(&self) -> &Arc<QletAgentInner> {
        &self.0
    }
}

impl QletAgent {
    pub fn New(nodeName: &str, svcAddr: &str, nodeCache: &CacheStore, podCache: &CacheStore) -> Result<Self> {
        let inner = QletAgentInner {
            closeNotify: Arc::new(Notify::new()),
            closed: AtomicBool::new(false),

            nodeName: nodeName.to_owned(),
            svcAddr: svcAddr.to_owned(),
            nodeClient: AggregateClient::New(nodeCache, "node", "", "")?,
            podClient: AggregateClient::New(podCache, "pod", "", "")?,
        };

        return Ok(Self(Arc::new(inner)))
    }

    pub fn Close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }

    pub async fn Process(&self) -> Result<()> {
        let listNotify = Arc::new(Notify::new());

        tokio::select! { 
            _ = self.closeNotify.notified() => {
                self.nodeClient.Close();
                self.podClient.Close();
                return Ok(())
            }
            _ = self.nodeClient.Process(vec![self.svcAddr.clone()], listNotify.clone()) => {

            }
            _ = self.podClient.Process(vec![self.svcAddr.clone()], listNotify.clone()) => {

            }
        }

        return Ok(())
    }
}