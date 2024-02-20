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

use std::sync::Arc;
use core::ops::Deref;

use async_trait::async_trait;
use etcd_client::{GetOptions, PutOptions};
use etcd_client::{
    Client, CompactionOptions, Compare, CompareOp, DeleteOptions, Txn, TxnOp, TxnOpResponse,
};
use tokio::sync::Notify;
use std::fmt::Debug;

use crate::etcd::etcd_client::EtcdClient;
use crate::etcd::watch::{WatchReader, Watcher};
use crate::common::*;
use crate::metastore::cache_store::{BackendStore, CacheStore};
use crate::metastore::selection_predicate::*;
use crate::qmeta::*;
use crate::metastore::data_obj::*;

pub const PATH_PREFIX: &str = "/registry";

#[derive(Debug)]
pub struct EtcdStoreInner {
    pub client: EtcdClient,
    pub pathPrefix: String,
    pub pagingEnable: bool,
}

impl Deref for EtcdStore {
    type Target = Arc<EtcdStoreInner>;

    fn deref(&self) -> &Arc<EtcdStoreInner> {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct EtcdStore(Arc<EtcdStoreInner>);

#[async_trait]
impl BackendStore for EtcdStore {
    async fn Get(&self, key: &str, minRevision: i64) -> Result<Option<DataObject>> {
        let preparedKey = self.PrepareKey(key)?;
        let getResp = self.client.lock().await.get(preparedKey, None).await?;
        let kvs = getResp.kvs();
        let actualRev = getResp.header().unwrap().revision();
        Self::ValidateMinimumResourceVersion(minRevision, actualRev)?;

        if kvs.len() == 0 {
            return Ok(None);
        }

        let kv = &kvs[0];
        let val = kv.value();

        let obj = Object::Decode(val)?;

        let obj = DataObject::NewFromObject(&obj, kv.mod_revision(), kv.mod_revision());

        return Ok(Some(obj));
    }

    async fn List(&self, prefix: &str, opts: &ListOption) -> Result<DataObjList> {
        let mut preparedKey = self.PrepareKey(prefix)?;

        let revision = opts.revision;
        let match_ = opts.revisionMatch;
        let pred = &opts.predicate;

        if !preparedKey.ends_with("/") {
            preparedKey = preparedKey + "/";
        }

        let keyPrefix = preparedKey.clone();

        let mut limit = pred.limit;
        let mut getOption = EtcdOption::default();
        let mut paging = false;
        if self.pagingEnable && pred.limit > 0 {
            paging = true;
            getOption.WithLimit(pred.limit as i64);
        }

        let fromRv = revision;

        let mut returnedRV = 0;
        let continueRv;
        let mut withRev = 0;
        let continueKey;
        if self.pagingEnable && pred.HasContinue() {
            //let (tk, trv) = pred.Continue(&keyPrefix)?;

            (continueKey, continueRv) = pred.Continue(&keyPrefix)?;

            let rangeEnd = Self::GetPrefixRangeEnd(&keyPrefix);
            getOption.WithRange(rangeEnd);
            preparedKey = continueKey.clone();

            // If continueRV > 0, the LIST request needs a specific resource version.
            // continueRV==0 is invalid.
            // If continueRV < 0, the request is for the latest resource version.
            if continueRv > 0 {
                withRev = continueRv.clone();
                returnedRV = continueRv;
            }
        } else if self.pagingEnable && pred.limit > 0 {
            if fromRv > 0 {
                match match_ {
                    RevisionMatch::None => (),
                    RevisionMatch::NotOlderThan => (),
                    RevisionMatch::Exact => {
                        returnedRV = fromRv;
                        withRev = returnedRV;
                    }
                }
            }

            let rangeEnd = Self::GetPrefixRangeEnd(&keyPrefix);
            getOption.WithRange(rangeEnd);
        } else {
            if fromRv > 0 {
                match match_ {
                    RevisionMatch::None => (),
                    RevisionMatch::NotOlderThan => (),
                    RevisionMatch::Exact => {
                        returnedRV = fromRv;
                        withRev = returnedRV;
                    }
                }
            }

            getOption.WithPrefix();
        }

        if withRev != 0 {
            getOption.WithRevision(withRev);
        }

        //let mut numFetched = 0;
        let mut hasMore;
        let mut v = Vec::new();
        let mut lastKey = Vec::new();
        //let mut numEvald = 0;
        let mut getResp;

        loop {
            //error!("key is {}, option is {:?}", &preparedKey, &getOption);
            let option = getOption.ToGetOption();
            getResp = self
                .client
                .lock()
                .await
                .get(preparedKey.clone(), Some(option))
                .await?;
            let actualRev = getResp.header().unwrap().revision();
            Self::ValidateMinimumResourceVersion(revision, actualRev)?;

            //numFetched += getResp.kvs().len();
            hasMore = getResp.more();

            if getResp.kvs().len() == 0 && hasMore {
                return Err(Error::CommonError(
                    "no results were found, but etcd indicated there were more values remaining"
                        .to_owned(),
                ));
            }

            for kv in getResp.kvs() {
                if paging && v.len() >= pred.limit {
                    hasMore = true;
                    break;
                }

                lastKey = kv.key().to_vec();
                let obj = Object::Decode(kv.value())?;
                let obj = DataObject::NewFromObject(&obj, kv.mod_revision(), kv.mod_revision());

                if pred.Match(&obj)? {
                    v.push(obj)
                }

                //numEvald += 1;
            }

            // indicate to the client which resource version was returned
            if returnedRV == 0 {
                returnedRV = getResp.header().unwrap().revision();
            }

            // no more results remain or we didn't request paging
            if !hasMore || !paging {
                break;
            }

            // we're paging but we have filled our bucket
            if v.len() >= pred.limit {
                break;
            }

            if limit < MAX_LIMIT {
                // We got incomplete result due to field/label selector dropping the object.
                // Double page size to reduce total number of calls to etcd.
                limit *= 2;
                if limit > MAX_LIMIT {
                    limit = MAX_LIMIT;
                }

                getOption.WithLimit(limit as i64);
            }
        }

        // instruct the client to begin querying from immediately after the last key we returned
        // we never return a key that the client wouldn't be allowed to see
        if hasMore {
            let newKey = String::from_utf8(lastKey).unwrap();
            let next = EncodeContinue(&(newKey + "\x000"), &keyPrefix, returnedRV)?;
            let mut remainingItemCount = -1;
            if pred.Empty() {
                remainingItemCount = getResp.count() as i64 - pred.limit as i64;
            }

            return Ok(DataObjList::New(
                v,
                returnedRV,
                Some(next),
                remainingItemCount,
            ));
        }

        return Ok(DataObjList::New(v, returnedRV, None, -1));
    }

    fn Register(&self, cacher: CacheStore, rev: i64, prefix: String, ready: Arc<Notify>, notify: Arc<Notify>) -> Result<()> {
        let storeClone = self.clone();
        let _future = tokio::spawn(async move{
            storeClone.Process1(&cacher, rev, &prefix, &ready, &notify).await
        });

        return Ok(())
    }
}

impl EtcdStore {
    pub async fn Create(&self, obj: &DataObject, leaseId: i64) -> Result<DataObject> {
        let key = obj.StoreKey();
        let preparedKey = self.PrepareKey(&key)?;
        let keyVec: &str = &preparedKey;
        let putopt = if leaseId == 0 {
            None
        } else {
            Some(PutOptions::default().with_lease(leaseId))
        };
        let txn = Txn::new()
            .when(vec![Compare::mod_revision(keyVec, CompareOp::Equal, 0)])
            .and_then(vec![TxnOp::put(keyVec, obj.Object().Encode()?, putopt)]);

        let resp: etcd_client::TxnResponse = self.client.lock().await.txn(txn).await?;
        if !resp.succeeded() {
            return Err(Error::NewNewKeyExistsErr(preparedKey, 0));
        } else {
            match &resp.op_responses()[0] {
                TxnOpResponse::Put(getresp) => {
                    let actualRev = getresp.header().unwrap().revision();
                    return Ok(obj.CopyWithRev(actualRev, actualRev));
                }
                _ => {
                    panic!("create get unexpect response")
                }
            };
        }
    }

    pub async fn Update(
        &self,
        expectedRev: i64,
        obj: &DataObject,
    ) -> Result<DataObject> {
        let key = obj.StoreKey();
        let preparedKey = self.PrepareKey(&key)?;
        let keyVec: &str = &preparedKey;
        let txn = if expectedRev > 0 {
            Txn::new()
                .when(vec![Compare::mod_revision(
                    keyVec,
                    CompareOp::Equal,
                    expectedRev,
                )])
                .and_then(vec![TxnOp::put(keyVec, obj.Encode()?, None)])
                .or_else(vec![TxnOp::get(keyVec, None)])
        } else {
            Txn::new().and_then(vec![TxnOp::put(keyVec, obj.Encode()?, None)])
        };

        let resp = self.client.lock().await.txn(txn).await?;
        if !resp.succeeded() {
            match &resp.op_responses()[0] {
                TxnOpResponse::Get(getresp) => {
                    let actualRev = getresp.header().unwrap().revision();
                    return Err(Error::NewUpdateRevNotMatchErr(expectedRev, actualRev));
                }
                _ => {
                    panic!("Delete get unexpect response")
                }
            };
        } else {
            match &resp.op_responses()[0] {
                TxnOpResponse::Put(getresp) => {
                    let actualRev = getresp.header().unwrap().revision();
                    return Ok(obj.CopyWithRev(actualRev, actualRev));
                }
                _ => {
                    panic!("Delete get unexpect response")
                }
            };
        }
    }

    pub async fn Delete(&self, key: &str, expectedRev: i64) -> Result<i64> {
        let preparedKey = self.PrepareKey(key)?;
        let keyVec: &str = &preparedKey;
        let txn = if expectedRev != 0 {
            Txn::new()
                .when(vec![Compare::mod_revision(
                    keyVec,
                    CompareOp::Equal,
                    expectedRev,
                )])
                .and_then(vec![TxnOp::delete(keyVec, None)])
                .or_else(vec![TxnOp::get(keyVec, None)])
        } else {
            Txn::new().and_then(vec![TxnOp::delete(keyVec, None)])
        };

        let resp = self.client.lock().await.txn(txn).await?;
        if !resp.succeeded() {
            match &resp.op_responses()[0] {
                TxnOpResponse::Get(getresp) => {
                    let actualRev = getresp.kvs()[0].mod_revision();
                    return Err(Error::NewDeleteRevNotMatchErr(expectedRev, actualRev));
                }
                _ => {
                    panic!("Delete get unexpect response")
                }
            };
        } else {
            match &resp.op_responses()[0] {
                TxnOpResponse::Delete(resp) => {
                    let actualRev = resp.header().unwrap().revision();
                    return Ok(actualRev);
                }
                _ => {
                    panic!("Delete get unexpect response")
                }
            };
        }
    }
}

impl EtcdStore {
    pub async fn NewWithEndpoints(endpoints: &[String], pagingEnable: bool) -> Result<Self> {
        let client = Client::connect(endpoints, None).await?;
        let inner = EtcdStoreInner {
            client: EtcdClient::New(client),
            pathPrefix: PATH_PREFIX.to_string(),
            pagingEnable,
        };

        return Ok(Self(Arc::new(inner))); 
    }

    pub async fn New(addr: &str, pagingEnable: bool) -> Result<Self> {
        let client = Client::connect([addr], None).await?;

        let inner = EtcdStoreInner {
            client: EtcdClient::New(client),
            pathPrefix: PATH_PREFIX.to_string(),
            pagingEnable,
        };

        return Ok(Self(Arc::new(inner)));
    }

    pub async fn LeaseGrant(&self, ttl: i64) -> Result<i64> {
        let resp = self.client.client.lock().await.lease_client().grant(ttl, None).await?;
        return Ok(resp.id());
    }

    pub async fn LeaseRevoke(&self, leaseId: i64) -> Result<()> {
        let _resp = self.client.client.lock().await.lease_client().revoke(leaseId).await?;
        return Ok(())
    }

    pub async fn LeaseKeepalive(&self, leaseId: i64) -> Result<()> { 
        let _resp = self.client.client.lock().await.lease_client().keep_alive(leaseId).await?;
        return Ok(())
    }

    async fn InitCacheStore(&self, cs: &CacheStore, rev: i64, prefix: &str) -> Result<i64> {
        let list = self
            .List(
                prefix,
                &ListOption {
                    revision: rev,
                    ..Default::default()
                },
            )
            .await?;

        {
            let mut inner = cs.write().unwrap();

            // close all watches
            inner.watchers.clear();
            // clear all cached data
            inner.cacheStore.clear();
            inner.cache.Reset();

            let channelRev = inner.ChannelRev();
            inner.listRevision = channelRev;
            for o in list.objs {
                let revision = o.revision;
                error!("InitCacheStore obj is {:?}", &o);
                let obj = o.CopyWithRev(channelRev, revision);
                inner.Add(&obj)?;
            }
        }

        return Ok(list.revision)
    }

    async fn UpdateCacheStore(&self, cs: &CacheStore, prefix: &str, listRev: i64, notify: &Arc<Notify>) -> Result<()> {
        let (mut w, r) = self.Watch(&prefix, listRev, SelectionPredicate::default())?;

        loop {
            tokio::select! {
                processResult = w.Processing() => {
                    processResult?;
                }
                event = r.GetNextEvent() => {
                    match event {
                        None => break,
                        Some(event) => {
                            cs.ProcessEvent(&event)?;
                        }
                    }
                }
                _ = notify.notified() => {
                    return Ok(())
                }
            }
        }

        return Ok(())
    }

    async fn Process1(&self, cs: &CacheStore, rev: i64, prefix: &str, ready: &Arc<Notify>, notify: &Arc<Notify>) -> Result<()> {
        let mut listRev = self.InitCacheStore(cs, rev, &prefix).await?;

        ready.notify_one();
        loop {
            match self.UpdateCacheStore(cs, prefix, listRev, notify).await {
                Err(e) => {
                    error!("EtcdStore watch with error {:?}", e);
                }
                Ok(()) => {
                    // the watching stop by user
                    return Ok(())
                },
            }

            listRev = self.InitCacheStore(cs, rev, prefix).await?;
        }
    }

    pub fn PrepareKey(&self, key: &str) -> Result<String> {
        let mut key = key;
        if key == "." || key == "/" {
            return Err(Error::CommonError(format!("invalid key: {}", key)));
        }

        if key.starts_with("/") {
            key = key.get(1..).unwrap();
        }

        return Ok(format!("{}/{}", self.pathPrefix, key));
    }

    pub fn ValidateMinimumResourceVersion(minRevision: i64, actualRevision: i64) -> Result<()> {
        if minRevision == 0 {
            return Ok(());
        }

        if minRevision > actualRevision {
            return Err(Error::NewMinRevsionErr(minRevision, actualRevision));
        }

        return Ok(());
    }


    pub async fn Clear(&mut self, prefix: &str) -> Result<i64> {
        let preparedKey = self.PrepareKey(prefix)?;

        let keyVec: &str = &preparedKey;

        let mut options = DeleteOptions::new();
        options = options.with_prefix();
        let resp = self
            .client
            .lock()
            .await
            .delete(keyVec, Some(options))
            .await?;

        let rv = resp.header().unwrap().revision();
        return Ok(rv);
    }

    pub fn GetPrefixRangeEnd(prefix: &str) -> String {
        let arr: Vec<u8> = prefix.as_bytes().to_vec();
        let arr = Self::GetPrefix(&arr);
        return String::from_utf8(arr).unwrap();
    }

    pub fn GetPrefix(prefix: &[u8]) -> Vec<u8> {
        let mut end = Vec::with_capacity(prefix.len());
        for c in prefix {
            end.push(*c);
        }

        for i in (0..prefix.len()).rev() {
            if end[i] < 0xff {
                end[i] += 1;
                let _ = end.split_off(i + 1);
                return end;
            }
        }

        // next prefix does not exist (e.g., 0xffff);
        // default to WithFromKey policy
        return vec![0];
    }

    pub fn Watch(
        &self,
        key: &str,
        rev: i64,
        pred: SelectionPredicate,
    ) -> Result<(Watcher, WatchReader)> {
        let preparedKey = self.PrepareKey(key)?;
        return Ok(Watcher::New(&self.client, &preparedKey, rev, pred));
    }

    pub async fn Compaction(&self, revision: i64) -> Result<()> {
        let options = CompactionOptions::new().with_physical();
        self.client
            .lock()
            .await
            .compact(revision, Some(options))
            .await?;
        return Ok(());
    }
}

// maxLimit is a maximum page limit increase used when fetching objects from etcd.
// This limit is used only for increasing page size by kube-apiserver. If request
// specifies larger limit initially, it won't be changed.
pub const MAX_LIMIT: usize = 10000;

#[derive(Debug, Default)]
pub struct EtcdOption {
    pub limit: Option<i64>,
    pub endKey: Option<Vec<u8>>,
    pub revision: Option<i64>,
    pub prefix: bool,
}

impl EtcdOption {
    pub fn WithLimit(&mut self, limit: i64) {
        self.limit = Some(limit);
    }

    pub fn WithRange(&mut self, endKey: impl Into<Vec<u8>>) {
        self.endKey = Some(endKey.into());
    }

    pub fn WithRevision(&mut self, revision: i64) {
        self.revision = Some(revision);
    }

    pub fn WithPrefix(&mut self) {
        self.prefix = true;
    }

    pub fn ToGetOption(&self) -> GetOptions {
        let mut getOption = GetOptions::new();
        match self.limit {
            None => (),
            Some(l) => getOption = getOption.with_limit(l),
        }

        match &self.endKey {
            None => (),
            Some(k) => {
                getOption = getOption.with_range(k.to_vec());
            }
        }

        match self.revision {
            None => (),
            Some(rv) => getOption = getOption.with_revision(rv),
        }

        if self.prefix {
            getOption = getOption.with_prefix();
        }

        return getOption;
    }
}
