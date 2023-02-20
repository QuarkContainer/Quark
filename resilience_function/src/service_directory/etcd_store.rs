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


use std::sync::atomic::AtomicI64;
use std::{sync::Arc, collections::BTreeMap};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering;

use etcd_client::{Client, Compare, Txn, TxnOp, CompareOp, TxnOpResponse, DeleteOptions, CompactionOptions};
use etcd_client::GetOptions;
use prost::Message;

use crate::etcd_client::EtcdClient;
use crate::types::DeepCopy;
use crate::watch::{Watcher, WatchReader};
use crate::{shared::common::*, selector::Labels};
use crate::service_directory::*;
use crate::selection_predicate::*;

#[derive(Debug, Default)]
pub struct MetaDataInner {
    pub kind: String,
    pub namespace: String,
    pub name: String,
    pub lables: Labels,
    pub annotations: Labels,
    
    // revision number set by etcd
    pub reversion: AtomicI64,
}

impl DeepCopy for MetaDataInner {
    fn DeepCopy(&self) -> Self {
        return self.Copy();
    }
}

impl PartialEq for MetaDataInner {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind &&
        self.namespace == other.namespace &&
        self.lables == other.lables &&
        self.annotations == other.annotations &&
        self.reversion.load(Ordering::Relaxed) == other.reversion.load(Ordering::Relaxed)
    }
}
impl Eq for MetaDataInner {}

impl MetaDataInner {
    pub fn New(item: &Object) -> Self {
        let mut lables = BTreeMap::new();
        for l in &item.labels {
            lables.insert(l.key.clone(), l.val.clone());
        }

        let mut annotations = BTreeMap::new();
        for l in &item.annotations {
            annotations.insert(l.key.clone(), l.val.clone());
        }

        let inner = MetaDataInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: lables.into(),
            annotations: annotations.into(),
            reversion: AtomicI64::new(0),
        };

        return inner;
    }

    pub fn Key(&self) -> String {
        return format!("/{}/{}", &self.namespace, &self.name);
    }

    pub fn Revision(&self) -> i64 {
        return self.reversion.load(Ordering::Relaxed);
    }

    pub fn SetRevision(&self, rev: i64) {
        return self.reversion.store(rev, Ordering::SeqCst);
    }

    pub fn Copy(&self) -> Self {
        return Self {
            kind: self.kind.clone(),
            namespace: self.namespace.clone(),
            name: self.name.clone(),
            lables: self.lables.Copy(),
            annotations: self.annotations.Copy(),
            reversion: AtomicI64::new(self.Revision()),
        }
    }

    pub fn ToObject(&self) -> Object {
        let mut obj = Object::default();
        obj.kind = self.kind.clone();
        obj.namespace = self.namespace.clone();
        obj.name = self.name.clone();
        obj.labels = self.lables.ToVec();

        return obj;
    }
}

#[derive(Debug, Default)]
pub struct DataObjInner {
    pub metadata: MetaDataInner,

    pub obj: Object,
}

impl PartialEq for DataObjInner {
    fn eq(&self, other: &Self) -> bool {
        return self.metadata == other.metadata && self.obj.val == other.obj.val;
    }
}
impl Eq for DataObjInner {}

impl Deref for DataObjInner {
    type Target = MetaDataInner;

    fn deref(&self) -> &MetaDataInner {
        &self.metadata
    }
}

impl DeepCopy for DataObjInner {
    fn DeepCopy(&self) -> Self {
        return Self {
            metadata: self.metadata.DeepCopy(),
            obj: self.obj.clone(),
        }
    }
}

impl DerefMut for DataObjInner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.metadata
    }
}

impl From<Object> for DataObjInner {
    fn from(item: Object) -> Self {
        let metadata : MetaDataInner = MetaDataInner::New(&item);

        let inner = Self {
            metadata: metadata,
            obj: item,
        };

        return inner;
    }
}

#[derive(Debug, Default)]
pub struct DataObjList {
    pub objs: Vec<DataObject>,
    pub revision: i64,
    pub continue_: Option<Continue>,
    pub remainCount: i64,
}

impl DataObjList {
    pub fn New(objs: Vec<DataObject>, revision: i64, continue_: Option<Continue>, remainCount: i64) -> Self {
        return Self {
            objs: objs,
            revision:  revision,
            continue_: continue_,
            remainCount: remainCount,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct DataObject(Arc<DataObjInner>);

impl PartialEq for DataObject {
    fn eq(&self, other: &Self) -> bool {
        return self.0 == other.0;
    }
}
impl Eq for DataObject {}

impl From<Object> for DataObject {
    fn from(item: Object) -> Self {
        let inner = item.into();

        return Self(Arc::new(inner));
    }
}

impl From<DataObjInner> for DataObject {
    fn from(inner: DataObjInner) -> Self {
        return Self(Arc::new(inner));
    }
}

impl Deref for DataObject {
    type Target = Arc<DataObjInner>;

    fn deref(&self) -> &Arc<DataObjInner> {
        &self.0
    }
}

impl DeepCopy for DataObject {
    fn DeepCopy(&self) -> Self {
        return Self(Arc::new(self.0.DeepCopy()));
    }
}

impl DataObject {
    pub fn Namespace(&self) -> String {
        return self.metadata.namespace.clone();
    }

    pub fn Name(&self) -> String {
        return self.metadata.name.clone();
    }

    pub fn Key(&self) -> String {
        return self.metadata.Key();
    }

    pub fn Obj(&self) -> Object {
        return self.obj.clone();
    }

    pub fn Revision(&self) -> i64 {
        return self.metadata.Revision();
    }

    pub fn Decode(buf: &[u8]) -> Result<Self> {
        let obj = Object::decode(buf)?;
        return Ok(Self(Arc::new(obj.into())))
    }

    pub fn Encode(&self) -> Result<Vec<u8>> {
        let mut buf : Vec<u8> = Vec::new();
        buf.reserve(self.obj.encoded_len());
        self.obj.encode(&mut buf)?;
        return Ok(buf)
    }

    pub fn Labels(&self) -> Labels {
        let lables = self.lables.clone();
        return lables
    }

    pub fn SetRevision(&self, rev: i64) {
        self.metadata.SetRevision(rev)
    }
    
}

async fn test() -> Result<()> {
    let mut client = Client::connect(["localhost:2379"], None).await?;
    // put kv
    client.put("foo", "bar", None).await?;
    // get kv
    let resp = client.get("foo", None).await?;
    if let Some(kv) = resp.kvs().first() {
        println!("Get kv: {{{}: {}}}", kv.key_str()?, kv.value_str()?);
    }

    Ok(())
}

impl Object {
    pub fn Encode(&self) -> Result<Vec<u8>> {
        let mut buf : Vec<u8> = Vec::new();
        buf.reserve(self.encoded_len());
        self.encode(&mut buf)?;
        return Ok(buf)
    }
}

pub const PATH_PREFIX : &str = "/registry";

pub struct EtcdStore {
    pub client: EtcdClient,
    pub pathPrefix: String,
    pub pagingEnable: bool,
}

impl EtcdStore {
    pub async fn New(addr: &str, pagingEnable: bool) -> Result<Self> {
        let client = Client::connect([addr], None).await?;

        return Ok(Self {
            client: EtcdClient::New(client),
            pathPrefix: PATH_PREFIX.to_string(),
            pagingEnable,
        })
    }

    pub fn Copy(&self) -> Self {
        return Self {
            client: self.client.clone(),
            pathPrefix: self.pathPrefix.clone(),
            pagingEnable: self.pagingEnable,
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
            return Ok(())
        }

        if minRevision > actualRevision {
            return Err(Error::NewMinRevsionErr(minRevision, actualRevision))
        }
        

        return Ok(())
    }

    pub async fn Get(&self, key: &str, minRevision: i64) -> Result<Option<DataObject>> {
        let preparedKey = self.PrepareKey(key)?;
        let getResp = self.client.lock().await.get(preparedKey, None).await?;
        let kvs = getResp.kvs();
        let actualRev = getResp.header().unwrap().revision();
        Self::ValidateMinimumResourceVersion(minRevision, actualRev)?;
        
        if kvs.len() == 0 {
            return Ok(None)
        }

        let kv = &kvs[0];
        let val = kv.value();

        let obj = Object::decode(val)?;
        let inner : DataObjInner = obj.into();
        inner.SetRevision(kv.mod_revision());
        
        return Ok(Some(inner.into()))
    }

    pub async fn Create(&self, key: &str, obj: &DataObject) -> Result<()> {
        let preparedKey = self.PrepareKey(key)?;
        let keyVec: &str = &preparedKey;
        let txn = Txn::new()
            .when(vec![Compare::mod_revision(keyVec, CompareOp::Equal, 0)])
            .and_then(vec![TxnOp::put(keyVec, obj.obj.Encode()?, None)]);

        let resp = self.client.lock().await.txn(txn).await?;
        if !resp.succeeded() {
            return Err(Error::NewNewKeyExistsErr(preparedKey, 0));
        } else {
            match &resp.op_responses()[0] {
                TxnOpResponse::Put(getresp) => {
                    let actualRev = getresp.header().unwrap().revision();
                    obj.SetRevision(actualRev);
                    return Ok(())
                }
                _ => {
                    panic!("create get unexpect response")
                }
            };
        }
    }

    pub async fn Clear(&mut self, prefix: &str) -> Result<i64> {
        let preparedKey = self.PrepareKey(prefix)?;

        let keyVec: &str = &preparedKey;

        let mut options = DeleteOptions::new();
        options = options.with_prefix();
        let resp = self.client.lock().await.delete(keyVec, Some(options)).await?;

        let rv = resp.header().unwrap().revision();
        return Ok(rv)
    }

    pub async fn Delete(&self, key: &str, expectedRev: i64) -> Result<()> {
        let preparedKey = self.PrepareKey(key)?;
        let keyVec: &str = &preparedKey;
        let txn = if expectedRev != 0 {
            Txn::new()
            .when(vec![Compare::mod_revision(keyVec, CompareOp::Equal, expectedRev)])
            .and_then(vec![TxnOp::delete(keyVec, None)])
            .or_else(vec![TxnOp::get(keyVec, None)])
        } else {
            Txn::new()
            .and_then(vec![TxnOp::delete(keyVec, None)])
        };
           
        let resp = self.client.lock().await.txn(txn).await?;
        if !resp.succeeded() {
            match &resp.op_responses()[0] {
                TxnOpResponse::Get(getresp) => {
                    let actualRev = getresp.header().unwrap().revision();
                    return Err(Error::NewDeleteRevNotMatchErr(expectedRev, actualRev));
                }
                _ => {
                    panic!("Delete get unexpect response")
                }
            };
        }
        return Ok(())
    } 

    pub async fn Update(&self, key: &str, expectedRev: i64, obj: &DataObject) -> Result<()> {
        let preparedKey = self.PrepareKey(key)?;
        let keyVec: &str = &preparedKey;
        let txn = if expectedRev > 0 {
            Txn::new()
            .when(vec![Compare::mod_revision(keyVec, CompareOp::Equal, expectedRev)])
            .and_then(vec![TxnOp::put(keyVec, obj.Encode()?, None)])
            .or_else(vec![TxnOp::get(keyVec, None)])
        } else {
            Txn::new()
            .and_then(vec![TxnOp::put(keyVec, obj.Encode()?, None)])
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
                    obj.SetRevision(actualRev);
                    return Ok(())
                }
                _ => {
                    panic!("Delete get unexpect response")
                }
            };
        }
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
                let _ = end.split_off(i+1);
                return end;
            }
        }

        // next prefix does not exist (e.g., 0xffff);
	    // default to WithFromKey policy
        return vec![0];
    }

    pub async fn List(&self, prefix: &str, opts: &ListOption) -> Result<DataObjList> {
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
            //println!("key is {}, option is {:?}", &preparedKey, &getOption);
            let option = getOption.ToGetOption();
            getResp = self.client.lock().await.get(preparedKey.clone(), Some(option)).await?;
            let actualRev = getResp.header().unwrap().revision();
            Self::ValidateMinimumResourceVersion(revision, actualRev)?;

            //numFetched += getResp.kvs().len();
            hasMore = getResp.more();

            if getResp.kvs().len() == 0 && hasMore {
                return Err(Error::CommonError("no results were found, but etcd indicated there were more values remaining".to_owned()));
            }

            for kv in getResp.kvs() {
                if paging && v.len() >= pred.limit {
                    hasMore = true;
                    break;
                }

                lastKey = kv.key().to_vec();
                let obj = Object::decode(kv.value())?;
                let inner : DataObjInner = obj.into();
                inner.SetRevision(kv.mod_revision());
                let obj = inner.into();

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
            let next = EncodeContinue(&(newKey+ "\x000"), &keyPrefix, returnedRV)?;
            let mut remainingItemCount = -1;
            if pred.Empty() {
                remainingItemCount = getResp.count() as i64 - pred.limit as i64;
            }
            
            return Ok(DataObjList::New(v, returnedRV, Some(next), remainingItemCount));
        }

        return Ok(DataObjList::New(v, returnedRV, None, -1));
    }

    pub fn Watch(&self, key: &str, rev: i64, pred: SelectionPredicate) -> Result<(Watcher, WatchReader)> {
        let preparedKey = self.PrepareKey(key)?;
        return Ok(Watcher::New(&self.client, &preparedKey, rev, pred))
    }

    pub async fn Compaction(&self, revision: i64) -> Result<()> {
        let options = CompactionOptions::new().with_physical();
        self.client.lock().await.compact(revision, Some(options)).await?;
        return Ok(())
    }
    
}

// maxLimit is a maximum page limit increase used when fetching objects from etcd.
// This limit is used only for increasing page size by kube-apiserver. If request
// specifies larger limit initially, it won't be changed.
pub const MAX_LIMIT : usize = 10000;

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

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use super::super::selector::*;

    pub fn ComputePodKey(obj: &DataObject) -> String {
        return format!("/pods/{}/{}", &obj.Namespace(), &obj.Name());
    }    

    // SeedMultiLevelData creates a set of keys with a multi-level structure, returning a resourceVersion
    // from before any were created along with the full set of objects that were persisted
    async fn SeedMultiLevelData(store: &mut EtcdStore) -> Result<(i64, Vec<DataObject>)> {
        // Setup storage with the following structure:
        //  /
        //   - first/
        //  |         - bar
        //  |
        //   - second/
        //  |         - bar
        //  |         - foo
        //  |
        //   - third/
        //  |         - barfoo
        //  |         - foo
        let barFirst = DataObject::NewPod("first", "bar", "", "")?;
        let barSecond = DataObject::NewPod("second", "bar", "", "")?;
        let fooSecond = DataObject::NewPod("second", "foo", "", "")?;
        let barfooThird = DataObject::NewPod("third", "barfoo", "", "")?;
        let fooThird = DataObject::NewPod("third", "foo", "", "")?;

        struct Test {
            key: String,
            obj: DataObject,
        }

        let mut tests = [
        Test {
            key: ComputePodKey(&barFirst),
            obj: barFirst,
        },
        Test {
            key: ComputePodKey(&barSecond),
            obj: barSecond,
        },
        Test {
            key: ComputePodKey(&fooSecond),
            obj: fooSecond,
        },
        Test {
            key: ComputePodKey(&barfooThird),
            obj: barfooThird,
        },
        Test {
            key: ComputePodKey(&fooThird),
            obj: fooThird,
        },
    ];

        let initRv = store.Clear("pods").await?;

        for t in &mut tests {
            store.Create(&t.key, &t.obj).await?;
        }

        let mut pods = Vec::new();
        for t in tests {
            pods.push(t.obj);
        }

        return Ok((initRv, pods))
    }

    pub async fn RunTestListWithoutPaging() -> Result<()> {
        let mut store = EtcdStore::New("localhost:2379", true).await?;

        let (_, preset) = SeedMultiLevelData(&mut store).await?;
        
        let listOptions = ListOption {
            revision: 0,
            revisionMatch: RevisionMatch::Exact,
            predicate: SelectionPredicate { limit:2, ..Default::default() },
        };

        let list = store.List("/pods/second", &listOptions).await?;
        assert!(list.continue_.is_some()==false);

        assert!(list.objs.len()==2, "objs is {:#?}", list);
        for i in 0..list.objs.len() {
            assert!(preset[i+1]==list.objs[i], 
                "expect {:#?}, actual {:#?}", preset[i+1], &list.objs[i]);
        }

        return Ok(())
    }

    //#[test]
    pub fn RunTestListWithoutPagingSync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        
        rt.block_on(RunTestListWithoutPaging()).unwrap();
    }

    pub async fn RunTestListContinuation() -> Result<()> {
        // Setup storage with the following structure:
        //  /
        //   - first/
        //  |         - bar
        //  |
        //   - second/
        //  |         - bar
        //  |         - foo
        let barFirst = DataObject::NewPod("first", "bar", "", "")?;
        let barSecond = DataObject::NewPod("second", "bar", "", "")?;
        let fooSecond = DataObject::NewPod("second", "foo", "", "")?;
        
        struct Test {
            key: String,
            obj: DataObject,
        }

        let mut tests = [
            Test {
                key: ComputePodKey(&barFirst),
                obj: barFirst,
            },
            Test {
                key: ComputePodKey(&barSecond),
                obj: barSecond,
            },
            Test {
                key: ComputePodKey(&fooSecond),
                obj: fooSecond,
            },
        ];

        let mut store = EtcdStore::New("localhost:2379", true).await?;

        let _initRv = store.Clear("pods").await?;

        for t in &mut tests {
            store.Create(&t.key, &t.obj).await?;
        }

        let mut preset = Vec::new();
        for t in tests {
            preset.push(t.obj);
        }

        let listOptions = ListOption {
            revision: 0,
            revisionMatch: RevisionMatch::None,
            predicate: SelectionPredicate { limit:1, ..Default::default() },
        };

        let mut list = store.List("/pods", &listOptions).await?;
        
        assert!(list.objs.len()==1, "objs is {:#?}", list);
        assert!(list.continue_.is_some()==true);
        for i in 0..list.objs.len() {
            assert!(preset[i]==list.objs[i], 
                "expect {:#?}, actual {:#?}", preset[i+1], &list.objs[i]);
        }

        let continueFromSecondItem = list.continue_.take().unwrap();

        let listOptions = ListOption {
            revision: 0,
            revisionMatch: RevisionMatch::None,
            predicate: SelectionPredicate { 
                limit:0, 
                continue_: Some(continueFromSecondItem.clone()),
                ..Default::default()
            },
        };

        let list = store.List("/pods", &listOptions).await?;

        assert!(list.continue_.is_some()==false, "list is {:#?}", &list);
        for i in 0..list.objs.len() {
            assert!(preset[i+1]==list.objs[i], 
                "expect {:#?}, actual {:#?}", preset[i+1], &list.objs[i]);
        }

        let listOptions = ListOption {
            revision: 0,
            revisionMatch: RevisionMatch::None,
            predicate: SelectionPredicate { 
                limit:1, 
                continue_: Some(continueFromSecondItem.clone()),
                ..Default::default()
            },
        };

        let mut list = store.List("/pods", &listOptions).await?;

        assert!(list.continue_.is_some()==true, "list is {:#?}", &list);
        for i in 0..list.objs.len() {
            assert!(preset[i+1]==list.objs[i], 
                "expect {:#?}, actual {:#?}", preset[i+1], &list.objs[i]);
        }

        let listOptions = ListOption {
            revision: 0,
            revisionMatch: RevisionMatch::None,
            predicate: SelectionPredicate { 
                limit:1, 
                continue_: list.continue_.take(),
                ..Default::default()
            },
        };

        let list = store.List("/pods", &listOptions).await?;
        assert!(list.continue_.is_some()==false, "list is {:#?}", &list);
        for i in 0..list.objs.len() {
            assert!(preset[i+2]==list.objs[i], 
                "expect {:#?}, actual {:#?}", preset[i+1], &list.objs[i]);
        }

        let _initRv = store.Clear("pods").await?;
        return Ok(())
    }

    //#[test]
    pub fn RunTestListContinuationSync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        
        rt.block_on(RunTestListContinuation()).unwrap();
    }

    pub async fn RunTestListPaginationRareObject() -> Result<()> {
        let mut store = EtcdStore::New("localhost:2379", true).await?;
        let _initRv = store.Clear("pods").await?;

        let mut pods = Vec::new();

        for i in 0..1000 {
            let pod = DataObject::NewPod("first", &format!("pod-{}", i), "", "")?;
            store.Create(&ComputePodKey(&pod), &pod).await?;
            pods.push(pod);
        }

        let listOptions = ListOption {
            revision: 0,
            revisionMatch: RevisionMatch::None,
            predicate: SelectionPredicate { 
                limit:1, 
                field: Selector(vec![  
                    Requirement::New("metadata.name", SelectionOp::Equals, vec!["pod-999".to_owned()]).unwrap(),         
                ]),
                continue_: None,
                ..Default::default()
            },
        };

        let list = store.List("/pods", &listOptions).await?;
        assert!(list.continue_.is_some()==false, "list is {:#?}", &list);
        assert!(pods[999]==list.objs[0], 
            "expect {:#?}, actual {:#?}", pods[999], &list.objs[0]);

        let _initRv = store.Clear("pods").await?;
        return Ok(())
    }

    //#[test]
    pub fn RunTestListPaginationRareObjectSync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        
        rt.block_on(RunTestListPaginationRareObject()).unwrap();
    }
}