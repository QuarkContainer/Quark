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


use std::{sync::Arc, collections::BTreeMap};
use core::ops::Deref;

use etcd_client::{Client, Compare, Txn, TxnOp, CompareOp, TxnOpResponse};
use etcd_client::GetOptions;
use prost::Message;

use crate::{shared::common::*, selector::Labels};
use super::service_directory::*;
use super::selection_predicate::*;

#[derive(Debug)]
pub struct DataObjInner {
    pub kind: String,
    pub namespace: String,
    pub name: String,
    pub lables: Labels,
    pub fields: Labels,
    
    // revision number set by etcd
    pub reversion: i64,

    pub obj: Object,
}

impl From<Object> for DataObjInner {
    fn from(item: Object) -> Self {
        let mut map = BTreeMap::new();
        for l in &item.labels {
            map.insert(l.key.clone(), l.val.clone());
        }

        let mut fields = BTreeMap::new();
        fields.insert("metadata.name".to_owned(), item.name.clone());
        fields.insert("metadata.namespace".to_owned(), item.namespace.clone());

        let inner = DataObjInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: map.into(),
            fields: fields.into(),
            reversion: 0,
            obj: item,
        };

        return inner;
    }
}

pub struct DataObjList {
    pub objs: Vec<DataObject>,
    pub revision: i64,
    pub next: Option<Continue>,
    pub remainCount: i64,
}

impl DataObjList {
    pub fn New(objs: Vec<DataObject>, revision: i64, next: Option<Continue>, remainCount: i64) -> Self {
        return Self {
            objs: objs,
            revision:  revision,
            next: next,
            remainCount: remainCount,
        }
    }
}

#[derive(Debug)]
pub struct DataObject(Arc<DataObjInner>);

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

impl DataObject {
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

    pub fn Attributes(&self) -> (Labels, Labels) {
        return (self.lables.clone(), self.fields.clone())
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

pub const PATH_PREFIX : &str = "/registry";

pub struct EtcdStore {
    pub client: Client,
    pub pathPrefix: String,
    pub pagingEnable: bool,
}

impl EtcdStore {
    pub async fn New(addr: &str, pagingEnable: bool) -> Result<Self> {
        let client = Client::connect([addr], None).await?;

        return Ok(Self {
            client: client,
            pathPrefix: PATH_PREFIX.to_string(),
            pagingEnable,
        })
    }

    pub fn PrepareKey(&self, key: &str) -> Result<String> {
        if key == "." || key == "/" {
            return Err(Error::CommonError(format!("invalid key: {}", key)));
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

    pub async fn Get(&mut self, key: &str, minRevision: i64) -> Result<Option<DataObject>> {
        let preparedKey = self.PrepareKey(key)?;
        let getResp = self.client.get(preparedKey, None).await?;
        let kvs = getResp.kvs();
        let actualRev = getResp.header().unwrap().revision();
        Self::ValidateMinimumResourceVersion(minRevision, actualRev)?;
        
        if kvs.len() == 0 {
            return Ok(None)
        }

        let kv = &kvs[0];
        let val = kv.value();

        let obj = Object::decode(val)?;
        let mut inner : DataObjInner = obj.into();
        inner.reversion = kv.mod_revision();
        
        return Ok(Some(DataObject(Arc::new(inner))))
    }

    pub async fn Create(&mut self, key: &str, obj: &DataObject) -> Result<i64> {
        let preparedKey = self.PrepareKey(key)?;
        let keyVec: &str = &preparedKey;
        let txn = Txn::new()
            .when(vec![Compare::mod_revision(keyVec, CompareOp::Equal, 0)])
            .and_then(vec![TxnOp::put(keyVec, obj.Encode()?, None)]);

        let resp = self.client.txn(txn).await?;
        if !resp.succeeded() {
            return Err(Error::NewNewKeyExistsErr(preparedKey, 0));
        } else {
            match &resp.op_responses()[0] {
                TxnOpResponse::Put(getresp) => {
                    let actualRev = getresp.header().unwrap().revision();
                    return Ok(actualRev)
                }
                _ => {
                    panic!("Delete get unexpect response")
                }
            };
        }
    }

    pub async fn Delete(&mut self, key: &str, expectedRev: i64) -> Result<()> {
        let preparedKey = self.PrepareKey(key)?;
        let keyVec: &str = &preparedKey;
        let txn = Txn::new()
            .when(vec![Compare::mod_revision(keyVec, CompareOp::Equal, expectedRev)])
            .and_then(vec![TxnOp::delete(keyVec, None)])
            .or_else(vec![TxnOp::get(keyVec, None)]);
        let resp = self.client.txn(txn).await?;
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

    pub async fn Update(&mut self, key: &str, expectedRev: i64, obj: &mut DataObject) -> Result<i64> {
        let preparedKey = self.PrepareKey(key)?;
        let keyVec: &str = &preparedKey;
        let txn = Txn::new()
            .when(vec![Compare::mod_revision(keyVec, CompareOp::Equal, expectedRev)])
            .and_then(vec![TxnOp::put(keyVec, obj.Encode()?, None)])
            .or_else(vec![TxnOp::get(keyVec, None)]);
        let resp = self.client.txn(txn).await?;
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
                    return Ok(actualRev)
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

    pub async fn List(&mut self, prefix: &str, opts: &ListOption) -> Result<DataObjList> {
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
                    RevisionMatch::NotOlderThan => (),
                    RevisionMatch::Exact => {
                        returnedRV = fromRv;
                        withRev = returnedRV;
                    }
                }
            }
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
            let option = getOption.ToGetOption();
            getResp = self.client.get(preparedKey.clone(), Some(option)).await?;
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
                let mut inner : DataObjInner = obj.into();
                inner.reversion = kv.mod_revision();
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
            let next = EncodeContinue(&(newKey+ "\x00"), &keyPrefix, returnedRV)?;
            let mut remainingItemCount = -1;
            if pred.Empty() {
                remainingItemCount = getResp.count() as i64 - pred.limit as i64;
            }
            
            return Ok(DataObjList::New(v, returnedRV, Some(next), remainingItemCount));
        }

        return Ok(DataObjList::New(v, returnedRV, None, -1));
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
    pub revision: Option<i64>
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

        return getOption;
    }
}