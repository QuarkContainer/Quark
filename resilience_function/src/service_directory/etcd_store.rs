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

use etcd_client::{Client, Compare, Txn, TxnOp, CompareOp, TxnOpResponse};
use prost::Message;

use crate::{shared::common::*, selector::Labels};
use super::service_directory::*;

#[derive(Debug)]
pub struct DataObjInner {
    pub kind: String,
    pub namespace: String,
    pub name: String,
    pub reversion: i64,
    pub lables: Labels,
    pub obj: Object,
}

impl From<Object> for DataObjInner {
    fn from(item: Object) -> Self {
        let mut labels = Labels::default();
        for l in &item.labels {
            labels.0.insert(l.key.clone(), l.val.clone());
        }

        let inner = DataObjInner {
            kind: item.kind.clone(),
            namespace: item.namespace.clone(),
            name: item.name.clone(),
            lables: labels,
            reversion: item.reversion,
            obj: item,
        };

        return inner;
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
}

impl EtcdStore {
    pub async fn New(addr: &str) -> Result<Self> {
        let client = Client::connect([addr], None).await?;

        return Ok(Self {
            client: client,
            pathPrefix: PATH_PREFIX.to_string(),
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

        let mut obj = Object::decode(val)?;
        obj.reversion = actualRev;
        
        return Ok(Some(obj.into()))
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

    /*pub async fn List(&mut self, prefix: &str, opts: &ListOption) -> Result<Vec<DataObject>> {
        let mut keyPrefix = self.PrepareKey(prefix)?;

        if !keyPrefix.ends_with("/") {
            keyPrefix = keyPrefix + "/";
        }

        let 

        return Ok(Vec::new())
    }*/
}

#[derive(Debug)]
pub enum RevisionMatch {
    NotOlderThan,
    Exact
}


#[derive(Debug)]
pub struct ListOption {
    pub revision: i64,

    pub revisionMatch: RevisionMatch,

}
