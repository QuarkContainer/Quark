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

use core::marker::PhantomData;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use rocksdb::{DB, Options, SingleThreaded, SliceTransform, WriteBatch};
use rocksdb::DBWithThreadMode;

use qobjs::common::*;

pub struct RocksStore {
    pub db: Arc<Mutex<DBWithThreadMode<SingleThreaded>>>,
    pub registryPath: String,
    pub initRev: i64,
}

pub const NODE_AGENT_STORE_PATH: &str = "/var/log/quark/data";
pub const REVISION_KEY: &str = "revision";
pub const REGISTRY_PATH: &str = "/registry/";

impl RocksStore {
    pub fn New() -> Result<Self> {
        let registryPath = REGISTRY_PATH;
        let path = NODE_AGENT_STORE_PATH;

        if !Path::new(path).exists() {
            fs::create_dir_all(path)?;
        }
    
        let prefix_extractor = SliceTransform::create_fixed_prefix(100);
    
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_prefix_extractor(prefix_extractor);

        let db = DB::open(&opts, path).unwrap();

        let revisionPath = format!("{}{}", registryPath, REVISION_KEY);
        let read = db.get(&revisionPath)?;
        let revision = match read {
            Some(v) => {
                assert!(v.len() == 8);
                unsafe {*(&v[0] as * const u8 as u64 as * const i64)}
            }
            None => {
                let rev : i64 = 0;
                db.put(&revisionPath, rev.to_ne_bytes())?;
                0
            }
        };
        
        return Ok(Self {
            db: Arc::new(Mutex::new(db)),
            registryPath: registryPath.to_string(),
            initRev: revision,
        });
    }

    // example: objType = "pods"
    pub fn NewObjStore<T: for<'a> Deserialize<'a> + Serialize>(&self, objType: &str ) -> RocksObjStore<T> {
        return RocksObjStore {
            prefix: format!("{}{}/", &self.registryPath, objType),
            revision: format!("{}{}", &self.registryPath, REVISION_KEY),
            db: self.db.clone(),
            phantom: PhantomData::default(),
        }
    }
}


#[derive(Debug)]
pub struct RocksObjStore<T: for<'a> Deserialize<'a> + Serialize > {
    pub prefix: String,
    pub revision: String,
    pub db: Arc<Mutex<DBWithThreadMode<SingleThreaded>>>,
    phantom: PhantomData<T>,
}

impl <T: for<'a> Deserialize<'a> + Serialize> RocksObjStore<T> {
    pub fn Save(&self, revision: i64, key: &str, obj: &T) -> Result<()> {
        let mut batch = WriteBatch::default();
        batch.put(self.revision.clone(), revision.to_ne_bytes());

        let data =  serde_json::to_string(obj)?;
        let key = format!("{}{}", self.prefix, key);
        batch.put(key, data);
        self.db.lock().unwrap().write(batch)?;
        return Ok(())
    }

    pub fn Remove(&self, revision: i64, key: &str) -> Result<()> {
        let mut batch = WriteBatch::default();
        batch.put(self.revision.clone(), revision.to_ne_bytes());
        
        let key = format!("{}{}", self.prefix, key);
        batch.delete(key);

        self.db.lock().unwrap().write(batch)?;
        return Ok(())
    }

    pub fn Load(&self) -> Result<Vec<T>> {
        let db = self.db.lock().unwrap();
        let mut iter = db.prefix_iterator(self.prefix.clone());
        let mut ret = Vec::new();
        loop {
            match iter.next() {
                None => break,
                Some(v) => {
                    let v = v?;
                    let str = String::from_utf8(v.1.to_vec())?;
                    let obj: T = serde_json::from_str(&str)?;
                    ret.push(obj);
                }
            }
        }

        return Ok(ret);
    }
}

 