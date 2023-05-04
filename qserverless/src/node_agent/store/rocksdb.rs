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
use std::sync::Arc;
use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use rocksdb::{DB, Options, SingleThreaded, SliceTransform};
use rocksdb::DBWithThreadMode;

use qobjs::common::*;

pub struct RocksStore {
    pub db: Arc<Mutex<DBWithThreadMode<SingleThreaded>>>,
    pub registryPath: String,
}

pub const NODE_AGENT_STORE_PATH: &str = "/var/run/quark";

impl RocksStore {
    // example: registryPath = "/registry/"
    pub fn New(registryPath: &str) -> Self {
        let path = NODE_AGENT_STORE_PATH;
    
        let prefix_extractor = SliceTransform::create_fixed_prefix(3);
    
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_prefix_extractor(prefix_extractor);

        let db = DB::open(&opts, path).unwrap();

        return Self {
            db: Arc::new(Mutex::new(db)),
            registryPath: registryPath.to_string()
        }
    }

    // example: objType = "pods"
    pub fn NewObjStore<T: for<'a> Deserialize<'a> + Serialize>(&self, objType: &str ) -> RocksObjStore<T> {
        return RocksObjStore {
            prefix: format!("{}{}/", &self.registryPath, objType),
            db: self.db.clone(),
            phantom: PhantomData::default(),
        }
    }
}

pub struct RocksObjStore<T: for<'a> Deserialize<'a> + Serialize > {
    pub prefix: String,
    pub db: Arc<Mutex<DBWithThreadMode<SingleThreaded>>>,
    phantom: PhantomData<T>,
}

impl <T: for<'a> Deserialize<'a> + Serialize> RocksObjStore<T> {
    pub fn Save(&self, key: &str, obj: &T) -> Result<()> {
        let data =  serde_json::to_string(obj)?;
        let key = format!("{}{}", self.prefix, key);
        self.db.lock().unwrap().put(key, data)?;
        return Ok(())
    }

    pub fn Load(self) -> Result<Vec<T>> {
        let db = self.db.lock().unwrap();
        let mut iter = db.prefix_iterator(self.prefix);
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

 