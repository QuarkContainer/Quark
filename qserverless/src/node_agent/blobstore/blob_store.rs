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
use std::sync::Mutex;
use std::sync::Arc;
use std::ops::Deref;
use std::path::Path;
use std::fs;
use rocksdb::{DB, Options, SingleThreaded, SliceTransform, IteratorMode};
use rocksdb::DBWithThreadMode;

use qobjs::common::*;
use crate::blobstore::blob::{BlobState, BlobInner};

use super::blob::BlobHandler;
use super::blob::{Blob, WriteBlob, ReadBlob};
use super::blob_fs::BlobFs;

lazy_static::lazy_static! {
    pub static ref BLOB_STORE: BlobStore = {
        BlobStore::NewFromLoad().unwrap()
    };
}

pub const BLOB_STORE_META_PATH: &str = "/var/lib/quark/blobstore/meta";
pub const BLOB_STORE_DATA_PATH: &str = "/var/lib/quark/blobstore/data";

#[derive(Debug)]
pub struct BlobStoreSessionInner {
    pub sessionId: u64,
    pub blobHandlers: BTreeMap<u64, BlobHandler>,
    pub lastSessionId: u64,
}

#[derive(Debug, Clone)]
pub struct BlobStoreSession(Arc<Mutex<BlobStoreSessionInner>>);

impl Deref for BlobStoreSession {
    type Target = Arc<Mutex<BlobStoreSessionInner>>;

    fn deref(&self) -> &Arc<Mutex<BlobStoreSessionInner>> {
        &self.0
    }
}

impl BlobStoreSession {
    pub fn New(sessionId: u64) -> Self {
        let inner = BlobStoreSessionInner {
            sessionId: sessionId,
            blobHandlers: BTreeMap::new(),
            lastSessionId: 0,
        };

        return Self(Arc::new(Mutex::new(inner)));
    }

    fn NextSessionId(&self) -> u64 {
        let mut inner = self.lock().unwrap();
        inner.lastSessionId += 1;
        return inner.lastSessionId;
    }

    pub fn Create(&self, namespace: &str, name: &str) -> Result<u64> {
        let id = self.NextSessionId();
        let writeBlob = BLOB_STORE.CreateBlob(id, namespace, name)?;
        self.lock().unwrap().blobHandlers.insert(id, BlobHandler::NewWrite(writeBlob));
        return Ok(id)
    }

    pub fn Open(&self, namespace: &str, name: &str) -> Result<u64> {
        let id = self.NextSessionId();
        let b = BLOB_STORE.Open(id, namespace, name)?;
        self.lock().unwrap().blobHandlers.insert(id, BlobHandler::NewRead(b));
        return Ok(id)
    }
}

pub struct BlobStore {
    pub db: Mutex<DBWithThreadMode<SingleThreaded>>,
    pub blobs: Mutex<BTreeMap<String, Blob>>,
    pub blobfs: BlobFs,
}

impl BlobStore {
    pub fn NewFromLoad() -> Result<Self> {
        let path = BLOB_STORE_META_PATH;

        if !Path::new(path).exists() {
            fs::create_dir_all(path)?;
        }
    
        let prefix_extractor = SliceTransform::create_fixed_prefix(256);
    
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_prefix_extractor(prefix_extractor);

        let blobfs = BlobFs::New(BLOB_STORE_DATA_PATH);

        let db = DB::open(&opts, path).unwrap();
        let mut blobs = BTreeMap::new();
        let mut cleanBlobs = Vec::new();
        {
            let mut iter = db.iterator(IteratorMode::Start);
            loop {
                match iter.next() {
                    None => break,
                    Some(v) => {
                        let v = v?;
                        let key = String::from_utf8(v.0.to_vec())?;
                        let data = String::from_utf8(v.1.to_vec())?;
                        let blobInner: BlobInner = serde_json::from_str(&data)?;
                        let blob : Blob = blobInner.into();
                        assert!(key==blob.Address());
                        if blob.State() == BlobState::Created {
                            // the blob is been sealed before crash, clean it
                            blobfs.Remove(&key)?;
                            cleanBlobs.push(key);
                            continue;
                        }
                        blobs.insert(key, blob);
                    }
                }
            }
        }
        
        for b in cleanBlobs {
            db.delete(b)?;
        }

        let store = Self {
            db: Mutex::new(db),
            blobs: Mutex::new(blobs),
            blobfs: blobfs,

        };

        return Ok(store)
    }

    pub fn CreateBlob(&self, id: u64, namespace: &str, name: &str) -> Result<WriteBlob> {
        let blob = Blob::Create(namespace, name)?;
        let file = self.blobfs.Create(&blob.Address())?;
        self.db.lock().unwrap().put(blob.Address(), blob.ToString()?)?;
        let blob = WriteBlob::New(id, &blob, file);
        return Ok(blob)
    }

    pub fn Seal(&self, writeBlob: &WriteBlob) -> Result<()> {
        let blob = &writeBlob.blob;
        self.db.lock().unwrap().put(blob.Address(), blob.ToString()?)?;
        return Ok(())
    }

    pub fn Open(&self, id: u64, namespace: &str, name: &str) -> Result<ReadBlob> {
        let addr = format!("/{}{}", namespace, name);
        let blob = match self.blobs.lock().unwrap().get(&addr) {
            None => return Err(Error::ENOENT(format!("BlobStore Open blob doesn't exisit {}", &addr))),
            Some(b) => b.clone(),
        };

        let file = self.blobfs.Open(&blob.Address())?;
        return Ok(ReadBlob {
            id: id,
            blob: blob,
            file: file,
        });
    }
}