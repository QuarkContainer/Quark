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

use qobjs::common::*;

use crate::blobstore::blob::BlobHandler;
use crate::blobstore::blob_store::BLOB_STORE;

#[derive(Debug)]
pub struct BlobSessionInner {
    pub blobHandlers: BTreeMap<u64, BlobHandler>,
    pub lastSessionId: u64,
}

#[derive(Debug, Clone)]
pub struct BlobSession(Arc<Mutex<BlobSessionInner>>);

impl Deref for BlobSession {
    type Target = Arc<Mutex<BlobSessionInner>>;

    fn deref(&self) -> &Arc<Mutex<BlobSessionInner>> {
        &self.0
    }
}

impl BlobSession {
    pub fn New() -> Self {
        let inner = BlobSessionInner {
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

    pub fn Get(&self, id: u64) -> Result<BlobHandler> {
        match self.lock().unwrap().blobHandlers.get(&id) {
            None => return Err(Error::EPERM(format!("BlobSession can't find handler with id {}", id))),
            Some(h) => {
                return Ok(h.clone());
            }
        }
    }

    pub fn Read(&self, id: u64, len: u64) -> Result<Vec<u8>> {
        let handler = self.Get(id)?;
        let mut buf = Vec::with_capacity(len as usize);
        buf.resize(len as usize, 0u8);
        let size = handler.Read(&mut buf)?;
        buf.resize(size, 0);
        return Ok(buf);
    }
}