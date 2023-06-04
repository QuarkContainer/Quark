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

use crate::BLOB_SVC_ADDR;
use crate::BLOB_SVC_CLIENT_MGR;
use crate::blobstore::blob::BlobHandler;
use crate::blobstore::blob_store::BLOB_STORE;

use super::blob::Blob;

#[derive(Debug, Default)]
pub struct BlobSessionInner {
    pub blobHandlers: BTreeMap<u64, BlobHandler>,
    pub lastBlobId: u64,
}

#[derive(Debug, Clone, Default)]
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
            lastBlobId: 0,
        };

        return Self(Arc::new(Mutex::new(inner)));
    }

    fn NextBlobId(&self) -> u64 {
        let mut inner = self.lock().unwrap();
        inner.lastBlobId += 1;
        return inner.lastBlobId;
    }

    pub fn Create(&self, namespace: &str, name: &str) -> Result<u64> {
        let id = self.NextBlobId();
        let writeBlob = BLOB_STORE.CreateBlob(id, namespace, name)?;
        self.lock().unwrap().blobHandlers.insert(id, BlobHandler::NewWrite(writeBlob));
        return Ok(id)
    }

    pub async fn Write(&self, id: u64, buf: &[u8]) -> Result<()> {
        let handler = self.Get(id)?;
        return handler.Write(buf).await;
    }

    pub async fn Seal(&self, id: u64) -> Result<()> {
        let handler = self.Get(id)?;
        return handler.Seal().await;
    }

    pub async fn Open(&self, svcAddr: &str, namespace: &str, name: &str) -> Result<(u64, Blob)> {
        let id = self.NextBlobId();

        if svcAddr == BLOB_SVC_ADDR.get().unwrap() {
            let b = BLOB_STORE.Open(id, namespace, name)?;
            let blob = b.blob.clone();
            self.lock().unwrap().blobHandlers.insert(id, BlobHandler::NewRead(b));
            return Ok((id, blob))
        } else {
            let b = BLOB_SVC_CLIENT_MGR.Open(svcAddr, namespace, name).await?;
            let blob = b.blob.clone();
            self.lock().unwrap().blobHandlers.insert(id, BlobHandler::NewRemoteRead(b));
            return Ok((id, blob))
        }
    }

    pub fn Get(&self, id: u64) -> Result<BlobHandler> {
        match self.lock().unwrap().blobHandlers.get(&id) {
            None => return Err(Error::EINVAL(format!("BlobSession can't find handler with id {}", id))),
            Some(h) => {
                return Ok(h.clone());
            }
        }
    }

    pub async fn Read(&self, id: u64, len: u64) -> Result<Vec<u8>> {
        let handler = self.Get(id)?;
        let buf = handler.Read(len).await?;
        return Ok(buf);
    }

    pub async fn Seek(&self, id: u64, seekType: u32, pos: i64) -> Result<u64> {
        let handler = self.Get(id)?;
        return handler.Seek(seekType, pos).await;
    }

    pub async fn Close(&self, id: u64) -> Result<()> {
        let handler = self.lock().unwrap().blobHandlers.remove(&id);
        match handler {
            None => return Err(Error::EINVAL(format!("BlobSession can't find handler with id {}", id))),
            Some(h) => {
                h.Close().await?;
                return Ok(());
            }
        }
    }
}