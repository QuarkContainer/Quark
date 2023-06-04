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

use std::path::Path;
use std::sync::Mutex;
use std::sync::Arc;
use std::ops::Deref;
use std::time::SystemTime;
use std::fs::File;
use std::io::SeekFrom;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::io::prelude::*;

use qobjs::common::*;

use super::blob_store::BLOB_STORE;

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
pub enum BlobState {
    Created,
    Sealed,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BlobInner {
    pub namespace: String,
    pub name: String,
    pub size: usize,
    pub checksum: String,
    pub state: BlobState,
    pub createTime: SystemTime,
    pub lastAccessTime: SystemTime,
}

#[derive(Debug, Clone)]
pub struct Blob(pub Arc<Mutex<BlobInner>>);

impl Deref for Blob {
    type Target = Arc<Mutex<BlobInner>>;

    fn deref(&self) -> &Arc<Mutex<BlobInner>> {
        &self.0
    }
}

impl From<BlobInner> for Blob {
    fn from(item: BlobInner) -> Self {
        return Self(Arc::new(Mutex::new(item)))
    }
}

impl Blob {
    pub fn Create(namespace: &str, name: &str) -> Result<Self> {
        let path = Path::new(name);
        if !path.has_root() {
            return Err(Error::EINVAL(format!("Blob name '{:?}' must has root", name)))
        }

        if path.is_dir() {
            return Err(Error::EINVAL(format!("Blob name '{:?}' must not be a directory", name)))
        }

        if namespace.contains("/") {
            return Err(Error::EINVAL(format!("Blob namespace '{}' must not containers '/' ", namespace)))
        }

        let inner = BlobInner {
            namespace: namespace.to_string(),
            name: name.to_string(),
            size: 0,
            checksum: String::new(),
            createTime: SystemTime::now(),
            lastAccessTime: SystemTime::now(),
            state: BlobState::Created,
        };

        return Ok(Self(Arc::new(Mutex::new(inner))))
    }

    pub fn State(&self) -> BlobState {
        return self.lock().unwrap().state;
    }

    pub fn Address(&self) -> String {
        let inner = self.lock().unwrap();
        return format!("/{}{}", &inner.namespace, &inner.name);
    }

    pub fn Access(&self) {
        let mut inner = self.lock().unwrap();
        inner.lastAccessTime = SystemTime::now();
    }

    pub fn Seal(&self, size: usize, checksum: String) {
        let mut inner = self.lock().unwrap();
        inner.size = size;
        inner.checksum = checksum;
        assert!(inner.state == BlobState::Created);
        inner.state = BlobState::Sealed;
    }

    pub fn ToString(&self) -> Result<String> {
        let str = serde_json::to_string(&*self.lock().unwrap())?;
        return Ok(str);
    }
}

#[derive(Debug)]
pub struct WriteBlob {
    pub id: u64,
    pub blob: Blob,
    pub file: File,
    pub size: usize,
    pub hasher: Option<Sha256>,
}

impl Drop for WriteBlob {
    fn drop(&mut self) {
        if self.hasher.is_some() {
            self.Seal().unwrap();
        }
    }
}

impl WriteBlob {
    pub fn New(id: u64, meta: &Blob, file: File) -> Self {
        return Self {
            id: id,
            blob: meta.clone(),
            file: file,
            size: 0,
            hasher: Some(Sha256::new()),
        }
    }

    pub fn Write(&mut self, buf: &[u8]) -> Result<()> {
        self.file.write_all(buf)?;
        match &mut self.hasher {
            None => return Err(Error::EINVAL(format!("WriteBlob::Write the blob has been sealled"))),
            Some(h) => {
                h.update(buf);
            }
        }
        self.size += buf.len();
        return Ok(())
    }

    pub fn Seal(&mut self) -> Result<()> {
        self.file.flush().unwrap();
        let hash = self.hasher.take().unwrap().finalize();
        self.blob.Seal(self.size, format!("{:?}", hash));
        return BLOB_STORE.Seal(self);
    }
}

#[derive(Debug)]
pub struct ReadBlob {
    pub id: u64,
    pub blob: Blob,
    pub file: File,
}

impl ReadBlob {
    pub fn New(id: u64, meta: &Blob, file: File) -> Self {
        return Self {
            id: id,
            blob: meta.clone(),
            file: file,
        }
    }

    pub fn Read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let count = self.file.read(buf)?;
        return Ok(count);
    }

    pub fn Seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let off = self.file.seek(pos)?;
        return Ok(off);
    }
}

#[derive(Debug)]
pub enum BlobHandlerInner {
    Write(WriteBlob),
    Read(ReadBlob),
}

#[derive(Debug, Clone)]
pub struct BlobHandler(Arc<Mutex<BlobHandlerInner>>);

impl Deref for BlobHandler {
    type Target = Arc<Mutex<BlobHandlerInner>>;

    fn deref(&self) -> &Arc<Mutex<BlobHandlerInner>> {
        &self.0
    }
}

impl BlobHandler {
    pub fn NewWrite(b: WriteBlob) -> Self {
        return Self(Arc::new(Mutex::new(BlobHandlerInner::Write(b))));
    }

    pub fn NewRead(b: ReadBlob) -> Self {
        return Self(Arc::new(Mutex::new(BlobHandlerInner::Read(b))));
    }

    pub fn Read(&self, buf: &mut [u8]) -> Result<usize> {
        let mut inner = self.lock().unwrap();
        match &mut *inner {
            BlobHandlerInner::Write(_) => return Err(Error::EINVAL(format!("can't read a writeable blob"))),
            BlobHandlerInner::Read(b) => {
                return b.Read(buf);
            }
        }
    }

    pub fn Seek(&self, pos: SeekFrom) -> Result<u64> {
        let mut inner = self.lock().unwrap();
        match &mut *inner {
            BlobHandlerInner::Write(_) => return Err(Error::EINVAL(format!("can't Seek a writeable blob"))),
            BlobHandlerInner::Read(b) => {
                return b.Seek(pos);
            }
        }
    }

    pub fn Write(&self, buf: &[u8]) -> Result<()> {
        let mut inner = self.lock().unwrap();
        match &mut *inner {
            BlobHandlerInner::Read(_) => return Err(Error::EINVAL(format!("can't write a readonly blob"))),
            BlobHandlerInner::Write(b) => {
                return b.Write(buf);
            }
        }
    }

    pub fn Seal(&self) -> Result<()> {
        let mut inner = self.lock().unwrap();
        match &mut *inner {
            BlobHandlerInner::Read(_) => return Err(Error::EINVAL(format!("can't write a readonly blob"))),
            BlobHandlerInner::Write(b) => {
                return b.Seal();
            }
        }
    }
}
