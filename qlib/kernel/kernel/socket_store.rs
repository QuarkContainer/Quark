// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use super::super::fs::file::*;

#[derive(Clone)]
pub struct SocketRecord {
    pub id: u64,
    pub socket: FileWeak,
}

#[derive(Default)]
pub struct SocketStoreIntern {
    pub nextRecord: u64,
    pub sockets: BTreeMap<u64, SocketRecord>,
}

#[derive(Default)]
pub struct SocketStore(Arc<QMutex<SocketStoreIntern>>);

impl Deref for SocketStore {
    type Target = Arc<QMutex<SocketStoreIntern>>;

    fn deref(&self) -> &Arc<QMutex<SocketStoreIntern>> {
        &self.0
    }
}

impl SocketStore {
    pub fn AddSocket(&self, sock: &File) {
        let mut store = self.lock();
        let rid = store.nextRecord;

        let sockId = sock.UniqueId();
        if store.sockets.contains_key(&sockId) {
            panic!("SocketStore::AddSocket Socket {} added twice", sockId);
        }

        store.nextRecord += 1;
        store.sockets.insert(
            sockId,
            SocketRecord {
                id: rid,
                socket: sock.Downgrade(),
            },
        );
    }

    pub fn DeleteSocket(&self, sock: &File) {
        let mut store = self.lock();
        let sockId = sock.UniqueId();
        store.sockets.remove(&sockId);
    }

    pub fn ListSockets(&self) -> Vec<(u64, File)> {
        let mut socks = Vec::new();
        let store = self.lock();

        for (id, sock) in &store.sockets {
            match sock.socket.Upgrade() {
                None => panic!("SocketStore::ListSockets get empty socket {}", id),
                Some(s) => socks.push((sock.id, s)),
            }
        }

        return socks;
    }
}
