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
use lazy_static::lazy_static;
use hashbrown::HashMap;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::kernel::fs::dirent::*;
use super::super::socket::unix::transport::unix::*;
use crate::qlib::kernel::kernel::fd_table::GapMgr;
use crate::qlib::kernel::socket::hostinet::uring_socket::UringSocketOperationsWeak;
use crate::qlib::socket_buf::AcceptQueue;

lazy_static! {
    pub static ref ABSTRACT_SOCKET: AbstractSocketNamespace = AbstractSocketNamespace::default();
    pub static ref TCP_SOCKET: TCPSocketNamespace = TCPSocketNamespace::New();
    pub static ref UNIX_SOCKET_PINS: UnixSocketPins = UnixSocketPins::default();
}

#[derive(Clone, Default)]
pub struct AbstractSocketNamespace(Arc<QMutex<BTreeMap<Vec<u8>, BoundEndpointWeak>>>);

impl Deref for AbstractSocketNamespace {
    type Target = Arc<QMutex<BTreeMap<Vec<u8>, BoundEndpointWeak>>>;

    fn deref(&self) -> &Arc<QMutex<BTreeMap<Vec<u8>, BoundEndpointWeak>>> {
        &self.0
    }
}

impl AbstractSocketNamespace {
    // BoundEndpoint retrieves the endpoint bound to the given name. The return
    // value is nil if no endpoint was bound.
    pub fn BoundEndpoint(&self, name: &Vec<u8>) -> Option<BoundEndpoint> {
        let mut a = self.lock();
        let weak = match a.get(name) {
            None => return None,
            Some(b) => b.clone(),
        };

        match weak.Upgrade() {
            None => {
                a.remove(name);
                return None;
            }
            Some(b) => return Some(b),
        }
    }

    // Bind binds the given socket.
    //
    // When the last reference managed by rc is dropped, ep may be removed from the
    // namespace.
    pub fn Bind(&self, name: Vec<u8>, ep: &BoundEndpoint) -> Result<()> {
        let mut a = self.lock();

        match a.get(&name) {
            None => (),
            Some(b) => match b.Upgrade() {
                None => {
                    a.remove(&name);
                }
                Some(_) => return Err(Error::SysError(SysErr::EADDRINUSE)),
            },
        };

        a.insert(name, ep.Downgrade());
        return Ok(());
    }

    // Remove removes the specified socket at name from the abstract socket
    // namespace, if it has not yet been replaced.
    pub fn Remove(&self, name: &Vec<u8>, ep: &BoundEndpoint) {
        let mut a = self.lock();
        let weak = match a.get(name) {
            None => {
                // We never delete a map entry apart from a socket's destructor (although the
                // map entry may be overwritten). Therefore, a socket should exist, even if it
                // may not be the one we expect.
                return
            }
            Some(b) => b.clone(),
        };

        let boundep = match weak.Upgrade() {
            None => {
                // We never delete a map entry apart from a socket's destructor (although the
                // map entry may be overwritten). Therefore, a socket should exist, even if it
                // may not be the one we expect.
                panic!("expected socket to exist at {:?} in abstract socket namespace", name);
            }
            Some(b) => b,
        };

        if boundep == *ep {
            a.remove(name);
        }
    }
}

#[derive(Clone, Default)]
pub struct UnixSocketPins(Arc<QMutex<BTreeMap<Vec<u8>, Dirent>>>);

impl Deref for UnixSocketPins {
    type Target = Arc<QMutex<BTreeMap<Vec<u8>, Dirent>>>;

    fn deref(&self) -> &Arc<QMutex<BTreeMap<Vec<u8>, Dirent>>> {
        &self.0
    }
}

impl UnixSocketPins {
    pub fn Pin(&self, name: Vec<u8>, dirent: &Dirent) {
        let mut a = self.lock();

        a.insert(name, dirent.clone());
    }

    pub fn Unpin(&self, name: &Vec<u8>) {
        let mut a = self.lock();

        a.remove(name);
    }
}

#[derive(Clone)]
pub enum InnerSocket {
    UringSocketServer(UringSocketOperationsWeak), 
}

impl From<UringSocketOperationsWeak> for InnerSocket  {
    fn from(inner: UringSocketOperationsWeak) -> Self {
        return Self::UringSocketServer(inner);
    }
}

pub struct TCPSocketNamespace(Arc<QMutex<TCPSocketNamespaceInner>>);

impl Deref for TCPSocketNamespace {
    type Target = Arc<QMutex<TCPSocketNamespaceInner>>;

    fn deref(&self) -> &Arc<QMutex<TCPSocketNamespaceInner>> {
        &self.0
    }
}

impl TCPSocketNamespace {
    pub fn New() -> Self {
        return Self(Arc::new(QMutex::new(TCPSocketNamespaceInner::New())))
    }

    pub fn Add(&self, port: u16, q: AcceptQueue) -> Result<()> {
        return self.lock().Add(port, q)
    }

    pub fn Remove(&self, port: u16) -> Result<()> {
        return self.lock().Remove(port);
    }

    pub fn Get(&self, port: u16) -> Option<AcceptQueue> {
        match self.lock().descTbl.get(&port) {
            None => return None,
            Some(q) => {
                return Some(q.clone())
            }
        }
    }
}

pub struct TCPSocketNamespaceInner {
    pub gaps: GapMgr,
    pub descTbl: HashMap<u16, AcceptQueue>,
}

impl TCPSocketNamespaceInner {
    pub fn New() -> Self {
        return Self {
            gaps: GapMgr::New(0, i16::MAX as u64),
            descTbl: HashMap::new(),
        };
    }

    pub fn Add(&mut self, port: u16, q: AcceptQueue) -> Result<()> {
        match self.descTbl.insert(port, q) {
            None => return Ok(()),
            _ => return Ok(()), // double listen in nginx panic!("TCPSocketNamespaceInner::NewSocketAt for port {}", port)
        }
    }

    pub fn Remove(&mut self, port: u16) -> Result<()> {
        match self.descTbl.remove(&port) {
            None => return Ok(()), //panic!("TCPSocketNamespaceInner::RemoveSocket for port {}", port),
            _ => return Ok(())
        }
    }
}