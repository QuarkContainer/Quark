// Copyright (c) 2021 QuarkSoft LLC
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

use alloc::sync::Arc;
use spin::Mutex;
use alloc::collections::btree_map::BTreeMap;
use core::ops::Deref;
use lazy_static::lazy_static;
use alloc::vec::Vec;

use super::super::socket::unix::transport::unix::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;

lazy_static! {
    pub static ref ABSTRACT_SOCKET: AbstractSocketNamespace = AbstractSocketNamespace::default();
}

pub fn BoundEndpoint(name: &Vec<u8>) -> Option<BoundEndpoint> {
    return ABSTRACT_SOCKET.BoundEndpoint(name);
}

pub fn Bind(name: Vec<u8>, ep: &BoundEndpoint) -> Result<()> {
    return ABSTRACT_SOCKET.Bind(name, ep);
}

#[derive(Clone, Default)]
pub struct AbstractSocketNamespace(Arc<Mutex<BTreeMap<Vec<u8>, BoundEndpointWeak>>>);

impl Deref for AbstractSocketNamespace {
    type Target = Arc<Mutex<BTreeMap<Vec<u8>, BoundEndpointWeak>>>;

    fn deref(&self) -> &Arc<Mutex<BTreeMap<Vec<u8>, BoundEndpointWeak>>> {
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
            Some(b) => {
                return Some(b)
            }
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
            Some(b) => {
                match b.Upgrade() {
                    None => {
                        a.remove(&name);
                    }
                    Some(_) => {
                        return Err(Error::SysError(SysErr::EADDRINUSE))
                    }
                }
            }
        };

        a.insert(name, ep.Downgrade());
        return Ok(())
    }
}