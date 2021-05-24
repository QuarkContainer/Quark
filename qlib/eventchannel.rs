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

use lazy_static::lazy_static;
use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;
use alloc::string::ToString;

use super::common::*;

lazy_static! {
    pub static ref EMITTERS: Mutex<Emitters> = Mutex::new(Emitters(BTreeMap::new()));
}

pub struct Emitters(BTreeMap<u64, Arc<Mutex<Emitter>>>);

#[derive(Clone, Debug)]
pub struct UncaughtSignal {
    pub Tid: i32,
    pub Pid: i32,
    pub SignalNumber: i32,
    pub FaultAddr: u64,
}

#[derive(Clone, Debug)]
pub enum Event {
    UncaughtSignal(UncaughtSignal)
}

pub trait Emitter: Send + Sync {
    fn Uid(&self) -> u64;
    // Emit writes a single eventchannel message to an emitter. Emit should
    // return hangup = true to indicate an emitter has "hung up" and no further
    // messages should be directed to it.
    fn Emit(&mut self, event: &Event) -> (bool, Result<()>);

    // Close closes this emitter. Emit cannot be used after Close is called.
    fn Close(&mut self) -> Result<Event>;
}

pub fn Emit(event: &Event) -> Result<()> {
    let mut errMsg = "".to_string();
    let mut removes = Vec::new();
    for (_, e) in &EMITTERS.lock().0 {
        let (hangup, err) = e.lock().Emit(event);
        match err {
            Err(e) => errMsg = errMsg + format!("error emitting {:?}: on {:?}", event, e).as_str(),
            Ok(()) => (),
        }

        if hangup {
            removes.push(e.lock().Uid())
        }
    }

    for id in removes {
        EMITTERS.lock().0.remove(&id);
    }

    return Err(Error::Common(errMsg))
}

pub fn AddEmiiter(e: &Arc<Mutex<Emitter>>) {
    let id = e.lock().Uid();
    EMITTERS.lock().0.insert(id, e.clone());
}