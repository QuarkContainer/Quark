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
// limitations under

use std::sync::{Arc, Mutex};
use core::ops::Deref;
//use tokio::sync::mpsc;
use std::sync::mpsc;

use qshare::qactor;

use crate::http_actor::HttpActor;

#[derive(Debug, Clone)]
pub enum Actor { 
    PyActor(PyActor),
    HttpActor(HttpActor),
}

impl Actor {
    pub fn Tell(&self, req: qactor::TellReq) {
        match self {
            Actor::PyActor(a) => a.Tell(req),
            Actor::HttpActor(a) => a.Tell(req),
        }
    }

    pub fn Recv(&self) -> Option<qactor::TellReq> {
        match self {
            Actor::PyActor(a) => return a.Recv(),
            Actor::HttpActor(_) => {
                unreachable!()
                //return a.Recv().await,
            }
        }
    }
}


#[derive(Debug)]
pub struct PyActorInner {
    pub id: String,
    pub modName: String,
    pub className: String,

    pub location: LocationId,
    pub inputTx: mpsc::Sender<qactor::TellReq>,
    pub inputRx: Mutex<mpsc::Receiver<qactor::TellReq>>,
}

#[derive(Debug, Clone)]
pub struct PyActor(Arc<PyActorInner>);

impl Deref for PyActor {
    type Target = Arc<PyActorInner>;

    fn deref(&self) -> &Arc<PyActorInner> {
        &self.0
    }
}

impl PyActor {
    pub fn New(id: &str, modName: &str, className: &str) -> Self {
        //let (tx, rx) = mpsc::channel::<qactor::TellReq>(30);
        let (tx, rx) = mpsc::channel();
        let inner = PyActorInner {
            id: id.to_owned(),
            modName: modName.to_owned(),
            className: className.to_owned(),
            location: LocationId::default(),
            inputRx: Mutex::new(rx),
            inputTx: tx,
        };

        return Self(Arc::new(inner))
    }

    pub fn Tell(&self, req: qactor::TellReq) {
        self.inputTx.send(req).unwrap();
    }

    pub fn Recv(&self) -> Option<qactor::TellReq> {
        match self.inputRx.lock().unwrap().recv() {
            Err(e) => {
                error!("PyActor::Recv get error {:?}", e);
                return None;
            }
            Ok(r) => {
                return Some(r)
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct LocationId {
    pub podIp: String,
    pub processId: u16,
    pub threadId: u16,
}
