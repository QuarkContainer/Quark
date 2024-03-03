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

use std::sync::Arc;
use core::ops::Deref;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TMutex;

use qshare::qactor;

use crate::gateway_actor::GatewayActor;

#[derive(Debug, Clone)]
pub enum Actor {
    PyActor(PyActor),
    GatewayActor(GatewayActor),
}

impl Actor {
    pub fn Tell(&self, req: qactor::TellReq) {
        match self {
            Actor::PyActor(a) => a.Tell(req),
            Actor::GatewayActor(a) => a.Tell(req),
        }
    }

    pub async fn Recv(&self) -> Option<qactor::TellReq> {
        match self {
            Actor::PyActor(a) => return a.Recv().await,
            Actor::GatewayActor(a) => return a.Recv().await,
        }
    }
}


#[derive(Debug)]
pub struct PyActorInner {
    pub name: String,
    pub modName: String,
    pub className: String,

    pub location: LocationId,
    pub inputTx: mpsc::Sender<qactor::TellReq>,
    pub inputRx: TMutex<mpsc::Receiver<qactor::TellReq>>,
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
    pub fn Tell(&self, req: qactor::TellReq) {
        self.inputTx.try_send(req).unwrap();
    }

    pub async fn Recv(&self) -> Option<qactor::TellReq> {
        let mut rx = self.inputRx.lock().await;
        let req = rx.recv().await;
        return req;
    }
}

#[derive(Debug)]
pub struct LocationId {
    pub podIp: String,
    pub processId: u16,
    pub threadId: u16,
}
