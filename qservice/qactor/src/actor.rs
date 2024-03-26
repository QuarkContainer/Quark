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

use pyo3::prelude::*;

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
}


#[derive(Debug)]
pub struct PyActorInner {
    pub id: String,
    pub modName: String,
    pub className: String,

    pub location: LocationId,
    pub queue: Py<PyAny>
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
    pub fn New(id: &str, modName: &str, className: &str, queue: &PyAny) -> Self {
        let inner = PyActorInner {
            id: id.to_owned(),
            modName: modName.to_owned(),
            className: className.to_owned(),
            location: LocationId::default(),
            queue: queue.into()
        };

        return Self(Arc::new(inner))
    }
    

    pub fn Tell(&self, req: qactor::TellReq) {
        use pyo3::types::PyTuple;

        Python::with_gil(|py| {
            let data = (&req.func, &req.req_id, &req.data);
    
            let put : Py<PyAny> = self.queue.getattr(py, "put").unwrap().into();
            let args = PyTuple::new(py, &[data]);
            put.call1(py, args).unwrap();
        });
    }
}

#[derive(Debug, Default)]
pub struct LocationId {
    pub podIp: String,
    pub processId: u16,
    pub threadId: u16,
}
