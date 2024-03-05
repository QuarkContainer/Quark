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


#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(deprecated)]

#[macro_use]
extern crate log;

extern crate simple_logging;

#[macro_use]
extern crate scopeguard;

pub mod qobject;
pub mod actor;
pub mod actor_system;
pub mod http_actor;

use std::thread;

use actor_system::ACTOR_SYSTEM;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qobject::*;
use qshare::qactor::TellReq;

#[pyfunction]
fn send(_target: String, _args: QObjectList) -> PyResult<()> {
    //Ok((a + b).to_string())
    return Ok(())
}

#[pyclass]
pub struct RustStruct {
    #[pyo3(get, set)]
    pub data: String,
    #[pyo3(get, set)]
    pub vector: Vec<u8>,
}

#[pyclass]
#[derive(Debug, Default)]
pub struct Tell {
    #[pyo3(get, set)]
    pub actor_id: String,
    #[pyo3(get, set)]
    pub func: String,
    #[pyo3(get, set)]
    pub req_id: u64,
    #[pyo3(get, set)]
    pub data: Vec<u8>
}

impl From<TellReq> for Tell {
    fn from(msg: TellReq) -> Self {
        let ret = Tell {
            actor_id: msg.actor_id.clone(),
            func: msg.func.clone(),
            req_id: msg.req_id,
            data: msg.data,
        };

        return ret
    }
}

impl Into<TellReq> for Tell {
    fn into(self) -> TellReq {
        return TellReq {
            actor_id: self.actor_id.clone(),
            func: self.func.clone(),
            req_id: self.req_id.clone(),
            // there is memory copy here,
            // todo: optimize later
            data: self.data
        }
    }
}

#[pymethods]
impl RustStruct {
    #[new]
    pub fn new(data: String, vector: Vec<u8>) -> RustStruct {
        RustStruct { data, vector }
    }
    pub fn printer(&self) {
        println!("{}", self.data);
        for i in &self.vector {
            println!("{}", i);
        }
    }
    pub fn extend_vector(&mut self, extension: Vec<u8>) {
        println!("{}", self.data);
        for i in extension {
            self.vector.push(i);
        }
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    defer!(error!("asdf"));
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn qactor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(depolyment, m)?)?;
    m.add_function(wrap_pyfunction!(recvfrom, m)?)?;
    m.add_function(wrap_pyfunction!(sendto, m)?)?;
    m.add_function(wrap_pyfunction!(new_http_actor, m)?)?;
    m.add_function(wrap_pyfunction!(new_py_actor, m)?)?;
    m.add_class::<RustStruct>()?;
    Ok(())
}

#[pyfunction]
fn depolyment(_py: Python) -> PyResult<()> {
    thread::spawn(move || {
        use tokio::runtime;
        let rt = runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
        rt.block_on(async move {
            let _ = ACTOR_SYSTEM.wait().await;
        });
    });
   
    return Ok(())
}

#[pyfunction]
fn recvfrom(_py: Python, actor_id: String) -> PyResult<Tell> {
    // println!("recvfrom ... {}", actor_id.clone());
    // let clone = actor_id.clone();
    // defer!(println!("recvfrom 2... {}", clone));
    // pyo3_asyncio::tokio::future_into_py(py, async {
    //     error!("recvfrom 3");
    //     let req: Tell = match ACTOR_SYSTEM.Recv(actor_id).await {
    //         Err(e) => return Err(e.into()),
    //         Ok(r) => r.into()
    //     };

    //     Ok(req)
    // })

    let req: Tell = match ACTOR_SYSTEM.Recv(actor_id) {
        Err(e) => return Err(e.into()),
        Ok(r) => r.into()
    };
    
    Ok(req)
}

#[pyfunction]
fn sendto(_py: Python, actorId: String, func: String, reqId: u64, data: Vec<u8>) -> PyResult<()> {
    let req = TellReq {
        actor_id: actorId,
        func: func,
        req_id: reqId,
        data: data 
    };
    match ACTOR_SYSTEM.Send(req) {
        Err(e) => return Err(e.into()),
        Ok(()) => return Ok(())
    }
}

#[pyfunction]
fn new_http_actor(
    _py: Python, 
    proxyActorId: &str, 
    gatewayActorId: &str, 
    gatewayFunc: &str, 
    httpPort: u16
) -> PyResult<()> {
    println!("new_http_actor ...");
    match ACTOR_SYSTEM.NewHttpProxyActor(proxyActorId, gatewayActorId, gatewayFunc, httpPort) {
        Err(e) => return Err(e.into()),
        Ok(()) => return Ok(())
    }
}

#[pyfunction]
fn new_py_actor(
    _py: Python, 
    id: &str, 
    modName: &str, 
    className: &str
) -> PyResult<()> {
    match ACTOR_SYSTEM.NewPyActor(id, modName, className) {
        Err(e) => return Err(e.into()),
        Ok(()) => return Ok(())
    }
}
