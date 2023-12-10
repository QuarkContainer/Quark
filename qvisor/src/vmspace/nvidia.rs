// Copyright (c) 2021 Quark Container Authors
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
use spin::Mutex;
use core::ops::Deref;
use std::ffi::CString;

use crate::qlib::common::*;
use crate::qlib::linux_def::SysErr;
use crate::qlib::proxy::*;

lazy_static! {
    pub static ref NVIDIA_HANDLERS: NvidiaHandlers = NvidiaHandlers::New();
}

pub fn NvidiaProxy(cmd: ProxyCommand, parameters: &ProxyParameters) -> Result<i64> {
    let handler = NVIDIA_HANDLERS.GetFuncHandler(cmd)?;
    match cmd {
        ProxyCommand::None => {
            panic!("get impossible proxy command");
        }
        ProxyCommand::CudaSetDevice => {
            let func: extern "C" fn(libc::c_int) -> u32 = unsafe {
                std::mem::transmute(handler)
            }; 

            let ret = func(parameters.para1 as i32);
            return Ok(ret as i64);
        }
        ProxyCommand::CudaDeviceSynchronize => {
            let func: extern "C" fn() -> i32 = unsafe {
                std::mem::transmute(handler)
            }; 

            let ret = func();
            return Ok(ret as i64);
        }
        _ => todo!()
    }
}

pub struct NvidiaHandlersInner {
    pub cudaRuntimeHandler: u64, //*mut libc::c_void,
    pub handlers: BTreeMap<ProxyCommand, u64>,
}

impl NvidiaHandlersInner {
    pub fn GetFuncHandler(&mut self, cmd: ProxyCommand) -> Result<u64> {
        match self.handlers.get(&cmd) {
            None => {
                let func = self.DLSym(cmd)?;
                self.handlers.insert(cmd, func);
                return Ok(func);
            }
            Some(func) => {
                return Ok(*func);
            }
        }
    }

    pub fn DLSym(&self, cmd: ProxyCommand) -> Result<u64> {
        match cmd {
            ProxyCommand::CudaSetDevice => {
                let func_name = CString::new("cudaSetDevice").unwrap();
                let handler: extern "C" fn(libc::c_int) -> i32 = unsafe {
                    std::mem::transmute(libc::dlsym(self.cudaRuntimeHandler as *mut libc::c_void, func_name.as_ptr()))
                };
                if handler as u64 != 0 {
                    return Ok(handler as u64)
                }
            }
            _ => ()
        }

        return Err(Error::SysError(SysErr::ENOTSUP))
    }
}

pub struct NvidiaHandlers(Mutex<NvidiaHandlersInner>);

impl Deref for NvidiaHandlers {
    type Target = Mutex<NvidiaHandlersInner>;

    fn deref(&self) -> &Mutex<NvidiaHandlersInner> {
        &self.0
    }
}

impl NvidiaHandlers {
    pub fn New() -> Self {
        unsafe { cuda_driver_sys::cuInit(0) };

        let cudart = format!("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so");
        let cudartlib = CString::new(&*cudart).unwrap();
        let cudaRuntimeHandler = unsafe {
            libc::dlopen(
            cudartlib.as_ptr(), 
            libc::RTLD_LAZY
            )
        } as u64;

        let inner = NvidiaHandlersInner {
            cudaRuntimeHandler: cudaRuntimeHandler,
            handlers: BTreeMap::new(),
        };

        return Self(Mutex::new(inner));  
    }

    // trigger the NvidiaHandlers initialization
    pub fn Trigger(&self) {}

    pub fn GetFuncHandler(&self, cmd: ProxyCommand) -> Result<u64> {
        let mut inner = self.lock();
        return inner.GetFuncHandler(cmd);
    } 
}