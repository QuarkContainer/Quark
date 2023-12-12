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
use crate::qlib::range::Range;

lazy_static! {
    pub static ref NVIDIA_HANDLERS: NvidiaHandlers = NvidiaHandlers::New();
    pub static ref FUNC_MAP: BTreeMap<ProxyCommand, &'static str> = BTreeMap::from(
        [
            (ProxyCommand::CudaSetDevice, "cudaSetDevice"),
            (ProxyCommand::CudaDeviceSynchronize, "cudaDeviceSynchronize"),
            (ProxyCommand::CudaMalloc, "cudaMalloc"),
            (ProxyCommand::CudaMemcpy, "cudaMemcpy"),
        ]
    );
}

pub fn  NvidiaProxy(cmd: ProxyCommand, parameters: &ProxyParameters) -> Result<i64> {
    let handler = NVIDIA_HANDLERS.GetFuncHandler(cmd)?;
    match cmd {
        ProxyCommand::None => {
            panic!("get impossible proxy command");
        }
        ProxyCommand::CudaSetDevice => {
            let func: extern "C" fn(libc::c_int) -> i32 = unsafe {
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
        ProxyCommand::CudaMalloc => {
            let func: extern "C" fn(u64, usize) -> i32 = unsafe {
                std::mem::transmute(handler)
            }; 

            let ret = func(parameters.para1, parameters.para2 as usize);
            return Ok(ret as i64);
        }
        ProxyCommand::CudaMemcpy => {
            return CudaMemcpy(handler, parameters);
        }
        _ => todo!()
    }
}

pub fn CudaMemcpy(handle: u64, parameters: &ProxyParameters) -> Result<i64> {
    let func: extern "C" fn(u64, u64, u64, u64) -> u32 = unsafe {
        std::mem::transmute(handle)
    };

    let kind = parameters.para5;
    
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => todo!(),
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            let dst = parameters.para1;
            let ptr = parameters.para2 as *const Range;
            let len = parameters.para3 as usize;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = func(dst + offset, r.start, r.len, kind);
                if ret != 0 {
                    error!(
                        "CUDA_MEMCPY_HOST_TO_DEVICE ret is {:x}/{:x}/{:x}/{} {}", 
                        dst+offset, r.start, r.len, kind, ret);
                    return Ok(ret as i64);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0)   
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            let ptr = parameters.para1 as *const Range;
            let len = parameters.para2 as usize;
            let src = parameters.para3;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = func(r.start, src + offset, r.len, kind);
                if ret != 0 {
                    return Ok(ret as i64);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0)   
        }
        _ => todo!()
    }
}

pub struct NvidiaHandlersInner {
    pub cudaHandler: u64,
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
        match FUNC_MAP.get(&cmd) {
            Some(&name) => {
                let func_name = CString::new(name).unwrap();
                let handler: u64 = unsafe {
                    std::mem::transmute(libc::dlsym(self.cudaRuntimeHandler as *mut libc::c_void, func_name.as_ptr()))
                };
                if handler != 0 {
                    return Ok(handler as u64)
                }
            }
            None => (),
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
        //unsafe { cuda_driver_sys::cuInit(0) };
        let cuda = format!("libcuda.so");
        let cudalib = CString::new(&*cuda).unwrap();
        let cudaHandler = unsafe {
            libc::dlopen(
                cudalib.as_ptr(), 
            libc::RTLD_LAZY
            )
        } as u64;

        let func_name = CString::new("cuInit").unwrap();
        let cuInitFunc: extern "C" fn(i32) -> i32 = unsafe {
            std::mem::transmute(libc::dlsym(cudaHandler as _, func_name.as_ptr()))
        };

        let mut handlers = BTreeMap::new();
        handlers.insert(ProxyCommand::CuInit, cuInitFunc as u64);

        cuInitFunc(0);
        
        let cudart = format!("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so");
        let cudartlib = CString::new(&*cudart).unwrap();
        let cudaRuntimeHandler = unsafe {
            libc::dlopen(
            cudartlib.as_ptr(), 
            libc::RTLD_LAZY
            )
        } as u64;

        let inner = NvidiaHandlersInner {
            cudaHandler: cudaHandler,
            cudaRuntimeHandler: cudaRuntimeHandler,
            handlers: handlers,
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