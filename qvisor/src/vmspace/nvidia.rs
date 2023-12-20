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
use std::os::raw::*;

use crate::qlib::common::*;
use crate::qlib::linux_def::SysErr;
use crate::qlib::proxy::*;
use crate::qlib::range::Range;

use cuda_driver_sys::*;

lazy_static! {
    pub static ref NVIDIA_HANDLERS: NvidiaHandlers = NvidiaHandlers::New();
    pub static ref FUNC_MAP: BTreeMap<ProxyCommand,(u64, &'static str)> = BTreeMap::from(
        [
            (ProxyCommand::CudaSetDevice, (0, "cudaSetDevice")),
            (ProxyCommand::CudaDeviceSynchronize, (0, "cudaDeviceSynchronize")),
            (ProxyCommand::CudaMalloc, (0, "cudaMalloc")),
            (ProxyCommand::CudaMemcpy, (0, "cudaMemcpy")),
            (ProxyCommand::CudaRegisterFatBinary, (1, "cuModuleLoadData")),
        ]
    );
}

pub fn  NvidiaProxy(cmd: ProxyCommand, parameters: &ProxyParameters) -> Result<i64> {
    error!("NvidiaProxy 0 cmd {:?}", cmd);
    let handler = NVIDIA_HANDLERS.GetFuncHandler(cmd)?;
    error!("NvidiaProxy 0 cmd {:?} After getting handler", cmd);
    match cmd {
        ProxyCommand::None => {
            panic!("get impossible proxy command");
        }
        ProxyCommand::CudaSetDevice => {
            error!("hochan CudaSetDevice 1");
            let func: extern "C" fn(libc::c_int) -> i32 = unsafe {
                std::mem::transmute(handler)
            }; 

            error!("hochan CudaSetDevice 2");
            let ret = func(parameters.para1 as i32);
            error!("hochan CudaSetDevice 3");
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
        ProxyCommand::CudaRegisterFatBinary => {
            error!("hochan ProxyCommand::CudaRegisterFatBinary parameters: {:x?}", parameters);

            let addr=parameters.para2 as *const u8;
            let len = parameters.para1 as usize;
            let bytes:_= unsafe{            
                std::slice::from_raw_parts(addr, len)
            };
            error!("hochan !!!2 {:x?}", &bytes);

            let flags:*mut std::os::raw::c_uint= std::ptr::null_mut();
            let active:*mut std::os::raw::c_int= std::ptr::null_mut();
            let retContextState = unsafe{ cuda_driver_sys::cuDevicePrimaryCtxGetState(0,flags,active)};
            error!("hochan retContextState {:?}",retContextState);
            error!("hochan active {:?}",active);

            let pctx   :*mut CUcontext= std::ptr::null_mut();
            let retContextCreate = unsafe{ cuda_driver_sys::cuCtxCreate_v2(pctx, 0, 0)};
            error!("hochan retContextCreate {:?}",retContextCreate);

            let module:*mut CUmodule = std::ptr::null_mut();
            let p=std::ptr::addr_of!(parameters.para2);
            let ret = unsafe{ cuda_driver_sys::cuModuleLoadData(module, p as *const c_void)};

            // let func: extern "C" fn(*mut CUmodule, *const c_void) -> CUresult = unsafe {
            //     std::mem::transmute(handler)
            // };
            
            // let ret = func(module, p as *const c_void);
            error!("hochan called func ret {:?}",ret);
            return Ok(ret as i64);
            // Ok(0)
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
        CUDA_MEMCPY_DEVICE_TO_DEVICE => {
            let dst = parameters.para1;
            let src = parameters.para3;
            let count = parameters.para4;
            let ret = func(dst, src, count, kind);
            return Ok(ret as i64);
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
            Some(&pair) => {
                let func_name = CString::new(pair.1).unwrap();
                let handler = if pair.0 ==0 { self.cudaRuntimeHandler } else { self.cudaHandler };
                let handler: u64 = unsafe {
                    std::mem::transmute(libc::dlsym(handler as *mut libc::c_void, func_name.as_ptr()))
                };
                
                if handler != 0 {
                    error!("hochan got handler {:x}", handler);
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
        let initResult = unsafe { cuda_driver_sys::cuInit(0) };
        error!("hochan initResult {:?}", initResult);
        let cuda = format!("/usr/lib/x86_64-linux-gnu/libcuda.so");
        let cudalib = CString::new(&*cuda).unwrap();
        let cudaHandler = unsafe {
            libc::dlopen(
                cudalib.as_ptr(), 
            libc::RTLD_LAZY
            )
        } as u64;

        assert!(cudaHandler != 0, "can't open libcuda.so");

        let func_name = CString::new("cuInit").unwrap();
        let cuInitFunc: extern "C" fn(i32) -> i32 = unsafe {
            std::mem::transmute(libc::dlsym(cudaHandler as _, func_name.as_ptr()))
        };

        assert!(cuInitFunc as u64 != 0, "can't open func cuInit");
        
        let mut handlers = BTreeMap::new();
        handlers.insert(ProxyCommand::CuInit, cuInitFunc as u64);

        cuInitFunc(0);
        
        let cudart = format!("/usr/lib/x86_64-linux-gnu/libcudart.so");
        let cudartlib = CString::new(&*cudart).unwrap();
        let cudaRuntimeHandler = unsafe {
            libc::dlopen(
            cudartlib.as_ptr(), 
            libc::RTLD_LAZY
            )
        } as u64;

        assert!(cudaRuntimeHandler != 0, "/usr/lib/x86_64-linux-gnu/libcudart.so");

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