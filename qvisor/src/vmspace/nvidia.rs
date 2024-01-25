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

use core::ops::Deref;
use spin::Mutex;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::os::raw::*;

use crate::qlib::common::*;
use crate::qlib::linux_def::SysErr;
use crate::qlib::proxy::*;
use crate::qlib::range::Range;
use crate::xpu::cuda::*;

use cuda_driver_sys::*;

//yiwang
use std::sync::Arc;

lazy_static! {
    pub static ref NVIDIA_HANDLERS: NvidiaHandlers = NvidiaHandlers::New();
    pub static ref FUNC_MAP: BTreeMap<ProxyCommand, (XpuLibrary, &'static str)> = BTreeMap::from([
        (
            ProxyCommand::CudaSetDevice,
            (XpuLibrary::CudaRuntime, "cudaSetDevice")
        ),
        (
            ProxyCommand::CudaDeviceSynchronize,
            (XpuLibrary::CudaRuntime, "cudaDeviceSynchronize")
        ),
        (
            ProxyCommand::CudaMalloc,
            (XpuLibrary::CudaRuntime, "cudaMalloc")
        ),
        (
            ProxyCommand::CudaMemcpy,
            (XpuLibrary::CudaRuntime, "cudaMemcpy")
        ),
        (
            ProxyCommand::CudaRegisterFatBinary,
            (XpuLibrary::CudaDriver, "cuModuleLoadData")
        ),
        (
            ProxyCommand::CudaRegisterFunction,
            (XpuLibrary::CudaDriver, "cuModuleGetFunction")
        ),
        (
            ProxyCommand::CudaLaunchKernel,
            (XpuLibrary::CudaDriver, "cuLaunchKernel")
        ),
    ]);
}

pub fn NvidiaProxy(cmd: ProxyCommand, parameters: &ProxyParameters) -> Result<i64> {
    error!("NvidiaProxy 0 cmd {:?}", cmd);
    let handler = NVIDIA_HANDLERS.GetFuncHandler(cmd)?;
    error!("NvidiaProxy 0 cmd {:?} After getting handler", cmd);
    match cmd {
        ProxyCommand::None => {
            panic!("get impossible proxy command");
        }
        ProxyCommand::CudaSetDevice => {
            error!("hochan CudaSetDevice 1");
            let func: extern "C" fn(libc::c_int) -> i32 = unsafe { std::mem::transmute(handler) };

            error!("hochan CudaSetDevice 2");
            let ret = func(parameters.para1 as i32);
            error!("hochan CudaSetDevice 3");
            return Ok(ret as i64);
        }
        ProxyCommand::CudaDeviceSynchronize => {
            let func: extern "C" fn() -> i32 = unsafe { std::mem::transmute(handler) };

            let ret = func();
            return Ok(ret as i64);
        }
        ProxyCommand::CudaMalloc => {
            let func: extern "C" fn(*mut *mut ::std::os::raw::c_void, usize) -> i32 =
                unsafe { std::mem::transmute(handler) };

            error!("hochan CudaMalloc before parameters:{:x?}", parameters);
            let mut para1 = parameters.para1 as *mut ::std::os::raw::c_void;
            let addr = &mut para1;

            error!(
                "hochan before cuda_runtime_sys::cudaMalloc addr {:x}",
                *addr as u64
            );
            let ret = func(addr, parameters.para2 as usize);
            error!(
                "hochan cuda_runtime_sys::cudaMalloc ret {:x?} addr {:x}",
                ret, *addr as u64
            );

            unsafe {
                *(parameters.para1 as *mut u64) = *addr as u64;
            }

            error!(
                "hochan CudaMalloc after parameters:{:x?} ret {:x?}",
                parameters, ret
            );
            return Ok(ret as i64);
        }
        ProxyCommand::CudaMemcpy => {
            return CudaMemcpy(handler, parameters);
        }
        ProxyCommand::CudaRegisterFatBinary => {
            let fatElfHeader = unsafe { &*(parameters.para2 as *const u8 as *const FatElfHeader) };
            let moduleKey = parameters.para3;
            error!("hochan moduleKey:{:x}", moduleKey);
            match GetFatbinInfo(parameters.para2, *fatElfHeader) {
                Ok(_) => {}
                Err(e) => {
                    return Err(e);
                }
            }

            let mut module: u64 = 0;
            let ret = unsafe {
                cuda_driver_sys::cuModuleLoadData(
                    &mut module as *mut _ as u64 as *mut CUmodule,
                    parameters.para2 as *const c_void,
                )
            };
            MODULES.lock().insert(moduleKey, module);
            error!(
                "hochan called func ret {:?} module ptr {:x?} MODULES {:x?}",
                ret,
                module,
                MODULES.lock()
            );
            return Ok(ret as i64);
        }
        ProxyCommand::CudaRegisterFunction => {
            let info = unsafe { &*(parameters.para1 as *const u8 as *const RegisterFunctionInfo) };

            let bytes = unsafe {
                std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize)
            };
            let deviceName = std::str::from_utf8(bytes).unwrap();
             // yiwang
            //  let mut module = *MODULES.lock().get(&info.fatCubinHandle).unwrap();

           
             let mut module = match MODULES.lock().get(&info.fatCubinHandle){
                 Some(module) => {
                     error!("yiwang module: {:x} for this fatCubinHandle:{} has been found", module, info.fatCubinHandle);
                     *module// as *const _ as u8 as u64
                 }
                 None => {
                     error!("yiwang no module found with this fatCubin"); 
                     0
                 }
             };
            error!(
                "hochan deviceName {}, parameters {:x?} module {:x}",
                deviceName, parameters, module
            );

            let mut hfunc: u64 = 0;
            let ret = unsafe {
                cuda_driver_sys::cuModuleGetFunction(
                    &mut hfunc as *mut _ as u64 as *mut CUfunction,
                    *(&mut module as *mut _ as u64 as *mut CUmodule),
                    CString::new(deviceName).unwrap().clone().as_ptr(),
                )
            };
            FUNCTIONS.lock().insert(info.hostFun, hfunc);
            error!(
                "hochan cuModuleGetFunction ret {:x?}, hfunc {:x}, &hfunc {:x}, FUNCTIONS  {:x?}",
                ret,
                hfunc,
                &hfunc,
                FUNCTIONS.lock()
            );

            
            //  let kernelInfo = KERNEL_INFOS.lock().get(&deviceName.to_string()).unwrap().clone();
            let kernelInfo = match KERNEL_INFOS.lock().get(&deviceName.to_string()) {
                Some(kernelInformations) => {
                    error!("yiwang found kernel {:?}", kernelInformations);
                    kernelInformations.clone()
                }
                None => {
                    error!(
                        "yiwang No kernel infos found with this deviceName : {}",
                        deviceName
                    );
                    Arc::new(KernelInfo::default())
                }
            };

            let paramInfo = parameters.para3 as *const u8 as *mut ParamInfo;
            unsafe {
                (*paramInfo).paramNum = kernelInfo.paramNum;
                for i in 0..(*paramInfo).paramNum {
                    (*paramInfo).paramSizes[i] = kernelInfo.paramSizes[i];
                }
                error!("hochan paramInfo in nvidia {:x?}", (*paramInfo));
            }
            return Ok(ret as i64);
        }
        ProxyCommand::CudaLaunchKernel => {
            error!(
                "hochan CudaLaunchKernel in host parameters {:x?}",
                parameters
            );
            let info = unsafe { &*(parameters.para1 as *const u8 as *const LaunchKernelInfo) };
            // let func = FUNCTIONS.lock().get(&info.func).unwrap().clone();
            let func =  match FUNCTIONS.lock().get(&info.func){
                Some(func) => {
                    error!("yiwang func has been found: {:x}",func);
                    func.clone()
                }
                None => {
                    error!("yiwang no CUfunction has been found");
                    0 
                }
            };
            error!(
                "hochan CudaLaunchKernel in host info {:x?}, func {:x}",
                info, func
            );

            let ret = unsafe {
                cuda_driver_sys::cuLaunchKernel(
                    func as CUfunction,
                    info.gridDim.x,
                    info.gridDim.y,
                    info.gridDim.z,
                    info.blockDim.x,
                    info.blockDim.y,
                    info.blockDim.z,
                    info.sharedMem as u32,
                    info.stream as *mut CUstream_st,
                    info.args as *mut *mut ::std::os::raw::c_void,
                    0 as *mut *mut ::std::os::raw::c_void,
                )
            };
            error!("hochan cuLaunchKernel ret {:x?}", ret);

            return Ok(0 as i64);
        }
        _ => todo!(),
    }
}

pub fn CudaMemcpy(handle: u64, parameters: &ProxyParameters) -> Result<i64> {
    let func: extern "C" fn(u64, u64, u64, u64) -> u32 = unsafe { std::mem::transmute(handle) };

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
                        dst + offset,
                        r.start,
                        r.len,
                        kind,
                        ret
                    );
                    return Ok(ret as i64);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0);
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

            return Ok(0);
        }
        CUDA_MEMCPY_DEVICE_TO_DEVICE => {
            let dst = parameters.para1;
            let src = parameters.para3;
            let count = parameters.para4;
            let ret = func(dst, src, count, kind);
            return Ok(ret as i64);
        }
        _ => todo!(),
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
                //yw
                // let handler = XPU_LIBRARY_HANDLERS.lock().get(&pair.0).unwrap().clone();
                let handler = match XPU_LIBRARY_HANDLERS.lock().get(&pair.0) {
                    Some(functionHandler) => {
                        error!("hochan function handler got {:?}", functionHandler);
                        functionHandler.clone()
                    }
                    None => {
                        error!("hochan no function handler found");
                        0
                    }
                };

                let handler: u64 = unsafe {
                    std::mem::transmute(libc::dlsym(
                        handler as *mut libc::c_void,
                        func_name.as_ptr(),
                    ))
                };

                if handler != 0 {
                    error!("hochan got handler {:x}", handler);
                    return Ok(handler as u64);
                }
            }
            None => (),
        }

        return Err(Error::SysError(SysErr::ENOTSUP));
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
        // This code piece is necessary. Otherwise cuModuleLoadData will return CUDA_ERROR_JIT_COMPILER_NOT_FOUND
        let lib = CString::new("libnvidia-ptxjitcompiler.so").unwrap();
        let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };
        error!("hochan libnvidia-ptxjitcompiler.so handle {:?}", handle);

        let initResult = unsafe { cuda_driver_sys::cuInit(0) };
        error!("hochan initResult {:?}", initResult);

        let mut ctx: MaybeUninit<CUcontext> = MaybeUninit::uninit();
        let ptr_ctx = ctx.as_mut_ptr();
        let ret = unsafe { cuda_driver_sys::cuCtxCreate_v2(ptr_ctx, 0, 0) };
        error!("hochan cuCtxCreate ret {:?}", ret);

        let ret = unsafe { cuCtxPushCurrent_v2(*ptr_ctx) };
        error!("hochan cuCtxPushCurrent ret {:?}", ret);

        let cuda = format!("/usr/lib/x86_64-linux-gnu/libcuda.so");
        let cudalib = CString::new(&*cuda).unwrap();
        let cudaHandler = unsafe { libc::dlopen(cudalib.as_ptr(), libc::RTLD_LAZY) } as u64;
        assert!(cudaHandler != 0, "can't open libcuda.so");
        XPU_LIBRARY_HANDLERS
            .lock()
            .insert(XpuLibrary::CudaDriver, cudaHandler);

        let handlers = BTreeMap::new();

        let cudart = format!("/usr/lib/x86_64-linux-gnu/libcudart.so");
        let cudartlib = CString::new(&*cudart).unwrap();
        let cudaRuntimeHandler =
            unsafe { libc::dlopen(cudartlib.as_ptr(), libc::RTLD_LAZY) } as u64;

        assert!(
            cudaRuntimeHandler != 0,
            "/usr/lib/x86_64-linux-gnu/libcudart.so"
        );
        XPU_LIBRARY_HANDLERS
            .lock()
            .insert(XpuLibrary::CudaRuntime, cudaRuntimeHandler);

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
