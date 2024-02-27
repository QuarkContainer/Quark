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
use std::sync::Arc;

use crate::qlib::common::*;
use crate::qlib::linux_def::SysErr;
use crate::qlib::proxy::*;
use crate::qlib::range::Range;
use crate::xpu::cuda::*;

use cuda_driver_sys::*;
use cuda_runtime_sys::{cudaDeviceP2PAttr, cudaStreamCaptureStatus, cudaStream_t};  
use cuda_runtime_sys::{cudaDeviceProp, cudaDeviceAttr,cudaFuncCache,cudaLimit,cudaSharedMemConfig};


lazy_static! {
    pub static ref NVIDIA_HANDLERS: NvidiaHandlers = NvidiaHandlers::New();
    pub static ref FUNC_MAP: BTreeMap<ProxyCommand, (XpuLibrary, &'static str)> = BTreeMap::from(
        [
        (ProxyCommand::CudaChooseDevice,(XpuLibrary::CudaRuntime, "cudaChooseDevice")),
        (ProxyCommand::CudaDeviceGetAttribute,(XpuLibrary::CudaRuntime, "cudaDeviceGetAttribute")),
        (ProxyCommand::CudaDeviceGetByPCIBusId,(XpuLibrary::CudaRuntime, "cudaDeviceGetByPCIBusId")),
        (ProxyCommand::CudaDeviceGetCacheConfig,(XpuLibrary::CudaRuntime, "cudaDeviceGetCacheConfig")),
        (ProxyCommand::CudaDeviceGetLimit,(XpuLibrary::CudaRuntime, "cudaDeviceGetLimit")),
        (ProxyCommand::CudaDeviceGetP2PAttribute,(XpuLibrary::CudaRuntime, "cudaDeviceGetP2PAttribute")),
        (ProxyCommand::CudaDeviceGetPCIBusId, (XpuLibrary::CudaRuntime, "cudaDeviceGetPCIBusId")),
        (ProxyCommand::CudaDeviceGetSharedMemConfig, (XpuLibrary::CudaRuntime, "cudaDeviceGetSharedMemConfig")),
        (ProxyCommand::CudaDeviceGetStreamPriorityRange,(XpuLibrary::CudaRuntime, "cudaDeviceGetStreamPriorityRange")),
        (ProxyCommand::CudaDeviceReset,(XpuLibrary::CudaRuntime, "cudaDeviceReset")),
        (ProxyCommand::CudaDeviceSetCacheConfig, (XpuLibrary::CudaRuntime, "cudaDeviceSetCacheConfig")),
        (ProxyCommand::CudaSetDevice,(XpuLibrary::CudaRuntime, "cudaSetDevice")),
        (ProxyCommand::CudaSetDeviceFlags,(XpuLibrary::CudaRuntime, "cudaSetDeviceFlags")),
        (ProxyCommand::CudaDeviceSynchronize,(XpuLibrary::CudaRuntime, "cudaDeviceSynchronize")),
        
        (ProxyCommand::CudaGetDeviceCount,(XpuLibrary::CudaRuntime, "cudaGetDeviceCount")),
       

        (ProxyCommand::CudaGetDeviceProperties,(XpuLibrary::CudaRuntime, "cudaGetDeviceProperties")),

        (ProxyCommand::CudaMalloc,(XpuLibrary::CudaRuntime, "cudaMalloc")),
        (ProxyCommand::CudaMemcpy,(XpuLibrary::CudaRuntime, "cudaMemcpy")),
        (ProxyCommand::CudaMemcpyAsync,(XpuLibrary::CudaRuntime,"cudaMemcpyAsync")),
        (ProxyCommand::CudaRegisterFatBinary,(XpuLibrary::CudaDriver, "cuModuleLoadData")),
        (ProxyCommand::CudaRegisterFunction,(XpuLibrary::CudaDriver, "cuModuleGetFunction")),
        (ProxyCommand::CudaRegisterVar,(XpuLibrary::CudaDriver, "cuModuleGetGlobal")),
        (ProxyCommand::CudaLaunchKernel,(XpuLibrary::CudaDriver, "cuLaunchKernel")),
        (ProxyCommand::CudaFree,(XpuLibrary::CudaRuntime,"cudaFree")),
        (ProxyCommand::CudaUnregisterFatBinary,(XpuLibrary::CudaDriver,"cuModuleUnload")),

        (ProxyCommand::CudaStreamSynchronize,(XpuLibrary::CudaRuntime,"cudaStreamSynchronize")),
        (ProxyCommand::CudaStreamCreate,(XpuLibrary::CudaRuntime,"cudaStreamCreate")),
        (ProxyCommand::CudaStreamDestroy,(XpuLibrary::CudaRuntime,"cudaStreamDestroy")),
        (ProxyCommand::CudaStreamIsCapturing,(XpuLibrary::CudaRuntime,"cudaStreamIsCapturing")),  
        (ProxyCommand::CuModuleGetLoadingMode,(XpuLibrary::CudaDriver,"cuModuleGetLoadingMode")), 

        (ProxyCommand::CudaGetLastError,(XpuLibrary::CudaRuntime,"cudaGetLastError")),

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
        ProxyCommand::CudaChooseDevice => {

            let mut device: ::std::os::raw::c_int = Default::default() ; 

            let deviceProp= unsafe { *(parameters.para2 as *const u8 as *const cudaDeviceProp)};

            error!("yiwang deviceProp is(
                deviceProp.luidDeviceNodeMask: {:x},
                deviceProp.totalGlobalMem: {:x},
                deviceProp.sharedMemPerBlock: {:x}
                deviceProp.regsPerBlock: {:x}
            )...",
            deviceProp.luidDeviceNodeMask,
            deviceProp.totalGlobalMem,
            deviceProp.sharedMemPerBlock,
            deviceProp.regsPerBlock,
             );

            let func: extern "C" fn(*mut ::std::os::raw::c_int, *const cudaDeviceProp) -> i32 = 
                unsafe { std::mem::transmute(handler)};

            let ret = func(&mut device, &deviceProp);

            error!("yiwang device after function call is :{}", device);

            error!("yiwang result is: {:?}, {}", ret, ret);

            unsafe {
                *(parameters.para1 as *mut i32 ) = device
            };

            return Ok(ret as i64);
        }
        ProxyCommand::CudaDeviceGetAttribute => {
            let attribute: cudaDeviceAttr;

            // beacuse cannot cast a primitive type to a struct 
            unsafe{
                attribute = *(&parameters.para2 as *const _ as u64 as *mut cudaDeviceAttr);
            }

            let mut value: c_int = 0; 

            let device: c_int = parameters.para3 as c_int; 

            let func: extern "C" fn(*mut ::std::os::raw::c_int, cudaDeviceAttr, ::std::os::raw::c_int) -> i32 = 
            unsafe { std::mem::transmute(handler)};
            

            let ret = func(&mut value, attribute, device);

            error!("yiwang value after function call is :{:x}", value);
            error!("yiwang cudaDeviceGetAttribute result is: {:?}, {}", ret,ret);
            
            unsafe{
            *(parameters.para1 as *mut i32) = value as i32
            };

            return Ok(ret as i64);

        }
        ProxyCommand::CudaDeviceGetByPCIBusId => {
            let func: extern "C" fn(*mut ::std::os::raw::c_int, *const ::std::os::raw::c_char) -> i32 = 
            unsafe { std::mem::transmute(handler)};

            let mut device: c_int = 0;

            let bytes = unsafe {
                std::slice::from_raw_parts(parameters.para2 as *const u8, parameters.para3 as usize)
            };

            let PCIBusId = std::str::from_utf8(bytes).unwrap();

            // Store the CString in a variable to extend its lifetime
            let cstring = CString::new(PCIBusId).unwrap();

            let ret = func(&mut device, cstring.as_ptr());
            error!("the cstring: {:?}",cstring);

            error!("yiwang device after function call is :{:x}", device);
            error!("yiwang cudaDeviceGetByPCIBusId result is: {:?}, {}", ret,ret);

            unsafe{
                *(parameters.para1 as *mut i32) = device as i32
                };

            return Ok(ret as i64);

        }
        ProxyCommand::CudaDeviceGetCacheConfig => {
            
            let func: extern "C" fn(*mut cudaFuncCache) -> i32 = unsafe{std::mem::transmute(handler)};

            let mut cacheConfig:cudaFuncCache;
            unsafe {
                cacheConfig = *(parameters.para1 as *mut _);        
            }
            
            let ret = func(&mut cacheConfig);

            error!("ret is :{}, {:?}",ret, ret);
            error!("now cacheConfig should change: {}, {:?}",cacheConfig as u32,cacheConfig);

            unsafe{
            *(parameters.para1 as *mut _) = cacheConfig as u32;
            }

            return Ok(ret as i64);

        }
        ProxyCommand::CudaDeviceGetLimit => {
            let limitType: cudaLimit;
            unsafe{
                limitType = *(&parameters.para2 as *const _ as u64 as *mut cudaLimit);
            }

            let mut limit: usize = 0;

            let func: extern "C" fn(*mut usize, cudaLimit) -> i32 = unsafe { std::mem::transmute(handler)};

            let ret = func(&mut limit, limitType);

            error!("yiwang limit after function call is :{:x}", limit);
            error!("yiwang cudaDeviceGetAttribute result is: {:?}, {}", ret,ret);

            unsafe{
                *(parameters.para1 as *mut _ ) = limit
                 };

            return Ok(ret as i64);
        }
        ProxyCommand::CudaDeviceGetP2PAttribute => {
            let mut value: libc::c_int = 0; // _host__​cudaError_t cudaDeviceGetP2PAttribute ( int* value, cudaDeviceP2PAttr attr, int  srcDevice, int  dstDevice )
            let attribute = unsafe { *(&parameters.para2 as *const _ as u64 as *mut cudaDeviceP2PAttr) };

            let func: extern "C" fn(*mut ::std::os::raw::c_int, cudaDeviceP2PAttr, ::std::os::raw::c_int, ::std::os::raw::c_int) -> i32 = unsafe { std::mem::transmute(handler)};
            let ret = func(&mut value, attribute, parameters.para3 as c_int, parameters.para4 as c_int);

            error!("yiwang value after function call is :{:x}", value);
            error!("yiwang cudaDeviceGetP2PAttribute result is: {:?}, {}", ret,ret);

            unsafe{
                *(parameters.para1 as *mut _ ) = value  as i32
                 };

            return Ok(ret as i64);  
        }
        ProxyCommand::CudaDeviceGetPCIBusId => {
 
            let func: extern "C" fn(*mut libc::c_char, ::std::os::raw::c_int, ::std::os::raw::c_int,) -> i32 = 
            unsafe { std::mem::transmute(handler)};
            
            let  mut pciBusId: Vec<c_char> = vec![0; parameters.para2 as usize];
           
            // Convert the Rust string to a C string
            // let mut c_str = CString::new(pciBusId.clone()).expect("Failed to create CString");
            // Access the raw pointer to the C string
            // let c_str_ptr = c_str.as_ptr() as *mut c_char;
            // let c_str_ptr = c_str.into_raw();

            let ret = func(pciBusId.as_mut_ptr(), parameters.para2 as ::std::os::raw::c_int, parameters.para3 as ::std::os::raw::c_int);

            // let modifiedPciBusId = c_str.to_string_lossy().into_owned();
            // let modifiedPciBusId = unsafe { CString::from_raw(c_str_ptr) }
            // .into_string()
            // .expect("Failed to convert CString to String");
            // let PCIBusId = std::str::from_utf8(pciBusId).unwrap();

            error!("yiwang PciBusId after C function call: {:#?}", pciBusId);

            error!("yiwang CudaDeviceGetPCIBusId result is: {:?}, {}", ret,ret);

            let pciBusString = String::from_utf8(pciBusId.iter().map(|&c| c as u8).collect()).unwrap();

            unsafe {
                // (*(parameters.para1 as *mut String)).clear();
                // (*(parameters.para1 as *mut String)).push_str(&pciBusString);
                *(parameters.para1 as *mut String) = pciBusString.to_owned();
            }

            return Ok(ret as i64);

        }
        ProxyCommand::CudaDeviceGetSharedMemConfig => {
            let func: extern "C" fn(*mut cudaSharedMemConfig) -> i32 = unsafe { std::mem::transmute(handler) };
            let mut sharedMemConfig: cudaSharedMemConfig = unsafe { *(parameters.para1 as *mut _)};

            let ret = func(&mut sharedMemConfig);

            error!("yiwang ret is :{}, {:#?}",ret, ret);
            error!("sharedMemConfig after function call is: {}, {:?}",sharedMemConfig as u32, sharedMemConfig);
            
            unsafe { *(parameters.para1 as *mut _) = sharedMemConfig as u32};

            return Ok(ret as i64);

        }
        ProxyCommand::CudaDeviceGetStreamPriorityRange => {
            let mut leastPriority: ::std::os::raw::c_int;
            let mut greatestPriority: ::std::os::raw::c_int;

            unsafe{
                leastPriority = *(parameters.para1 as *mut _); 
                greatestPriority = *(parameters.para2 as *mut _) ;
            } 

            error!("yiwang leastPriority before function call is: {}",leastPriority);
            error!("yiwang greatestPriority before function call is: {}",greatestPriority);

            let func: extern "C" fn(*mut ::std::os::raw::c_int, *mut ::std::os::raw::c_int) -> i32 = 
                unsafe { std::mem::transmute(handler)};

            let ret = func(&mut leastPriority,&mut greatestPriority);

            error!("yiwang leastPriority after function call is :{}", leastPriority);
            error!("yiwang greatestPriority after function call is:{}",greatestPriority);

            error!("yiwang result is: {:?}, {}", ret, ret);
            
            unsafe{
                *(parameters.para1 as *mut _) = leastPriority;
                *(parameters.para2 as *mut _) = greatestPriority;
            }

            return Ok(ret as i64);

        }
        ProxyCommand::CudaDeviceSetCacheConfig => {
            let func: extern "C" fn (cudaFuncCache) -> i32 = unsafe { std::mem::transmute(handler)};

            let cacheConfig  = unsafe { std::mem::transmute(parameters.para1 as u32) };
            let ret = func(cacheConfig);

            error!("result is {}, {:?}", ret, ret);

            return Ok(ret as i64);

        }
        ProxyCommand::CudaSetDevice => {
            error!("CudaSetDevice 1");
            let func: extern "C" fn(libc::c_int) -> i32 = unsafe { std::mem::transmute(handler) };

            error!("CudaSetDevice 2");
            let ret = func(parameters.para1 as i32);
            error!("CudaSetDevice 3");
            return Ok(ret as i64);
        }
        ProxyCommand::CudaSetDeviceFlags => {
            error!("CudaSetDeviceFlags");
            let func: extern "C" fn(libc::c_uint) -> i32 = unsafe { std::mem::transmute(handler)};

            let ret = func(parameters.para1 as u32);
            error!("result is {}, {:?}", ret, ret);

            return Ok(ret as i64);
        } 
        ProxyCommand::CudaDeviceReset |
        ProxyCommand::CudaDeviceSynchronize => {
            let func: extern "C" fn() -> i32 = unsafe { std::mem::transmute(handler) };

            let ret = func();
            return Ok(ret as i64);
        }
        ProxyCommand::CudaGetDeviceCount => {
            let mut deviceCount: libc::c_int = 0; 

            let func: extern "C" fn(*mut ::std::os::raw::c_int) -> i32 = 
                unsafe { std::mem::transmute(handler)};

            let ret = func(&mut deviceCount);

            error!("yiwang device count after function call is :{:x}", deviceCount);
            error!("yiwang cudaGetDeviceCount result is: {:?}, {}", ret,ret);
            
            unsafe{
            *(parameters.para1 as *mut i32) = deviceCount as i32
            };

            return Ok(ret as i64);

        }
        ProxyCommand::CudaGetDeviceProperties => {
            let device = parameters.para2; 

            let mut deviceProp: cudaDeviceProp = Default::default();

            let func: extern "C" fn(*mut cudaDeviceProp, ::std::os::raw::c_int) -> i32 = 
                unsafe { std::mem::transmute(handler)};
            
            let ret = func(&mut deviceProp, device as ::std::os::raw::c_int);

            error!("yiwang deviceProp after function call, deviceProp.name:{:?}, deviceProp.uuid:{:?}, deviceProp.luid:{:?}", deviceProp.name, deviceProp.uuid, deviceProp.luid);
            error!("yiwang result is {}, {:?}",ret,ret);


            unsafe{
                *(parameters.para1 as * mut _ ) = deviceProp
            };

            return Ok(ret as i64);

        }
        ProxyCommand::CudaMalloc => {
            let func: extern "C" fn(*mut *mut ::std::os::raw::c_void, usize) -> i32 =
                unsafe { std::mem::transmute(handler) };

            error!("CudaMalloc before parameters:{:x?}", parameters);
            let mut para1 = parameters.para1 as *mut ::std::os::raw::c_void;
            let addr = &mut para1;

            error!(
                 "before cuda_runtime_sys::cudaMalloc addr {:x}",
                 *addr as u64
             );
            let ret = func(addr, parameters.para2 as usize);
            error!( 
                 "cuda_runtime_sys::cudaMalloc ret {:x?} addr {:x}",
                 ret, *addr as u64
             );

            unsafe {
                *(parameters.para1 as *mut u64) = *addr as u64;
            }

             error!(
                 "CudaMalloc after parameters:{:x?} ret {:x?}",
                 parameters, ret
             );
            return Ok(ret as i64);
        }

        ProxyCommand::CudaFree => {
            let func: extern "C" fn(*mut ::std::os::raw::c_void) -> i32 =
                unsafe {std::mem::transmute(handler)};
                  
            let ret = func( parameters.para1 as *mut ::std::os::raw::c_void);
            error!("cuda free memory return value: {} at location: {:x}", ret, parameters.para1);

            return Ok(ret as i64);

        }
        ProxyCommand::CudaMemcpy => {
            return CudaMemcpy(handler, parameters);
        }
        ProxyCommand::CudaMemcpyAsync => {
            return CudaMemcpyAsync(handler, parameters);
        }
        ProxyCommand::CudaRegisterFatBinary => {
            // reference to a FatElfHeader based on the memory location described by para2
            let fatElfHeader = unsafe { &*(parameters.para2 as *const u8 as *const FatElfHeader) };
            let moduleKey = parameters.para3;
            error!("moduleKey:{:x}", moduleKey);

            match GetFatbinInfo(parameters.para2, fatElfHeader) {
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
                 "called func ret {:?} module ptr {:x?} MODULES {:x?}",
                 ret,
                 module,
                 MODULES.lock()
             );
            return Ok(ret as i64);
        }
        ProxyCommand::CudaUnregisterFatBinary => {
            let moduleKey = parameters.para1;
            error!("yiwang unregister module key:{:x}", moduleKey);

            let module = match MODULES.lock().get(&moduleKey){
                Some(module) => {
                    error!("yiwang module: {:x} for this module key: {:x} has been found",module, moduleKey);
                    *module 
                }
                None => {
                    error!("yiwang no module be found with this fatcubinHandle:{:x}",moduleKey);
                    0
                }
            };
            let ret = unsafe{
                cuda_driver_sys::cuModuleUnload((module as *const u64) as  CUmodule) 
            };
            error!("ret: {:?}", ret);
            // delete the module 
            MODULES.lock().remove(&moduleKey);
            error!("modules treepmap should be empty:{:x?}",MODULES.lock());
            //TODO: may need to remove the FUNCTIONS key value pair 
            return Ok(ret as i64);

        }
        ProxyCommand::CudaRegisterFunction => {
            let info = unsafe { &*(parameters.para1 as *const u8 as *const RegisterFunctionInfo) };

            let bytes = unsafe {
                std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize)
            };
            let deviceName = std::str::from_utf8(bytes).unwrap();
           
            let mut module = match MODULES.lock().get(&info.fatCubinHandle){
                 Some(module) => {
                     error!("module: {:x} for this fatCubinHandle:{} has been found", module, info.fatCubinHandle);
                     *module
                 }
                 None => {
                     error!("no module found with this fatCubin"); 
                     0
                 }
             };
             error!(
                "deviceName {}, parameters {:x?} module {:x}",
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
                 "cuModuleGetFunction ret {:x?}, hfunc {:x}, &hfunc {:x}, FUNCTIONS  {:x?}",
                 ret,
                 hfunc,
                 &hfunc,
                 FUNCTIONS.lock()
             );

            let kernelInfo = match KERNEL_INFOS.lock().get(&deviceName.to_string()) {
                Some(kernelInformations) => {
                    error!("found kernel {:?}", kernelInformations);
                    kernelInformations.clone()
                }
                None => {
                    error!(
                         "No kernel infos found with this deviceName : {}",
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
                error!("paramInfo in nvidia {:x?}", (*paramInfo));
            }
            return Ok(ret as i64);
        }
        ProxyCommand::CudaRegisterVar  => {
            let info = unsafe { *(parameters.para1 as *const u8 as *const RegisterVarInfo)};

            
            let bytes = unsafe {
                std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize)
            };

            let deviceName = std::str::from_utf8(bytes).unwrap();

            let deviceName_cString = CString::new(deviceName).unwrap();

            let module = match MODULES.lock().get(&info.fatCubinHandle){
                Some(module) => {
                    error!("module: {:x} for this fatCubinHandle:{:x} has been found", module, info.fatCubinHandle);
                    *module
                }
                None => {
                    error!("no module found with this fatCubin"); 
                    0
                }
            };
            error!("deviceName {}, parameters {:x?} module {:x}", deviceName, parameters, module);

            let mut devicePtr: CUdeviceptr = 0; 
            let mut d_size: usize = 0 ; 
            let ret = unsafe {
                cuda_driver_sys::cuModuleGetGlobal_v2(
                    &mut devicePtr, 
                    &mut d_size, 
                    (module as *const u64) as  CUmodule, 
                    deviceName_cString.as_ptr())
            };

            GLOBALS.lock().insert(info.hostVar, devicePtr);
            error!(
                "cuModuleGetGlobal_v2 ret {:x?}, devicePtr {:x}, GLOBALS {:x?}",
                ret,
                devicePtr,
                GLOBALS.lock()
            );
         
            return Ok(ret as i64);
        }
        ProxyCommand::CudaLaunchKernel => {
            error!(
                "CudaLaunchKernel in host parameters {:x?}",
                 parameters
             );
            let info = unsafe { &*(parameters.para1 as *const u8 as *const LaunchKernelInfo) };
          
            let func =  match FUNCTIONS.lock().get(&info.func){
                Some(func) => {
                    error!("func has been found: {:x}",func);
                    func.clone()
                }
                None => {
                    error!("no CUfunction has been found");
                    0 
                }
            };
            error!(
                 "CudaLaunchKernel in host info {:x?}, func {:x}",
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
            error!("cuLaunchKernel ret {:x?}", ret);

            return Ok(ret as i64);
        }
        ProxyCommand::CudaStreamSynchronize => {
            error!("yiwang CudaStreamSynchronize stream parameter is:{:?}",parameters.para1);
            let func: extern "C" fn(cudaStream_t) -> i32 = unsafe {std::mem::transmute(handler)};
              
            let ret = func( parameters.para1 as cudaStream_t);

            error!("yiwang cudaStreamSynchronize result {:x?}", ret);

            return Ok(ret as i64);
        }
        ProxyCommand::CudaStreamCreate => {
                   
            error!("yiwang parameters para1 value(address of stream):{:x}",parameters.para1);
                 
            let func: extern "C" fn(*mut cudaStream_t) -> i32 = unsafe {std::mem::transmute(handler)};
            unsafe{
            let mut a: cudaStream_t = *(parameters.para1 as *mut u64) as cudaStream_t;
            
            error!("yiwang a value(content of stream):{:x}", a as u64);
           
            let ret = func(&mut a);
            //let ret = func(&mut a as * mut u64 as cudaStream_t)
            error!("yiwang a value(content of stream) after function call:{:x}", a as u64);
            error!("yiwang stream value:{:x}", *(parameters.para1 as *mut u64));
    
            // let ret = func(parameters.para1 as *mut u64 as *mut cudaStream_t);
            // error!("yiwang CudaStreamCreate pStream content is: {:?}",pStream);
            // error!("yiwang CudaStreamCreate a value after call:{:x}", a);
            // error!("yiwang CudaStreamCreate a address after call:{:p}", &a);
            error!("yiwang CudaStreamCreate result: {:x?}",ret);
    
            *(parameters.para1 as *mut u64) = a as u64;

            return Ok(ret as i64);}

        }
        ProxyCommand::CudaStreamDestroy => {
            error!("yiwang cudaStreamDestroy stream value to be destroy: {:x}", parameters.para1);
            let func: extern "C" fn(cudaStream_t) -> i32 = unsafe { std::mem::transmute(handler)};

            let ret = func(parameters.para1 as cudaStream_t);
            error!("yiwang CudaStreamDestory result: {:x?}", ret);

            return Ok(ret as i64);
        }
        ProxyCommand::CudaStreamIsCapturing => {
            error!("yiwang CudaStreamIsCapturing stream value is :{:x}", parameters.para1);
            let func: extern "C" fn(cudaStream_t, *mut cudaStreamCaptureStatus) -> i32 = unsafe{std::mem::transmute(handler)};

            let mut captureStatus:cudaStreamCaptureStatus;
            unsafe {
                captureStatus = *(parameters.para2 as *mut _);        
            }
            error!("captureStatus is: {}, {:?}", captureStatus as u32,captureStatus);

            let ret = func(parameters.para1 as cudaStream_t, &mut captureStatus);

            error!("ret is :{:x?}",ret);
            error!("now captureStatus should change: {}, {:?}",captureStatus as u32,captureStatus);

            unsafe{
            *(parameters.para2 as *mut _) = captureStatus as u32;
            }

            return Ok(ret as i64);
        }
        ProxyCommand::CuModuleGetLoadingMode => {
             let mut loadingMode:CUmoduleLoadingMode; 
             unsafe{
                loadingMode = *(parameters.para1 as *mut _ )
             };

             error!("before function call, loading mode is: {}, {:?}", loadingMode as i32, loadingMode );


             let func: extern "C" fn(*mut CUmoduleLoadingMode) -> i32  = unsafe{std::mem::transmute(handler)};

             let ret = func(&mut loadingMode);

             error!("ret is : {}, {:x?}",ret, ret);
             error!("loading mode should change: {}, {:?}",loadingMode as i32,loadingMode);

             return Ok(ret as i64);

            // let initResult = unsafe { cuda_driver_sys::cuInit(0) };
            // error!("initResult {:?}", initResult);

            // let func: extern "C" fn(*mut ::std::os::raw::c_void) -> i32 =
            //     unsafe {std::mem::transmute(handler)};
                  
            // let ret = func( parameters.para1 as *mut ::std::os::raw::c_void);
        
            // let mut mode: CUmoduleLoadingMode  = CUmoduleLoadingMode_enum::CU_MODULE_EAGER_LOADING;
            // let func: extern "C" fn(*mut CUmoduleLoadingMode) -> i32  = unsafe{std::mem::transmute(handler)};
            // let ret = func(&mut mode);

            // error!("ret is 3: {}, {:x?}",ret, ret);
            // error!("loading mode should change: {} {:?}, " ,mode as i32, mode );

            // return Ok(ret as i64);
         }
         ProxyCommand::CudaGetLastError => {
            let func: extern "C" fn() -> i32 = unsafe { std::mem::transmute(handler) };

            let ret = func();
            return Ok(ret as i64);
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

pub fn CudaMemcpyAsync(handle: u64, parameters: &ProxyParameters) -> Result<i64> {
    
    let func: extern "C" fn(u64, u64, u64, u64, cudaStream_t) -> i32 = unsafe { std::mem::transmute(handle) };
    let kind = parameters.para5;
    let stream = parameters.para6 as cudaStream_t;
    match kind{
        CUDA_MEMCPY_HOST_TO_HOST => todo!(),
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            let dst = parameters.para1;  // device 
            let ptr = parameters.para2 as *const Range;
            // length of the vector 
            let len = parameters.para3 as usize;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = func(dst + offset, r.start, r.len, kind, stream);
                if ret != 0 {
                    error!(
                        "CUDA_MEMCPY_ASYNC_HOST_TO_DEVICE ret is {:x}/{:x}/{:x}/{}/{}/{:?}",
                        dst + offset,
                        r.start,
                        r.len,
                        kind,
                        ret,
                        stream
                    );
                    return Ok(ret as i64);
                }
                offset += r.len;
            }
            
            assert!(offset == count);

            return Ok(0);
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            // dst is host(virtual address)
            let ptr = parameters.para1 as *const Range;
            let len = parameters.para2 as usize;

            let src = parameters.para3;       //device 
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = func(r.start, src + offset, r.len, kind,stream);
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
            let ret = func(dst, src, count, kind,stream);
            return Ok(ret as i64);
        }
        _ => todo!(),
    }
}

pub struct NvidiaHandlersInner {
    pub cudaHandler: u64,
    pub cudaRuntimeHandler: u64,
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
             
                let handler = match XPU_LIBRARY_HANDLERS.lock().get(&pair.0) {
                    Some(functionHandler) => {
                        error!("function handler got {:?}", functionHandler);
                        functionHandler.clone()
                    }
                    None => {
                        error!("no function handler found");
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
                    error!("got handler {:x}", handler);
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
        error!("libnvidia-ptxjitcompiler.so handle {:?}", handle);

        let initResult = unsafe { cuda_driver_sys::cuInit(0) };
        error!("initResult {:?}", initResult);

        let mut ctx: MaybeUninit<CUcontext> = MaybeUninit::uninit();
        let ptr_ctx = ctx.as_mut_ptr();
        let ret = unsafe { cuda_driver_sys::cuCtxCreate_v2(ptr_ctx, 0, 0) };
        error!("cuCtxCreate ret {:?}", ret);

        let ret = unsafe { cuCtxPushCurrent_v2(*ptr_ctx) };
        error!("cuCtxPushCurrent ret {:?}", ret);

        let cuda = format!("/usr/lib/x86_64-linux-gnu/libcuda.so");
        let cudalib = CString::new(&*cuda).unwrap();
        let cudaHandler = unsafe { libc::dlopen(cudalib.as_ptr(), libc::RTLD_LAZY) } as u64;
        assert!(cudaHandler != 0, "can't open libcuda.so");
        XPU_LIBRARY_HANDLERS
            .lock()
            .insert(XpuLibrary::CudaDriver, cudaHandler);

        let handlers = BTreeMap::new();

        // let cudart = format!("/usr/lib/x86_64-linux-gnu/libcudart.so");
        let cudart = format!("/usr/local/cuda/lib64/libcudart.so");
        
        let cudartlib = CString::new(&*cudart).unwrap();
        let cudaRuntimeHandler =
            unsafe { libc::dlopen(cudartlib.as_ptr(), libc::RTLD_LAZY) } as u64;

        // assert!(
        //     cudaRuntimeHandler != 0,
        //     "/usr/lib/x86_64-linux-gnu/libcudart.so"
        // );
        assert!(
            cudaRuntimeHandler != 0,
            "/usr/local/cuda/lib64/libcudart.so"
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
