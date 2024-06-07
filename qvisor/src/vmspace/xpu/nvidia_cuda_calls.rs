
use std::sync:: Arc;
use crate::qlib::common::*;

use crate::qlib::config::*;
use crate::qlib::proxy::*;
use crate::qlib::range::Range;

use crate::{PMA_KEEPER, QUARK_CONFIG};

use cuda_driver_sys::{CUmodule,CUfunction,CUdeviceptr,CUresult,CUstream_st,CUfunction_attribute};


use super::super::{IoVec, MemoryDef};


use std::ffi::CString;
use std::os::raw::*;
use std::ptr::copy_nonoverlapping;
use std::slice;


use crate::qlib::proxy::ProxyParameters;
use crate::qlib::common::Result;

use crate::xpu::cuda_api::*;
use crate::xpu::cuda::*;
use crate::nvidia::{MEM_RECORDER, MEM_MANAGER};

use cuda_runtime_sys::{
    cudaDeviceAttr, cudaDeviceP2PAttr, cudaDeviceProp, cudaEvent_t, cudaFuncAttributes,
    cudaFuncCache, cudaLimit, cudaMemAttachGlobal, cudaSharedMemConfig, cudaStreamCaptureMode,
    cudaStreamCaptureStatus, cudaStream_t,
};


pub fn CudaChooseDevice(parameters: &ProxyParameters) -> Result<u32> {
    let mut device: c_int = Default::default();
    let deviceProp = unsafe { *(parameters.para2 as *const u8 as *const cudaDeviceProp) };

    let ret = unsafe { cudaChooseDevice(&mut device, &deviceProp) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaChooseDevice: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut i32) = device };
    Ok(ret as u32)
}

pub fn CudaDeviceGetAttribute(parameters: &ProxyParameters) -> Result<u32> {
    let attribute: cudaDeviceAttr =
        unsafe { *(&parameters.para2 as *const _ as u64 as *mut cudaDeviceAttr) };
    let mut value: c_int = 0;
    let device: c_int = parameters.para3 as c_int;

    let ret = unsafe { cudaDeviceGetAttribute(&mut value, attribute, device) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaDeviceGetAttribute: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut i32) = value as i32 };
    Ok(ret as u32)
}

pub fn CudaDeviceGetByPCIBusId(parameters: &ProxyParameters) -> Result<u32> {
    let mut device: c_int = 0;
    let bytes = unsafe {
        std::slice::from_raw_parts(parameters.para2 as *const u8, parameters.para3 as usize)
    };
    let PCIBusId = std::str::from_utf8(bytes).unwrap();
    let cstring = CString::new(PCIBusId).unwrap();

    let ret = unsafe { cudaDeviceGetByPCIBusId(&mut device, cstring.as_ptr()) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceGetByPCIBusId: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut i32) = device as i32 };
    Ok(ret as u32)
}

pub fn CudaDeviceGetCacheConfig(parameters: &ProxyParameters) -> Result<u32> {
    let mut cacheConfig: u32 = 0;

    let ret = unsafe { cudaDeviceGetCacheConfig(&mut cacheConfig as *mut _ as u64) };

    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceGetCacheConfig: {}",
            ret as u32
        );
    }

    unsafe {
        *(parameters.para1 as *mut u32) = cacheConfig;
    }
    Ok(ret as u32)
}

pub fn CudaDeviceGetLimit(parameters: &ProxyParameters) -> Result<u32> {
    let limitType: cudaLimit =
        unsafe { *(&parameters.para2 as *const _ as *mut cudaLimit) };
    let mut limit: usize = 0;

    let ret = unsafe { cudaDeviceGetLimit(&mut limit, limitType) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceGetLimit: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut _) = limit };
    Ok(ret as u32)
}

pub fn CudaDeviceGetP2PAttribute(parameters: &ProxyParameters) -> Result<u32> {
    let mut value: libc::c_int = 0;
    let attribute =
        unsafe { *(&parameters.para2 as *const _ as u64 as *mut cudaDeviceP2PAttr) };

    let ret = unsafe {
        cudaDeviceGetP2PAttribute(
            &mut value,
            attribute,
            parameters.para3 as c_int,
            parameters.para4 as c_int,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceGetP2PAttribute: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut _) = value as i32 };
    Ok(ret as u32)
}

pub fn CudaDeviceGetPCIBusId(parameters: &ProxyParameters) -> Result<u32> {
    let mut pciBusId: Vec<c_char> = vec![0; parameters.para2 as usize];

    let ret = unsafe {
        cudaDeviceGetPCIBusId(
            pciBusId.as_mut_ptr(),
            parameters.para2 as c_int,
            parameters.para3 as c_int,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceGetPCIBusId: {}",
            ret as u32
        );
    }

    let pciBusString =
        String::from_utf8(pciBusId.iter().map(|&c| c as u8).collect()).unwrap();
    unsafe { *(parameters.para1 as *mut String) = pciBusString.to_owned() };
    Ok(ret as u32)
}

pub fn CudaDeviceGetSharedMemConfig(parameters: &ProxyParameters) -> Result<u32> {
    let mut sharedMemConfig: u32 = 0;

    let ret = unsafe { cudaDeviceGetSharedMemConfig(&mut sharedMemConfig as *mut _ as u64) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceGetSharedMemConfig: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut u32) = sharedMemConfig as u32 };
    Ok(ret as u32)
}
pub fn CudaDeviceGetStreamPriorityRange(parameters: &ProxyParameters) -> Result<u32> {
    let mut leastPriority: c_int = unsafe { *(parameters.para1 as *mut _) };
    let mut greatestPriority: c_int = unsafe { *(parameters.para2 as *mut _) };

    let ret = unsafe {
        cudaDeviceGetStreamPriorityRange(&mut leastPriority, &mut greatestPriority)
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceGetStreamPriorityRange: {}",
            ret as u32
        );
    }

    unsafe {
        *(parameters.para1 as *mut _) = leastPriority;
        *(parameters.para2 as *mut _) = greatestPriority;
    }
    Ok(ret as u32)
}
pub fn CudaDeviceSetCacheConfig(parameters: &ProxyParameters) -> Result<u32> {
    let ret = unsafe { cudaDeviceSetCacheConfig(parameters.para1 as u32) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaDeviceSetCacheConfig: {}",
            ret as u32
        );
    }

    Ok(ret as u32)
}
pub fn CudaDeviceSetLimit(parameters: &ProxyParameters) -> Result<u32> {
    let ret = unsafe {
        cudaDeviceSetLimit(
            parameters.para1 as usize,
            parameters.para2 as usize,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceSetLimit: {}",
            ret as u32
        );
    }

    Ok(ret as u32)
}
pub fn CudaDeviceSetSharedMemConfig(parameters: &ProxyParameters) -> Result<u32> {
    let ret = unsafe {
        cudaDeviceSetSharedMemConfig(
            parameters.para1,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaDeviceSetSharedMemConfig: {}",
            ret as u32
        );
    }

    Ok(ret as u32)
}
pub fn CudaSetDevice(parameters: &ProxyParameters) -> Result<u32> {
    let ret = unsafe { cudaSetDevice(parameters.para1 as i32) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaSetDevice: {}", ret as u32);
    }

    Ok(ret as u32)
}
pub fn CudaSetDeviceFlags(parameters: &ProxyParameters) -> Result<u32> {
    let ret = unsafe { cudaSetDeviceFlags(parameters.para1 as u32) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaSetDeviceFlags: {}",
            ret as u32
        );
    }

    Ok(ret as u32)
}
pub fn CudaSetValidDevices(parameters: &ProxyParameters) -> Result<u32> {
    let ret = unsafe {
        cudaSetValidDevices(parameters.para1 as *mut c_int, parameters.para2 as i32)
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaSetValidDevices: {}",
            ret as u32
        );
    }

    Ok(ret as u32)
}

pub fn CudaDeviceReset() -> Result<u32> {
    let ret = unsafe { cudaDeviceReset() };
    if ret as u32 != 0 {
        error!("Nvidia.rs: error caused by CudaDeviceReset: {}", ret as u32);
    }

    Ok(ret as u32)
}
pub fn CudaDeviceSynchronize() -> Result<u32> {
    let ret = unsafe { cudaDeviceSynchronize() };
    if ret as u32 != 0 {
        error!(
            "Nvidia.rs: error caused by CudaDeviceSynchronize: {}",
            ret as u32
        );
    }

    Ok(ret as u32)
}

pub fn CudaGetDevice(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaGetDevice");
    let mut device: c_int = Default::default();

    let ret = unsafe { cudaGetDevice(&mut device) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaGetDevice: {}", ret as u32);
    }

    unsafe { *(parameters.para1 as *mut i32) = device };
    return Ok(ret as u32);
}
pub fn CudaGetDeviceCount(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaGetDeviceCount");
    let mut deviceCount: c_int = 0;

    let ret = unsafe { cudaGetDeviceCount(&mut deviceCount) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaGetDeviceCount: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut i32) = deviceCount as i32 };
    return Ok(ret as u32);
}

pub fn CudaGetDeviceProperties(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaGetDeviceProperties");
    let device = parameters.para2;
    let mut deviceProp: cudaDeviceProp = Default::default();

    let ret = unsafe { cudaGetDeviceProperties(&mut deviceProp, device as c_int) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaGetDeviceProperties: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut _) = deviceProp };
    return Ok(ret as u32);
}
pub fn CudaGetErrorString(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaGetErrorString");
    let ptr = unsafe { cudaGetErrorString(parameters.para1 as u32) };

    let cStr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
    let errorString = errorStr.to_string();
    unsafe { *(parameters.para2 as *mut String) = errorString };
    return Ok(0 as u32);
}
pub fn CudaGetErrorName(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaGetErrorName");
    let ptr = unsafe { cudaGetErrorName(parameters.para1 as u32) };

    let cStr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
    let errorString = errorStr.to_string();
    unsafe { *(parameters.para2 as *mut String) = errorString };
    return Ok(0 as u32);
}
pub fn CudaGetDeviceFlags() -> Result<u32> {
    //error!("nvidia.rs: cudaGetDeviceFlags");
    let mut deviceFlags = 0;

    let ret = unsafe { cudaGetDeviceFlags(&mut deviceFlags) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaGetDeviceFlags: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}

pub fn CudaPeekAtLastError() -> Result<u32> {
    //error!("nvidia.rs: cudaPeekAtLastError");
    let ret = unsafe { cudaPeekAtLastError() };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaPeekAtLastError: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaMalloc(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: cudaMalloc, mode{:?}", QUARK_CONFIG.lock().CudaMode);
    if QUARK_CONFIG.lock().CudaMemType == CudaMemType::UM {
        // error!("nvidia.rs: CudaMode::fastswitch()");
        let mut para1 = parameters.para1 as *mut c_void;

        let ret = unsafe {
            cudaMallocManaged(
                &mut para1 as *mut _ as u64,
                parameters.para2 as usize,
                cudaMemAttachGlobal,
            )
        };
        if ret as u32 != 0 {
            error!(
                "nvidia.rs: error caused by cudaMallocManaged: {}",
                ret as u32
            );
        } else {
            MEM_RECORDER
                .lock()
                .push((para1 as u64, parameters.para2 as usize));
        }

        // while let Some(element) = iterator.next() {
        //     total_mem = total_mem + element.1;
        // }

        // let location = cudaMemLocation {
        //     type_: cudaMemLocationType::cudaMemLocationTypeDevice,
        //     id: 0, // device 0
        // };
        // let ret = unsafe { cudaMemAdvise_v2( para1, parameters.para2 as usize, cudaMemoryAdvise::cudaMemAdviseSetReadMostly, location) };
        // if ret as u32 != 0 {
        //     error!("nvidia.rs: error caused by cudaMalloc(cudaMemAdvise_v2): {}", ret as u32);
        // }
        // let ret = unsafe { cudaMemAdvise_v2( para1, parameters.para2 as usize, cudaMemoryAdvise::cudaMemAdviseSetPreferredLocation, location) };
        // if ret as u32 != 0 {
        //     error!("nvidia.rs: error caused by cudaMalloc(cudaMemAdvise_v2): {}", ret as u32);
        // }
        unsafe { *(parameters.para1 as *mut u64) = para1 as u64 };
        // error!("nvidia.rs: malloc ptr:{:x}, size:{:x}", para1 as u64, parameters.para2);
        return Ok(ret as u32);
    } else if QUARK_CONFIG.lock().CudaMemType == CudaMemType::MemPool {
        let (addr, ret) = MEM_MANAGER
            .lock()
            .gpuManager
            .alloc(parameters.para2 as usize);
        unsafe { *(parameters.para1 as *mut u64) = addr };
        if ret != 0 {
            error!("mem pool failed to alloc");
        }
        return Ok(ret as u32);
    } else {
        // error!("nvidia.rs: CudaMode::Default()");
        let mut addr: u64 = 0;

        let ret = unsafe {
            cudaMalloc(
                &mut addr as *mut _ as u64 as *mut *mut libc::c_void,
                parameters.para2 as usize,
            )
        };
        if ret as u32 != 0 {
            error!("nvidia.rs: error caused by cudaMalloc: {}", ret as u32);
        }

        unsafe { *(parameters.para1 as *mut u64) = addr };
        return Ok(ret as u32);
    }
}
pub fn CudaFree(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: cudaFree addr is {:x?}", parameters.para1);
    if QUARK_CONFIG.lock().CudaMemType == CudaMemType::UM {
        let memRecorder = MEM_RECORDER.lock();
        let index = memRecorder
            .iter()
            .position(|&r| r.0 == parameters.para1 as u64)
            .unwrap();
        MEM_RECORDER.lock().remove(index);
    }

    let ret = unsafe { cudaFree(parameters.para1 as *mut c_void) };
    if ret as u32 != 0 {
        let ptr = unsafe { cudaGetErrorString(1) };

        let cStr = unsafe { std::ffi::CStr::from_ptr(ptr) };
        let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
        let errorString = errorStr.to_string();
        error!(
            "nvidia.rs: error caused by cudaFree: {} errorstring {}",
            ret as u32, errorString
        );
    }

    return Ok(ret as u32);
}

pub fn CudaRegisterFatBinary(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: cudaRegisterFatBinary");

    let fatElfHeader = unsafe { &*(parameters.para2 as *const u8 as *const FatElfHeader) };
    // error!("fatElfHeader magic is :{:x}, version is :{:x}, header size is :{:x}, size is :{:x}", fatElfHeader.magic, fatElfHeader.version, fatElfHeader.header_size, fatElfHeader.size);
    let moduleKey = parameters.para3;
    match GetFatbinInfo(parameters.para2, fatElfHeader) {
        Ok(_) => {}
        Err(e) => {
            return Err(e);
        }
    }

    let mut module: u64 = 0;
    let ret = unsafe {
        cuModuleLoadData(
            &mut module as *mut _ as u64 as *mut CUmodule,
            parameters.para2 as *const c_void,
        )
    };
    // TODO: try this https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaRegisterFatBinary(cuModuleLoadData): {}",
            ret as u32
        );
    }

    // unsafe{ cudaDeviceSynchronize();}
    MODULES.lock().insert(moduleKey, module);
    let fatbinSize: usize = parameters.para1 as usize;
    let mut fatbinData = Vec::with_capacity(parameters.para1 as usize);
    unsafe { copy_nonoverlapping(parameters.para2 as *const u8, fatbinData.as_mut_ptr(), fatbinSize); }
    //MEM_MANAGER.lock().fatBinManager.fatBinVec.push(fatbinData);
    //MEM_MANAGER.lock().fatBinManager.fatBinHandleVec.push((moduleKey, module));
    //MEM_MANAGER.lock().fatBinManager.fatBinFuncVec.push(Vec::new());
    // error!("insert module: {:x} -> {:x}", moduleKey, module);
    return Ok(ret as u32);
}
pub fn CudaUnregisterFatBinary(parameters: &ProxyParameters) -> Result<u32> {
    let moduleKey = parameters.para1;

    let module = match MODULES.lock().get(&moduleKey) {
        Some(module) => module.clone(),
        None => {
            error!("CudaUnregisterFatBinary: no module be found with this fatcubinHandle: {:x}", moduleKey);
            moduleKey.clone()
        }
    };

    let ret = unsafe { cuModuleUnload(module as *const u64 as CUmodule) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaUnregisterFatBinary(cuModuleUnload): {}",
            ret as u32
        );
    }

    // delete the module
    //MODULES.lock().remove(&moduleKey);
    return Ok(ret as u32);
        
}
pub fn CudaRegisterFunction(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaRegisterFunction");
    let info = unsafe { &*(parameters.para1 as *const u8 as *const RegisterFunctionInfo) };
    let bytes = unsafe {
        std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize)
    };
    let deviceName = std::str::from_utf8(bytes).unwrap();

    let mut module = match MODULES.lock().get(&info.fatCubinHandle) {
        Some(module) => module.clone(),
        None => {
            error!(
                "CudaRegisterFunction: no module be found with this fatcubinHandle: {:x}",
                info.fatCubinHandle
            );
            info.fatCubinHandle.clone()
        }
    };

    let mut hfunc: u64 = 0;
    let func_name = CString::new(deviceName).unwrap();
    let ret = unsafe {
        // cuda_driver_sys::
        cuModuleGetFunction(
            &mut hfunc as *mut _ as u64 as *mut CUfunction,
            *(&mut module as *mut _ as u64 as *mut CUmodule),
            func_name.clone().as_ptr(),
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaRegisterFunction(cuModuleGetFunction): {}",
            ret as u32
        );
    }

    //unsafe{ cudaDeviceSynchronize(); }
    FUNCTIONS.lock().insert(info.hostFun, hfunc);

    let kernelInfo = match KERNEL_INFOS.lock().get(&deviceName.to_string()) {
        Some(kernelInformations) => kernelInformations.clone(),
        None => {
            //error!("No kernel infos found with this deviceName : {}", deviceName);
            Arc::new(KernelInfo::default())
        }
    };
    let paramInfo = parameters.para3 as *const u8 as *mut ParamInfo;
    unsafe {
        (*paramInfo).paramNum = kernelInfo.paramNum;
        for i in 0..(*paramInfo).paramNum {
            (*paramInfo).paramSizes[i] = kernelInfo.paramSizes[i];
        }
    }

    //let index = MEM_MANAGER.lock().fatBinManager.fatBinHandleVec.iter().position(|&r| r.0 == info.fatCubinHandle).unwrap();
    //MEM_MANAGER.lock().fatBinManager.fatBinFuncVec[index].push((info.hostFun, func_name));

    return Ok(ret as u32);
}
pub fn CudaRegisterVar(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaRegisterFunction");
    let info = unsafe { *(parameters.para1 as *const u8 as *const RegisterVarInfo) };
    let bytes = unsafe {
        std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize)
    };
    let deviceName = std::str::from_utf8(bytes).unwrap();

    let module = match MODULES.lock().get(&info.fatCubinHandle) {
        Some(module) => module.clone(),
        None => {
            error!(
                "CudaRegisterVar: no module be found with this fatcubinHandle: {}",
                info.fatCubinHandle
            );
            info.fatCubinHandle.clone()
        }
    };

    let mut devicePtr: CUdeviceptr = 0;
    let mut dSize: usize = 0;
    let ownded_name = CString::new(deviceName).unwrap();
    let name = ownded_name.as_ptr();
    let ret = unsafe {
        cuModuleGetGlobal_v2(
            &mut devicePtr,
            &mut dSize,
            (module as *const u64) as CUmodule,
            name,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaRegisterVar(cuModuleGetGlobal_v2): {}",
            ret as u32
        );
    }

    //GLOBALS.lock().insert(info.hostVar, devicePtr);
    // TODO: cudaMemcpyToSymbol may need to store this info, no need for pytorch for now

    return Ok(ret as u32);
}
pub fn CudaLaunchKernel(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaLaunchKernel");
    let info = unsafe { &*(parameters.para1 as *const u8 as *const LaunchKernelInfo) };
    let func = match FUNCTIONS.lock().get(&info.func) {
        Some(func) => func.clone(),
        None => {
            //error!("no CUfunction has been found {:x}", info.func);
            0
        }
    };

    let ret: CUresult = unsafe {
        cuLaunchKernel(
            func as CUfunction,
            info.gridDim.x,
            info.gridDim.y,
            info.gridDim.z,
            info.blockDim.x,
            info.blockDim.y,
            info.blockDim.z,
            info.sharedMem as u32,
            info.stream as *mut CUstream_st,
            info.args as *mut *mut c_void,
            0 as *mut *mut c_void,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaLaunchKernel(cuLaunchKernel): {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaStreamSynchronize(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaStreamSynchronize");
    let ret = unsafe { cudaStreamSynchronize(parameters.para1 as cudaStream_t) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamSynchronize: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaStreamCreate(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaStreamCreate");
    let mut s: cudaStream_t = unsafe { *(parameters.para1 as *mut u64) as cudaStream_t };

    let ret = unsafe { cudaStreamCreate(&mut s) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamCreate: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut u64) = s as u64 };
    return Ok(ret as u32);
}

pub fn CudaStreamCreateWithFlags(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaStreamCreateWithFlags");
    let mut stream: cudaStream_t =
        unsafe { *(parameters.para1 as *mut u64) as cudaStream_t };

    let ret = unsafe { cudaStreamCreateWithFlags(&mut stream, parameters.para2 as u32) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamCreateWithFlags: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut u64) = stream as u64 };
    return Ok(ret as u32);
}
pub fn CudaStreamCreateWithPriority(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaStreamCreateWithPriority");
    let mut stream: cudaStream_t =
        unsafe { *(parameters.para1 as *mut u64) as cudaStream_t };

    let ret = unsafe {
        cudaStreamCreateWithPriority(
            &mut stream,
            parameters.para2 as u32,
            parameters.para3 as i32,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamCreateWithPriority: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut u64) = stream as u64 };
    return Ok(ret as u32);
}
pub fn CudaStreamDestroy(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaStreamDestroy");
    let ret = unsafe { cudaStreamDestroy(parameters.para1 as cudaStream_t) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamDestroy: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaStreamGetFlags(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaStreamGetFlags");
    let mut flags: u32 = 0;

    let ret = unsafe { cudaStreamGetFlags(parameters.para1 as u64, &mut flags) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamGetFlags: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para2 as *mut u32) = flags };
    return Ok(ret as u32);
}
pub fn CudaStreamGetPriority(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: CudaStreamGetPriority");
    let mut priority: i32 = 0;

    let ret =
        unsafe { cudaStreamGetPriority(parameters.para1 as u64, &mut priority) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamGetPriority: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para2 as *mut i32) = priority };
    return Ok(ret as u32);
}
pub fn CudaStreamIsCapturing(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaStreamIsCapturing");
    let mut captureStatus: cudaStreamCaptureStatus =
        unsafe { *(parameters.para2 as *mut _) };

    let ret = unsafe {
        cudaStreamIsCapturing(parameters.para1 as cudaStream_t, &mut captureStatus)
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamIsCapturing: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para2 as *mut _) = captureStatus as u32 };
    return Ok(ret as u32);
}
pub fn CudaStreamQuery(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaStreamQuery");
    let ret = unsafe { cudaStreamQuery(parameters.para1 as cudaStream_t) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaStreamQuery: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CudaStreamWaitEvent(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaStreamWaitEvent");
    let ret = unsafe {
        cudaStreamWaitEvent(
            parameters.para1 as cudaStream_t,
            parameters.para2 as cudaEvent_t,
            parameters.para3 as c_uint,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaStreamWaitEvent: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaThreadExchangeStreamCaptureMode(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaThreadExchangeStreamCaptureMode");
    let mut mode: cudaStreamCaptureMode =
        unsafe { *(parameters.para1 as *mut cudaStreamCaptureMode) };

    let ret = unsafe { cudaThreadExchangeStreamCaptureMode(&mut mode) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaThreadExchangeStreamCaptureMode: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut _) = mode as u32 };
    return Ok(ret as u32);
}
pub fn CudaEventCreate(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaEventCreate");
    let mut event: cudaEvent_t = unsafe { *(parameters.para1 as *mut u64) as cudaEvent_t };

    let ret = unsafe { cudaEventCreate(&mut event) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaEventCreate: {}", ret as u32);
    }

    unsafe { *(parameters.para1 as *mut u64) = event as u64 };
    return Ok(ret as u32);
}
pub fn CudaEventCreateWithFlags(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaEventCreateWithFlags");
    let mut event: cudaEvent_t = unsafe { *(parameters.para1 as *mut u64) as cudaEvent_t };

    let ret = unsafe { cudaEventCreateWithFlags(&mut event, parameters.para2 as c_uint) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by CudaEventCreateWithFlags: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut u64) = event as u64 };
    return Ok(ret as u32);
}
pub fn CudaEventDestroy(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaEventDestroy");
    let ret = unsafe { cudaEventDestroy(parameters.para1 as cudaEvent_t) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaEventDestroy: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaEventElapsedTime(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaEventElapsedTime");
    let mut time: f32 = 0.0;

    let ret = unsafe {
        cudaEventElapsedTime(
            &mut time,
            parameters.para2 as cudaEvent_t,
            parameters.para3 as cudaEvent_t,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaEventElapsedTime: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut _) = time };
    return Ok(ret as u32);
}
pub fn CudaEventQuery(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaEventQuery");
    let ret = unsafe { cudaEventQuery(parameters.para1 as cudaEvent_t) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaEventQuery: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CudaEventRecord(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaEventRecord");
    let ret = unsafe {
        cudaEventRecord(
            parameters.para1 as cudaEvent_t,
            parameters.para2 as cudaStream_t,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaEventRecord: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CudaEventSynchronize(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaEventSynchronize");
    let ret = unsafe { cudaEventSynchronize(parameters.para1 as cudaEvent_t) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaEventSynchronize: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaFuncGetAttributes(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaFuncGetAttributes");
    //error!("map is {:?}", FUNCTIONS.lock());
    let dev_func = match FUNCTIONS.lock().get(&parameters.para2) {
        Some(func) => func.clone(),
        None => {
            //error!("CudaFuncGetAttributes not find for func: {:x}", parameters.para2);
            0
        }
    };
    let mut attr: cudaFuncAttributes = Default::default();
    let mut tmpAttr: i32 = 0;
    let mut ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.sharedSizeBytes = tmpAttr as usize;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.constSizeBytes = tmpAttr as usize;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.localSizeBytes = tmpAttr as usize;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.maxThreadsPerBlock = tmpAttr as i32;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.numRegs = tmpAttr as i32;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.ptxVersion = tmpAttr as i32;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.binaryVersion = tmpAttr as i32;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.cacheModeCA = tmpAttr as i32;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.maxDynamicSharedSizeBytes = tmpAttr as i32;
    }
    ret = unsafe {
        cuFuncGetAttribute(
            &mut tmpAttr,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT as u32,
            dev_func,
        )
    };
    if ret as i32 != 0 {
        error!(
            "nvidia.rs: error caused by cuFuncGetAttribute: {}",
            ret as u32
        );
        return Ok(ret as u32);
    } else {
        attr.preferredShmemCarveout = tmpAttr as i32;
    }

    unsafe { *(parameters.para1 as *mut cudaFuncAttributes) = attr };
    return Ok(0 as u32);
}
pub fn CudaFuncSetAttribute(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaFuncSetAttribute");
    let dev_func = match FUNCTIONS.lock().get(&parameters.para1) {
        Some(func) => func.clone(),
        None => 0,
    };
    let ret = unsafe {
        cuFuncSetAttribute(
            dev_func as u64,
            parameters.para2 as u32,
            parameters.para3 as i32,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaFuncSetAttribute(cuFuncSetAttribute): {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaFuncSetCacheConfig(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaFuncSetCacheConfig");
    let ret = unsafe {
        cudaFuncSetCacheConfig(
            parameters.para1 as *const c_void,
            *(&parameters.para2 as *const _ as *const cudaFuncCache),
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaFuncSetCacheConfig: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaFuncSetSharedMemConfig(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaFuncSetSharedMemConfig");
    let ret = unsafe {
        cudaFuncSetSharedMemConfig(
            parameters.para1 as *const c_void,
            *(&parameters.para2 as *const _ as *const cudaSharedMemConfig),
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaFuncSetSharedMemConfig: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CudaGetLastError(_parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cuModuleGetLoadingMode");
    let ret = unsafe { cudaGetLastError() };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaGetLastError: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}

pub fn CudaMemset(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaMemset");
    let ret = unsafe {
        cudaMemset(
            parameters.para1 as *const c_void,
            parameters.para2 as c_int,
            parameters.para3 as usize,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaMemset: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CudaMemsetAsync(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaMemsetAsync");
    let ret = unsafe {
        cudaMemsetAsync(
            parameters.para1 as *const c_void,
            parameters.para2 as c_int,
            parameters.para3 as usize,
            parameters.para4 as cudaStream_t,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cudaMemsetAsync: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CudaHostAlloc(parameters: &ProxyParameters) -> Result<u32> {
    let size = parameters.para2 as usize;
    let flags = parameters.para3 as u32;
    let arrAddr = parameters.para4;
    let cnt = unsafe { &mut *(parameters.para5 as *mut usize) };
    let hostAddr = unsafe { &mut *(parameters.para6 as *mut u64) };

    let mut addr: u64 = 0;

    let ret = unsafe { cudaHostAlloc(&mut addr as *mut _ as u64, size, flags) };
    if ret != 0 {
        error!("nvidia.rs: error caused by CudaHostAlloc: {}", ret as u32);
        return Ok(ret as u32);
    }

    assert!(addr % MemoryDef::PAGE_SIZE_2M == 0);
    *hostAddr = addr;

    let iovs = unsafe { slice::from_raw_parts_mut(arrAddr as *mut IoVec, *cnt) };
    let hugePageCnt =
        ((size as u64 + MemoryDef::PAGE_SIZE_2M - 1) / MemoryDef::PAGE_SIZE_2M) as usize;
    let mut pages = Vec::new();
    for _i in 0..hugePageCnt {
        let pageAddr = PMA_KEEPER
            .AllocHugePage()
            .expect("CudaHostAlloc can't alloc huge page");
        pages.push(pageAddr);
    }
    pages.sort();

    let mut iov = IoVec::NewFromAddr(pages[0], MemoryDef::PAGE_SIZE_2M as usize);
    let mut idx = 0;
    for i in 1..hugePageCnt {
        if pages[i] == iov.End() {
            iov.len += MemoryDef::PAGE_SIZE_2M as usize;
        } else {
            iovs[idx] = iov;
            idx += 1;
            iov = IoVec::NewFromAddr(pages[i], MemoryDef::PAGE_SIZE_2M as usize);
        }
    }

    iovs[idx] = iov;

    // in case the size < 2MB
    let left = size % MemoryDef::PAGE_SIZE_2M as usize;
    if left != 0 {
        let gap = MemoryDef::PAGE_SIZE_2M as usize - left;
        iovs[idx].len -= gap;
    }

    *cnt = idx + 1;

    let remapFlags = libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED | libc::MREMAP_DONTUNMAP;
    // let remapFlags = libc::MREMAP_FIXED;
    let mut offset = 0;
    for i in 0..*cnt {
        let iov = iovs[i];
        let ret = unsafe {
            libc::mremap(
                (addr + offset) as _,
                iov.Len() as _,
                iov.Len() as _,
                remapFlags,
                iov.Start() as u64,
            )
        } as i64;

        if ret == -1 {
            return Err(Error::SystemErr(-errno::errno().0));
        }

        offset += iov.Len() as u64;
    }

    return Ok(0);
}

pub fn CudeFreeHost(parameters: &ProxyParameters) -> Result<u32> {
    let hostAddr = parameters.para2;
    let arrAddr = parameters.para3;
    let cnt = parameters.para4 as usize;

    let iovs = unsafe { slice::from_raw_parts(arrAddr as *const IoVec, cnt) };
    for iov in iovs {
        let mut addr = iov.Start();
        let len = iov.Len();
        let ret = unsafe { libc::munmap(addr as _, len) };

        if ret == -1 {
            return Err(Error::SystemErr(-errno::errno().0));
        }

        while addr < iov.End() {
            PMA_KEEPER.FreeHugePage(addr);
            addr += MemoryDef::PAGE_SIZE_2M;
        }
    }

    let ret = unsafe { cudaFreeHost(hostAddr) };
    if ret != 0 {
        error!("nvidia.rs: error caused by CudeFreeHost: {}", ret as u32);
        return Ok(ret as u32);
    }

    return Ok(0);
}

pub fn CudaMemcpy(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaMemcpy");
    // let func: extern "C" fn(u64, u64, u64, u64) -> u32 = unsafe { std::mem::transmute(handle) };
    //error!("nvidia.rs: CudaMemcpy: count:{}, len:{}", parameters.para4 as usize, parameters.para3 as usize);
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
                // let ret = func(dst + offset, r.start, r.len, kind);
                let ret = unsafe { cudaMemcpy(dst + offset, r.start, r.len, kind) };
                if ret != 0 {
                    error!("nvidia.rs: error caused by cudaMemcpy: {}", ret as u32);
                    return Ok(ret as u32);
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
                // let ret = func(r.start, src + offset, r.len, kind);
                let ret = unsafe { cudaMemcpy(r.start, src + offset, r.len, kind) };
                if ret != 0 {
                    error!("nvidia.rs: error caused by cudaMemcpy: {}", ret as u32);
                    return Ok(ret as u32);
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
            // let ret = func(dst, src, count, kind);

            let ret = unsafe { cudaMemcpy(dst, src, count, kind) };
            if ret != 0 {
                error!("nvidia.rs: error caused by cudaMemcpy: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        _ => todo!(),
    }
}

pub fn CudaMemcpyAsync(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaMemcpyAsync");
    //error!("nvidia.rs: CudaMemcpy: count:{}, len:{}", parameters.para4 as usize, parameters.para3 as usize);
    let kind = parameters.para5;
    let stream = parameters.para6 as cudaStream_t;
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => todo!(),
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            let dst = parameters.para1; // device
            let ptr = parameters.para2 as *const Range;
            // length of the vector
            let len = parameters.para3 as usize;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = unsafe { cudaMemcpyAsync(dst + offset, r.start, r.len, kind, stream) };
                if ret as u32 != 0 {
                    error!("nvidia.rs: error caused by CudaMemcpyAsync: {}", ret as u32);
                    return Ok(ret as u32);
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

            let src = parameters.para3; //device
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = unsafe { cudaMemcpyAsync(r.start, src + offset, r.len, kind, stream) };

                if ret != 0 {
                    error!("nvidia.rs: error caused by CudaMemcpyAsync: {}", ret as u32);
                    return Ok(ret as u32);
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

            let ret = unsafe { cudaMemcpyAsync(dst, src, count, kind, stream) };
            if ret != 0 {
                error!("nvidia.rs: error caused by CudaMemcpyAsync: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        _ => todo!(),
    }
}
pub fn CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    let dev_func = match FUNCTIONS.lock().get(&parameters.para2) {
        Some(func) => func.clone(),
        None => 0,
    };
    let mut numBlocks = 0;

    let ret = unsafe {
        cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &mut numBlocks,
            dev_func as CUfunction,
            parameters.para3 as c_int,
            parameters.para4 as usize,
            parameters.para5 as c_uint,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: {}", ret as u32);
    }

    unsafe { *(parameters.para1 as *mut i32) = numBlocks as i32 };
    return Ok(ret as u32);
}

