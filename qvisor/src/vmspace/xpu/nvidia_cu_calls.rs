use crate::qlib::common::*;
use crate::qlib::proxy::*;
use crate::xpu::cuda_api::*;
use crate::xpu::cuda::*;
use std::os::raw::*;
use std::sync::Arc;
use std::ffi::CString;

use cuda_driver_sys::{
    CUcontext, CUdevice, CUfunction, CUmodule, CUresult,
    CUstream_st,
};


pub fn CuModuleGetLoadingMode(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cuModuleGetLoadingMode");
    let mut loadingMode: CumoduleLoadingModeEnum = unsafe { *(parameters.para1 as *mut _) };

    let ret = unsafe { cuModuleGetLoadingMode(&mut loadingMode) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cuModuleGetLoadingMode: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut _) = loadingMode as u32 };
    return Ok(ret as u32);
}
pub fn CuInit(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CuInit");
    let ret = unsafe { cuInit(parameters.para1 as c_uint) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cuInit: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CuDevicePrimaryCtxGetState(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cuDevicePrimaryCtxGetState");
    let mut flags: c_uint = Default::default();
    let mut active: c_int = Default::default();

    let ret = unsafe {
        cuDevicePrimaryCtxGetState(
            parameters.para1 as CUdevice,
            &mut flags as *mut c_uint,
            &mut active as *mut c_int,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cuDevicePrimaryCtxGetState: {}",
            ret as u32
        );
    }

    unsafe {
        *(parameters.para2 as *mut _) = flags;
        *(parameters.para3 as *mut _) = active;
    };
    return Ok(ret as u32);
}
pub fn CuCtxGetCurrent(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CuCtxGetCurrent");
    let mut ctx: u64 = 0;

    let ret = unsafe { cuCtxGetCurrent(&mut ctx as *mut _ as *mut CUcontext) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cuCtxGetCurrent: {}", ret as u32);
    }

    unsafe { *(parameters.para1 as *mut u64) = ctx };
    return Ok(ret as u32);
}
pub fn CuModuleLoadData(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CuModuleLoadData");
    match GetParameterInfo(parameters.para2, parameters.para3) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let mut module: u64 = 0;

    let ret = unsafe {
        cuModuleLoadData(
            &mut module as *mut _ as *mut CUmodule,
            parameters.para2 as *const c_void,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cuModuleLoadData: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut u64) = module };
    return Ok(ret as u32);
}
pub fn CuModuleGetFunction(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: CuModuleGetFunction");
    let bytes = unsafe {
        std::slice::from_raw_parts(parameters.para3 as *const u8, parameters.para4 as usize)
    };
    let funcName = std::str::from_utf8(bytes).unwrap();
    let mut hfunc: u64 = 0;

    let ret = unsafe {
        // cuda_driver_sys::
        cuModuleGetFunction(
            &mut hfunc as *mut _ as u64 as *mut CUfunction,
            parameters.para2 as CUmodule,
            CString::new(funcName).unwrap().clone().as_ptr(),
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cuModuleGetFunction: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut u64) = hfunc };

    let kernelInfo = match KERNEL_INFOS.lock().get(&funcName.to_string()) {
        Some(kernelInformations) => kernelInformations.clone(),
        None => {
            //error!("No kernel infos found with this funcName : {}",funcName);
            Arc::new(KernelInfo::default())
        }
    };

    let paramInfo = parameters.para5 as *const u8 as *mut ParamInfo;
    unsafe {
        (*paramInfo).paramNum = kernelInfo.paramNum;
        for i in 0..(*paramInfo).paramNum {
            (*paramInfo).paramSizes[i] = kernelInfo.paramSizes[i];
        }
    }

    // FUNCTIONS.lock().insert(info.hostFun, hfunc);
    return Ok(ret as u32);
}
pub fn CuModuleUnload(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: CuModuleUnload");
    let ret = unsafe { cuModuleUnload(parameters.para1 as CUmodule) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cuModuleUnload: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CuLaunchKernel(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CuLaunchKernel");
    let info = unsafe { &*(parameters.para1 as *const u8 as *const CuLaunchKernelInfo) };
    let stream = match STREAMS.lock().get(&info.hStream) {
        Some(s)=> s.clone(),
        None => 0,
    };

    let ret: CUresult = unsafe {
        // cuda_driver_sys::
        cuLaunchKernel(
            info.f as CUfunction,
            info.gridDimX,
            info.gridDimY,
            info.gridDimZ,
            info.blockDimX,
            info.blockDimY,
            info.blockDimZ,
            info.sharedMemBytes as u32,
            stream as *mut CUstream_st,
            info.kernelParams as *mut *mut ::std::os::raw::c_void,
            0 as *mut *mut ::std::os::raw::c_void,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cuLaunchKernel: {}", ret as u32);
    }

    return Ok(ret as u32);
}