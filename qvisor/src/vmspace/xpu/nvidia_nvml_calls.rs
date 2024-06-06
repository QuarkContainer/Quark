
use crate::qlib::common::*;
use crate::qlib::proxy::*;
use crate::xpu::cuda_api::*;
use std::os::raw::*;


pub fn NvmlDeviceGetCountV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: nvmlDeviceGetCountV2");
    let mut deviceCount: c_int = Default::default();

    let ret = unsafe { cudaGetDeviceCount(&mut deviceCount) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cudaGetDeviceCount: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para1 as *mut _) = deviceCount as u32 };
    return Ok(ret as u32);
}
pub fn NvmlInitWithFlags(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: nvmlInitWithFlags");
    let ret = unsafe { nvmlInitWithFlags(parameters.para1 as c_uint) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by nvmlInitWithFlags: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn NvmlInit(_parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: NvmlInit");
    let ret = unsafe { nvmlInitWithFlags(0 as c_uint) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by NvmlInit(nvmlInitWithFlags): {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn NvmlInitV2(_parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: NvmlInitV2");
    let ret = unsafe { nvmlInit_v2() };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by nvmlInit_v2: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn NvmlShutdown(_parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: NvmlShutdown");
    let ret = unsafe { nvmlShutdown() };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by nvmlShutdown: {}", ret as u32);
    }

    return Ok(ret as u32);
}