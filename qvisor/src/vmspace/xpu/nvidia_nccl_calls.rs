use crate::qlib::proxy::*;
use crate::vmspace::xpu::cuda_api::*;
use std::os::raw::*;
use std::ffi::CStr;
use std::ptr::null_mut;
use cuda_runtime_sys::cudaStream_t;
use crate::qlib::proxy::ProxyParameters;
use crate::qlib::common::Result;

use super::cuda::STREAMS;

pub fn NcclGetVersion(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclGetVersion");
    let mut version: c_int = 0;
    let ret = unsafe { ncclGetVersion(&mut version) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclGetVersion: {}", ret as u32);
    } else {
        unsafe { *(parameters.para1 as *mut c_int)= version };
    }
    return Ok(ret as u32);
}

pub fn NcclGetUniqueId(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclGetUniqueId");
    let mut ncclUniqueId_ns: NcclUniqueId = NcclUniqueId::default();
    let ret = unsafe { ncclGetUniqueId(&mut ncclUniqueId_ns) };
    // error!("nvidia.rs: ncclGetUniqueId_after ncclGetUniqueId call");

    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclGetUniqueId: {}", ret as u32);
    } else {
        unsafe { *(parameters.para1 as *mut _)= ncclUniqueId_ns };
    }
    return Ok(ret as u32);  
}

pub fn NcclCommInitRank(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommInitRank");
    let mut ncclComm_t_: NcclCommT = null_mut();
    let ncclUniqueId_ns = unsafe { *(parameters.para3 as *const NcclUniqueId) };
    let ret = unsafe { ncclCommInitRank(&mut ncclComm_t_, parameters.para2 as i32, ncclUniqueId_ns, parameters.para4 as i32) };

    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommInitRank: {}", ret as u32);
    } else {
        unsafe { *(parameters.para1 as *mut _)= ncclComm_t_ };
    }
    return Ok(ret as u32);
}

pub fn NcclCommInitRankConfig(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommInitRankConfig");
    let mut ncclComm_t_: NcclCommT = null_mut();
    let ncclUniqueId_ns = unsafe { *(parameters.para3 as *const NcclUniqueId) };
    // let config = unsafe { *(parameters.para5 as *const NcclConfig) };

    let ret = unsafe { ncclCommInitRankConfig(&mut ncclComm_t_, parameters.para2 as i32, ncclUniqueId_ns, parameters.para4 as i32, parameters.para5 as *const NcclConfig) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommInitRankConfig: {}", ret as u32);
    } else {
        unsafe { *(parameters.para1 as *mut _)= ncclComm_t_ };
    }
    return Ok(ret as u32);
}

pub fn NcclCommInitAll(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommInitAll");
    let mut ncclComm_t_s: Vec<NcclCommT> = vec![null_mut(); parameters.para2 as usize];  

    let ret = unsafe { ncclCommInitAll(ncclComm_t_s.as_mut_ptr(), parameters.para2 as i32, parameters.para3 as *const c_int) };

    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommInitAll: {}", ret as u32);
    } else {
        for i in 0..parameters.para2 as usize {
            unsafe {
                *(parameters.para1 as *mut NcclCommT).add(i) = ncclComm_t_s[i];
            }
        }
    }

    return Ok(ret as u32);
}

pub fn NcclCommDestroy(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommDestroy");
    let ret = unsafe { ncclCommDestroy(parameters.para1 as NcclCommT) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommDestroy: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclCommAbort(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommAbort");
    let ret = unsafe { ncclCommAbort(parameters.para1 as NcclCommT) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommAbort: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclCommCount(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommCount");
    let mut count: c_int = 0;
    let ret = unsafe { ncclCommCount(parameters.para1 as NcclCommT, &mut count) };

    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommCount: {}", ret as u32);
    } else {
        unsafe { *(parameters.para2 as *mut c_int) = count };
    }
    return Ok(ret as u32);
    
}

pub fn NcclCommUserRank(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommUserRank");
    let mut rank: c_int = 0;
    let ret = unsafe { ncclCommUserRank(parameters.para1 as NcclCommT, &mut rank) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommUserRank: {}", ret as u32);
    } else {
        unsafe { *(parameters.para2 as *mut c_int) = rank };
    }
    return Ok(ret as u32);
}

pub fn NcclCommCuDevice(parameters: &ProxyParameters) -> Result<u32> {
    // error!("nvidia.rs: ncclCommCuDevice");
    let mut device: c_int = 0;
    let ret = unsafe { ncclCommCuDevice(parameters.para1 as NcclCommT, &mut device) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommCuDevice: {}", ret as u32);
    } else {
        unsafe { *(parameters.para2 as *mut c_int) = device };
    }
    return Ok(ret as u32);
    
}

pub fn NcclCommGetAsyncError(parameters: &ProxyParameters) -> Result<u32> {
    let mut result = NcclResultT::NcclSuccess;
    let ret = unsafe { ncclCommGetAsyncError(parameters.para1 as NcclCommT, &mut result) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclCommGetAsyncError: {}", ret as u32);
    } else {
        unsafe { *(parameters.para2 as *mut NcclResultT) = result };
    }
    return Ok(ret as u32);
}

pub fn NcclSend(parameters: &ProxyParameters) -> Result<u32> {
    let info = unsafe { *(parameters.para2 as *const u8 as *const NcclSendRecvInfo) };
    let stream = match STREAMS.lock().get(&info.stream) {
        Some(s)=> s.clone(),
        None => panic!(),
    };
    let ret = unsafe { ncclSend(parameters.para1 as *const c_void, info.count as usize, info.datatype, info.peer, info.comm as NcclCommT, stream as cudaStream_t) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclSend: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclRecv(parameters: &ProxyParameters) -> Result<u32> {
    let info = unsafe { *(parameters.para2 as *const u8 as *const NcclSendRecvInfo) };
    let stream = match STREAMS.lock().get(&info.stream) {
        Some(s)=> s.clone(),
        None => panic!(),
    };
    let ret = unsafe { ncclRecv(parameters.para1 as *mut c_void, info.count as usize, info.datatype, info.peer, info.comm as NcclCommT, stream as cudaStream_t) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclRecv: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclGroupStart() -> Result<u32> {
    let ret = unsafe { ncclGroupStart() };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclGroupStart: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclGroupEnd() -> Result<u32> {
    let ret = unsafe { ncclGroupEnd() };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclGroupEnd: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclAllGather(parameters: &ProxyParameters) -> Result<u32> {
    let info = unsafe { *(parameters.para3 as *const u8 as *const NcclAllGatherReduceInfo) };
    let stream = match STREAMS.lock().get(&info.stream) {
        Some(s)=> s.clone(),
        None => panic!(),
    };
    let ret = unsafe { ncclAllGather(parameters.para1 as *const c_void, parameters.para2 as *mut c_void, info.count as usize, info.datatype, info.comm as NcclCommT, stream as cudaStream_t) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclAllGather: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclAllReduce(parameters: &ProxyParameters) -> Result<u32> {
    let info = unsafe { *(parameters.para3 as *const u8 as *const NcclAllGatherReduceInfo) };
    let stream = match STREAMS.lock().get(&info.stream) {
        Some(s)=> s.clone(),
        None => panic!(),
    };
    let ret = unsafe { ncclAllReduce(parameters.para1 as *const c_void, parameters.para2 as *mut c_void, info.count as usize, info.datatype, info.op, info.comm as NcclCommT, stream as cudaStream_t) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclAllReduce: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclReduceScatter(parameters: &ProxyParameters) -> Result<u32> {
    let info = unsafe { *(parameters.para3 as *const u8 as *const NcclAllGatherReduceInfo) };
    let stream = match STREAMS.lock().get(&info.stream) {
        Some(s)=> s.clone(),
        None => panic!(),
    };
    let ret = unsafe { ncclReduceScatter(parameters.para1 as *const c_void, parameters.para2 as *mut c_void, info.count as usize, info.datatype, info.op, info.comm as NcclCommT, stream as cudaStream_t) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by ncclReduceScatter: {}", ret as u32);
    }
    return Ok(ret as u32);
}

pub fn NcclGetErrorString(parameters: &ProxyParameters) -> Result<u32> {
    let ret = unsafe {
        ncclGetErrorString(parameters.para1 as u32)
    };
    
    let c_str: &CStr = unsafe { CStr::from_ptr(ret) };

    // Convert CStr to &str and handle potential UTF-8 errors
    let error_str= c_str.to_str().expect("Invalid UTF-8 data");

    let error_string = error_str.to_string();
    unsafe { *(parameters.para2 as *mut String) = error_string };
    return Ok(0 as u32);
}
