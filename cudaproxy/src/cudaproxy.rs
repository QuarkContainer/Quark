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

use std::ffi::CString;
use std::mem::transmute;
use std::os::raw::*;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32};
use lazy_static::lazy_static;
// use libc::{c_int, SK_MEMINFO_SNDBUF};
use libc::c_int;
use libc::dlsym;
use libelf::raw::*;

use cuda_driver_sys::*;
use cuda_runtime_sys::{
    cudaStreamCaptureMode, cudaMemcpyKind, cudaEvent_t, cudaStreamCaptureStatus, 
    cudaStream_t,cudaFuncAttributes,cudaFuncAttribute,
    cudaDeviceAttr, cudaDeviceP2PAttr, cudaDeviceProp, cudaFuncCache, cudaLimit,
    cudaSharedMemConfig,cudaError_t
};
use rcublas_sys::{cublasHandle_t,cublasMath_t,cublasOperation_t, cublasComputeType_t, cublasGemmAlgo_t, cudaDataType_t};
use cuda11_cublasLt_sys::{cublasLtHandle_t,cublasLtMatmulDesc_t,cublasLtMatrixLayout_t,cublasLtMatmulAlgo_t,cublasLtMatmulPreference_t,cublasLtMatmulHeuristicResult_t};

use crate::proxy::*;
use crate::syscall::*;

extern crate alloc;

pub const SYS_PROXY: usize = 10003;

pub static mut DLOPEN_ORIG: Option<unsafe extern "C" fn(*const libc::c_char, libc::c_int) -> *mut libc::c_void> = None;
pub static mut DLCLOSE_ORIG: Option<unsafe extern "C" fn(*mut libc::c_void) -> libc::c_int> = None;
pub static mut DL_HANDLE: *mut libc::c_void = ptr::null_mut();
lazy_static! {
    static ref IS_GETDEVICE_UPDATED: AtomicBool = AtomicBool::new(false);
    static ref CURRENT_GPU_ID: AtomicI32 = AtomicI32::new(0);
    static ref LAST_ERROR: AtomicU32 = AtomicU32::new(0);
}
// thread_local!(static ERROR_CODE: AtomicU32 = AtomicU32::new(0));

#[no_mangle]
pub extern "C" fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void {
    //let mut ret: *mut c_void  = std::ptr::null_mut();
    // if filename.is_null() {
    //     println!("intercepted dlopen(Null, {})", flag);
    // } else {
    //     let c_str = unsafe { std::ffi::CStr::from_ptr(filename) };
    //     let filename_string = c_str.to_string_lossy().to_string();
    //     println!("intercepted dlopen again({} {})", filename_string, flag);
    // }

    if filename.is_null() {
        return unsafe { DLOPEN_ORIG.unwrap()(filename, flag) };  
    }
    
    if unsafe { DLOPEN_ORIG.is_none() } {
        let symbol = CString::new("dlopen").unwrap();
        unsafe { DLOPEN_ORIG = Some(std::mem::transmute(libc::dlsym(libc::RTLD_NEXT, symbol.as_ptr())))};
        //  unsafe {DLOPEN_ORIG =  Some(std::mem::transmute(
        //       ::libc::dlsym(::libc::RTLD_NEXT, std::mem::transmute(b"dlopen\x00".as_ptr())) )) };
    }
    if unsafe { DLOPEN_ORIG.is_none() } {
        println!("DLopen_Orig is still none");
    }

    let replace_libs = [
        "libcuda.so.1",
        "libcuda.so",
        "libnvidia-ml.so.1",
        "libcudnn_cnn_infer.so.8",
    ];

    if !filename.is_null() {
        let c_str = unsafe { std::ffi::CStr::from_ptr(filename) };
        let filename_string = c_str.to_string_lossy().to_string();

        for libs in &replace_libs{
            if filename_string == *libs {
                //println!("replacing dlopen call to {} with libcudaproxy.so", libs);
                let cudaProxy = CString::new("libcudaproxy.so").unwrap();
                unsafe { DL_HANDLE = DLOPEN_ORIG.unwrap()(cudaProxy.as_ptr(),flag) };
                //unsafe {println!("DL_handle be replaced: {:x?}",DL_HANDLE ) };
                if unsafe { DL_HANDLE.is_null() } {
                    println!("failed to replaced dlopen call to libcudaproxy.so");
                }
                return unsafe { DL_HANDLE };
            }
        }
    }

    let ret = unsafe { DLOPEN_ORIG.unwrap_or_else(||{ panic!("DLOPEN_ORIG IS None")})(filename,flag)};

    if ret.is_null() {
        //let err = unsafe { libc::dlerror() };
        //let errMesg = unsafe { CString::from_raw(err) };
        // println!(
        //     "dlopen failed: {}",
        //     errMesg.to_str().unwrap_or("unknown error")
        // );
    }
     return ret;
}

#[no_mangle]
pub extern "C" fn dlclose(handle: *mut c_void) -> c_int{
    if handle.is_null() {
        //println!("[dlclose] handle NULL");
        return -1;
    } else if unsafe{ DLCLOSE_ORIG.is_none() } {
        let symbol= CString::new("dlclose").unwrap();
        let orig = unsafe { dlsym(libc::RTLD_NEXT, symbol.as_ptr()) };
        if orig.is_null() {
            println!("[dlclose] dlsym failed");
        }else {
            // unsafe{ DLCLOSE_ORIG =  Some(std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_void) -> *mut c_void>(orig)) };
            unsafe { DLCLOSE_ORIG = Some(transmute(orig)); }
        }
    }
    if unsafe{ DL_HANDLE == handle } {
        //println!("[dlclose] ignore close");
        return 0;
    } else{
        return unsafe{ DLCLOSE_ORIG.unwrap()(handle) };
    }
}
#[no_mangle]
pub extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> usize {
    println!("Hijacked cudaMemGetInfo");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaMemGetInfo as usize, free as *mut _ as usize, total as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn ncclGetVersion(version: *mut c_int) -> usize {
    println!("Hijacked ncclGetVersion");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::NcclGetVersion as usize, version as *mut _ as usize);
}
// ncclgetuniqueid
#[no_mangle]
pub extern "C" fn ncclGetUniqueId(ncclUniqueId_p: *mut NcclUniqueId) -> usize {
    println!("Hijacked ncclGetUniqueId");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::NcclGetUniqueId as usize, ncclUniqueId_p as *mut _ as usize, 0);
}
// #nncccommitinitrank
#[no_mangle]
pub extern "C" fn ncclCommInitRank(
    comm: *mut NcclCommT,
    n: c_int,
    comm_id: NcclUniqueId,
    rank: c_int,
) -> usize {
    println!("Hijacked ncclCommInitRank");
    return cudaSyscall5(
        SYS_PROXY,
        ProxyCommand::NcclCommInitRank as usize,
        comm as *mut _ as usize,
        n as usize,
        &comm_id as *const _ as usize,
        rank as usize,
    );
}
#[no_mangle]
pub extern "C" fn ncclCommInitAll(comms: *mut NcclCommT, n: c_int, devlist: *const c_int) -> usize {
    println!("Hijacked ncclCommInitAll");
    return cudaSyscall4(
        SYS_PROXY,
        ProxyCommand::NcclCommInitAll as usize,
        comms as *mut _ as usize,
        n as usize,
        devlist as *const _ as usize,
    );
}

#[no_mangle]
pub extern "C" fn ncclCommDestroy(comm: *mut NcclCommT) -> usize {
    println!("Hijacked ncclCommDestroy");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::NcclCommDestroy as usize, comm as *mut _ as usize);
}
#[no_mangle]
pub extern "C" fn ncclCommAbort(comm: NcclCommT) -> usize {
    println!("Hijacked ncclCommAbort");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::NcclCommAbort as usize, comm as usize);
}

#[no_mangle]
pub extern "C" fn ncclCommInitRankConfig(
    comm: *mut NcclCommT,
    n: c_int,
    comm_id: NcclUniqueId,
    rank: c_int,
    config: *const NcclConfig,
) -> usize {
    println!("Hijacked ncclCommInitRankConfig");
    return cudaSyscall6(
        SYS_PROXY,
        ProxyCommand::NcclCommInitRankConfig as usize,
        comm as *mut _ as usize,
        n as usize,
        &comm_id as *const _ as usize,
        rank as usize,
        config as *const _ as usize,
    );
}
#[no_mangle]
pub extern "C" fn ncclCommCount(comm: NcclCommT, count: *mut c_int) -> usize {
    println!("Hijacked ncclCommCount");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::NcclCommCount as usize, comm as usize, count as usize);
}
#[no_mangle]
pub extern "C" fn ncclCommUserRank(comm: NcclCommT, rank: *mut c_int) -> usize {
    println!("Hijacked ncclCommUserRank");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::NcclCommUserRank as usize, comm as usize, rank as usize);
}
#[no_mangle]
pub extern "C" fn ncclCommCuDevice(comm: NcclCommT, device: *mut c_int) -> usize {
    println!("Hijacked ncclCommCuDevice");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::NcclCommCuDevice as usize, comm as usize, device as usize);
}
#[no_mangle]
pub extern "C" fn ncclCommGetAsyncError(comm: NcclCommT, NcclResultT_p: *mut NcclResultT) -> usize {
    println!("Hijacked ncclCommGetAsyncError");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::NcclCommGetAsyncError as usize, comm as usize, NcclResultT_p as usize);
    
}
#[no_mangle]
pub extern "C" fn ncclSend(sendbuff: *const c_void,
    count: usize,
    datatype: NcclDataTypeT,
    peer: c_int,
    comm: NcclCommT,
    stream: cudaStream_t,) -> usize {
    println!("Hijacked ncclSend");

    let send_info = NcclSendRecvInfo {
        count: count,
        datatype: datatype,
        peer: peer,
        comm: comm as u64,
        stream: stream as u64

    };

    return cudaSyscall3(SYS_PROXY, ProxyCommand::NcclSend as usize, sendbuff as usize, &send_info as *const _ as usize);
    
}
#[no_mangle]
pub extern "C" fn ncclRecv(recvbuff: *mut c_void,
    count: usize,
    datatype: NcclDataTypeT,
    peer: c_int,
    comm: NcclCommT,
    stream: cudaStream_t,) -> usize {
    println!("Hijacked ncclRecv");
    let recv_info = NcclSendRecvInfo {
        count: count,
        datatype: datatype,
        peer: peer,
        comm: comm as u64,
        stream: stream as u64
    };
    return cudaSyscall3(SYS_PROXY, ProxyCommand::NcclRecv as usize, recvbuff as usize, &recv_info as *const _ as usize);
    }    
#[no_mangle]
pub extern "C" fn ncclGroupStart() -> usize {
    println!("Hijacked ncclGroupStart");
    return cudaSyscall1(SYS_PROXY, ProxyCommand::NcclGroupStart as usize);
}
#[no_mangle]
pub extern "C" fn ncclGroupEnd() -> usize {
    println!("Hijacked ncclGroupEnd");
    return cudaSyscall1(SYS_PROXY, ProxyCommand::NcclGroupEnd as usize);
}
#[no_mangle]
pub extern "C" fn ncclAllReduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: NcclDataTypeT,
    op: NcclRedOpT,
    comm: NcclCommT,
    stream: cudaStream_t,
)  -> usize {
    println!("Hijacked NcclAllReduce");
    let send_info: NcclAllGatherReduceInfo = NcclAllGatherReduceInfo {
        count: count,
        datatype: datatype,
        op: op,
        comm: comm as u64,
        stream: stream as u64
    };
    return cudaSyscall4(SYS_PROXY, ProxyCommand::NcclAllReduce as usize, sendbuff as usize, recvbuff as usize, &send_info as *const _ as usize);
}
#[no_mangle]
pub extern "C" fn ncclReduceScatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: usize,
    datatype: NcclDataTypeT,
    op: NcclRedOpT,
    comm: NcclCommT,
    stream: cudaStream_t,
)    -> usize {
    println!("Hijacked ncclReduceScatter");
    let send_info: NcclAllGatherReduceInfo = NcclAllGatherReduceInfo {
        count: recvcount,
        datatype: datatype,
        op: op,
        comm: comm as u64,
        stream: stream as u64
    };
    return cudaSyscall4(SYS_PROXY, ProxyCommand::NcclReduceScatter as usize, sendbuff as usize, recvbuff as usize, &send_info as *const _ as usize);
}
#[no_mangle]
pub extern "C" fn ncclAllGather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: NcclDataTypeT,
    comm: NcclCommT,
    stream: cudaStream_t,
)    -> usize {
    println!("Hijacked ncclAllGather"); 
    let send_info: NcclAllGatherReduceInfo = NcclAllGatherReduceInfo {
        count: count,
        datatype: datatype,
        op: NcclRedOpT::NcclSum,
        comm: comm as u64,
        stream: stream as u64
    };
    return cudaSyscall4(SYS_PROXY, ProxyCommand::NcclAllGather as usize, sendbuff as usize, recvbuff as usize, &send_info as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn ncclGetErrorString(
    error: NcclResultT,
) -> *const c_char {
    let mut errorString:[i8; 128] = [1; 128];
    cudaSyscall3(SYS_PROXY,ProxyCommand::NcclGetErrorString as usize, error as usize, &mut errorString as *mut _ as usize);
    let cStr = unsafe { std::ffi::CStr::from_ptr(&errorString as *const c_char) };
    // let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
    // let ptr = errorStr.to_string().as_ptr() as *const i8;
    // return ptr;
    let error_str = match cStr.to_str() {
        Ok(s) => s,
        Err(_) => "Invalid UTF-8 data",
    };
    let c_string = std::ffi::CString::new(error_str).unwrap();
    let ptr = c_string.into_raw();
    return ptr;
    // if errorString.is_null() {
    //     println!("Error: Error string is null");
    //     return ptr::null(); // Return null pointer to indicate error
    // }
    // return errorString;
    
}
// #[no_mangle]
// pub extern "C" fn ncclCommGetAsyncError(
//     comm: NcclCommT,
//     async_error: *mut NcclResultT,
// ) -> usize {
//     return unsafe {
//         syscall3(SYS_PROXY, ProxyCommand::NcclCommGetAsyncError as usize, comm as usize, async_error as usize)
//     };
// }

// #[no_mangle]
// pub extern "C" fn ncclCommCount(
//     comm: NcclCommT,
//     count: *mut c_int,
// ) -> usize {
//     return unsafe {
//         syscall3(SYS_PROXY, ProxyCommand::NcclCommCount as usize, comm as usize, count as usize)
//     };
// }
// #[no_mangle]
// pub extern "C" fn ncclCommUserRank(
//     comm: NcclCommT,
//     rank: *mut c_int,
// ) -> usize {
//     return unsafe {
//         syscall3(SYS_PROXY, ProxyCommand::NcclCommUserRank as usize, comm as usize, rank as usize)
//     };
// }

// #[no_mangle]
// pub extern "C" fn ncclReduce(
//     send_buff: *const c_void,
//     recv_buff: *mut c_void,
//     count: usize,
//     dataType: ncclDataType_t,
//     reduceOp: ncclRedOp_t,
//     root: c_int,
//     comm: NcclCommT,
//     stream: cudaStream_t,
// ) -> usize {
//     return unsafe {
//         syscall9(SYS_PROXY, ProxyCommand::NcclReduce as usize, send_buff as usize, recv_buff as usize, count as usize, dataType as usize, reduceOp as usize, root as usize, comm as usize, stream as usize)
//     };
// }
// #[no_mangle]
// pub extern "C" fn ncclBcast(
//     buff: *mut c_void,
//     count: usize,
//     dataType: ncclDataType_t,
//     root: c_int,
//     comm: NcclCommT,
//     stream: cudaStream_t,
// ) -> usize {
//     return unsafe {
//         syscall7(SYS_PROXY, ProxyCommand::NcclBcast as usize, buff as usize, count as usize, dataType as usize, root as usize, comm as usize, stream as usize)
//     };
// }
// #[no_mangle]
// pub extern "C" fn ncclReduceScatter(
//     send_buff: *const c_void,
//     recv_buff: *mut c_void,
//     recv_count: usize,
//     dataType: ncclDataType_t,
//     reduceOp: ncclRedOp_t,
//     comm: NcclCommT,
//     stream: cudaStream_t,
// ) -> usize {
//     return unsafe {
//         syscall8(SYS_PROXY, ProxyCommand::NcclReduceScatter as usize, send_buff as usize, recv_buff as usize, recv_count as usize, dataType as usize, reduceOp as usize, comm as usize, stream as usize)
//     };
// }
// #[no_mangle]
// pub extern "C" fn ncclAllGather(
//     send_buff: *const c_void,
//     recv_buff: *mut c_void,
//     send_count: usize,
//     dataType: ncclDataType_t,
//     comm: NcclCommT,
//     stream: cudaStream_t,
// ) -> usize {
//     return unsafe {
//         syscall7(SYS_PROXY, ProxyCommand::NcclAllGather as usize, send_buff as usize, recv_buff as usize, send_count as usize, dataType as usize, comm as usize, stream as usize)
//     };
// }
// #[no_mangle]
// pub extern "C" fn ncclSend(
//     send_buff: *const c_void,
//     count: usize,
//     dataType: ncclDataType_t,
//     peer: c_int,
//     comm: NcclCommT,
//     stream: cudaStream_t,
// ) -> usize {
//     return unsafe {
//         syscall7(SYS_PROXY, ProxyCommand::NcclSend as usize, send_buff as usize, count as usize, dataType as usize, peer as usize, comm as usize, stream as usize)
//     };
// }

// device management
#[no_mangle]
pub extern "C" fn cudaChooseDevice(device: *mut c_int, prop: *const cudaDeviceProp) -> usize {
    //println!("Hijacked cudaChooseDevice");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaChooseDevice as usize, device as *mut _ as usize, prop as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> usize {
    //println!("Hijacked cudaDeviceGetAttribute");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CudaDeviceGetAttribute as usize, value as *mut _ as usize, attr as usize, device as usize);
}


#[no_mangle]
pub extern "C" fn cudaDeviceGetByPCIBusId(device: *mut c_int, pciBusId: *const c_char) -> usize {
    //println!("Hijacked cudaDeviceGetByPCIBusId");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetByPCIBusId as usize, device as *mut _ as usize, pciBusId as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> usize {
    //println!("Hijacked cudaDeviceGetCacheConfig");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaDeviceGetCacheConfig as usize, pCacheConfig as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> usize {
    //println!("Hijacked cudaDeviceGetLimit");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetLimit as usize, pValue as *mut _ as usize, limit as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetP2PAttribute(
    value: *mut c_int,
    attr: cudaDeviceP2PAttr,
    srcDevice: c_int,
    dstDevice: c_int,
) -> usize {
    //println!("Hijacked cudaDeviceGetP2PAttribute");
    return cudaSyscall5(SYS_PROXY,ProxyCommand::CudaDeviceGetP2PAttribute as usize, value as *mut _ as usize,
            attr as usize, srcDevice as usize, dstDevice as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, device: c_int) -> usize {
    //println!("Hijacked cudaDeviceGetPCIBusId");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CudaDeviceGetPCIBusId as usize, pciBusId as *const _ as usize, len as usize, device as usize);
}

pub extern "C" fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> usize {
    //println!("Hijacked cudaDeviceGetSharedMemConfig");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaDeviceGetSharedMemConfig as usize, pConfig as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> usize {
    //println!("Hijacked cudaDeviceGetStreamPriorityRange");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetStreamPriorityRange as usize,
            leastPriority as *mut _ as usize, greatestPriority as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceReset() -> usize {
    //println!("Hijacked cudaDeviceReset()");
    return cudaSyscall1(SYS_PROXY, ProxyCommand::CudaDeviceReset as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> usize {
    //println!("Hijacked cudaDeviceSetCacheConfig");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaDeviceSetCacheConfig as usize, cacheConfig as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceSetLimit(limit: cudaLimit, value: u64) -> usize {
    //println!("Hijacked cudaDeviceSetLimit");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaDeviceSetLimit as usize, limit as usize, value as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> usize {
    //println!("Hijacked cudaDeviceSetLimit");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaDeviceSetSharedMemConfig as usize, config as usize);

}

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> usize {
    //println!("Hijacked1 cudaSetDevice");
    CURRENT_GPU_ID.store(device, std::sync::atomic::Ordering::SeqCst);
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaSetDevice as usize, device as usize);
}

#[no_mangle]
pub extern "C" fn cudaSetDeviceFlags(flags: c_uint) -> usize {
    //println!("Hijacked cudaSetDeviceFlags");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaSetDeviceFlags as usize, flags as usize);
}

#[no_mangle]
pub extern "C" fn cudaSetValidDevices(device_arr: *mut c_int, len: c_int) -> usize{
    //println!("Hijacked cudaSetValidDevices");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaSetValidDevices as usize, device_arr as *mut _ as usize, len as usize);
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> usize {
    //println!("Hijacked cudaDeviceSynchronize()");
    return cudaSyscall1(SYS_PROXY, ProxyCommand::CudaDeviceSynchronize as usize);
}

#[no_mangle]
pub extern "C" fn cudaGetDevice(device: *mut c_int) -> usize {
    //println!("Hijacked cudaGetDevice");
    // return unsafe {
    //     cudaSyscall2(SYS_PROXY, ProxyCommand::CudaGetDevice as usize, device as *mut _ as usize)
    // };
    let devId = CURRENT_GPU_ID.load(std::sync::atomic::Ordering::Relaxed);
    unsafe { *device = devId };
    return 0;
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> usize {
    //println!("Hijacked cudaGetDeviceCount");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaGetDeviceCount as usize, count as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceFlags(flags: *mut c_uint) -> usize{
    //println!("Hijacked cudaGetDeviceFlags");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaGetDeviceFlags as usize, flags as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceProperties(prop: u64, device: c_int) -> usize {
    //println!("Hijacked cudaGetDeviceProperties");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaGetDeviceProperties as usize, prop as usize, device as usize);
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceProperties_v2(prop: u64, device: c_int) -> usize {
    //println!("Hijacked cudaGetDeviceProperties_v2");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaGetDeviceProperties as usize, prop as usize, device as usize);
}

#[no_mangle]
pub extern "C" fn cudaGetErrorString(error: cudaError_t) -> *const c_char {
    //println!("Hijacked cudaGetErrorString");
    let mut errorString:[i8; 128] = [1; 128];
    cudaSyscall3(SYS_PROXY,ProxyCommand::CudaGetErrorString as usize, error as usize, &mut errorString as *mut _ as usize);
    let cStr = unsafe { std::ffi::CStr::from_ptr(&errorString as *const c_char) };
    let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
    let ptr = errorStr.to_string().as_ptr() as *const i8;
    return ptr;
}

#[no_mangle]
pub extern "C" fn cudaGetErrorName(error: cudaError_t) -> *const c_char{
     //println!("Hijacked cudaGetErrorName");
    let mut errorName:[i8; 128] = [1; 128];
    cudaSyscall3(SYS_PROXY,ProxyCommand::CudaGetErrorName as usize, error as usize, &mut errorName as *mut _ as usize);
    let cStr = unsafe { std::ffi::CStr::from_ptr(&errorName as *const c_char) };
    let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
    let ptr = errorStr.to_string().as_ptr() as *const i8;
    return ptr;
}

#[no_mangle]
pub extern "C" fn cudaGetLastError() -> usize {
    //println!("Hijacked cudaGetLastError");
    return LAST_ERROR.load(std::sync::atomic::Ordering::Relaxed) as usize;
}

#[no_mangle]
pub extern "C" fn cudaPeekAtLastError() -> usize {
    //println!("Hijacked cudaPeekAtLastError");
    return cudaSyscall1(SYS_PROXY, ProxyCommand::CudaPeekAtLastError as usize);
}

// we assume all cuda libraries locate in /usr/local/cuda/compat, if its not the case, need modify this
pub fn findPtxJitCompilerLibrary(path: &mut String) -> std::io::Result<()> {
    let libPath = "/usr/local/cuda/compat/libnvidia-ptxjitcompiler.so.1";
    if std::path::Path::new(&libPath).exists() {
        let res = std::fs::canonicalize(libPath)?;
        path.push_str(res.into_os_string().to_str().unwrap());
    } else {
        let res = std::fs::canonicalize("/usr/local/cuda/compat/lib.real/libnvidia-ptxjitcompiler.so.1")?;
        path.push_str(res.into_os_string().to_str().unwrap());
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinary(fatCubin: &FatHeader) -> *mut *mut c_void {
    //println!("Hijacked __cudaRegisterFatBinary");
    let mut ptxlibPath: String = "".to_string();
    findPtxJitCompilerLibrary(&mut ptxlibPath).unwrap();

    let len = fatCubin.text.header_size as usize + fatCubin.text.size as usize;
    //let mut result: *mut *mut c_void = ptr::null_mut();
    let result = unsafe { libc::calloc(1, 0x58) as *mut *mut c_void };
    if result.is_null() {
        panic!("CUDA register an atexit handler for fatbin cleanup, but is failed!");
    }
    // println!("param 4 is {:x}", &ptxlibPath as *const _ as usize);
    cudaSyscall5(SYS_PROXY, ProxyCommand::CudaRegisterFatBinary as usize, len, fatCubin.text as *const _ as usize, 
            result as usize, &(ptxlibPath.as_bytes()[0]) as *const _ as usize);
    return result;
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle: u64) {
    //println!("Hijacked __cudaUnregisterFatBinary()");
    cudaSyscall2(SYS_PROXY, ProxyCommand::CudaUnregisterFatBinary as usize, fatCubinHandle as usize);
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(_fatCubinHandle: u64) {
    //println!("Hijacked __cudaRegisterFatBinaryEnd");
    //panic!("TODO: __cudaRegisterFatBinaryEnd not yet implemented");
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFunction(
    fatCubinHandle: u64,
    hostFun: u64,
    deviceFun: u64,  //same thing as deviceName
    deviceName: u64,
    thread_limit: i32,
    tid: u64,
    bid: u64,
    bDim: u64,
    gDim: u64,
    wSize: usize
)  {
    //println!("Hijacked __cudaRegisterFunction");
        let info = RegisterFunctionInfo {
        fatCubinHandle: fatCubinHandle,
        hostFun: hostFun,
        deviceFun: deviceFun,
        deviceName: deviceName,
        thread_limit: thread_limit,
        tid: tid,
        bid: bid,
        bDim: bDim,
        gDim: gDim,
        wSize: wSize,
    };
    cudaSyscall2(SYS_PROXY, ProxyCommand::CudaRegisterFunction as usize, &info as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn __cudaRegisterVar(
    fatCubinHandle: u64,
    hostVar: u64,       // char pointer
    deviceAddress: u64, // char pointer
    deviceName: u64,    // char pointer
    ext: i32,
    size: usize,
    constant: i32,
    global: i32,
) {
    //println!("Hijacked __cudaRegisterVar");
    let info = RegisterVarInfo {
        fatCubinHandle: fatCubinHandle,
        hostVar: hostVar,
        deviceAddress: deviceAddress,
        deviceName: deviceName,
        ext: ext,
        size: size,
        constant: constant,
        global: global,
    };
    // println!("RegisterVarInfo {:x?}", info);
    cudaSyscall2(SYS_PROXY, ProxyCommand::CudaRegisterVar as usize, &info as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    func: u64,
    gridDim: Qdim3,
    blockDim: Qdim3,
    args: u64,
    sharedMem: usize,
    stream: u64,
) {
    //println!("Hijacked cudaLaunchKernel");
    let info = LaunchKernelInfo {
        func: func,
        gridDim: gridDim,
        blockDim: blockDim,
        args: args,
        sharedMem: sharedMem,
        stream: stream,
    };
    cudaSyscall2(SYS_PROXY, ProxyCommand::CudaLaunchKernel as usize, &info as *const _ as usize);
}

// #[no_mangle]
// pub extern "C" fn cudaLaunchCooperativeKernel(
//     func: u64,
//     gridDim: Qdim3,
//     blockDim: Qdim3,
//     args: u64,
//     sharedMem: u64,
//     stream: u64,
// ) -> usize {
//     //println!("Hijacked cudaLaunchCooperativeKernel");
//     // let info = LaunchCooperativeKernelInfo {
//     //     func: func,
//     //     gridDim: gridDim,
//     //     blockDim: blockDim,
//     //     args: args,
//     //     sharedMem: sharedMem,
//     //     stream: stream,
//     // };
//     panic!("todo cudaLaunchCooperativeKernel")
// }

// #[no_mangle]
// pub extern "C" fn cudaLaunchCooperativeKernelMultiDevice(
//     launchParamsList: *mut cudaLaunchParams,
//     numDevices: c_uint,
//     flags: c_uint
// ) -> usize {
//     //println!("Hijacked cudaLaunchCooperativeKernelMultiDevice");
//     panic!("todo cudaLaunchCooperativeKernelMultiDevice")
// }

// #[no_mangle]
// pub extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//     numBlocks: *mut c_int, 
//     func: u64, 
//     blockSize: c_int, 
//     dynamicSMemSize:u64
// ) -> usize {
//     //println!("Hijacked cudaOccupancyMaxActiveBlocksPerMultiprocessor");
//     panic!("TODO: cudaOccupancyMaxActiveBlocksPerMultiprocessor");
//     // return unsafe {
//     //   cudaSyscall5(SYS_PROXY, ProxyCommand::cudaOccupancyMaxActiveBlocksPerMultiprocessor as usize, 
//     //     numBlocks as *mut _ as usize, func as usize, blockSize as usize, dynamicSMemSize as usize)
//     // };
// }

#[no_mangle]
pub extern "C" fn cudaHostAlloc(dev_ptr: *mut *mut c_void, size: usize, flags: u32) -> usize {
    //println!("Hijacked cudaMalloc");
    let ret = cudaSyscall4(
        SYS_PROXY,
        ProxyCommand::CudaHostAlloc as usize,
        dev_ptr as *const _ as usize,
        size,
        flags as usize,
    );
    //unsafe { println!("cudaHostAlloc ptr{:x}, size: {}, flags {:x}", *(dev_ptr as *mut _ as *mut u64) as u64, size, flags); }
    return ret;
}

#[no_mangle]
pub extern "C" fn cudaFreeHost(dev_ptr: *mut c_void) -> usize {
    //println!("Hijacked cudaFreeHost");
    // println!("cudaFreeHost ptr: {:x}", dev_ptr as *mut _ as u64);
    return cudaSyscall2(
        SYS_PROXY,
        ProxyCommand::CudaFreeHost as usize,
        dev_ptr as *const _ as usize,
    );
}

#[no_mangle]
pub extern "C" fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> usize {
    //println!("Hijacked cudaMalloc");
    let ret = cudaSyscall3(SYS_PROXY,ProxyCommand::CudaMalloc as usize, dev_ptr as *const _ as usize, size);
    //unsafe { println!("malloc ptr{:x}, size: {}", *(dev_ptr as *mut _ as *mut u64) as u64, size); }
    return ret;
}

#[no_mangle]
pub extern "C" fn cudaFree(dev_ptr: *mut c_void) -> usize {
    //println!("Hijacked cudaFree");
    // println!("cudaFree ptr: {:x}", dev_ptr as *mut _ as u64);
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaFree as usize, dev_ptr as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> usize {
    //println!("Hijacked cudaMemcpy");
    if kind == cudaMemcpyKind::cudaMemcpyHostToHost {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
        }
        return 0;
    }
    return cudaSyscall5(SYS_PROXY, ProxyCommand::CudaMemcpy as usize, dst as *const _ as usize, src as usize, count as usize, kind as usize);
}

#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> usize {
    //println!("Hijacked cudaMemcpyAsync");
    //println!("memcpy from {:x} to {:x}, with size {} and stream {}",src as *const _ as u64, dst as *mut _ as u64, count, stream as u64);
    if kind == cudaMemcpyKind::cudaMemcpyHostToHost {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, count);
        }
        return 0;
    }
    let info = cudaMemcpyAsyncInfo {
        dst: dst as u64,
        src: src as u64,
        count: count,
        kind: kind as u32,
        stream: stream as u64,
    };
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaMemcpyAsync as usize, &info as *const _ as usize);
}

// Stream Management API
#[no_mangle]
pub extern "C" fn cudaStreamSynchronize(stream: cudaStream_t) -> usize {
    //println!("Hijacked cudaStreamSynchronize");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaStreamSynchronize as usize, stream as usize);
}

// create an asynchronous stream
#[no_mangle]
pub extern "C" fn cudaStreamCreate(pStream: *mut cudaStream_t) -> usize {
    //println!("Hijacked cudaStreamCreate, stream address: {:x}");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaStreamCreate as usize, pStream as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> usize {
    //println!("Hijacked cudaStreamCreateWithFlags");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaStreamCreateWithFlags as usize, pStream as *const _ as usize, flags as usize);
}

#[no_mangle]
pub extern "C" fn cudaStreamCreateWithPriority(pStream: *mut cudaStream_t, flags: c_uint, priority: c_int) -> usize {
    //println!("Hijacked cudaStreamCreateWithPriority");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CudaStreamCreateWithPriority as usize, 
            pStream as *const _ as usize, flags as usize, priority as usize);
}

// Destroys and cleans up an asynchronous stream.
#[no_mangle]
pub extern "C" fn cudaStreamDestroy(stream: cudaStream_t) -> usize {
    //println!("Hijacked cudaStreamDestroy");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaStreamDestroy as usize, stream as usize);
}

#[no_mangle]
pub extern "C" fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut c_uint) -> usize {
    //println!("Hijacked cudaStreamGetFlags");s
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaStreamGetFlags as usize, hStream as usize, flags as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut c_int) -> usize{
    //println!("Hijacked cudaStreamGetPriority");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaStreamGetPriority as usize, hStream as usize, priority as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> usize {
    //println!("Hijacked cudaStreamIsCapturing");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaStreamIsCapturing as usize, stream as usize, pCaptureStatus as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaStreamQuery(stream: cudaStream_t) -> usize {
    //println!( "Hijacked cudaStreamQuery");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaStreamQuery as usize, stream as usize);
}

#[no_mangle]
pub extern "C" fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> usize {
    //println!("Hijacked cudaStreamWaitEvent()");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CudaStreamWaitEvent as usize, stream as usize, event as usize, flags as usize);
}

#[no_mangle]
pub extern "C" fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> usize{
    //println!("Hijacked cudaThreadExchangeStreamCaptureMode");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaThreadExchangeStreamCaptureMode as usize, mode as *mut _ as usize);

}

#[no_mangle]
pub extern "C" fn cudaEventCreate(event: *mut cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventCreate()");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaEventCreate as usize, event as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> usize{
    //println!("Hijacked cudaEventCreateWithFlags()");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaEventCreateWithFlags as usize, event as *mut _ as usize, flags as usize);
}

#[no_mangle]
pub extern "C" fn cudaEventDestroy(event: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventDestroy()");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaEventDestroy as usize, event as usize);
}

#[no_mangle]
pub extern "C" fn cudaEventElapsedTime(ms: *mut c_float, start: cudaEvent_t, end: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventElapsedTime()");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CudaEventElapsedTime as usize, ms as *mut _ as usize , start as usize, end as usize);
}

#[no_mangle]
pub extern "C" fn cudaEventQuery(event: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventQuery");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaEventQuery as usize, event as usize);
}

#[no_mangle]
pub extern "C" fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> usize{
    //println!("Hijacked cudaEventRecord()");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaEventRecord as usize, event as usize, stream as usize);
}

#[no_mangle]
pub extern "C" fn cudaEventSynchronize(event: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventSynchronize()");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaEventSynchronize as usize, event as usize);
}

#[no_mangle]
pub extern "C" fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: u64) -> usize {
    //println!("Hijacked cudaFuncGetAttributes");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaFuncGetAttributes as usize, attr as *mut _ as usize, func as usize);
}

#[no_mangle]
pub extern "C" fn cudaFuncSetAttribute(func: u64, attr:cudaFuncAttribute, value: c_int) -> usize{
    //println!("Hijacked cudaFuncSetAttribute");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CudaFuncSetAttribute as usize, func as usize, attr as usize, value as usize);
}

#[no_mangle]
pub extern "C" fn cudaFuncSetCacheConfig(func: u64,cacheConfig: cudaFuncCache) -> usize{
    //println!("Hijacked cudaFuncSetCacheConfig");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaFuncSetCacheConfig as usize, func as usize, cacheConfig as usize);
}

#[no_mangle]
pub extern "C" fn cudaFuncSetSharedMemConfig(func: u64, config: cudaSharedMemConfig) -> usize{
    //println!("Hijacked cudaFuncSetSharedMemConfig");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CudaFuncSetCacheConfig as usize, func as usize, config as usize);
}

#[no_mangle]
pub extern "C" fn cuModuleGetLoadingMode(
    mode: *mut CumoduleLoadingModeEnum
) -> usize {
    //println!("Hijacked cuModuleGetLoadingMode");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CuModuleGetLoadingMode as usize, mode as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn cuInit(flags: c_uint) -> usize {
    //println!("Hijacked cuInit");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CuInit as usize, flags as usize);
}

// #[no_mangle]
// pub extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> usize {
//     //println!("Hijacked cuDeviceGet");
//     panic!("TODO: cuDeviceGet not yet implemented");
// }

// #[no_mangle]
// pub extern "C" fn cuDeviceGetCount(count: *mut c_int) -> usize {
//     //println!("Hijacked cuDeviceGetCount");
//     panic!("TODO: cuDeviceGetCount not yet implemented");
// }

// #[no_mangle]
// pub extern "C" fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> usize {
//     //println!("Hijacked cuDeviceGetName");
//     panic!("TODO: cuDeviceGetName not yet implemented");
// }

// #[no_mangle]
// pub extern "C" fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> usize {
//     //println!("Hijacked cuDeviceGetUuid");
//     panic!("TODO: cuDeviceGetUuid not yet implemented");
// }

// #[no_mangle]
// pub extern "C" fn cuDeviceGetAttribute(
//     pi: *mut c_int,
//     attrub: CUdevice_attribute,
//     dev: CUdevice,
// ) -> usize {
//     //println!("Hijacked cuDeviceGetAttribute");
//     panic!("TODO: cuDeviceGetAttribute not yet implemented");
// }

// #[no_mangle]
// pub extern "C" fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> usize {
//     //println!("Hijacked cuDeviceGetAttribute");
//     panic!("TODO: cuDeviceGetProperties not yet implemented");
// }

// #[no_mangle]
// pub extern "C" fn cuDeviceComputeCapability(
//     major: *mut c_int,
//     minor: *mut c_int,
//     dev: CUdevice,
// ) -> usize {
//     //println!("Hijacked cuDeviceComputeCapability");
//     panic!("TODO: cuDeviceComputeCapability not yet implemented");
// }

#[no_mangle]
pub extern "C" fn cuDevicePrimaryCtxGetState(
    dev: CUdevice,
    flags: *mut c_uint,
    active: *mut c_int,
) -> usize {
    //println!("Hijacked cuDevicePrimaryCtxGetState");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CuDevicePrimaryCtxGetState as usize, 
            dev as usize, flags as *mut _ as usize, active as *mut _ as usize);
}

//NVML
#[no_mangle]
pub extern "C" fn nvmlInitWithFlags(flags: u32) -> u32 {
    //println!("Hijacked nvmlInitWithFlags");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::NvmlInitWithFlags as usize, flags as usize) as u32;
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> usize {
    //println!("Hijacked nvmlDeviceGetCount_v2()");
    // Workaround for pytorch expecting nvmlDeviceGetCount and cudaGetDeviceCount to be the same
    return cudaSyscall2(SYS_PROXY, ProxyCommand::NvmlDeviceGetCountV2 as usize, deviceCount as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn nvmlInit() -> usize {
    //println!("Hijacked nvmlInit()");
    return cudaSyscall1(SYS_PROXY, ProxyCommand::NvmlInit as usize);
}

#[no_mangle]
pub extern "C" fn nvmlInit_v2() -> usize {
    //println!("Hijacked nvmlDeviceGetCount_v2()");
    return cudaSyscall1(SYS_PROXY, ProxyCommand::NvmlInitV2 as usize);
}

#[no_mangle]
pub extern "C" fn nvmlShutdown() -> usize {
    //println!("Hijacked nvmlShutdown()");
    return cudaSyscall1( SYS_PROXY, ProxyCommand::NvmlShutdown as usize);
}

#[no_mangle]
pub extern "C" fn cublasCreate_v2(handle: *mut cublasHandle_t) -> usize{
    //println!("Hijacked cublasCreate_v2()");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CublasCreateV2 as usize, handle as *mut _ as usize);
}

#[no_mangle]
pub extern "C" fn cublasDestroy_v2(handle: cublasHandle_t) -> usize{
    //println!("Hijacked cublasDestroy_v2");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CublasDestroyV2 as usize, handle as usize);
}

#[no_mangle]
pub extern "C" fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> usize{
    //println!("Hijacked cublasSetStream");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CublasSetStreamV2 as usize, handle as usize, streamId as usize);
}

#[no_mangle]
pub extern "C" fn cublasSetWorkspace_v2(handle: cublasHandle_t, workspace: *mut c_void, workspaceSizeInByte: u64) -> usize{
    //println!("Hijacked cublasSetWorkspace_v2");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CublasSetWorkspaceV2 as usize, handle as usize, 
            workspace as *mut _ as usize, workspaceSizeInByte as usize);
}

#[no_mangle]
pub extern "C" fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> usize {
    //println!("Hijacked cublasSetMathMode");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CublasSetMathMode as usize, handle as usize, mode as usize);
}

#[no_mangle]
pub extern "C" fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> usize {
    //println!("Hijacked cublasGetMathMode");
    return cudaSyscall3(SYS_PROXY, ProxyCommand::CublasGetMathMode as usize, handle as usize, mode as usize);
}

#[no_mangle]
pub extern "C" fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: *const f32,
    lda: c_int,
    B: *const f32,
    ldb: c_int,
    beta: *const f32,
    C: *mut f32,
    ldc: c_int,
    ) -> usize {
        //println!("Hijacked cublasSgemm_v2");
        let info = CublasSgemmV2Info {
            handle: handle as u64,
            transa: transa as u32,
            transb: transb as u32,
            m: m,
            n: n,
            k: k,
            alpha: alpha,
            A: A,
            lda: lda,
            B: B,
            ldb: ldb,
            beta:beta,
            C: C,
            ldc: ldc,
        };
        //println!("SgemmStridedBatchedInfo {:x?}", info);
        return cudaSyscall4(SYS_PROXY, ProxyCommand::CublasSgemmV2 as usize, &info as *const _ as usize, 
                alpha as *const _ as usize, beta as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: u64,
    A: u64,
    Atype: cudaDataType_t,
    lda: c_int,
    B: u64,
    Btype: cudaDataType_t,
    ldb: c_int,
    beta: u64,
    C: u64,
    Ctype: cudaDataType_t,
    ldc: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t) -> usize {
        let info: GemmExInfo = GemmExInfo {
            handle: handle as u64,
            transa: transa as u32,
            transb: transb as u32,
            m: m as i32,
            n: n as i32,
            k: k as i32,
            alpha: alpha,
            A: A,
            Atype: Atype as u32,
            lda: lda as i32,
            B: B,
            Btype: Btype as u32,
            ldb: ldb as i32,
            beta: beta,
            C: C,
            Ctype: Ctype as u32,
            ldc: ldc as i32,
            computeType: computeType as u32,
            algo: algo as u32,
        };
    //  println!("SgemmStridedBatchedInfo {:x?}", info);
    return cudaSyscall4(SYS_PROXY,ProxyCommand::CublasGemmEx as usize,
            &info as *const _ as usize, alpha as usize, beta as usize);
}

#[no_mangle]
pub extern "C" fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: u64,
    A: u64,
    Atype: cudaDataType_t,
    lda: i32,
    strideA: i64,
    B: u64,
    Btype: cudaDataType_t,
    ldb: i32,
    strideB: i64,
    beta: u64,
    C: u64,
    Ctype: cudaDataType_t,
    ldc: i32,
    strideC: i64,
    batchCount: i32,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t) -> usize {
        let info: GemmStridedBatchedExInfo = GemmStridedBatchedExInfo {
            handle: handle as u64,
            transa: transa as u32,
            transb: transb as u32,
            m: m,
            n: n,
            k: k,
            alpha: alpha,
            A: A,
            Atype: Atype as u32,
            lda: lda,
            strideA: strideA,
            B: B,
            Btype: Btype as u32,
            ldb: ldb,
            strideB: strideB,
            beta: beta,
            C: C,
            Ctype: Ctype as u32,
            ldc: ldc,
            strideC: strideC,
            batchCount: batchCount,
            computeType: computeType as u32,
            algo: algo as u32,
        };
    return cudaSyscall4(SYS_PROXY,ProxyCommand::CublasGemmStridedBatchedEx as usize,
            &info as *const _ as usize, alpha as usize, beta as usize);
}

// #[no_mangle]
// pub extern "C" fn cublasLtCreate(lighthandle: *mut cublasLtHandle_t) -> usize {
//     //println!("cublasLtCreate , but not yet implemented. ");
//     panic!("TODO: cublasLtCreate");
// }

#[no_mangle]
pub extern "C" fn cudaMemset(devPtr: *const c_void, value: c_int, count: usize) -> usize {
    //println!("cudaMemset");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CudaMemset as usize, devPtr as usize, value as usize, count as usize);
}

#[no_mangle]
pub extern "C" fn cudaMemsetAsync(devPtr: *const c_void, value: c_int, count: usize, stream: cudaStream_t) -> usize {
    //println!("cudaMemsetAsync");
    return cudaSyscall5(SYS_PROXY, ProxyCommand::CudaMemsetAsync as usize, devPtr as usize, value as usize, count as usize, stream as usize);
}

#[no_mangle]
pub extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut c_int,
    func: *const c_void, blockSize: c_int, dynamicSMemSize: usize, flags: c_uint) -> usize {
    //println!("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    let info = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInfo {
        numBlocks: numBlocks as u64,
        func: func as u64,
        blockSize: blockSize as u32,
        dynamicSMemSize: dynamicSMemSize,
        flags: flags as u32,
    };
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags as usize,
        &info as *const _ as usize);
}

// Following function is a inline function
// #[no_mangle]
// pub extern "C" fn cudaOccupancyMaxPotentialBlockSize(minGridSize: *mut c_int,
//     blockSize: *mut c_int, func: CUfunction, dynamicSMemSize: usize, blockSizeLimit: c_int) -> usize {
//     //println!("cudaOccupancyMaxPotentialBlockSize");
//     panic!();
//     return 1;
// }

#[no_mangle]
pub extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> usize {
    //println!("cuCtxGetCurrent");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CuCtxGetCurrent as usize, pctx as usize);
}

#[no_mangle]
pub extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void ) -> usize {
    //println!("hijeck cuModuleLoadData");
    if image.is_null() {
        //println!("image is NULL!");
        return cuda_driver_sys::cudaError_enum::CUDA_ERROR_INVALID_IMAGE as usize;
    }
    let ehdr = image as *const _ as *const Elf64_Ehdr;
    unsafe {
        if  (*ehdr).e_ident[EI_MAG0 as usize] != ELFMAG0 as u8 ||
            (*ehdr).e_ident[EI_MAG1 as usize] != ELFMAG1 as u8 ||
            (*ehdr).e_ident[EI_MAG2 as usize] != ELFMAG2 as u8 ||
            (*ehdr).e_ident[EI_MAG3 as usize] != ELFMAG3 as u8 {
            //println!("invalid image!");
            return cuda_driver_sys::cudaError_enum::CUDA_ERROR_INVALID_IMAGE as usize;
        }
    }
    let image_len:u64 = unsafe { (*ehdr).e_shoff + (*ehdr).e_shnum as u64 * (*ehdr).e_shentsize as u64 };
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CuModuleLoadData as usize, module as usize, image as usize, image_len as usize);
}

#[no_mangle]
pub extern "C" fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *mut c_char ) -> usize {
    //println!("hijeck cuModuleGetFunction");
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CuModuleGetFunction as usize, hfunc as usize, hmod as usize, name as usize);
}

#[no_mangle]
pub extern "C" fn cuModuleUnload(hmod: CUmodule) -> usize {
    //println!("hijeck cuModuleUnload");
    return cudaSyscall2(SYS_PROXY, ProxyCommand::CuModuleUnload as usize, hmod as usize);
}

#[no_mangle]
pub extern "C" fn cuLaunchKernel(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, 
    blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, hStream: CUstream,
    kernelParams: *mut *mut c_void, extra: *mut *mut c_void) -> usize {
        //println!("hijeck cuLaunchKernel");
        let info = CuLaunchKernelInfo {
            f: f as u64,
            gridDimX: gridDimX, 
            gridDimY: gridDimY, 
            gridDimZ: gridDimZ, 
            blockDimX: blockDimX, 
            blockDimY: blockDimY, 
            blockDimZ: blockDimZ, 
            sharedMemBytes: sharedMemBytes, 
            hStream: hStream as u64, 
            kernelParams: kernelParams as u64, 
            extra: extra as u64
        };
        return cudaSyscall2(SYS_PROXY, ProxyCommand::CuLaunchKernel as usize, &info as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: *const f32,
    lda: c_int,
    strideA: c_longlong,
    B: *const f32,
    ldb: c_int,
    strideB: c_longlong,
    beta: *const f32,
    C: *mut f32,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
    ) -> usize {
        //println!("Hijacked cublasSgemmStridedBatched");
        let info = SgemmStridedBatchedInfo {
            handle: handle as u64,
            transa: transa as u32,
            transb: transb as u32,
            m: m,
            n: n,
            k: k,
            alpha: alpha,
            A: A,
            lda: lda,
            strideA: strideA,
            B: B,
            ldb: ldb,
            strideB:strideB,
            beta:beta,
            C: C,
            ldc: ldc,
            strideC: strideC,
            batchCount: batchCount,
        };
    //  println!("SgemmStridedBatchedInfo {:x?}", info);
    return cudaSyscall4(SYS_PROXY,ProxyCommand::CublasSgemmStridedBatched as usize,
            &info as *const _ as usize, alpha as *const _ as usize, beta as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn cublasLtMatmul(
    lightHandle: cublasLtHandle_t,
    computeDesc: cublasLtMatmulDesc_t,
    alpha: *const c_void,
    A: *const c_void,
    Adesc: cublasLtMatrixLayout_t,
    B: *const c_void,
    Bdesc: cublasLtMatrixLayout_t,
    beta: *const c_void,
    C: *const c_void,
    Cdesc: cublasLtMatrixLayout_t,
    D: *mut c_void,
    Ddesc: cublasLtMatrixLayout_t,
    algo: *const cublasLtMatmulAlgo_t,
    workspace: *mut c_void,
    workspaceSizeInBytes: usize,
    stream: cudaStream_t,
) -> usize {
    //println!("Hijacked cublasSgemmStridedBatched()");
    let info = CublasLtMatmulInfo {
        lightHandle:lightHandle as u64,
        computeDesc: computeDesc as u64,
        alpha: alpha,
        A: A,
        Adesc: Adesc as u64,
        B: B,
        Bdesc: Bdesc as u64,
        beta: beta,
        C: C,
        Cdesc: Cdesc as u64,
        D: D,
        Ddesc: Ddesc as u64,
        algo: algo as *const CublasLtMatmulAlgoT,
        workspace: workspace,
        workspaceSizeInBytes: workspaceSizeInBytes,
        stream: stream as u64,
    };
    //println!("CublasLtMatmulInfo {:x?}", info);
    return cudaSyscall4(SYS_PROXY, ProxyCommand::CublasLtMatmul as usize,
            &info as *const _ as usize, alpha as *const _ as usize, beta as *const _ as usize);
}

#[no_mangle]
pub extern "C" fn cublasLtMatmulAlgoGetHeuristic(
    lightHandle: cublasLtHandle_t,
    operationDesc: cublasLtMatmulDesc_t,
    Adesc: cublasLtMatrixLayout_t,
    Bdesc: cublasLtMatrixLayout_t,
    Cdesc: cublasLtMatrixLayout_t,
    Ddesc: cublasLtMatrixLayout_t,
    preference: cublasLtMatmulPreference_t,
    requestedAlgoCount: ::libc::c_int,
    heuristicResultsArray: *mut cublasLtMatmulHeuristicResult_t,
    returnAlgoCount: *mut ::libc::c_int,
) -> usize {
    //println!("Hijacked cublasLtMatmulAlgoGetHeuristic");
    let info: CublasLtMatmulAlgoGetHeuristicInfo = CublasLtMatmulAlgoGetHeuristicInfo {
        lightHandle: lightHandle as u64,
        operationDesc: operationDesc as u64,
        Adesc: Adesc as u64,
        Bdesc: Bdesc as u64,
        Cdesc: Cdesc as u64,
        Ddesc: Ddesc as u64,
        preference: preference as u64,
        requestedAlgoCount: requestedAlgoCount
    };
    //println!("CublasLtMatmulAlgoGetHeuristicInfo {:x?}", info);
    return cudaSyscall4(SYS_PROXY,ProxyCommand::CublasLtMatmulAlgoGetHeuristic as usize,
            &info as *const _ as usize, heuristicResultsArray as *mut _ as usize, returnAlgoCount as *mut _ as usize);
}

#[derive(Debug)]
#[repr(C)]
pub struct NvidiaRes {
    pub res: u32,
    pub lasterr: u32, 
}

impl NvidiaRes {
    pub fn FromU64(v: u64) -> Self {
        let res = (v >> 32) as u32;
        let lasterr = v as u32;
        return NvidiaRes {
            res: res,
            lasterr: lasterr
        }
    }

    pub fn ToU64(&self) -> u64 {
        return (self.res as u64) << 32 | (self.lasterr as u64)
    }
}

#[inline(always)]
fn updateLastError(ret: usize) -> usize {
    let nvidiaRes = NvidiaRes::FromU64(ret as u64);
    LAST_ERROR.store(nvidiaRes.lasterr, std::sync::atomic::Ordering::SeqCst);
    // println!("updateLastError nvidiaRes is {:?}", &nvidiaRes);
    nvidiaRes.res as usize
}

#[inline(always)]
fn cudaSyscall1(n: usize, a1: usize) -> usize {
    let ret = unsafe { syscall1(n, a1) };
    updateLastError(ret)
}

#[inline(always)]
fn cudaSyscall2(n: usize, a1: usize, a2: usize) -> usize {
    let ret = unsafe { syscall2(n, a1, a2) };
    updateLastError(ret)
}

#[inline(always)]
fn cudaSyscall3(n: usize, a1: usize, a2: usize, a3: usize) -> usize {
    let ret = unsafe { syscall3(n, a1, a2, a3) };
    updateLastError(ret)
}

#[inline(always)]
fn cudaSyscall4(n: usize, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
    let ret = unsafe { syscall4(n, a1, a2, a3, a4) };
    updateLastError(ret)
}

#[inline(always)]
fn cudaSyscall5(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize) -> usize {
    let ret = unsafe { syscall5(n, a1, a2, a3, a4, a5) };
    updateLastError(ret)
}

#[inline(always)]
fn cudaSyscall6(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize) -> usize {
    let ret = unsafe { syscall6(n, a1, a2, a3, a4, a5, a6) };
    updateLastError(ret)
}