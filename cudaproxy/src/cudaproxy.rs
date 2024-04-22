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
use rcublas_sys::{cublasHandle_t,cublasMath_t,cublasOperation_t};
use cuda11_cublasLt_sys::{cublasLtHandle_t,cublasLtMatmulDesc_t,cublasLtMatrixLayout_t,cublasLtMatmulAlgo_t,cublasLtMatmulPreference_t,cublasLtMatmulHeuristicResult_t};

use crate::proxy::*;
use crate::syscall::*;

extern crate alloc;

pub const SYS_PROXY: usize = 10003;

pub static mut DLOPEN_ORIG: Option<unsafe extern "C" fn(*const libc::c_char, libc::c_int) -> *mut libc::c_void> = None;
pub static mut DLCLOSE_ORIG: Option<unsafe extern "C" fn(*mut libc::c_void) -> libc::c_int> = None;
pub static mut DL_HANDLE: *mut libc::c_void = ptr::null_mut();

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

// device management
#[no_mangle]
pub extern "C" fn cudaChooseDevice(device: *mut c_int, prop: *const cudaDeviceProp) -> usize {
    //println!("Hijacked cudaChooseDevice");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaChooseDevice as usize, device as *mut _ as usize, prop as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> usize {
    //println!("Hijacked cudaDeviceGetAttribute");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaDeviceGetAttribute as usize, value as *mut _ as usize, attr as usize, device as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetByPCIBusId(device: *mut c_int, pciBusId: *const c_char) -> usize {
    //println!("Hijacked cudaDeviceGetByPCIBusId");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetByPCIBusId as usize, device as *mut _ as usize, pciBusId as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> usize {
    //println!("Hijacked cudaDeviceGetCacheConfig");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaDeviceGetCacheConfig as usize, pCacheConfig as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> usize {
    //println!("Hijacked cudaDeviceGetLimit");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetLimit as usize, pValue as *mut _ as usize, limit as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetP2PAttribute(
    value: *mut c_int,
    attr: cudaDeviceP2PAttr,
    srcDevice: c_int,
    dstDevice: c_int,
) -> usize {
    //println!("Hijacked cudaDeviceGetP2PAttribute");
    return unsafe {
        syscall5(SYS_PROXY,ProxyCommand::CudaDeviceGetP2PAttribute as usize, value as *mut _ as usize,
            attr as usize, srcDevice as usize, dstDevice as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, device: c_int) -> usize {
    //println!("Hijacked cudaDeviceGetPCIBusId");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaDeviceGetPCIBusId as usize, pciBusId as *const _ as usize, len as usize, device as usize)
    };
}

pub extern "C" fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> usize {
    //println!("Hijacked cudaDeviceGetSharedMemConfig");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaDeviceGetSharedMemConfig as usize, pConfig as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> usize {
    //println!("Hijacked cudaDeviceGetStreamPriorityRange");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetStreamPriorityRange as usize,
            leastPriority as *mut _ as usize, greatestPriority as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceReset() -> usize {
    //println!("Hijacked cudaDeviceReset()");
    return unsafe {
        syscall1(SYS_PROXY, ProxyCommand::CudaDeviceReset as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> usize {
    //println!("Hijacked cudaDeviceSetCacheConfig");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaDeviceSetCacheConfig as usize, cacheConfig as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceSetLimit(limit: cudaLimit, value: u64) -> usize {
    //println!("Hijacked cudaDeviceSetLimit");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaDeviceSetLimit as usize, limit as usize, value as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> usize {
    //println!("Hijacked cudaDeviceSetLimit");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaDeviceSetSharedMemConfig as usize, config as usize)
    };

}

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> usize {
    //println!("Hijacked1 cudaSetDevice");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaSetDevice as usize, device as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaSetDeviceFlags(flags: c_uint) -> usize {
    //println!("Hijacked cudaSetDeviceFlags");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaSetDeviceFlags as usize, flags as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaSetValidDevices(device_arr: *mut c_int, len: c_int) -> usize{
    //println!("Hijacked cudaSetValidDevices");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaSetValidDevices as usize, device_arr as *mut _ as usize, len as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> usize {
    //println!("Hijacked cudaDeviceSynchronize()");
    return unsafe {
        syscall1(SYS_PROXY, ProxyCommand::CudaDeviceSynchronize as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaGetDevice(device: *mut c_int) -> usize {
    //println!("Hijacked cudaGetDevice");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaGetDevice as usize, device as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> usize {
    //println!("Hijacked cudaGetDeviceCount");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaGetDeviceCount as usize, count as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceFlags(flags: *mut c_uint) -> usize{
    //println!("Hijacked cudaGetDeviceFlags");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaGetDeviceFlags as usize, flags as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceProperties(prop: u64, device: c_int) -> usize {
    //println!("Hijacked cudaGetDeviceProperties");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaGetDeviceProperties as usize, prop as usize, device as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceProperties_v2(prop: u64, device: c_int) -> usize {
    //println!("Hijacked cudaGetDeviceProperties_v2");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaGetDeviceProperties as usize, prop as usize, device as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaGetErrorString(error: cudaError_t) -> *const c_char {
    //println!("Hijacked cudaGetErrorString");
    let errorString: *const c_char = std::ptr::null();
    unsafe {
        syscall3(SYS_PROXY,ProxyCommand::CudaGetErrorString as usize, error as usize, errorString as *mut c_char as usize)
    };
    return errorString;
}

#[no_mangle]
pub extern "C" fn cudaGetErrorName(error: cudaError_t) -> *const c_char{
     //println!("Hijacked cudaGetErrorName");
    let errorName: *const c_char = std::ptr::null();
    unsafe {
        syscall3(SYS_PROXY,ProxyCommand::CudaGetErrorName as usize, error as usize, errorName as *mut c_char as usize)
    };
    return errorName;
}

#[no_mangle]
pub extern "C" fn cudaGetLastError() -> usize {
    //println!("Hijacked cudaGetLastError");
    return unsafe {
        syscall1(SYS_PROXY, ProxyCommand::CudaGetLastError as usize) 
    };
}

#[no_mangle]
pub extern "C" fn cudaPeekAtLastError() -> usize {
    //println!("Hijacked cudaPeekAtLastError");
    return unsafe { 
        syscall1(SYS_PROXY, ProxyCommand::CudaPeekAtLastError as usize) 
    };
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
    unsafe {
        syscall5(SYS_PROXY, ProxyCommand::CudaRegisterFatBinary as usize, len, fatCubin.text as *const _ as usize, 
            result as usize, &(ptxlibPath.as_bytes()[0]) as *const _ as usize);
    }
    return result;
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle: u64) {
    //println!("Hijacked __cudaUnregisterFatBinary()");
    unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaUnregisterFatBinary as usize, fatCubinHandle as usize);
    }
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
    println!("Hijacked __cudaRegisterFunction");
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
    unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaRegisterFunction as usize, &info as *const _ as usize);
    }
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
    unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaRegisterVar as usize, &info as *const _ as usize);
    }
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
    unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaLaunchKernel as usize, &info as *const _ as usize);
    }
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
//     //   syscall5(SYS_PROXY, ProxyCommand::cudaOccupancyMaxActiveBlocksPerMultiprocessor as usize, 
//     //     numBlocks as *mut _ as usize, func as usize, blockSize as usize, dynamicSMemSize as usize)
//     // };
// }

#[no_mangle]
pub extern "C" fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> usize {
    //println!("Hijacked cudaMalloc");
    return unsafe {
        syscall3(SYS_PROXY,ProxyCommand::CudaMalloc as usize, dev_ptr as *const _ as usize, size)
    };
}

#[no_mangle]
pub extern "C" fn cudaFree(dev_ptr: *mut c_void) -> usize {
    //println!("Hijacked cudaFree");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaFree as usize, dev_ptr as *const _ as usize)
    };
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
    return unsafe {
        syscall5(SYS_PROXY, ProxyCommand::CudaMemcpy as usize, dst as *const _ as usize, src as usize, count as usize, kind as usize)
    };
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
    if kind == cudaMemcpyKind::cudaMemcpyHostToHost {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, count);
        }
        return 0;
    }
    return unsafe {
        syscall6(SYS_PROXY, ProxyCommand::CudaMemcpyAsync as usize, dst as *const _ as usize,
            src as usize, count as usize, kind as usize, stream as usize)
    };
}

// Stream Management API
#[no_mangle]
pub extern "C" fn cudaStreamSynchronize(stream: cudaStream_t) -> usize {
    //println!("Hijacked cudaStreamSynchronize");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaStreamSynchronize as usize, stream as usize)
    };
}

// create an asynchronous stream
#[no_mangle]
pub extern "C" fn cudaStreamCreate(pStream: *mut cudaStream_t) -> usize {
    //println!("Hijacked cudaStreamCreate, stream address: {:x}");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaStreamCreate as usize, pStream as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> usize {
    //println!("Hijacked cudaStreamCreateWithFlags");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaStreamCreateWithFlags as usize, pStream as *const _ as usize, flags as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaStreamCreateWithPriority(pStream: *mut cudaStream_t, flags: c_uint, priority: c_int) -> usize {
    //println!("Hijacked cudaStreamCreateWithPriority");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaStreamCreateWithPriority as usize, 
            pStream as *const _ as usize, flags as usize, priority as usize)
    };
}

// Destroys and cleans up an asynchronous stream.
#[no_mangle]
pub extern "C" fn cudaStreamDestroy(stream: cudaStream_t) -> usize {
    //println!("Hijacked cudaStreamDestroy");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaStreamDestroy as usize, stream as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut c_uint) -> usize {
    //println!("Hijacked cudaStreamGetFlags");s
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaStreamGetFlags as usize, hStream as usize, flags as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut c_int) -> usize{
    //println!("Hijacked cudaStreamGetPriority");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaStreamGetPriority as usize, hStream as usize, priority as *mut _ as usize)
    }
}

#[no_mangle]
pub extern "C" fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> usize {
    //println!("Hijacked cudaStreamIsCapturing");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaStreamIsCapturing as usize, stream as usize, pCaptureStatus as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaStreamQuery(stream: cudaStream_t) -> usize {
    //println!( "Hijacked cudaStreamQuery");
    return unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CudaStreamQuery as usize, stream as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> usize {
    //println!("Hijacked cudaStreamWaitEvent()");
    return unsafe{
        syscall4(SYS_PROXY, ProxyCommand::CudaStreamWaitEvent as usize, stream as usize, event as usize, flags as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> usize{
    //println!("Hijacked cudaThreadExchangeStreamCaptureMode");
    return unsafe{
       syscall2(SYS_PROXY, ProxyCommand::CudaThreadExchangeStreamCaptureMode as usize, mode as *mut _ as usize)
    };

}

#[no_mangle]
pub extern "C" fn cudaEventCreate(event: *mut cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventCreate()");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaEventCreate as usize, event as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> usize{
    //println!("Hijacked cudaEventCreateWithFlags()");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaEventCreateWithFlags as usize, event as *mut _ as usize, flags as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaEventDestroy(event: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventDestroy()");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaEventDestroy as usize, event as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaEventElapsedTime(ms: *mut c_float, start: cudaEvent_t, end: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventElapsedTime()");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaEventElapsedTime as usize, ms as *mut _ as usize , start as usize, end as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaEventQuery(event: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventQuery");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaEventQuery as usize, event as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> usize{
    //println!("Hijacked cudaEventRecord()");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaEventRecord as usize, event as usize, stream as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaEventSynchronize(event: cudaEvent_t) -> usize{
    //println!("Hijacked cudaEventSynchronize()");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaEventSynchronize as usize, event as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: u64) -> usize {
    //println!("Hijacked cudaFuncGetAttributes");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaFuncGetAttributes as usize, attr as *mut _ as usize, func as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaFuncSetAttribute(func: u64, attr:cudaFuncAttribute, value: c_int) -> usize{
    //println!("Hijacked cudaFuncSetAttribute");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaFuncSetAttribute as usize, func as usize, attr as usize, value as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaFuncSetCacheConfig(func: u64,cacheConfig: cudaFuncCache) -> usize{
    //println!("Hijacked cudaFuncSetCacheConfig");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaFuncSetCacheConfig as usize, func as usize, cacheConfig as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaFuncSetSharedMemConfig(func: u64, config: cudaSharedMemConfig) -> usize{
    //println!("Hijacked cudaFuncSetSharedMemConfig");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaFuncSetCacheConfig as usize, func as usize, config as usize)
    };
}

#[no_mangle]
pub extern "C" fn cuModuleGetLoadingMode(
    mode: *mut CumoduleLoadingModeEnum
) -> usize {
    println!("Hijacked cuModuleGetLoadingMode");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CuModuleGetLoadingMode as usize, mode as *const _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn cuInit(flags: c_uint) -> usize {
    //println!("Hijacked cuInit");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CuInit as usize, flags as usize)
    } ;
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
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CuDevicePrimaryCtxGetState as usize, 
            dev as usize, flags as *mut _ as usize, active as *mut _ as usize)
    };
}

//NVML
#[no_mangle]
pub extern "C" fn nvmlInitWithFlags(flags: u32) -> u32 {
    //println!("Hijacked nvmlInitWithFlags");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::NvmlInitWithFlags as usize, flags as usize)
    } as u32;
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> usize {
    //println!("Hijacked nvmlDeviceGetCount_v2()");
    // Workaround for pytorch expecting nvmlDeviceGetCount and cudaGetDeviceCount to be the same
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::NvmlDeviceGetCountV2 as usize, deviceCount as *mut _ as usize)
    };
}

#[no_mangle]
pub extern "C" fn nvmlInit() -> usize {
    //println!("Hijacked nvmlInit()");
    return unsafe {
        syscall1(SYS_PROXY, ProxyCommand::NvmlInit as usize)
    };
}

#[no_mangle]
pub extern "C" fn nvmlInit_v2() -> usize {
    //println!("Hijacked nvmlDeviceGetCount_v2()");
    return unsafe {
        syscall1(SYS_PROXY, ProxyCommand::NvmlInitV2 as usize)
    };
}

#[no_mangle]
pub extern "C" fn nvmlShutdown() -> usize {
    //println!("Hijacked nvmlShutdown()");
    return unsafe {
        syscall1( SYS_PROXY, ProxyCommand::NvmlShutdown as usize)
    };
}

#[no_mangle]
pub extern "C" fn cublasCreate_v2(handle: *mut cublasHandle_t) -> usize{
    //println!("Hijacked cublasCreate_v2()");
    return unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CublasCreateV2 as usize, handle as *mut _ as usize)
    }
}

#[no_mangle]
pub extern "C" fn cublasDestroy_v2(handle: cublasHandle_t) -> usize{
    //println!("Hijacked cublasDestroy_v2");
    return unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CublasDestroyV2 as usize, handle as usize)
    }
}

#[no_mangle]
pub extern "C" fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> usize{
    //println!("Hijacked cublasSetStream");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CublasSetStreamV2 as usize, handle as usize, streamId as usize)
    }
}

#[no_mangle]
pub extern "C" fn cublasSetWorkspace_v2(handle: cublasHandle_t, workspace: *mut c_void, workspaceSizeInByte: u64) -> usize{
    //println!("Hijacked cublasSetWorkspace_v2");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CublasSetWorkspaceV2 as usize, handle as usize, 
            workspace as *mut _ as usize, workspaceSizeInByte as usize)
    }
}

#[no_mangle]
pub extern "C" fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> usize {
    //println!("Hijacked cublasSetMathMode");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CublasSetMathMode as usize, handle as usize, mode as usize)
    };
}

#[no_mangle]
pub extern "C" fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> usize {
    //println!("Hijacked cublasGetMathMode");
    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CublasGetMathMode as usize, handle as usize, mode as usize)
    };
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
        return unsafe {
            syscall4(SYS_PROXY, ProxyCommand::CublasSgemmV2 as usize, &info as *const _ as usize, 
                alpha as *const _ as usize, beta as *const _ as usize)
        };
}

// #[no_mangle]
// pub extern "C" fn cublasLtCreate(lighthandle: *mut cublasLtHandle_t) -> usize {
//     //println!("cublasLtCreate , but not yet implemented. ");
//     panic!("TODO: cublasLtCreate");
// }

#[no_mangle]
pub extern "C" fn cudaMemset(devPtr: *const c_void, value: c_int, count: usize) -> usize {
    //println!("cudaMemset");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaMemset as usize, devPtr as usize, value as usize, count as usize)
    };
}

#[no_mangle]
pub extern "C" fn cudaMemsetAsync(devPtr: *const c_void, value: c_int, count: usize, stream: cudaStream_t) -> usize {
    //println!("cudaMemsetAsync");
    return unsafe {
       syscall5(SYS_PROXY, ProxyCommand::CudaMemsetAsync as usize, devPtr as usize, value as usize, count as usize, stream as usize)
   };
}

#[no_mangle]
pub extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut c_int,
    func: *const c_void, blockSize: c_int, dynamicSMemSize: usize, flags: c_uint) -> usize {
    //println!("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    return unsafe {
       syscall6(SYS_PROXY, ProxyCommand::CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags as usize,
        numBlocks as usize, func as usize, blockSize as usize, dynamicSMemSize as usize, flags as usize)
   };
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
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CuCtxGetCurrent as usize, pctx as usize)
    };
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
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CuModuleLoadData as usize, module as usize, image as usize, image_len as usize)
    };
}

#[no_mangle]
pub extern "C" fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *mut c_char ) -> usize {
    //println!("hijeck cuModuleGetFunction");
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CuModuleGetFunction as usize, hfunc as usize, hmod as usize, name as usize)
    };
}

#[no_mangle]
pub extern "C" fn cuModuleUnload(hmod: CUmodule) -> usize {
    //println!("hijeck cuModuleUnload");
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CuModuleUnload as usize, hmod as usize)
    };
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
        return unsafe {
            syscall2(SYS_PROXY, ProxyCommand::CuLaunchKernel as usize, &info as *const _ as usize)
        };
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
    return unsafe {
        syscall4(SYS_PROXY,ProxyCommand::CublasSgemmStridedBatched as usize,
            &info as *const _ as usize, alpha as *const _ as usize, beta as *const _ as usize)
    };
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
) -> usize{
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
    return unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CublasLtMatmul as usize,
            &info as *const _ as usize, alpha as *const _ as usize, beta as *const _ as usize)
    };
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
    let info = CublasLtMatmulAlgoGetHeuristicInfo {
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
    return unsafe {
        syscall4(SYS_PROXY,ProxyCommand::CublasLtMatmulAlgoGetHeuristic as usize,
            &info as *const _ as usize, heuristicResultsArray as *mut _ as usize, returnAlgoCount as *mut _ as usize)
    };
}