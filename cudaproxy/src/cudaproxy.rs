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

use std::os::raw::*;

use cuda_runtime_sys::{dim3, cudaStream_t, cudaError_t, cudaMemcpyKind};

use crate::syscall::*;
use crate::proxy::*;
pub const SYS_PROXY: usize = 10003;

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> usize {
    println!("Hijacked cudaSetDevice({})", device);

    let ret = unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaSetDevice as usize, device as usize) 
    };

    println!("Hijacked ret({})", ret);
    return ret;
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> usize {
    println!("Hijacked cudaDeviceSynchronize()");

    return unsafe {
        syscall1(SYS_PROXY, ProxyCommand::CudaDeviceSynchronize as usize) 
    };
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinary(fatCubin: &FatHeader) -> *mut u64 {
    println!("Hijacked __cudaRegisterFatBinary(fatCubin:{:#x?})", fatCubin);
    let addrFatCubin = fatCubin.text as *const _ as *const u64;
    unsafe {
        println!("hochan ProxyCommand::CudaRegisterFatBinary: {:x}, fatCubin.text.header_size:{:x}, fatCubin.text.size: {:x}, addrFatCubin: {:x}", 
        ProxyCommand::CudaRegisterFatBinary as usize, fatCubin.text.header_size as usize, fatCubin.text.size as usize, addrFatCubin as usize);
        let addr=addrFatCubin as *const u8;
        let len = fatCubin.text.header_size as usize + fatCubin.text.size as usize;
        // let bytes = std::slice::from_raw_parts(fatCubin.text as *const _ as u64 as *const u8, len);
        println!("hochan fatCubin.text addr {:x}", fatCubin.text as *const _ as u64);
        // println!("hochan !!!0 {:x?}", bytes);
        println!("hochan addr {:x?}, addrFatCubin {:x?}",addr,addrFatCubin);

        let result = 0 as *mut u64;
        syscall4(SYS_PROXY, ProxyCommand::CudaRegisterFatBinary as usize, len, fatCubin.text as *const _ as usize, result as usize);
        return result;
    }
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFunction(
    fatCubinHandle:u64, 
    hostFun:u64, 
    deviceFun:u64, 
    deviceName:u64, 
    thread_limit:usize, 
    tid:u64, 
    bid:u64, 
    bDim:u64, 
    gDim:u64, 
    wSize:usize
) {
    println!("Hijacked __cudaRegisterFunction(fatCubinHandle:{:x}, hostFun:{:x}, deviceFun:{:x}, deviceName:{:x}, thread_limit: {}, tid: {:x}, bid: {:x}, bDim: {:x}, gDim: {:x}, wSize: {})", fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);    
    let info = RegisterFactionInfo {
        fatCubinHandle: fatCubinHandle, 
        hostFun: hostFun, 
        deviceFun: deviceFun, 
        deviceName: deviceName, 
        thread_limit: thread_limit, 
        tid: tid, 
        bid: bid, 
        bDim: bDim, 
        gDim: gDim, 
        wSize: wSize
    };
    println!("hochan RegisterFactionInfo {:x?}", info);
    unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaRegisterFunction as usize, &info as *const _ as usize);
    }
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    func:u64, 
    gridDim:Qdim3, 
    blockDim:Qdim3, 
    args:u64, 
    sharedMem:usize, 
    stream:u64
) {
    println!("Hijacked cudaLaunchKernel(func:{:x}, gridDim:{:x?}, blockDim:{:x?}, args:{:x}, sharedMem: {}, stream: {:x?})", 
        func, gridDim, blockDim, args, sharedMem, stream);
    let info = LaunchKernelInfo {
        func: func, 
        gridDim: gridDim, 
        blockDim: blockDim, 
        args: args, 
        sharedMem: sharedMem, 
        stream: stream
    };
    unsafe {
        let bytes = std::slice::from_raw_parts(args as *const u64, 4);
        println!("hochan LaunchKernelInfo {:x?} args bytes {:x?}", info, bytes);

    
        syscall2(SYS_PROXY, ProxyCommand::CudaLaunchKernel as usize, &info as *const _ as usize);
    }
}

#[no_mangle]
pub extern "C" fn cudaMalloc(
        dev_ptr: *mut *mut c_void, 
        size: usize
    ) -> usize {
    println!("Hijacked cudaMalloc(size:{})", size);

    unsafe {
        println!("hochan cudaMalloc before dev_ptr {:x?} *dev_ptr {:x?})", dev_ptr, *dev_ptr as u64);
        let ret = syscall3(SYS_PROXY, ProxyCommand::CudaMalloc as usize, dev_ptr as * const _ as usize, size);
        println!("hochan cudaMalloc after dev_ptr {:x?} *dev_ptr {:x?})", dev_ptr, *dev_ptr as u64);
        println!("hochan cudaMalloc ret {:x}", ret);
        return ret;
    };    
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
        dst: *mut c_void, 
        src: *const c_void, 
        count: usize, 
        kind: cudaMemcpyKind
    ) -> usize {
    println!("Hijacked cudaMemcpy(size:{})", count);

    if kind == cudaMemcpyKind::cudaMemcpyHostToHost {
        unsafe {
            std::ptr::copy_nonoverlapping(src as * const u8, dst as * mut u8, count);
        }
        
        return 0;
    }

    return unsafe {
        syscall5(SYS_PROXY, ProxyCommand::CudaMemcpy as usize, dst as * const _ as usize, src as usize, count as usize, kind as usize) 
    };
}