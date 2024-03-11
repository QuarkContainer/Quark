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
use std::ffi::CStr;

use cuda_runtime_sys::cudaMemcpyKind;
use cuda_runtime_sys::{cudaStream_t, cudaStreamCaptureStatus}; 
use cuda_runtime_sys::{cudaDeviceProp, cudaDeviceAttr,cudaFuncCache,cudaLimit,cudaDeviceP2PAttr,cudaSharedMemConfig};

use crate::syscall::*;
use crate::proxy::*;
pub const SYS_PROXY: usize = 10003;

// device management
//not yet tested 
#[no_mangle]
pub extern "C" fn cudaChooseDevice(device:*mut c_int, prop: *const cudaDeviceProp) -> usize{
    println!("Hijacked cudaChooseDevice(prop address: {:x})", prop as u64);

    unsafe{
        println!("yiwang deviceProperties is(   
            prop.luidDeviceNodeMask: {:x},
            prop.totalGlobalMem: {:x},
            prop.sharedMemPerBlock: {:x}
            prop.regsPerBlock: {:x}
        )...",    
        (*prop).luidDeviceNodeMask,
        (*prop).totalGlobalMem,
        (*prop).sharedMemPerBlock,
        (*prop).regsPerBlock)
    };

    let ret = unsafe{
        syscall3(SYS_PROXY, ProxyCommand::CudaChooseDevice as usize, device as *const _ as usize, prop as usize) 
    };
    return ret; 
}

// not yet tested 
#[no_mangle]
pub extern "C" fn cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> usize{
    println!("Hijacked cudaDeviceGetAttribute(device: {}, cudaDeviceAttr: {:?}) ", device, attr);

    let ret = unsafe{
        syscall4(SYS_PROXY, ProxyCommand::CudaDeviceGetAttribute as usize , value as * const _ as usize, attr as usize , device as usize)
    };

    return ret;
}

//not yet tested
#[no_mangle]
pub extern "C" fn cudaDeviceGetByPCIBusId(device: *mut c_int, pciBusId: *const c_char) -> usize {

     // Convert *const c_char to &CStr
    let c_str = unsafe { CStr::from_ptr(pciBusId) };

     // Convert &CStr to &str
    if let Ok(str_val) = c_str.to_str() {
        println!("Hijacked cudaDeviceGetByPCIBusId(pciBusId: {})", str_val);
    } 

    let ret = unsafe{
        syscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetByPCIBusId as usize, device as *const _ as usize, pciBusId as usize)
    };

    return ret;
}

//not yet tested
#[no_mangle] 
pub extern "C" fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> usize {
    println!("Hijacked cudaDeviceGetCacheConfig");
        
    
    println!("cudaFuncCache struct address is: {:x}", pCacheConfig as usize);
    
    return unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CudaDeviceGetCacheConfig as usize, pCacheConfig as *const _ as usize)    
    };
}



// not yet tested 
#[no_mangle] 
pub extern "C" fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> usize {
    println!("Hijacked cudaDeviceGetLimit(limit want to query: {:?}) ", limit);

    let ret = unsafe{
        syscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetLimit as usize, pValue as *const _ as usize, limit as usize)
    };

    return ret;

}

//not yet tested 
#[no_mangle] 
pub extern "C" fn cudaDeviceGetP2PAttribute(value: *mut c_int, attr: cudaDeviceP2PAttr, srcDevice: c_int, dstDevice: c_int) -> usize {
    println!("Hijacked cudaDeviceGetP2PAttribute(Queries attribute: {:?} of the link between device {} and device {}.) ",attr, srcDevice, dstDevice);

    let ret = unsafe { 
        syscall5(SYS_PROXY, ProxyCommand::CudaDeviceGetP2PAttribute as usize, value as *const _ as usize, attr as usize, srcDevice as usize, dstDevice as usize)
    };

    return ret; 
}

// not yet tested 
#[no_mangle]
pub extern "C" fn cudaDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, device: c_int) -> usize {
    println!("Hijacked cudaDeviceGetPCIBusId(device: {})",device);

    let ret = unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaDeviceGetPCIBusId as usize, pciBusId as *const _ as usize , len as usize,  device as usize)
    };
    return ret; 
}

// not yet tested 
pub extern "C" fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> usize {
    unsafe{
        println!("Hijacked cudaDeviceGetSharedMemConfig(cudaSharedMemConfig: {:?})", *pConfig as u32);
    };

    let ret = unsafe {
        syscall2( SYS_PROXY, ProxyCommand::CudaDeviceGetSharedMemConfig as usize, pConfig as *const _ as usize)
    };
    
    return ret; 
    
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> usize{
    unsafe{
        println!("Hijacked cudaDeviceGetStreamPriorityRange, leastPriority is: {}, greatestPriority is: {}", *leastPriority,*greatestPriority)
    };

    return unsafe{
        syscall3(SYS_PROXY, ProxyCommand::CudaDeviceGetStreamPriorityRange as usize, leastPriority as *const _ as usize, greatestPriority as *const _ as usize)
    };
}

// not yet tested 
#[no_mangle]
pub extern "C" fn cudaDeviceReset() -> usize{
    println!("Hijacked cudaDeviceReset()");

    return unsafe{
        syscall1(SYS_PROXY,ProxyCommand::CudaDeviceReset as usize)
    };
}

//not yet tested 
#[no_mangle]
pub extern "C" fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> usize{

    println!("Hijacked cudaDeviceSetCacheConfig(cacheConfig setting: {:#?})",cacheConfig);
    return unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CudaDeviceSetCacheConfig as usize, cacheConfig as usize)
    };

}

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> usize {
    println!("Hijacked cudaSetDevice({})", device);

    let ret = unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaSetDevice as usize, device as usize) 
    };

    println!("Hijacked ret({})", ret);
    return ret;
}

// not yet tested
#[no_mangle]
pub extern "C" fn cudaSetDeviceFlags(flags: c_uint) -> usize{
    println!("Hijacked cudaSetDeviceFlags({})", flags);

    let ret = unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaSetDeviceFlags as usize,flags as usize)

    };
    return ret;
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> usize {
    println!("Hijacked cudaDeviceSynchronize()");

    return unsafe {
        syscall1(SYS_PROXY, ProxyCommand::CudaDeviceSynchronize as usize) 
    };
}

// not yet tested 
#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> usize{
    unsafe{
    println!("Hijacked cudaGetDeviceCount({})", *count)
    };
    
    return unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CudaGetDeviceCount as usize, count as *const _ as usize )
    };
}

// not yet tested 
#[no_mangle]
pub extern "C" fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> usize{
    println!("Hijacked cudaGetDevicePropoerties(device {})",device);
    
    return unsafe{
        syscall3(SYS_PROXY, ProxyCommand::CudaGetDeviceProperties as usize, prop as * const _ as usize , device as usize)
    };

}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinary(fatCubin: &FatHeader) -> *mut u64 {
    println!("Hijacked __cudaRegisterFatBinary(fatCubin:{:#x?})", fatCubin);
    let addrFatCubin = fatCubin.text as *const _ as *const u64;
    println!("ProxyCommand::CudaRegisterFatBinary: {:x}, fatCubin.text.header_size:{:x}, fatCubin.text.size: {:x}, addrFatCubin: {:x}", 
    ProxyCommand::CudaRegisterFatBinary as usize, fatCubin.text.header_size as usize, fatCubin.text.size as usize, addrFatCubin as usize);
    let addr=addrFatCubin as *const u8;
    let len = fatCubin.text.header_size as usize + fatCubin.text.size as usize;
    // let bytes = std::slice::from_raw_parts(fatCubin.text as *const _ as u64 as *const u8, len);
    println!("fatCubin.text addr {:x}", fatCubin.text as *const _ as u64);
    // println!("!!!0 {:x?}", bytes);
    println!("addr {:x?}, addrFatCubin {:x?}",addr,addrFatCubin);

    let result = &(0 as *mut u64) as *const _ as u64;
    println!("result {:x}", result);
    unsafe {
        syscall4(SYS_PROXY, ProxyCommand::CudaRegisterFatBinary as usize, len, fatCubin.text as *const _ as usize, result as usize);
    }
    return result as *mut u64;
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle:u64) {
    println!("Hijacked __cudaUnregisterFatBinary(fatCubinHandle = {:x})", fatCubinHandle);
    let fatCubinPtr: *const u64 = fatCubinHandle as *const u64;
    unsafe{
    println!("Hijacked __cudaUnregisterFatBinary( the content of fatCubinHandle = {:x})", *fatCubinPtr);
    }
   
    unsafe{
        //  if *fatCubinPtr != 0 {
         println!("the content of fatCubin poninter is not 0, need to unload the module");
         syscall2(SYS_PROXY, ProxyCommand::CudaUnregisterFatBinary as usize,fatCubinHandle as usize);
        //  }
    }
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(fatCubinHandle:u64){
    let fatCubinPtr: *const u64 = fatCubinHandle as *const u64;
    unsafe{
    println!("Hijacked __cudaUnregisterFatBinaryEnd( the content of fatCubinHandle = {:x})", *fatCubinPtr);
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
        wSize: wSize
    };
    println!("RegisterFunctionInfo {:x?}", info);
    unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaRegisterFunction as usize, &info as *const _ as usize);
    }
}

// __cudaRegisterVar
#[no_mangle]
pub extern "C" fn __cudaRegisterVar(
    fatCubinHandle:u64,
    hostVar:u64,   // char pointer
    deviceAddress:u64,  // char pointer 
    deviceName:u64, // char pointer 
    ext:i32,
    size: usize,
    constant: i32,
    global: i32,
){
    println!("Hijacked __cudaRegisterVar(fatCubinHandle:{:x}, hostVar:{:x}, 
        deviceAddress:{:x}, deviceName:{:x}, ext:{}, size:{:x}, constant:{}, 
        global:{})",fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);

    let info = RegisterVarInfo {
        fatCubinHandle: fatCubinHandle,
        hostVar: hostVar,
        deviceAddress: deviceAddress,
        deviceName: deviceName,
        ext: ext,
        size: size,
        constant: constant,
        global: global
    };

    println!("RegisterVarInfo {:x?}", info);

    unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaRegisterVar as usize, &info as *const _ as usize);
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
        syscall2(SYS_PROXY, ProxyCommand::CudaLaunchKernel as usize, &info as *const _ as usize);
    }
}

#[no_mangle]
pub extern "C" fn cudaMalloc(
        dev_ptr: *mut *mut c_void, 
        size: usize
    ) -> usize {
    println!("Hijacked cudaMalloc(size:{})", size);

    let ret = unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaMalloc as usize, dev_ptr as * const _ as usize, size)
    };
    return ret;
}

#[no_mangle]
pub extern "C" fn cudaFree(
    dev_ptr: *mut c_void,
) -> usize {
    println!("Hijacked cudaFree at {:?}",dev_ptr);
    
    let ret = unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CudaFree as usize,dev_ptr as * const _ as usize )
    };
    return ret;
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


#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream:cudaStream_t
) -> usize{
    println!("Hijacked cudaMemcpyAsync(size:{})",count);

    if kind == cudaMemcpyKind::cudaMemcpyHostToHost {
        unsafe{
            std::ptr::copy_nonoverlapping(src, dst, count);
        }

        return 0; 
    }

    return unsafe{
        syscall6(SYS_PROXY, ProxyCommand::CudaMemcpyAsync as usize, dst as * const _ as usize, src as usize, count as usize, kind as usize , stream as usize)
    };
}

// Stream Management API

//Waits for stream tasks to complete
#[no_mangle]
pub extern "C" fn cudaStreamSynchronize(
    stream:cudaStream_t) -> usize{
    println!("Hijacked cudaStreamSynchronize stream: {:?}", stream);

    return unsafe{
        syscall2(SYS_PROXY, ProxyCommand::CudaStreamSynchronize as usize, stream as usize)
    };

}

// create an asynchronous stream 
#[no_mangle]
pub extern "C" fn cudaStreamCreate(
    pStream: *mut cudaStream_t
) -> usize{
    
    println!("Hijacked cudaStreamCreate, stream address: {:x}",pStream as u64);
    unsafe{
        println!("pStream referecne to : {:x}",(*pStream) as u64);
    }
   
    return unsafe {
        syscall2(SYS_PROXY, ProxyCommand::CudaStreamCreate as usize, pStream as *const _ as usize)
    };
}

// Destroys and cleans up an asynchronous stream.
#[no_mangle]
pub extern "C" fn cudaStreamDestroy(
    stream: cudaStream_t
) -> usize {
    println!("Hijacked cudaStreamDestory, stream: {:x}",stream as u64);
    return unsafe{
        syscall2(SYS_PROXY,ProxyCommand::CudaStreamDestroy as usize,  stream as usize)
    };
}

//Return a stream's capture status 
#[no_mangle]
pub extern "C" fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus
) -> usize {

    unsafe{
    println!("Hijacked cudaStreamIsCapturing,, captureStatus is :{}", (*pCaptureStatus) as u64);
    };

    println!("captureStatus address is: {:x}", pCaptureStatus as usize);

    return unsafe{
        syscall3(SYS_PROXY, ProxyCommand::CudaStreamIsCapturing as usize, stream as usize, pCaptureStatus as *const _ as usize)
    };

}
  
// #[no_mangle]
// pub extern "C" fn cuModuleGetLoadingMode(
//     mode: *mut CUmoduleLoadingMode
// ) -> usize {
//     unsafe{
//     println!("Hijacked cuModuleGetLoadingMode, mode is {:x?}", (*mode) as u64);
//     }

//     return unsafe{
//         syscall2(SYS_PROXY, ProxyCommand::CuModuleGetLoadingMode as usize, mode as *const _ as usize )
//     };
// }

// Error handling 
// not yet tested 
#[no_mangle]
pub extern "C" fn cudaGetLastError() -> usize{
    println!("Hijacked cudaGetLatError()");

    return unsafe{
        syscall1(SYS_PROXY, ProxyCommand::CudaGetLastError as usize)
    };   
}