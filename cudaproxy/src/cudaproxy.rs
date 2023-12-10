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

use cuda_runtime_sys::*; // {dim3, cudaStream_t, cudaError_t, cudaMemcpyKind};

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
pub extern "C" fn cudaMalloc(
        dev_ptr: *mut *mut c_void, 
        size: usize
    ) -> usize {
    println!("Hijacked cudaMalloc(size:{})", size);

    return unsafe {
        syscall3(SYS_PROXY, ProxyCommand::CudaMalloc as usize, dev_ptr as * const _ as usize, size) 
    };
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
        dst: *mut c_void, 
        src: *const c_void, 
        count: usize, 
        kind: cudaMemcpyKind
    ) -> usize {
    println!("Hijacked cudaMemcpy(count:{})", count);

    return unsafe {
        syscall5(
            SYS_PROXY, 
            ProxyCommand::CudaMemcpy as usize, 
            dst as * const _ as usize, 
            src as * const _ as usize, 
            count, 
            kind as usize
        ) 
    };
}