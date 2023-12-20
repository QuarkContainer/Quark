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

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
#[repr(u64)]
pub enum ProxyCommand {
    None = 0 as u64,
    CudaSetDevice,
    CudaDeviceSynchronize,
    CudaMalloc,
    CudaMemcpy,
    CudaRegisterFatBinary,

    CuInit,
}

impl Default for ProxyCommand {
    fn default() -> Self {
        return Self::None;
    }
}

#[derive(Copy, Clone, Default, Debug)]
#[repr(C)]
pub struct ProxyParameters {
    pub para1: u64,
    pub para2: u64,
    pub para3: u64,
    pub para4: u64,
    pub para5: u64,
    pub para6: u64,
    pub para7: u64,
}

// from https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
// cudaMemcpyHostToHost = 0
// Host -> Host
// cudaMemcpyHostToDevice = 1
// Host -> Device
// cudaMemcpyDeviceToHost = 2
// Device -> Host
// cudaMemcpyDeviceToDevice = 3
// Device -> Device
// cudaMemcpyDefault = 4
// Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing

pub type CudaMemcpyKind = u64;

pub const CUDA_MEMCPY_HOST_TO_HOST: u64 = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: u64 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: u64 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: u64 = 3;
pub const CUDA_MEMCPY_DEFAULT: u64 = 4;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FatHeader {
    magic: u32,
    version: u32,
    pub text: &'static FatTextHeader, 
    data: u64,
    unknown: u64,
    text2: u64,
    zero: u64
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct FatTextHeader {
    kind: u16,
    unknown1: u16,
    header_size: u32,
    pub size: u64,
    compressed_size: u32,
    unknown2: u32,
    minor: u16,
    major: u16,
    arch: u32,
    obj_name_offset: u32,
    obj_name_len: u32,
    flags:u64,
    zero: u64,
    decompressed_size: u64
}