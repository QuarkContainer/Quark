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
    CudaRegisterFunction,
    CudaLaunchKernel,

    CuInit,
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
#[repr(u64)]
pub enum XpuLibrary {
    None = 0 as u64,
    CudaRuntime,
    CudaDriver,
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

pub const FATBIN_FLAG_64BIT: u64 = 0x0000000000000001;
pub const FATBIN_FLAG_DEBUG: u64 = 0x0000000000000002;
pub const FATBIN_FLAG_LINUX: u64 = 0x0000000000000010;
pub const FATBIN_FLAG_COMPRESS: u64 = 0x0000000000002000;

pub const EIATTR_PARAM_CBANK: u64 = 0xa;
pub const EIATTR_EXTERNS: u64 = 0xf;
pub const EIATTR_FRAME_SIZE: u64 = 0x11;
pub const EIATTR_MIN_STACK_SIZE: u64 = 0x12;
pub const EIATTR_KPARAM_INFO: u64 = 0x17;
pub const EIATTR_CBANK_PARAM_SIZE: u64 = 0x19;
pub const EIATTR_MAX_REG_COUNT: u64 = 0x1b;
pub const EIATTR_EXIT_INSTR_OFFSETS: u64 = 0x1c;
pub const EIATTR_S2RCTAID_INSTR_OFFSETS: u64 = 0x1d;
pub const EIATTR_CRS_STACK_SIZE: u64 = 0x1e;
pub const EIATTR_SW1850030_WAR: u64 = 0x2a;
pub const EIATTR_REGCOUNT: u64 = 0x2f;
pub const EIATTR_SW2393858_WAR: u64 = 0x30;
pub const EIATTR_INDIRECT_BRANCH_TARGETS: u64 = 0x34;
pub const EIATTR_CUDA_API_VERSION: u64 = 0x37;

pub const EIFMT_NVAL: u64 = 0x1;
pub const EIFMT_HVAL: u64 = 0x3;
pub const EIFMT_SVAL: u64 = 0x4;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FatHeader {
    magic: u32,
    version: u32,
    pub text: &'static FatElfHeader, 
    data: u64,
    unknown: u64,
    text2: u64,
    zero: u64
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct FatElfHeader {
    magic: u32,
    version: u16,
    pub header_size: u16,
    pub size: u64
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct FatTextHeader {
    pub kind: u16,
    unknown1: u16,
    pub header_size: u32,
    pub size: u64,
    compressed_size: u32,
    unknown2: u32,
    minor: u16,
    major: u16,
    arch: u32,
    obj_name_offset: u32,
    obj_name_len: u32,
    pub flags:u64,
    zero: u64,
    decompressed_size: u64
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct ParamInfo {
    pub addr: u64,
    pub paramNum: usize,
    pub paramSizes: [u16;30]
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct LaunchKernelInfo {
    pub func: u64,
    pub gridDim: Qdim3, 
    pub blockDim: Qdim3, 
    pub args: u64, 
    pub sharedMem: usize, 
    pub stream: u64
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct Qdim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct RegisterFunctionInfo {
    pub fatCubinHandle:u64, 
    pub hostFun:u64, 
    pub deviceFun:u64, 
    pub deviceName:u64, 
    pub thread_limit:usize, 
    pub tid:u64, 
    pub bid:u64, 
    pub bDim:u64, 
    pub gDim:u64, 
    pub wSize:usize
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct NvInfoKernelEntry {
    pub format: u8,
    pub attribute: u8,
    pub values_size: u16,
    pub values: u32
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct NvInfoKParamInfo {
    pub index: u32,
    pub ordinal: u16,
    pub offset: u16,
    pub comp: u32
}

impl NvInfoKParamInfo {
    pub fn GetSize(&self) -> u16 {
        (self.comp>>18 & 0x3fff) as u16
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct NvInfoEntry {
    pub format: u8,
    pub attribute: u8,
    pub values_size: u16,
    pub kernel_id: u32,
    pub value: u32
}