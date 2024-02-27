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
    //devcie management
    CudaChooseDevice,
    CudaDeviceGetAttribute,
    CudaDeviceGetByPCIBusId,
    CudaDeviceGetCacheConfig,
    CudaDeviceGetLimit,
    CudaDeviceGetP2PAttribute,
    CudaDeviceGetPCIBusId,
    CudaDeviceGetSharedMemConfig,
    CudaDeviceGetStreamPriorityRange,
    CudaDeviceReset,
    CudaDeviceSetCacheConfig,
    
    CudaSetDevice,
    CudaSetDeviceFlags,
    CudaDeviceSynchronize,
   
    CudaGetDeviceCount,
    
    CudaGetDeviceProperties,

    CudaMalloc,
    CudaMemcpy,
    CudaMemcpyAsync,
    CudaFree,
    CudaRegisterFatBinary,
    CudaUnregisterFatBinary,
    CudaRegisterFunction,
    CudaRegisterVar,
    CudaLaunchKernel,

    CuInit,
    // stream management 
    CudaStreamSynchronize,
    CudaStreamCreate,
    CudaStreamDestroy,
    CudaStreamIsCapturing,
    CuModuleGetLoadingMode,
    //Error handling
    CudaGetLastError, 
}


#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
#[repr(u64)]
pub enum XpuLibrary {
    None = 0 as u64,
    CudaRuntime,
    CudaDriver,
}

// #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
// #[repr(u32)]
// pub enum CUmoduleLoadingMode {
//     None = 0x0,
//     CU_MODULE_EAGER_LOADING = 0x1,
//     CU_MODULE_LAZY_LOADING = 0x2,
// }

// impl Default for CUmoduleLoadingMode {
//     fn default() -> Self {
//         return CUmoduleLoadingMode::None;
//     }
// }

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CumoduleLoadingModeEnum {
    CuModuleEagerLoading = 1,
    CuModuleLazyLoading = 2,
}
pub use self::CumoduleLoadingModeEnum as CUmoduleLoadingMode;

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
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CudaUUIDSt {
    pub bytes: [i8; 16usize],
}

pub type CudaUUID = CudaUUIDSt;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CudaDeviceProperties {
    pub name: [i8; 256usize],
    pub uuid: CudaUUID,   
    pub luid: [i8; 8usize],
    pub luidDeviceNodeMask: i8,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: i32,
    pub warpSize: i32,
    pub memPitch: usize,
    pub maxThreadsPerBlock: i32,
    pub maxThreadsDim: [i32; 3usize],
    pub maxGridSize: [i32; 3usize],
    pub clockRate: i32,
    pub totalConstMem: usize,
    pub major: i32,
    pub minor: i32,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: i32,
    pub multiProcessorCount: i32,
    pub kernelExecTimeoutEnabled: i32,
    pub integrated: i32,
    pub canMapHostMemory: i32,
    pub computeMode: i32,
    pub maxTexture1D: i32,
    pub maxTexture1DMipmap: i32,
    pub maxTexture1DLinear: i32,
    pub maxTexture2D: [i32; 2usize],
    pub maxTexture2DMipmap: [i32; 2usize],
    pub maxTexture2DLinear: [i32; 3usize],
    pub maxTexture2DGather: [i32; 2usize],
    pub maxTexture3D: [i32; 3usize],
    pub maxTexture3DAlt: [i32; 3usize],
    pub maxTextureCubemap: i32,
    pub maxTexture1DLayered: [i32; 2usize],
    pub maxTexture2DLayered: [i32; 3usize],
    pub maxTextureCubemapLayered: [i32; 2usize],
    pub maxSurface1D: i32,
    pub maxSurface2D: [i32; 2usize],
    pub maxSurface3D: [i32; 3usize],
    pub maxSurface1DLayered: [i32; 2usize],
    pub maxSurface2DLayered: [i32; 3usize],
    pub maxSurfaceCubemap: i32,
    pub maxSurfaceCubemapLayered: [i32; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: i32,
    pub ECCEnabled: i32,
    pub pciBusID: i32,
    pub pciDeviceID: i32,
    pub pciDomainID: i32,
    pub tccDriver: i32,
    pub asyncEngineCount: i32,
    pub unifiedAddressing: i32,
    pub memoryClockRate: i32,
    pub memoryBusWidth: i32,
    pub l2CacheSize: i32,
    pub maxThreadsPerMultiProcessor: i32,
    pub streamPrioritiesSupported: i32,
    pub globalL1CacheSupported: i32,
    pub localL1CacheSupported: i32,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: i32,
    pub managedMemory: i32,
    pub isMultiGpuBoard: i32,
    pub multiGpuBoardGroupID: i32,
    pub hostNativeAtomicSupported: i32,
    pub singleToDoublePrecisionPerfRatio: i32,
    pub pageableMemoryAccess: i32,
    pub concurrentManagedAccess: i32,
    pub computePreemptionSupported: i32,
    pub canUseHostPointerForRegisteredMem: i32,
    pub cooperativeLaunch: i32,
    pub cooperativeMultiDeviceLaunch: i32,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: i32,
    pub directManagedMemAccessFromHost: i32,
}

// impl Default for CudaDeviceProperties {
//     fn default() -> Self {
//         unsafe { ::std::mem::zeroed() }
//     }
// }

impl Default for CudaDeviceProperties {
    fn default() -> Self {
        Self {
            name: [0; 256],
            uuid: CudaUUID { bytes: [0; 16]},
            luid: [0; 8usize],
            luidDeviceNodeMask: 0,
            totalGlobalMem: 0,
            sharedMemPerBlock: 0,
            regsPerBlock: 0,
            warpSize: 0,
            memPitch: 0,
            maxThreadsPerBlock: 0,
            maxThreadsDim: [0; 3],
            maxGridSize: [0; 3],
            clockRate: 0,
            totalConstMem: 0,       
            major: 0,
            minor: 0,
            textureAlignment: 0,
            texturePitchAlignment: 0,
            deviceOverlap: 0,
            multiProcessorCount: 0,
            kernelExecTimeoutEnabled: 0,
            integrated: 0,
            canMapHostMemory: 0,
            computeMode: 0,
            maxTexture1D: 0,
            maxTexture1DMipmap: 0,
            maxTexture1DLinear: 0,
            maxTexture2D: [0; 2],
            maxTexture2DMipmap: [0; 2],
            maxTexture2DLinear: [0; 3],
            maxTexture2DGather: [0; 2],
            maxTexture3D: [0; 3],
            maxTexture3DAlt: [0; 3],
            maxTextureCubemap: 0,
            maxTexture1DLayered: [0; 2],
            maxTexture2DLayered: [0; 3],
            maxTextureCubemapLayered: [0; 2],
            maxSurface1D: 0,
            maxSurface2D: [0; 2],
            maxSurface3D: [0; 3],
            maxSurface1DLayered: [0; 2],
            maxSurface2DLayered: [0; 3],
            maxSurfaceCubemap: 0,
            maxSurfaceCubemapLayered: [0; 2],
            surfaceAlignment: 0,
            concurrentKernels: 0,
            ECCEnabled: 0,
            pciBusID: 0,
            pciDeviceID: 0,
            pciDomainID: 0,
            tccDriver: 0,
            asyncEngineCount: 0,
            unifiedAddressing: 0,
            memoryClockRate: 0,
            memoryBusWidth: 0,
            l2CacheSize: 0,
            maxThreadsPerMultiProcessor: 0,
            streamPrioritiesSupported: 0,
            globalL1CacheSupported: 0,
            localL1CacheSupported: 0,
            sharedMemPerMultiprocessor: 0,
            regsPerMultiprocessor: 0,
            managedMemory: 0,
            isMultiGpuBoard: 0,
            multiGpuBoardGroupID: 0,
            hostNativeAtomicSupported: 0,
            singleToDoublePrecisionPerfRatio: 0,
            pageableMemoryAccess: 0,
            concurrentManagedAccess: 0,
            computePreemptionSupported: 0,
            canUseHostPointerForRegisteredMem: 0,
            cooperativeLaunch: 0,
            cooperativeMultiDeviceLaunch: 0,
            sharedMemPerBlockOptin: 0,
            pageableMemoryAccessUsesHostPageTables: 0,
            directManagedMemAccessFromHost: 0,
        }
    }
}


#[repr(C)]
#[derive(Debug)]
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
#[derive(Default, Debug)]
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
#[derive(Default, Debug)]
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
#[derive(Default, Debug)]
pub struct NvInfoEntry {
    pub format: u8,
    pub attribute: u8,
    pub values_size: u16,
    pub kernel_id: u32,
    pub value: u32
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct RegisterVarInfo {
    pub fatCubinHandle:u64, 
    pub hostVar: u64,
    pub deviceAddress:u64,
    pub deviceName:u64,
    pub ext:i32,
    pub size: usize,
    pub constant: i32,
    pub global: i32,
}