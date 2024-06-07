use crate::qlib::proxy::*;
// use crate::vmspace::{CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE};
use crate::xpu::cuda::{FUNCTIONS, MODULES};
use crate::xpu::cuda_api::*;
use std::ffi::CString;
use std::os::raw::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use cuda_driver_sys::{CUfunction, CUmodule};
use cuda_runtime_sys::cudaStream_t;

pub const ONE_TB: u64 = 1 << 40; //0x10_000_000_000;
pub const ONE_GB: u64 = 1 << 30; //0x40_000_000;
pub const VADDR_START_ADDR: u64 = 7 * ONE_TB;
pub const MAX_MEM_RESERVE_IN_BYTES: u64 = 256 * ONE_GB;

use rcublas_sys::cublasHandle_t;
pub use u32 as CUmemLocationType;
pub use u32 as CUmemAllocationHandleType;
pub use u32 as CUmemAllocationType;
pub use u32 as CUmemAccess_flags;

use super::cuda::{BLASHANDLE, STREAMS};


#[repr(C)]
#[derive(Debug, Default)]
pub struct CUmemLocation {
    pub type_: CUmemLocationType,
    pub id: i32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAllocationProp_st__bindgen_ty_1 {
    pub compressionType: ::core::ffi::c_uchar,
    pub gpuDirectRDMACapable: ::core::ffi::c_uchar,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 4usize],
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct CUmemAllocationProp {
    pub type_: CUmemAllocationType,
    pub requestedHandleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32HandleMetaData: u64,
    pub allocFlags: CUmemAllocationProp_st__bindgen_ty_1,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: CUmemAccess_flags,
}

pub struct MemoryManager {
    pub gpuManager: GPUMemoryManager,
    pub cpuManager: CPUMemoryManager,
    pub fatBinManager: FatBinManager,
    pub ctxManager: CtxManager,
}

// #[derive(Debug, Default)]
// pub struct CudaDeviceLimit {
//     cudaLimitStackSize: usize, // 0x00
//     cudaLimitPrintfFifoSize: usize, // 0x01
//     cudaLimitMallocHeapSize: usize, // 0x02
//     cudaLimitDevRuntimeSyncDepth: usize, // 0x03
//     cudaLimitDevRuntimePendingLaunchCount: usize, // 0x04
//     cudaLimitMaxL2FetchGranularity: usize, // 0x05
//     cudaLimitPersistingL2CacheSize: usize, // 0x06
// }
#[derive(Debug, Default)]
pub struct DeviceStatus {
    currentDev: i32,
    cacheConfig: u32,
    // deviceLimit: CudaDeviceLimit, TODO:
    sharedMemConfig: u32,
    deviceFlags: u32,
}

#[repr(u32)]
#[derive(Debug)]
pub enum StreamType {
    None,
    Priority,
    Flag,
}

impl Default for StreamType {
    fn default() -> Self { StreamType::None }
}

#[derive(Debug, Default)]
pub struct StreamStatus {
    streamType: StreamType,
    priority: i32,
    flag: u32,
    // attribute:  cudaStreamGetAttribute
    // captureInfo :  cudaStreamIsCapturing + cudaStreamGetCaptureInfo 
}

#[derive(Debug, Default)]
pub struct CublasStatus {
    mathMode: u32,
    workspacePtr: u64,
    workspaceSize: usize,
    stream: u64,
}

#[derive(Debug, Default)]
pub struct FuncStatus {
    pub maxDynamicSharedSizeBytes: i32,
    pub preferedSharedMemoryCarveout: i32,
    pub requiredClusterWidth: i32,
    pub requiredClusterHeight: i32,
    pub requiredClusterDepth: i32,
    pub nonPortableClusterSizeAllowed: i32,
    pub clusterSchedulingPolicyPreference: i32,
}

pub struct CtxManager {
    pub deviceStatus: DeviceStatus,
    pub streamStatus: Vec<StreamStatus>,
    pub cublasStatus: BTreeMap<u64, CublasStatus>, // cannot be Vec because workspace need to keep changing
    pub funcStatus: Vec<FuncStatus>,
}

pub struct FatBinManager {
    pub fatBinVec: Vec<Vec<u8>>,
    pub fatBinHandleVec: Vec<(u64, u64)>, // (moduleKey, module)
    pub fatBinFuncVec: Vec<Vec<(u64, CString)>>, // (host_func, func_name)
}

pub struct CPUMemoryManager {
    pub reservedStartAddr: u64,
    pub usedLen: u64,
    pub memMappingTable: BTreeMap<u64, u64>, //gpuAddr - hostAddr
}

pub struct GPUMemoryManager {
    pub granularity: usize,
    pub reservedStartAddr: u64,
    pub nextAvailableAddr: u64,
    pub memAddrVec: Vec<u64>,
    pub memLenVec: Vec<usize>,
    pub memHandleVec: Vec<u64>,
    pub addrTable: BTreeMap<u64, u32>,
    pub idxCounter: AtomicU32,
    pub currentTotalMem: usize,
}

impl GPUMemoryManager {
    pub fn new() -> Self {
        let mut prop: CUmemAllocationProp = CUmemAllocationProp::default();
        prop.type_ = 0x1; //CU_MEM_ALLOCATION_TYPE_PINNED 
        prop.location.type_ = 0x1; //CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0x0 as i32; // hard-coded to 0 for now
        let mut gran: usize = 0;
        let mut res = unsafe { cuMemGetAllocationGranularity(&mut gran as *mut _ as u64, &prop as *const _ as u64, 0x1) };
        if res as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by cuMemGetAllocationGranularity: {}", res as u32);
        }
        let reserve_size = round_up(MAX_MEM_RESERVE_IN_BYTES as usize, gran);
        let mut dptr: u64 = 0;
        res = unsafe { cuMemAddressReserve(&mut dptr as *mut _ as u64, reserve_size, 0, VADDR_START_ADDR, 0) };
        if res as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by cuMemAddressReserve: {}", res as u32);
        }
        // error!("reserved addr: {:x} to {:x} of size {:x}",dptr as u64, dptr as u64 + reserve_size as u64,reserve_size as u64);
        Self {
            granularity: gran,
            reservedStartAddr: dptr,
            nextAvailableAddr: dptr,
            memAddrVec: Vec::new(),
            memLenVec: Vec::new(),
            memHandleVec: Vec::new(),
            addrTable: BTreeMap::new(),
            idxCounter: AtomicU32::new(0),
            currentTotalMem: 0,
        }
    }

    pub fn alloc(&mut self, size: usize) -> (u64, i64) {
        let return_addr: u64 = self.nextAvailableAddr;
        let mut prop = CUmemAllocationProp::default();
        prop.type_ = 0x1; //CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = 0x1; //CU_MEM_LOCATION_TYPE_DEVICE;
        let alloc_size = round_up(size, self.granularity);
        
        let mut handle = 0;
        let mut res = unsafe { cuMemCreate(&mut handle as *mut _ as u64, alloc_size, &prop as *const _ as u64, 0) };
        if res as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by cuMemCreate: {}", res as u32);
            return (0x0 ,res as i64);
        }
        res = unsafe { cuMemMap(return_addr as u64, alloc_size, 0, handle.clone(), 0) };
        if res as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by cuMemMap: {}", res as u32);
            return (0x0, res as i64);
        }

        let mut accessDescriptors = CUmemAccessDesc::default();
        accessDescriptors.location.type_ = 0x1; //CU_MEM_LOCATION_TYPE_DEVICE;
        accessDescriptors.flags = 0x3; //CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        res = unsafe { cuMemSetAccess(return_addr as u64, alloc_size, &accessDescriptors as *const _ as u64,1) };
        if res as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by cuMemSetAccess: {}", res as u32);
            return (0x0, res as i64);
        } else {
            // error!("successfully alloc mem {:x} for ptr {:x}",alloc_size as u64, return_addr);
            self.currentTotalMem += alloc_size;
            self.nextAvailableAddr += alloc_size as u64;
            self.memAddrVec.push(return_addr);
            self.memLenVec.push(alloc_size);
            self.memHandleVec.push(handle);
            self.addrTable.insert(return_addr, self.idxCounter.fetch_add(1, Ordering::Relaxed));
        }

        return (return_addr, 0)
    }

    pub fn offload(&mut self) {
        for idx in 0..self.memAddrVec.len() {
            let addr = self.memAddrVec[idx];
            let mapSize = self.memLenVec[idx];
            let res = unsafe { cuMemUnmap(addr, mapSize as usize) };
            if res as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by cuMemUnmap(offload): {}", res as u32)
            }
        }
        for elem in &self.memHandleVec {
            let res = unsafe { cuMemRelease(elem.clone()) };
            if res as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by cuMemRelease(offload): {}", res as u32)
            }
        }
    }
    pub fn restore(&mut self) {
        for idx in 0..self.memAddrVec.len() {
            let mut prop = CUmemAllocationProp::default();
            prop.type_ = 0x1; //CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type_ = 0x1; //CU_MEM_LOCATION_TYPE_DEVICE;
            let alloc_size = self.memLenVec[idx];
            
            let mut handle = 0;
            let mut res = unsafe { cuMemCreate(&mut handle as *mut _ as u64, alloc_size, &prop as *const _ as u64, 0) };
            if res as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by cuMemCreate(restore): {}", res as u32);
            }
            let addr = self.memAddrVec[idx];
            res = unsafe { cuMemMap(addr, alloc_size, 0, handle.clone(), 0) };
            if res as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by cuMemMap(restore): {}", res as u32);
            }
            let mut accessDescriptors = CUmemAccessDesc::default();
            accessDescriptors.location.type_ = 0x1; //CU_MEM_LOCATION_TYPE_DEVICE;
            accessDescriptors.flags = 0x3; //CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            res = unsafe { cuMemSetAccess(addr, alloc_size, &accessDescriptors as *const _ as u64,1) };
            if res as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by cuMemSetAccess(restore): {}", res as u32);
            }
            self.memHandleVec[idx] = handle;
        }
    }
}

fn round_up(x: usize, y: usize) -> usize {
    ((x - 1)/y + 1) * y
}

impl CPUMemoryManager {
    pub fn new() -> Self {
        Self{
            reservedStartAddr: 0x0,
            usedLen: 0x0,
            memMappingTable: BTreeMap::new(),
        }
    }

    pub fn alloc(&mut self, size: usize) -> u32 {
        let mut hptr: u64 = 0;
        let res = unsafe { cudaHostAlloc(&mut hptr as *mut _ as u64, size, 0x0) };
        if res as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by cudaHostAlloc: {}", res as u32);
        } else {
            self.reservedStartAddr = hptr;
        }
        return res as u32;
    }

    pub fn free(&mut self, ptr: u64) -> u32 {
        let res = unsafe { cudaFreeHost(ptr) };
        if res as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by cudaFreeHost: {}", res as u32);
        } else {
            self.reservedStartAddr = 0x0;
            self.usedLen = 0x0;
            self.memMappingTable = BTreeMap::new();
        }
        return res as u32;
    }
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            gpuManager: GPUMemoryManager::new(), 
            cpuManager: CPUMemoryManager::new(),
            fatBinManager: FatBinManager::new(),
            ctxManager: CtxManager::new(),
        }
    }

    pub fn offloadGPUMem(&mut self) {
        let res = self.cpuManager.alloc(self.gpuManager.currentTotalMem);
        if res != 0 {
            error!("cuda_mem_manager.rs: failed to offload GPU Memory: {}", res as u32);
        } else {
            for idx in 0..self.gpuManager.memAddrVec.len() {
                let hptr = self.cpuManager.reservedStartAddr + self.cpuManager.usedLen;
                let dptr = self.gpuManager.memAddrVec[idx];
                let cpySize = self.gpuManager.memLenVec[idx];
                let resCpy = unsafe { cudaMemcpyAsync(hptr, dptr, cpySize as u64, CUDA_MEMCPY_DEVICE_TO_HOST, 0 as cudaStream_t)};
                if resCpy != 0 {
                    error!("cuda_mem_manager.rs: error caused by cudaMemcpyAsync D2H: {}", resCpy as u32);
                } else {
                    // error!("memcpy {:x}->{:x}, size {:x}", dptr.clone(), hptr.clone(), cpySize.clone());
                    self.cpuManager.memMappingTable.insert(
                        dptr, self.cpuManager.reservedStartAddr+self.cpuManager.usedLen);
                    // error!("offload size {}, d:{}->h:{}", cpySize, dptr, hptr);
                    self.cpuManager.usedLen += cpySize as u64;
                }
            }
            unsafe{ cudaStreamSynchronize(0 as cudaStream_t)};
            self.gpuManager.offload();
        }
    }

    pub fn restoreGPUMem(&mut self) {
        self.gpuManager.restore();
        for idx in 0..self.gpuManager.memAddrVec.len() {
            let cpySize = self.gpuManager.memLenVec[idx];
            let dptr = self.gpuManager.memAddrVec[idx];
            let hptr = match self.cpuManager.memMappingTable.get(&dptr) {
                Some(p) => *p,
                None => panic!("impossible device ptr"),
            };
            
            // error!("memcpy {:x}->{:x}, size {:x}", hptr.clone(), dptr.clone(), cpySize.clone());
            let resCpy = unsafe { cudaMemcpyAsync(dptr, hptr, cpySize as u64, CUDA_MEMCPY_HOST_TO_DEVICE, 0 as cudaStream_t)};
            if resCpy != 0 {
                error!("cuda_mem_manager.rs: error caused by cudaMemcpyAsync H2D: {}", resCpy as u32);
            }
        }
        unsafe{ cudaStreamSynchronize(0 as cudaStream_t)};
        self.cpuManager.free(self.cpuManager.reservedStartAddr);
    }

    pub fn offloadGPUFatbin(&mut self) {
        self.fatBinManager.unregisterFatbin();
    }

    pub fn restoreGPUFatbin(&mut self) {
        self.fatBinManager.restoreFatbin();
    }
    
    pub fn offloadGPUContext(&mut self) {
        //need new ctx manager, cannot reuse
        self.ctxManager = CtxManager::new();
        self.ctxManager.checkpointCtx();
        //cuDevicePrimaryCtxRelease(0); // should release or no?
    }

    pub fn restoreGPUContext(&mut self) {
        self.ctxManager.restoreCtx();
    }
}


impl FatBinManager {
    pub fn new() -> Self {
        Self {
            fatBinHandleVec: Vec::new(),
            fatBinVec: Vec::new(),
            fatBinFuncVec: Vec::new(),
        }
    }

    pub fn unregisterFatbin(&mut self) {
        for elem in &self.fatBinHandleVec {
            let ret = unsafe { cuModuleUnload((*elem).1.clone()  as CUmodule) };
            if ret as u32 != 0 {
                error!(
                    "cuda_mem_manager.rs: error caused by unregisterFatbin(cuModuleUnload): {}",
                    ret as u32
                );
            } else {
                error!("successfully unregister fatbin {}", (*elem).1.clone());
            }
        }
    }

    pub fn restoreFatbin(&mut self) {
        for idx in 0..self.fatBinHandleVec.len() {
            let mut module: u64 = 0;
            let ret = unsafe {
                cuModuleLoadData(
                    &mut module as *mut _ as u64 as *mut CUmodule,
                    self.fatBinVec[idx].as_ptr() as *const c_void,
                )
            };
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreFatbin(cuModuleLoadData): {}", ret as u32);
            }
            self.fatBinHandleVec[idx].1 = module.clone();

            // update module
            if let Some(old_module) = MODULES.lock().get_mut(&self.fatBinHandleVec[idx].0) {
                *old_module = module.clone();
            }
            
            // update function
            for elem in self.fatBinFuncVec[idx].iter() {
                let mut hfunc: u64 = 0;
                let func_name = &elem.1;
                let ret = unsafe {
                    cuModuleGetFunction(
                        &mut hfunc as *mut _ as u64 as *mut CUfunction,
                        module as CUmodule,
                        func_name.clone().as_ptr(),
                    )
                };
                if ret as u32 != 0 {
                    error!("cuda_mem_manager.rs: error caused by restoreFatbin(cuModuleGetFunction): {}", ret as u32);
                }
                if let Some(old_func) = FUNCTIONS.lock().get_mut(&elem.0) {
                    *old_func = hfunc;
                }
            }   
        }
    }
}

impl CtxManager {
    // TODO: how to ckpt event
    // doing: 
    pub fn new() -> Self {
        Self {
            deviceStatus: DeviceStatus::default(),
            streamStatus: Vec::new(),
            cublasStatus: BTreeMap::new(),
            funcStatus: Vec::new(),
        }
    }

    pub fn checkpointCtx(&mut self) {
        self.getDeviceStatus();
        self.getStreamStatus();
        self.getBlasStatus();
        self.getFuncStatus()
    }

    pub fn getDeviceStatus(&mut self) {
        let mut device: c_int = Default::default();
        let ret = unsafe { cudaGetDevice(&mut device) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by ckptCtx(cudaGetDevice): {}", ret as u32);
        }
        self.deviceStatus.currentDev = device;

        let mut cacheConfig: u32 = 0;
        let ret = unsafe { cudaDeviceGetCacheConfig(&mut cacheConfig as *mut _ as u64) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by ckptCtx(cudaDeviceGetCacheConfig): {}", ret as u32);
        }
        self.deviceStatus.cacheConfig = cacheConfig;

        let mut sharedMemConfig: u32 = 0;
        let ret = unsafe { cudaDeviceGetSharedMemConfig(&mut sharedMemConfig as *mut _ as u64) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by ckptCtx(cudaDeviceGetSharedMemConfig): {}", ret as u32);
        }
        self.deviceStatus.sharedMemConfig = sharedMemConfig;
        
        let mut deviceFlag: u32 = 0;
        let ret = unsafe { cudaGetDeviceFlags(&mut deviceFlag) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by ckptCtx(cudaGetDeviceFlags): {}", ret as u32);
        }
        self.deviceStatus.deviceFlags = deviceFlag;
    }  

    pub fn getStreamStatus(&mut self) {
        for (_, value) in STREAMS.lock().iter() {
            let mut flag: u32 = 0;
            let ret = unsafe { cudaStreamGetFlags((*value).clone(), &mut flag) };
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cudaStreamGetFlags): {}", ret as u32);
            }

            let mut priority: i32 = 0;
            let ret = unsafe { cudaStreamGetPriority ((*value).clone(), &mut priority) };
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cudaStreamGetPriority): {}", ret as u32);
            }
            let mut streamType = StreamType::None;
            if flag != 0 {
                streamType = StreamType::Flag;
            }
            if priority != 0 {
                streamType = StreamType::Priority;
            }
            let streamStatus = StreamStatus {
                streamType: streamType,
                flag: flag,
                priority: priority,
            };
            self.streamStatus.push(streamStatus);
        }
    }

    pub fn getBlasStatus(&mut self) {
        for (_, value) in BLASHANDLE.lock().iter() {
            let mut mode: u32 = 0;
            let ret = unsafe { cublasGetMathMode((*value).clone(), &mut mode) };
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cublasGetMathMode): {}", ret as u32);
            }

            let mut stream: u64 = 0;
            let ret = unsafe { cublasGetStream((*value).clone(), &mut stream) };
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cublasGetStream): {}", ret as u32);
            }
            
            match self.cublasStatus.get_mut(value) {
                Some(status) => {
                    status.mathMode = mode;
                    status.stream = stream;
                },
                None => {
                    error!("impossible handle");
                    panic!();
                },
            }
        }
    }

    pub fn getFuncStatus(&mut self) {
        for (_, func) in FUNCTIONS.lock().iter() {
            let mut maxDynamicSharedSizeBytes: i32 = 0;
            let mut preferedSharedMemoryCarveout:i32 = 0;
            let mut requiredClusterWidth: i32 = 0;
            let mut requiredClusterHeight: i32 = 0;
            let mut requiredClusterDepth: i32 = 0;
            let mut nonPortableClusterSizeAllowed: i32 = 0;
            let mut clusterSchedulingPolicyPreference: i32 = 0;
            let ret = unsafe { cuFuncGetAttribute(&mut maxDynamicSharedSizeBytes, 8,func.clone()) }; //CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES  
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncGetAttribute(&mut preferedSharedMemoryCarveout, 9,func.clone()) }; //CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT 
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncGetAttribute(&mut requiredClusterWidth, 11,func.clone()) }; //CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH  
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncGetAttribute(&mut requiredClusterHeight, 12,func.clone()) }; //CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT   
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncGetAttribute(&mut requiredClusterDepth, 13,func.clone()) }; //CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH 
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncGetAttribute(&mut nonPortableClusterSizeAllowed, 14,func.clone()) }; //CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED  
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncGetAttribute(&mut clusterSchedulingPolicyPreference, 15,func.clone()) }; //CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE   
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }
            let status = FuncStatus {
                maxDynamicSharedSizeBytes: maxDynamicSharedSizeBytes,
                preferedSharedMemoryCarveout: preferedSharedMemoryCarveout,
                requiredClusterWidth: requiredClusterWidth,
                requiredClusterHeight: requiredClusterHeight,
                requiredClusterDepth: requiredClusterDepth,
                nonPortableClusterSizeAllowed: nonPortableClusterSizeAllowed,
                clusterSchedulingPolicyPreference: clusterSchedulingPolicyPreference,
            };
            self.funcStatus.push(status);
        }
    }
    
    pub fn restoreCtx(&mut self) {
        // cudaSetDevice
        self.restoreDeviceStatus();
        self.restoreStreamStatus();
        self.restoreCublasStatus();
        self.restoreFuncStatus()
    }

    pub fn restoreDeviceStatus(&mut self) {
        let ret = unsafe { cudaSetDevice(self.deviceStatus.currentDev) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by restoreCtx(restoreDeviceStatus): {}", ret as u32);
        }

        let ret = unsafe { cudaDeviceSetCacheConfig(self.deviceStatus.cacheConfig) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by restoreCtx(cudaDeviceSetCacheConfig): {}", ret as u32);
        }

        let ret = unsafe { cudaDeviceSetSharedMemConfig(self.deviceStatus.sharedMemConfig as u64) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by restoreCtx(cudaDeviceSetSharedMemConfig): {}", ret as u32);
        }

        let ret = unsafe { cudaSetDeviceFlags(self.deviceStatus.deviceFlags) };
        if ret as u32 != 0 {
            error!("cuda_mem_manager.rs: error caused by restoreCtx(cudaSetDeviceFlags): {}", ret as u32);
        }
    }

    pub fn restoreStreamStatus(&mut self) {
        let mut i = 0;
        error!("stream status : {:?}", self.streamStatus[0]);
        for (_, value) in STREAMS.lock().iter_mut() {
            let mut stream: u64 = 0;
            match self.streamStatus[i].streamType {
                StreamType::None => {
                    let ret = unsafe { cudaStreamCreate(&mut stream as *mut _ as *mut cudaStream_t) };
                    if ret as u32 != 0 {
                        error!("cuda_mem_manager.rs: error caused by restoreStreamStatus(cudaStreamCreate): {}", ret as u32);
                    }
                },
                StreamType::Flag => {
                    let ret = unsafe { 
                        cudaStreamCreateWithFlags(&mut stream as *mut _ as *mut cudaStream_t, self.streamStatus[i].flag)
                    };
                    if ret as u32 != 0 {
                        error!("cuda_mem_manager.rs: error caused by restoreStreamStatus(cudaStreamCreateWithFlags): {}", ret as u32);
                    }
                },
                StreamType::Priority => {
                    let ret = unsafe { 
                        cudaStreamCreateWithPriority(&mut stream as *mut _ as *mut cudaStream_t, self.streamStatus[i].flag, self.streamStatus[i].priority)
                    };
                    if ret as u32 != 0 {
                        error!("cuda_mem_manager.rs: error caused by restoreStreamStatus(cudaStreamCreateWithPriority): {}", ret as u32);
                    }
                },
            }
            *value = stream;
            i += 1;
        }
    }

    pub fn restoreCublasStatus(&mut self) {
        for (_, value) in BLASHANDLE.lock().iter_mut() {
            match self.cublasStatus.get(value) {
                Some(status) => {
                    let mut handle = 0;
                    let ret = unsafe { cublasCreate_v2(&mut handle) };
                    if ret as u32 != 0 {
                        error!("cuda_mem_manager.rs: error caused by restoreCublasStatus(cublasCreate_v2): {}", ret as u32);
                    }

                    let ret = unsafe { cublasSetMathMode(handle.clone() as cublasHandle_t, status.mathMode.clone()) };
                    if ret as u32 != 0 {
                        error!("cuda_mem_manager.rs: error caused by restoreCublasStatus(cublasSetMathMode): {}", ret as u32);
                    }

                    let ret = unsafe { cublasSetStream_v2(handle.clone() as cublasHandle_t, status.stream.clone()) };
                    if ret as u32 != 0 {
                        error!("cuda_mem_manager.rs: error caused by restoreCtx(cublasGetStream): {}", ret as u32);
                    }

                    let ret = unsafe { cublasSetWorkspace_v2(handle.clone() as cublasHandle_t, status.workspacePtr, status.workspaceSize) };
                    if ret as u32 != 0 {
                        error!("cuda_mem_manager.rs: error caused by restoreCtx(cublasGetStream): {}", ret as u32);
                    }

                    *value = handle.clone();
                },
                None => {
                    error!("impossible handle");
                    panic!();
                },
            }
        }
    }

    pub fn restoreFuncStatus(&mut self) {
        let mut i= 0;
        for (_, func) in FUNCTIONS.lock().iter_mut() {
            let ret = unsafe { cuFuncSetAttribute(func.clone(), 8,self.funcStatus[i].maxDynamicSharedSizeBytes) }; //CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES  
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncSetAttribute(func.clone(), 9,self.funcStatus[i].preferedSharedMemoryCarveout) }; //CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT 
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncSetAttribute(func.clone(), 11,self.funcStatus[i].requiredClusterWidth) }; //CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH  
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncSetAttribute(func.clone(), 12,self.funcStatus[i].requiredClusterHeight) }; //CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT   
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncSetAttribute(func.clone(), 13,self.funcStatus[i].requiredClusterDepth) }; //CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH 
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncSetAttribute(func.clone(), 14,self.funcStatus[i].nonPortableClusterSizeAllowed) }; //CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED  
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }

            let ret = unsafe { cuFuncSetAttribute(func.clone(), 15,self.funcStatus[i].clusterSchedulingPolicyPreference) }; //CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE   
            if ret as u32 != 0 {
                error!("cuda_mem_manager.rs: error caused by restoreCtx(cuFuncGetAttribute): {}", ret as u32);
            }
            i += 1;
        }
    }
}