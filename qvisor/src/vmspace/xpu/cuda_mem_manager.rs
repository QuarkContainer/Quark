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

pub use u32 as CUmemLocationType;
pub use u32 as CUmemAllocationHandleType;
pub use u32 as CUmemAllocationType;
pub use u32 as CUmemAccess_flags;
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