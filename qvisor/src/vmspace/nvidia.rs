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

use spin::Mutex;
use std::collections::HashMap;
//use std::collections::BTreeMap;
// use std::ffi::CString;
// use std::ffi::CStr;
// use std::os::raw::*;
// use std::ptr::copy_nonoverlapping;
// use std::slice;
// use std::sync::atomic::{AtomicUsize, Ordering};
// use std::sync::{mpsc, Arc};
use std::sync::mpsc;
use std::thread;
use crate::qlib::qmsg::*;
// use std::time::{Duration, Instant};
use crate::qlib::common::*;
//use crate::qlib::linux_def::SysErr;
use crate::qlib::config::*;
use crate::qlib::proxy::*;
// use crate::qlib::range::Range;
// use crate::xpu::cuda::*;
use crate::xpu::cuda_api::*;
use crate::xpu::cuda_mem_manager::*;
use crate::{QUARK_CONFIG};

use crate::xpu::nvidia_nccl_calls::*;
use crate::xpu::nvidia_cuda_calls::*;
use crate::xpu::nvidia_cu_calls::*;
use crate::xpu::nvidia_cublas_calls::*;
use crate::xpu::nvidia_nvml_calls::*;





// use cuda11_cublasLt_sys::{
//     cublasLtHandle_t, cublasLtMatmulAlgo_t, cublasLtMatmulDesc_t, cublasLtMatmulHeuristicResult_t,
//     cublasLtMatmulPreference_t, cublasLtMatrixLayout_t,
// };
// use cuda_driver_sys::{
//     CUcontext, CUdevice, CUfunction, CUmodule, CUresult,
//     CUstream_st,
// };
use cuda_runtime_sys::cudaStream_t;
// use rcublas_sys::cublasHandle_t;

// use super::{IoVec, MemoryDef};

use super::kernel::SHARESPACE;


lazy_static! {
    pub static ref MEM_RECORDER: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());
    pub static ref MEM_MANAGER: Mutex<MemoryManager> = Mutex::new(MemoryManager::new());
    static ref WORKER_MAP: Mutex<HashMap<u64, mpsc::Sender<u64>>> = Mutex::new(HashMap::new()); //pid - worker tid
    // static ref OFFLOAD_TIMER: AtomicUsize = AtomicUsize::new(0);
    // pub static ref NVIDIA_HANDLERS: NvidiaHandlers = NvidiaHandlers::New();
    // pub static ref FUNC_MAP: BTreeMap<ProxyCommand, (XpuLibrary, &'static str)> = BTreeMap::from([
    //     (ProxyCommand::CudaChooseDevice,(XpuLibrary::CudaRuntime, "cudaChooseDevice")),
    //     (ProxyCommand::CudaDeviceGetAttribute,(XpuLibrary::CudaRuntime, "cudaDeviceGetAttribute")),
    //     (ProxyCommand::CudaDeviceGetByPCIBusId,(XpuLibrary::CudaRuntime, "cudaDeviceGetByPCIBusId")),
    //     (ProxyCommand::CudaDeviceGetCacheConfig,(XpuLibrary::CudaRuntime, "cudaDeviceGetCacheConfig")),
    //     (ProxyCommand::CudaDeviceGetLimit,(XpuLibrary::CudaRuntime, "cudaDeviceGetLimit")),
    //     (ProxyCommand::CudaDeviceGetP2PAttribute,(XpuLibrary::CudaRuntime, "cudaDeviceGetP2PAttribute")),
    //     (ProxyCommand::CudaDeviceGetPCIBusId,(XpuLibrary::CudaRuntime, "cudaDeviceGetPCIBusId")),
    //     (ProxyCommand::CudaDeviceGetSharedMemConfig,(XpuLibrary::CudaRuntime, "cudaDeviceGetSharedMemConfig")),
    //     (ProxyCommand::CudaDeviceGetStreamPriorityRange,(XpuLibrary::CudaRuntime, "cudaDeviceGetStreamPriorityRange")),
    //     (ProxyCommand::CudaDeviceReset,(XpuLibrary::CudaRuntime, "cudaDeviceReset")),
    //     (ProxyCommand::CudaDeviceSetCacheConfig,(XpuLibrary::CudaRuntime, "cudaDeviceSetCacheConfig")),
    //     (ProxyCommand::CudaSetDevice,(XpuLibrary::CudaRuntime, "cudaSetDevice")),
    //     (ProxyCommand::CudaSetDeviceFlags,(XpuLibrary::CudaRuntime, "cudaSetDeviceFlags")),
    //     (ProxyCommand::CudaDeviceSynchronize,(XpuLibrary::CudaRuntime, "cudaDeviceSynchronize")),
    //     (ProxyCommand::CudaGetDevice,(XpuLibrary::CudaRuntime, "cudaGetDevice")),
    //     (ProxyCommand::CudaGetDeviceCount,(XpuLibrary::CudaRuntime, "cudaGetDeviceCount")),
    //     (ProxyCommand::CudaGetDeviceProperties,(XpuLibrary::CudaRuntime, "cudaGetDeviceProperties")),
    //     (ProxyCommand::CudaMalloc,(XpuLibrary::CudaRuntime, "cudaMalloc")),
    //     (ProxyCommand::CudaMemcpy,(XpuLibrary::CudaRuntime, "cudaMemcpy")),
    //     (ProxyCommand::CudaMemcpyAsync,(XpuLibrary::CudaRuntime, "cudaMemcpyAsync")),
    //     (ProxyCommand::CudaRegisterFatBinary,(XpuLibrary::CudaDriver, "cuModuleLoadData")),
    //     (ProxyCommand::CudaRegisterFunction,(XpuLibrary::CudaDriver, "cuModuleGetFunction")),
    //     (ProxyCommand::CudaRegisterVar,(XpuLibrary::CudaDriver, "cuModuleGetGlobal_v2")),
    //     (ProxyCommand::CudaLaunchKernel,(XpuLibrary::CudaDriver, "cuLaunchKernel")),
    //     (ProxyCommand::CudaFree,(XpuLibrary::CudaRuntime, "cudaFree")),
    //     (ProxyCommand::CudaUnregisterFatBinary,(XpuLibrary::CudaDriver, "cuModuleUnload")),
    //     (ProxyCommand::CudaStreamSynchronize,(XpuLibrary::CudaRuntime, "cudaStreamSynchronize")),
    //     (ProxyCommand::CudaStreamCreate,(XpuLibrary::CudaRuntime, "cudaStreamCreate")),
    //     (ProxyCommand::CudaStreamDestroy,(XpuLibrary::CudaRuntime, "cudaStreamDestroy")),
    //     (ProxyCommand::CudaStreamIsCapturing,(XpuLibrary::CudaRuntime, "cudaStreamIsCapturing")),
    //     (ProxyCommand::CuModuleGetLoadingMode,(XpuLibrary::CudaDriver, "cuModuleGetLoadingMode")),
    //     (ProxyCommand::CudaGetLastError,(XpuLibrary::CudaRuntime, "cudaGetLastError")),
    //     (ProxyCommand::CuInit,(XpuLibrary::CudaDriver, "cuInit")),
    //     (ProxyCommand::CuDevicePrimaryCtxGetState,(XpuLibrary::CudaDriver, "cuDevicePrimaryCtxGetState")),
    //     (ProxyCommand::NvmlInit,(XpuLibrary::Nvml, "nvmlInit")),
    //     (ProxyCommand::NvmlInitV2,(XpuLibrary::Nvml, "nvmlInit_v2")),
    //     (ProxyCommand::NvmlShutdown,(XpuLibrary::Nvml, "nvmlShutdown")),
    //     (ProxyCommand::NvmlInitWithFlags,(XpuLibrary::Nvml, "nvmlInitWithFlags")),
    //     (ProxyCommand::NvmlDeviceGetCountV2,(XpuLibrary::Nvml, "nvmlDeviceGetCount_v2")),
//     ]);
}
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
            lasterr: lasterr,
        };
    }

    pub fn ToU64(&self) -> u64 {
        return (self.res as u64) << 32 | (self.lasterr as u64);
    }
}

pub fn NvidiaProxy(
    qmsg: &QMsg,
    containerId: &str,
) -> Result<i64> {
    let mut workerMap = WORKER_MAP.lock();
    match workerMap.get(&qmsg.taskId.data) {
        Some(tx) => {
            tx.send(qmsg as *const _ as u64).unwrap();
        }
        None => {
            let (tx, rx) = mpsc::channel();
            workerMap.insert(qmsg.taskId.data, tx.clone());
            let containerId2 = containerId.to_owned();
            let qmsgPtr = qmsg as *const _ as u64;
            tx.send(qmsgPtr).unwrap();
            let _handle = thread::spawn(move || {
                // error!("start thread");
                let mut msg2 = unsafe {&mut *(qmsgPtr as *mut QMsg) };
                InitNvidia(&containerId2, msg2); // init per thread
                loop {
                    let tmpMsg = rx.recv().unwrap() as *mut QMsg;
                    msg2 = unsafe { &mut (*tmpMsg) };
                    if let Msg::Proxy(msg) = &msg2.msg {
                        if msg.cmd == ProxyCommand::ExitWorkerThread {
                            break;
                        } else {
                            let ret = NvidiaProxyExecute(msg2, &containerId2);
                            match ret {
                                Ok(_res) => {
                                    let currTaskId = msg2.taskId;
                                    SHARESPACE
                                    .scheduler
                                    .ScheduleQ(currTaskId, currTaskId.Queue(), true)
                                }
                                Err(e) => {
                                    // no error
                                    error!("nvidia proxy get error {:?}", e);
                                }
                            }
                        }
                    }
                }
            });
        }
    }
    return Ok(0);
}
pub fn NvidiaProxyExecute(
    qmsg: &QMsg,
    containerId: &str,
) -> Result<i64> {
    let mut result: NvidiaRes = NvidiaRes {
        res: 0,
        lasterr: 0,
    };
    match qmsg.msg {
        Msg::Proxy(msg) => {
            //error!("execute cmd: {:?}", msg.cmd.clone());
            let ret: Result<u32> = Execute(&msg.cmd, &msg.parameters, containerId);
            match ret {
                Ok(v) => {
                    result.res = v;
                }
                Err(_) => unreachable!(),
            }

            match &msg.cmd {
                ProxyCommand::CudaGetErrorName |
                ProxyCommand::CudaGetErrorString |
                ProxyCommand::CudaRegisterFatBinary |
                ProxyCommand::CudaRegisterFunction |
                ProxyCommand::CudaRegisterVar |
                ProxyCommand::CudaUnregisterFatBinary => (),
                _ => {
                    let err = unsafe { cudaGetLastError() };
                    result.lasterr = err as u32;
                }
            } 
        }
        _ => unreachable!(),
    }
    return Ok(result.ToU64() as i64);
}

pub fn Execute(
    cmd: &ProxyCommand,
    parameters: &ProxyParameters,
    _containerId: &str,
) -> Result<u32> {
    match cmd {
        ProxyCommand::None => {
            panic!("get impossible proxy command");
        }
        ProxyCommand::NcclGetVersion => return NcclGetVersion(parameters),
        ProxyCommand::NcclGetUniqueId => return NcclGetUniqueId(parameters),
        ProxyCommand::NcclCommInitRank => return NcclCommInitRank(parameters),
        ProxyCommand::NcclCommInitRankConfig => return NcclCommInitRankConfig(parameters),
        ProxyCommand::NcclCommInitAll => return NcclCommInitAll(parameters),
        ProxyCommand::NcclCommDestroy => return NcclCommDestroy(parameters),
        ProxyCommand::NcclCommAbort => return NcclCommAbort(parameters),
        ProxyCommand::NcclCommCount => return NcclCommCount(parameters),
        ProxyCommand::NcclCommUserRank => return NcclCommUserRank(parameters),
        ProxyCommand::NcclCommCuDevice => return NcclCommCuDevice(parameters),
        ProxyCommand::NcclCommGetAsyncError => return NcclCommGetAsyncError(parameters),
        ProxyCommand::NcclSend => return NcclSend(parameters),
        ProxyCommand::NcclRecv => return NcclRecv(parameters),
        ProxyCommand::NcclGroupStart => return NcclGroupStart(),
        ProxyCommand::NcclGroupEnd => return NcclGroupEnd(),
        ProxyCommand::NcclAllGather => return NcclAllGather(parameters),
        ProxyCommand::NcclAllReduce => return NcclAllReduce(parameters),
        ProxyCommand::NcclReduceScatter => return NcclReduceScatter(parameters),
        ProxyCommand::NcclGetErrorString => return NcclGetErrorString(parameters),

        ProxyCommand::CudaChooseDevice => return CudaChooseDevice(parameters),
        ProxyCommand::CudaDeviceGetAttribute => return CudaDeviceGetAttribute(parameters),
        ProxyCommand::CudaDeviceGetByPCIBusId => return CudaDeviceGetByPCIBusId(parameters),
        ProxyCommand::CudaDeviceGetCacheConfig => return CudaDeviceGetCacheConfig(parameters),
        ProxyCommand::CudaDeviceGetLimit => return CudaDeviceGetLimit(parameters),
        ProxyCommand::CudaDeviceGetP2PAttribute => return CudaDeviceGetP2PAttribute(parameters),
        ProxyCommand::CudaDeviceGetPCIBusId => return CudaDeviceGetPCIBusId(parameters),
        ProxyCommand::CudaDeviceGetSharedMemConfig => return CudaDeviceGetSharedMemConfig(parameters),
        ProxyCommand::CudaDeviceGetStreamPriorityRange => return CudaDeviceGetStreamPriorityRange(parameters),
        ProxyCommand::CudaDeviceSetCacheConfig => return CudaDeviceSetCacheConfig(parameters),
        ProxyCommand::CudaDeviceSetLimit => return CudaDeviceSetLimit(parameters),
        ProxyCommand::CudaDeviceSetSharedMemConfig => return CudaDeviceSetSharedMemConfig(parameters),
        ProxyCommand::CudaSetDevice => return CudaSetDevice(parameters),
        ProxyCommand::CudaSetDeviceFlags => return CudaSetDeviceFlags(parameters),
        ProxyCommand::CudaSetValidDevices => return CudaSetValidDevices(parameters),
        ProxyCommand::CudaDeviceReset => return CudaDeviceReset(),
        ProxyCommand::CudaDeviceSynchronize => return CudaDeviceSynchronize(),
        ProxyCommand::CudaGetDevice => return CudaGetDevice(parameters),
        ProxyCommand::CudaGetDeviceCount => return CudaGetDeviceCount(parameters),
        ProxyCommand::CudaGetDeviceFlags => return CudaGetDeviceFlags(),
        ProxyCommand::CudaGetDeviceProperties => return CudaGetDeviceProperties(parameters),
        ProxyCommand::CudaGetErrorString => return CudaGetErrorString(parameters),
        ProxyCommand::CudaGetErrorName => return CudaGetErrorName(parameters),
        ProxyCommand::CudaPeekAtLastError => return CudaPeekAtLastError(),
        ProxyCommand::CudaMalloc => return CudaMalloc(parameters),
        ProxyCommand::CudaFree => return CudaFree(parameters),
        ProxyCommand::CudaMemGetInfo => return CudaMemGetInfo(parameters),
        ProxyCommand::CudaMemcpy => return CudaMemcpy(parameters),
        ProxyCommand::CudaMemcpyAsync => return CudaMemcpyAsync(parameters),
        ProxyCommand::CudaRegisterFatBinary => return CudaRegisterFatBinary(parameters),
        ProxyCommand::CudaUnregisterFatBinary => return CudaUnregisterFatBinary(parameters),
        ProxyCommand::CudaRegisterFunction => return CudaRegisterFunction(parameters),
        ProxyCommand::CudaRegisterVar => return CudaRegisterVar(parameters),
        ProxyCommand::CudaLaunchKernel => return CudaLaunchKernel(parameters),
        ProxyCommand::CudaStreamSynchronize => return CudaStreamSynchronize(parameters),
        ProxyCommand::CudaStreamCreate => return CudaStreamCreate(parameters),
        ProxyCommand::CudaStreamCreateWithFlags => return CudaStreamCreateWithFlags(parameters),
        ProxyCommand::CudaStreamCreateWithPriority => return CudaStreamCreateWithPriority(parameters),
        ProxyCommand::CudaStreamDestroy => return CudaStreamDestroy(parameters),
        ProxyCommand::CudaStreamGetFlags => return CudaStreamGetFlags(parameters),
        ProxyCommand::CudaStreamGetPriority => return CudaStreamGetPriority(parameters),
        ProxyCommand::CudaStreamIsCapturing => return CudaStreamIsCapturing(parameters),
        ProxyCommand::CudaStreamQuery => return CudaStreamQuery(parameters),
        ProxyCommand::CudaStreamWaitEvent => return CudaStreamWaitEvent(parameters),
        ProxyCommand::CudaThreadExchangeStreamCaptureMode => return CudaThreadExchangeStreamCaptureMode(parameters),
        ProxyCommand::CudaEventCreate => return CudaEventCreate(parameters),
        ProxyCommand::CudaEventCreateWithFlags => return CudaEventCreateWithFlags(parameters),
        ProxyCommand::CudaEventDestroy => return CudaEventDestroy(parameters),
        ProxyCommand::CudaEventElapsedTime => return CudaEventElapsedTime(parameters),
        ProxyCommand::CudaEventQuery => return CudaEventQuery(parameters),
        ProxyCommand::CudaEventRecord => return CudaEventRecord(parameters),
        ProxyCommand::CudaEventSynchronize => return CudaEventSynchronize(parameters),
        ProxyCommand::CudaFuncGetAttributes => return CudaFuncGetAttributes(parameters),
        ProxyCommand::CudaFuncSetAttribute => return CudaFuncSetAttribute(parameters),
        ProxyCommand::CudaFuncSetCacheConfig => return CudaFuncSetCacheConfig(parameters),
        ProxyCommand::CudaFuncSetSharedMemConfig => return CudaFuncSetSharedMemConfig(parameters),
        ProxyCommand::CudaGetLastError => return CudaGetLastError(parameters),
        ProxyCommand::CudaMemset => return CudaMemset(parameters),
        ProxyCommand::CudaMemsetAsync => return CudaMemsetAsync(parameters),
        ProxyCommand::CudaHostAlloc => return CudaHostAlloc(parameters),
        ProxyCommand::CudaFreeHost => match CudeFreeHost(parameters) {
            Err(e) => {
                error!("CudeFreeHost fail with error {:?}", e);
                return Ok(2);
            }
            Ok(n) => return Ok(n),
        }
        ProxyCommand::CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags => return CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(parameters),
        ProxyCommand::CuModuleGetLoadingMode => return CuModuleGetLoadingMode(parameters),
        ProxyCommand::CuInit => return CuInit(parameters),
        ProxyCommand::CuDevicePrimaryCtxGetState => return CuDevicePrimaryCtxGetState(parameters),
        ProxyCommand::NvmlDeviceGetCountV2 => return NvmlDeviceGetCountV2(parameters),
        ProxyCommand::NvmlInitWithFlags => return NvmlInitWithFlags(parameters),
        ProxyCommand::NvmlInit => return NvmlInit(parameters),
        ProxyCommand::NvmlInitV2 => return NvmlInitV2(parameters),
        ProxyCommand::NvmlShutdown => return NvmlShutdown(parameters),
        ProxyCommand::CublasCreateV2 => return CublasCreateV2(parameters),
        ProxyCommand::CublasDestroyV2 => return CublasDestroyV2(parameters),
        ProxyCommand::CublasSetStreamV2 => return CublasSetStreamV2(parameters),
        ProxyCommand::CublasSetWorkspaceV2 => return CublasSetWorkspaceV2(parameters),
        ProxyCommand::CublasSetMathMode => return CublasSetMathMode(parameters),
        ProxyCommand::CublasSgemmStridedBatched => return CublasSgemmStridedBatched(parameters),
        ProxyCommand::CublasLtMatmul => return CublasLtMatmul(parameters),
        ProxyCommand::CublasLtMatmulAlgoGetHeuristic => return CublasLtMatmulAlgoGetHeuristic(parameters),
        ProxyCommand::CublasGetMathMode => return CublasGetMathMode(parameters),
        ProxyCommand::CuCtxGetCurrent => return CuCtxGetCurrent(parameters),
        ProxyCommand::CuModuleLoadData => return CuModuleLoadData(parameters),
        ProxyCommand::CuModuleGetFunction => return CuModuleGetFunction(parameters),
        ProxyCommand::CuModuleUnload => return CuModuleUnload(parameters),
        ProxyCommand::CuLaunchKernel => return CuLaunchKernel(parameters),
        ProxyCommand::CublasSgemmV2 => return CublasSgemmV2(parameters),
        ProxyCommand::CublasGemmEx => return CublasGemmEx(parameters),
        ProxyCommand::CublasGemmStridedBatchedEx => return CublasGemmStridedBatchedEx(parameters),
        _ => { return Ok(0); },
    }
}


fn InitNvidia(_containerId: &str, _qmsg: &QMsg) {
    error!("Init nvidia");
    // cuModuleLoadData requires libnvidia-ptxjitcompiler.so, and nvidia image will mount some host libraries into container,
    // the lib we want to use is locate in /usr/local/cuda/compat/, host version libraries will cause error.
    // if let Msg::Proxy(msg) = qmsg.msg {
        // { // looks like CUDA 12 doesn't need this anymore, comment out for now. TODO: test too see if lower cuda version needs
        //     error!("Init nvidia");
        //     let bytes = unsafe {
        //         std::slice::from_raw_parts(msg.parameters.para4 as *const _, msg.parameters.para5 as usize)
        //     };
        //     let ptxlibPath = std::str::from_utf8(bytes).unwrap();
        //     let ptxlibPathStr = format!("/{}{}", containerId, ptxlibPath);
        //     let ptxlibPath = CString::new(ptxlibPathStr).unwrap();
        //     let _handle = unsafe { libc::dlopen(ptxlibPath.as_ptr() as *const i8, libc::RTLD_LAZY) };
        // }
        let initResult1 = unsafe { cudaSetDevice(0) as u32 };
        let initResult2 = unsafe { cudaDeviceSynchronize() as u32 };
    
        if initResult1 | initResult2 != 0 {
            error!("cuda runtime init error");
        }
    // } // else is impossible
}
pub fn SwapOutMem() -> Result<i64> {
    error!("nvidia rs:SwapOutMem2");
    // let now = Instant::now();
    // let mut totalSize = 0;
    if QUARK_CONFIG.lock().CudaMemType == CudaMemType::UM {
        let memRecorder = MEM_RECORDER.lock();
        let mut iterator = memRecorder.iter();
        while let Some(element) = iterator.next() {
            let ret = unsafe { cudaMemPrefetchAsync(element.0, element.1, -1, 0 as cudaStream_t) }; // -1 means cudaCpuDeviceId
                                                                                                    // totalSize = totalSize + element.1;
                                                                                                    // error!("cudaMemPrefetchAsync to host, ptr: {:x}, count: {}", element.0, element.1);

            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaMemPrefetchAsync to host: {}",
                    ret as u32
                );
            }
        }
        let ret2 = unsafe { cudaStreamSynchronize(0 as cudaStream_t) };
        if ret2 as u32 != 0 {
            error!(
                "nvidia.rs: error caused by cudaMemPrefetchAsync to host: {}",
                ret2 as u32
            );
        }
    } else if QUARK_CONFIG.lock().CudaMemType == CudaMemType::MemPool {
        MEM_MANAGER.lock().checkpointGPUContext();
        MEM_MANAGER.lock().offloadGPUMem();
        MEM_MANAGER.lock().offloadGPUFatbin();
        MEM_MANAGER.lock().offloadGPUContext();
        // totalSize = MEM_MANAGER.lock().gpuManager.currentTotalMem.clone() as usize;
    }
    // error!("total mem is: {}, SwapOutMem time{:?}", totalSize, now.elapsed());
    return Ok(0);
}

pub fn SwapInMem() -> Result<i64> {
    error!("nvidia rs:SwapInMem2");
    // let now = Instant::now();
    // let mut totalSize = 0;
    if QUARK_CONFIG.lock().CudaMemType == CudaMemType::UM {
        let memRecorder = MEM_RECORDER.lock();
        let mut iterator = memRecorder.iter();
        while let Some(element) = iterator.next() {
            let ret = unsafe { cudaMemPrefetchAsync(element.0, element.1, 0, 0 as cudaStream_t) }; // for now, hard coded to device 0
                                                                                                   // totalSize = totalSize + element.1;
                                                                                                   // error!("cudaMemPrefetchAsync back to gpu, ptr: {:x}, count: {}", element.0, element.1);
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaMemPrefetchAsync to gpu: {}",
                    ret as u32
                );
            }
        }
        let ret2 = unsafe { cudaStreamSynchronize(0 as cudaStream_t) };
        if ret2 as u32 != 0 {
            error!(
                "nvidia.rs: error caused by cudaMemPrefetchAsync to gpu: {}",
                ret2 as u32
            );
        }
    } else if QUARK_CONFIG.lock().CudaMemType == CudaMemType::MemPool {
        MEM_MANAGER.lock().restoreGPUBasicContext();
        MEM_MANAGER.lock().restoreGPUFatbin();
        MEM_MANAGER.lock().restoreGPUMem();
        MEM_MANAGER.lock().restoreGPUFullContext();
        // totalSize = MEM_MANAGER.lock().cpuManager.usedLen.clone() as usize;
    }
    // error!("total mem is: {},SwapInMem time:{:?}", totalSize, now.elapsed());
    return Ok(0);
}

// pub struct NvidiaHandlersInner {
//     pub cudaHandler: u64,
//     pub cudaRuntimeHandler: u64,
//     pub nvmlHandler: u64,
//     pub handlers: BTreeMap<ProxyCommand, u64>,
// }

// impl NvidiaHandlersInner {
//     pub fn GetFuncHandler(&mut self, cmd: ProxyCommand) -> Result<u64> {
//         match self.handlers.get(&cmd) {
//             None => {
//                 let func = self.DLSym(cmd)?;
//                 self.handlers.insert(cmd, func);
//                 return Ok(func);
//             }
//             Some(func) => {
//                 return Ok(*func);
//             }
//         }
//     }

//     pub fn DLSym(&self, cmd: ProxyCommand) -> Result<u64> {
//         match FUNC_MAP.get(&cmd) {
//             Some(&pair) => {
//                 let func_name = CString::new(pair.1).unwrap();

//                 let handler = match XPU_LIBRARY_HANDLERS.lock().get(&pair.0) {
//                     Some(functionHandler) => {
//                         error!("function handler got {:?}", functionHandler);
//                         functionHandler.clone()
//                     }
//                     None => {
//                         error!("no function handler found");
//                         0
//                     }
//                 };

//                 let handler: u64 = unsafe {
//                     std::mem::transmute(libc::dlsym(
//                         handler as *mut libc::c_void,
//                         func_name.as_ptr(),
//                     ))
//                 };

//                 if handler != 0 {
//                     error!("got handler {:x}", handler);
//                     return Ok(handler as u64);
//                 }
//             }
//             None => (),
//         }

//         return Err(Error::SysError(SysErr::ENOTSUP));
//     }
// }

// pub struct NvidiaHandlers(Mutex<NvidiaHandlersInner>);

// impl Deref for NvidiaHandlers {
//     type Target = Mutex<NvidiaHandlersInner>;

//     fn deref(&self) -> &Mutex<NvidiaHandlersInner> {
//         &self.0
//     }
// }

// impl NvidiaHandlers {
//     pub fn New() -> Self {
//         let handlers = BTreeMap::new();
//         let inner = NvidiaHandlersInner {
//             cudaHandler: 0,
//             cudaRuntimeHandler: 0,
//             nvmlHandler:0,
//             handlers: handlers,
//         };
//         return Self(Mutex::new(inner));
//     }

//     // trigger the NvidiaHandlers initialization
//    pub fn Trigger(&self, containerId: &str, ptxlibPath: &str) {
//         // cuModuleLoadData requires libnvidia-ptxjitcompiler.so, and nvidia image will mount some host libraries into container,
//         // the lib we want to use is locate in /usr/local/cuda/compat/, host version libraries will cause error.
//         let ptxlibPathStr = format!("/{}{}", containerId, ptxlibPath);
//         let ptxlibPath = CString::new(ptxlibPathStr).unwrap();
//         let handle = unsafe { libc::dlopen(ptxlibPath.as_ptr() as *const i8, libc::RTLD_LAZY) };

//         let initResult1 = unsafe {cudaSetDevice(0) as u32};
//         let initResult2 = unsafe {cudaDeviceSynchronize() as u32};

//         if initResult1 | initResult2 != 0 {
//             error!("cuda runtime init error");
//         }
//     }

//     pub fn GetFuncHandler(&self, cmd: ProxyCommand) -> Result<u64> {
//         let mut inner = self.lock();
//         return inner.GetFuncHandler(cmd);
//     }
// }
