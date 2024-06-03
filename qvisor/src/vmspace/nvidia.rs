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

use serde::ser::Impossible;
//use core::ops::Deref;
use spin::Mutex;
use core::ops::Deref;
use std::collections::HashMap;
//use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::*;
use std::ptr::{copy_nonoverlapping,null_mut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use crate::qlib::qmsg::*;
use crate::qlib::ShareSpace;
// use std::time::{Duration, Instant};

use crate::qlib::common::*;
//use crate::qlib::linux_def::SysErr;
use crate::qlib::config::*;
use crate::qlib::proxy::*;
use crate::qlib::range::Range;
use crate::runc::container::nvidia;
use crate::xpu::cuda::*;
use crate::xpu::cuda_api::*;
use crate::xpu::cuda_mem_manager::*;
use crate::QUARK_CONFIG;
use std::ffi::CStr;
use cuda11_cublasLt_sys::{
    cublasLtHandle_t, cublasLtMatmulAlgo_t, cublasLtMatmulDesc_t, cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t, cublasLtMatrixLayout_t,
};
use cuda_driver_sys::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUfunction_attribute, CUmodule, CUresult,
    CUstream_st,
};
use cuda_runtime_sys::{
    cudaDeviceAttr, cudaDeviceP2PAttr, cudaDeviceProp, cudaError_t, cudaEvent_t,
    cudaFuncAttributes, cudaFuncCache, cudaLimit, cudaMemAttachGlobal, cudaStream_t,
    cudaSharedMemConfig, cudaStreamCaptureMode, cudaStreamCaptureStatus,
};
use rcublas_sys::{cublasHandle_t};

use super::kernel::SHARESPACE;

lazy_static! {
    static ref MEM_RECORDER: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());
    static ref MEM_MANAGER: Mutex<MemoryManager> = Mutex::new(MemoryManager::new());
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
            lasterr: lasterr
        }
    }

    pub fn ToU64(&self) -> u64 {
        return (self.res as u64) << 32 | (self.lasterr as u64)
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
            let handle = thread::spawn(move || {
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
                                Ok(res) => {
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
    error!("proxy return");
    return Ok(0);
}
pub fn NvidiaProxyExecute(
    qmsg: &QMsg,
    containerId: &str,
) -> Result<i64> {
    error!("in NvidiaProxyExecute");
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
    containerId: &str,
) -> Result<u32> {
    match cmd {
        ProxyCommand::None => {
            panic!("get impossible proxy command");
        }
        ProxyCommand::NcclGetUniqueId => {
            error!("nvidia.rs: ncclGetUniqueId");
            let mut ncclUniqueId_ns: NcclUniqueId = NcclUniqueId::default();
            let ret = unsafe { ncclGetUniqueId(&mut ncclUniqueId_ns) };
            // error!("nvidia.rs: ncclGetUniqueId_after ncclGetUniqueId call");

            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclGetUniqueId: {}", ret as u32);
            } else {
                unsafe { *(parameters.para1 as *mut _)= ncclUniqueId_ns };
            }
            return Ok(ret as u32);  

        }
        ProxyCommand::NcclCommInitRank => {
            error!("nvidia.rs: ncclCommInitRank");
            let mut ncclComm_t_: NcclCommT = null_mut();
            let ncclUniqueId_ns = unsafe { *(parameters.para3 as *const NcclUniqueId) };
            let ret = unsafe { ncclCommInitRank(&mut ncclComm_t_, parameters.para2 as i32, ncclUniqueId_ns, parameters.para4 as i32) };

            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclCommInitRank: {}", ret as u32);
            } else {
                unsafe { *(parameters.para1 as *mut _)= ncclComm_t_ };
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclCommInitRankConfig => {
            error!("nvidia.rs: ncclCommInitRankConfig");
            let mut ncclComm_t_: NcclCommT = null_mut();
            let ncclUniqueId_ns = unsafe { *(parameters.para3 as *const NcclUniqueId) };
            // let config = unsafe { *(parameters.para5 as *const NcclConfig) };

            let ret = unsafe { ncclCommInitRankConfig(&mut ncclComm_t_, parameters.para2 as i32, ncclUniqueId_ns, parameters.para4 as i32, parameters.para5 as *const NcclConfig) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclCommInitRankConfig: {}", ret as u32);
            } else {
                unsafe { *(parameters.para1 as *mut _)= ncclComm_t_ };
            }
            return Ok(ret as u32);

        }
        ProxyCommand::NcclCommInitAll => {
            error!("nvidia.rs: ncclCommInitAll");
            // let mut ncclComm_t_s:Vec<NcclCommT> = Vec::with_capacity(parameters.para2 as usize);
            // unsafe {
            //     ncclComm_t_s.set_len(parameters.para2 as usize);
            // }
            let mut ncclComm_t_s: Vec<NcclCommT> = vec![null_mut(); parameters.para2 as usize];  
            // error!("nvidia.rs: ncclCommInitAlbefrel  call");
            // let ret = unsafe { ncclCommInitAll(&mut ncclComm_t_s[0] as *mut NcclCommT, parameters.para2 as i32, parameters.para3 as *const c_int) };
            let ret = unsafe { ncclCommInitAll(ncclComm_t_s.as_mut_ptr(), parameters.para2 as i32, parameters.para3 as *const c_int) };
            // error!("nvidia.rs: ncclCommInitAl after call");
            // if ret as u32 != 0 {
            //     error!("nvidia.rs: error caused by ncclCommInitAll: {}", ret as u32);
            // }
            // for i in 0..parameters.para2 as u64 {
            //     unsafe { *((parameters.para1 + i) as *mut u8 as *mut NcclCommT)= ncclComm_t_s[i as usize] };
            // }
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclCommInitAll: {}", ret as u32);
            } else {
                for i in 0..parameters.para2 as usize {
                    unsafe {
                        *(parameters.para1 as *mut NcclCommT).add(i) = ncclComm_t_s[i];
                    }
                }
                // error!("nvidia.rs: ncclCommInitAll after nccl comm vec copy");
            }
            // error!("nvidia.rs: ncclCommInitAl after nccl comm vec copy");
            return Ok(ret as u32);

        }
        ProxyCommand::NcclCommDestroy => {
            error!("nvidia.rs: ncclCommDestroy");
            let ret = unsafe { ncclCommDestroy(parameters.para1 as NcclCommT) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclCommDestroy: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclCommAbort => {
            error!("nvidia.rs: ncclCommAbort");
            let ret = unsafe { ncclCommAbort(parameters.para1 as NcclCommT) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclCommAbort: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclCommCount => {
            error!("nvidia.rs: ncclCommCount");
            let mut count: c_int = 0;
            let ret = unsafe { ncclCommCount(parameters.para1 as NcclCommT, &mut count) };

            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclCommCount: {}", ret as u32);
            } else {
                unsafe { *(parameters.para2 as *mut c_int) = count };
            }
            return Ok(ret as u32);
            
        }
        ProxyCommand::NcclCommUserRank => {
            error!("nvidia.rs: ncclCommUserRank");
            let mut rank: c_int = 0;
            let ret = unsafe { ncclCommUserRank(parameters.para1 as NcclCommT, &mut rank) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclCommUserRank: {}", ret as u32);
            } else {
                unsafe { *(parameters.para2 as *mut c_int) = rank };
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclSend => {
            error!("nvidia.rs: ncclSend");
            let info = unsafe { *(parameters.para2 as *const u8 as *const NcclSendRecvInfo) };

            let ret = unsafe { ncclSend(parameters.para1 as *const c_void, info.count as usize, info.datatype, info.peer, info.comm as NcclCommT, info.stream as cudaStream_t) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclSend: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclRecv => {
            error!("nvidia.rs: ncclRecv");
            let info = unsafe { *(parameters.para2 as *const u8 as *const NcclSendRecvInfo) };
            let ret = unsafe { ncclRecv(parameters.para1 as *mut c_void, info.count as usize, info.datatype, info.peer, info.comm as NcclCommT, info.stream as cudaStream_t) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclRecv: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclGroupStart => {
            error!("nvidia.rs: ncclGroupStart");
            let ret = unsafe { ncclGroupStart() };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclGroupStart: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclGroupEnd => {
            error!("nvidia.rs: ncclGroupEnd");
            let ret = unsafe { ncclGroupEnd() };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclGroupEnd: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclAllGather => {
            error!("nvidia.rs: ncclAllGather");
            let info = unsafe { *(parameters.para3 as *const u8 as *const NcclAllGatherReduceInfo) };
            let ret = unsafe { ncclAllGather(parameters.para1 as *const c_void, parameters.para2 as *mut c_void, info.count as usize, info.datatype, info.comm as NcclCommT, info.stream as cudaStream_t) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclAllGather: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclAllReduce => {
            error!("nvidia.rs: ncclAllReduce");
            let info = unsafe { *(parameters.para3 as *const u8 as *const NcclAllGatherReduceInfo) };
            let ret = unsafe { ncclAllReduce(parameters.para1 as *const c_void, parameters.para2 as *mut c_void, info.count as usize, info.datatype, info.op, info.comm as NcclCommT, info.stream as cudaStream_t) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclAllReduce: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclReduceScatter => {
            error!("nvidia.rs: ncclReduceScatter");
            let info = unsafe { *(parameters.para3 as *const u8 as *const NcclAllGatherReduceInfo) };
            let ret = unsafe { ncclReduceScatter(parameters.para1 as *const c_void, parameters.para2 as *mut c_void, info.count as usize, info.datatype, info.op, info.comm as NcclCommT, info.stream as cudaStream_t) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by ncclReduceScatter: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        ProxyCommand::NcclGetErrorString => {
            let ret = unsafe {
                ncclGetErrorString(parameters.para1 as u32)
            };
            
            let c_str: &CStr = unsafe { std::ffi::CStr::from_ptr(ret) };

            // Convert CStr to &str and handle potential UTF-8 errors
            let error_str= c_str.to_str().expect("Invalid UTF-8 data");

            let error_string = error_str.to_string();
            unsafe { *(parameters.para2 as *mut String) = error_string };
            return Ok(0 as u32);
        }
        ProxyCommand::CudaChooseDevice => {
            // error!("nvidia.rs: cudaChooseDevice");
            let mut device: c_int = Default::default();
            let deviceProp = unsafe { *(parameters.para2 as *const u8 as *const cudaDeviceProp) };

            let ret = unsafe { cudaChooseDevice(&mut device, &deviceProp) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaChooseDevice: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut i32) = device };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetAttribute => {
            // error!("nvidia.rs: cudaDeviceGetAttribute");
            let attribute: cudaDeviceAttr =
                unsafe { *(&parameters.para2 as *const _ as u64 as *mut cudaDeviceAttr) };
            let mut value: c_int = 0;
            let device: c_int = parameters.para3 as c_int;

            let ret = unsafe { cudaDeviceGetAttribute(&mut value, attribute, device) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaDeviceGetAttribute: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut i32) = value as i32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetByPCIBusId => {
            // error!("nvidia.rs: cudaDeviceGetByPCIBusId");
            let mut device: c_int = 0;
            let bytes = unsafe {
                std::slice::from_raw_parts(parameters.para2 as *const u8, parameters.para3 as usize)
            };
            let PCIBusId = std::str::from_utf8(bytes).unwrap();
            let cstring = CString::new(PCIBusId).unwrap();

            let ret = unsafe { cudaDeviceGetByPCIBusId(&mut device, cstring.as_ptr()) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceGetByPCIBusId: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut i32) = device as i32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetCacheConfig => {
            // error!("nvidia.rs: cudaDeviceGetCacheConfig");
            let mut cacheConfig: cudaFuncCache = unsafe { *(parameters.para1 as *mut _) };

            let ret = unsafe { cudaDeviceGetCacheConfig(&mut cacheConfig) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceGetCacheConfig: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = cacheConfig as u32; }   
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetLimit => {
            //error!("nvidia.rs: cudaDeviceGetLimit");
            let limitType: cudaLimit =
                unsafe { *(&parameters.para2 as *const _ as *mut cudaLimit) };
            let mut limit: usize = 0;

            let ret = unsafe { cudaDeviceGetLimit(&mut limit, limitType) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceGetLimit: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = limit };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetP2PAttribute => {
            //error!("nvidia.rs: cudaDeviceGetP2PAttribute");
            let mut value: libc::c_int = 0;
            let attribute =
                unsafe { *(&parameters.para2 as *const _ as u64 as *mut cudaDeviceP2PAttr) };

            let ret = unsafe {
                cudaDeviceGetP2PAttribute(
                    &mut value,
                    attribute,
                    parameters.para3 as c_int,
                    parameters.para4 as c_int,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceGetP2PAttribute: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = value as i32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetPCIBusId => {
            //error!("nvidia.rs: cudaDeviceGetPCIBusId");
            let mut pciBusId: Vec<c_char> = vec![0; parameters.para2 as usize];

            let ret = unsafe {
                cudaDeviceGetPCIBusId(
                    pciBusId.as_mut_ptr(),
                    parameters.para2 as c_int,
                    parameters.para3 as c_int,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceGetPCIBusId: {}",
                    ret as u32
                );
            }

            let pciBusString =
                String::from_utf8(pciBusId.iter().map(|&c| c as u8).collect()).unwrap();
            unsafe { *(parameters.para1 as *mut String) = pciBusString.to_owned() };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetSharedMemConfig => {
            //error!("nvidia.rs: cudaDeviceGetSharedMemConfig");
            let mut sharedMemConfig: cudaSharedMemConfig =
                unsafe { *(parameters.para1 as *mut cudaSharedMemConfig) };

            let ret = unsafe { cudaDeviceGetSharedMemConfig(&mut sharedMemConfig) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceGetSharedMemConfig: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut u32) = sharedMemConfig as u32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceGetStreamPriorityRange => {
            //error!("nvidia.rs: cudaDeviceGetStreamPriorityRange");
            let mut leastPriority: c_int = unsafe { *(parameters.para1 as *mut _) };
            let mut greatestPriority: c_int = unsafe { *(parameters.para2 as *mut _) };

            let ret = unsafe {
                cudaDeviceGetStreamPriorityRange(&mut leastPriority, &mut greatestPriority)
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceGetStreamPriorityRange: {}",
                    ret as u32
                );
            }

            unsafe {
                *(parameters.para1 as *mut _) = leastPriority;
                *(parameters.para2 as *mut _) = greatestPriority;
            }
            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceSetCacheConfig => {
            //error!("nvidia.rs: cudaDeviceSetCacheConfig");
            let ret = unsafe { cudaDeviceSetCacheConfig(parameters.para1 as u32) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaDeviceSetCacheConfig: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceSetLimit => {
            //error!("nvidia.rs: cudaDeviceSetLimit");
            let ret = unsafe {
                cudaDeviceSetLimit(
                    *(&parameters.para1 as *const _ as u64 as *mut cudaLimit),
                    parameters.para2 as usize,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceSetLimit: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceSetSharedMemConfig => {
            //error!("nvidia.rs: CudaDeviceSetSharedMemConfig");
            let ret = unsafe {
                cudaDeviceSetSharedMemConfig(
                    *(&parameters.para1 as *const _ as u64 as *mut cudaSharedMemConfig),
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceSetSharedMemConfig: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaSetDevice => {
            // error!("nvidia.rs: cudaSetDevice {}",parameters.para1 as i32);
            let ret = unsafe { cudaSetDevice(parameters.para1 as i32) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaSetDevice: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaSetDeviceFlags => {
            //error!("nvidia.rs: cudaSetDeviceFlags");
            let ret = unsafe { cudaSetDeviceFlags(parameters.para1 as u32) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaSetDeviceFlags: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaSetValidDevices => {
            //error!("nvidia.rs: cudaSetValidDevices");
            let ret = unsafe {
                cudaSetValidDevices(parameters.para1 as *mut c_int, parameters.para2 as i32)
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaSetValidDevices: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }

        ProxyCommand::CudaDeviceReset => {
            //error!("nvidia.rs: cudaDeviceReset");
            let ret = unsafe { cudaDeviceReset() };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaDeviceReset: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaDeviceSynchronize => {
            // error!("nvidia.rs: cudaDeviceSynchronize");
            let ret = unsafe { cudaDeviceSynchronize() };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaDeviceSynchronize: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaGetDevice => {
            //error!("nvidia.rs: cudaGetDevice");
            let mut device: c_int = Default::default();

            let ret = unsafe { cudaGetDevice(&mut device) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaGetDevice: {}", ret as u32);
            }

            unsafe { *(parameters.para1 as *mut i32) = device };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaGetDeviceCount => {
            //error!("nvidia.rs: cudaGetDeviceCount");
            let mut deviceCount: c_int = 0;

            let ret = unsafe { cudaGetDeviceCount(&mut deviceCount) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaGetDeviceCount: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut i32) = deviceCount as i32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaGetDeviceFlags => {
            //error!("nvidia.rs: cudaGetDeviceFlags");
            let mut deviceFlags = 0;

            let ret = unsafe { cudaGetDeviceFlags(&mut deviceFlags) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaGetDeviceFlags: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaGetDeviceProperties => {
            //error!("nvidia.rs: cudaGetDeviceProperties");
            let device = parameters.para2;
            let mut deviceProp: cudaDeviceProp = Default::default();

            let ret = unsafe { cudaGetDeviceProperties(&mut deviceProp, device as c_int) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaGetDeviceProperties: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = deviceProp };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaGetErrorString => {
            //error!("nvidia.rs: cudaGetErrorString");
            let ptr = unsafe {
                cudaGetErrorString(parameters.para1 as u32)
            };

            let cStr = unsafe { std::ffi::CStr::from_ptr(ptr) };
            let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
            let errorString = errorStr.to_string();
            unsafe { *(parameters.para2 as *mut String) = errorString };
            return Ok(0 as u32);
        }
        ProxyCommand::CudaGetErrorName => {
            //error!("nvidia.rs: cudaGetErrorName");
            let ptr = unsafe {
                cudaGetErrorName(parameters.para1 as u32)
            };

            let cStr = unsafe { std::ffi::CStr::from_ptr(ptr) };
            let errorStr = cStr.to_str().expect("Invalid UTF-8 data");
            let errorString = errorStr.to_string();
            unsafe { *(parameters.para2 as *mut String) = errorString };
            return Ok(0 as u32);
        }
        ProxyCommand::CudaPeekAtLastError => {
            //error!("nvidia.rs: cudaPeekAtLastError");
            let ret = unsafe { cudaPeekAtLastError() };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaPeekAtLastError: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaMalloc => {
            // error!("nvidia.rs: cudaMalloc, mode{:?}", QUARK_CONFIG.lock().CudaMode);
            if QUARK_CONFIG.lock().CudaMemType == CudaMemType::UM {
                // error!("nvidia.rs: CudaMode::fastswitch()");
                let mut para1 = parameters.para1 as *mut c_void;

                let ret = unsafe {
                    cudaMallocManaged(
                        &mut para1 as *mut _ as u64,
                        parameters.para2 as usize,
                        cudaMemAttachGlobal,
                    )
                };
                if ret as u32 != 0 {
                    error!(
                        "nvidia.rs: error caused by cudaMallocManaged: {}",
                        ret as u32
                    );
                } else {
                    MEM_RECORDER
                        .lock()
                        .push((para1 as u64, parameters.para2 as usize));
                }

                // while let Some(element) = iterator.next() {
                //     total_mem = total_mem + element.1;
                // }

                // let location = cudaMemLocation {
                //     type_: cudaMemLocationType::cudaMemLocationTypeDevice,
                //     id: 0, // device 0
                // };
                // let ret = unsafe { cudaMemAdvise_v2( para1, parameters.para2 as usize, cudaMemoryAdvise::cudaMemAdviseSetReadMostly, location) };
                // if ret as u32 != 0 {
                //     error!("nvidia.rs: error caused by cudaMalloc(cudaMemAdvise_v2): {}", ret as u32);
                // }
                // let ret = unsafe { cudaMemAdvise_v2( para1, parameters.para2 as usize, cudaMemoryAdvise::cudaMemAdviseSetPreferredLocation, location) };
                // if ret as u32 != 0 {
                //     error!("nvidia.rs: error caused by cudaMalloc(cudaMemAdvise_v2): {}", ret as u32);
                // }
                unsafe { *(parameters.para1 as *mut u64) = para1 as u64 };
                // error!("nvidia.rs: malloc ptr:{:x}, size:{:x}", para1 as u64, parameters.para2);
                return Ok(ret as u32);

                
            } else if QUARK_CONFIG.lock().CudaMemType == CudaMemType::MemPool {
                let (addr,ret) = MEM_MANAGER.lock().gpuManager.alloc(parameters.para2 as usize);
                unsafe { *(parameters.para1 as *mut u64) = addr};
                if ret != 0 {
                    error!("mem pool failed to alloc");
                }
                return Ok(ret as u32);
            } else {
                // error!("nvidia.rs: CudaMode::Default()");
                let mut para1 = parameters.para1 as *mut c_void;

                let ret = unsafe {
                    cudaMalloc(
                        &mut para1 as *mut _ as *mut *mut _ as *mut *mut c_void,
                        parameters.para2 as usize,
                    )
                };
                if ret as u32 != 0 {
                    error!("nvidia.rs: error caused by cudaMalloc: {}", ret as u32);
                }

                unsafe { *(parameters.para1 as *mut u64) = para1 as u64 };
                return Ok(ret as u32);
            }
        }
        ProxyCommand::CudaFree => {
            //error!("nvidia.rs: cudaFree");
            if QUARK_CONFIG.lock().CudaMemType == CudaMemType::UM {
                let memRecorder = MEM_RECORDER.lock();
                let index = memRecorder
                    .iter()
                    .position(|&r| r.0 == parameters.para1 as u64)
                    .unwrap();
                MEM_RECORDER.lock().remove(index);
            }

            let ret = unsafe { cudaFree(parameters.para1 as *mut c_void) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaFree: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaMemcpy => {
            return CudaMemcpy(parameters);
        }
        ProxyCommand::CudaMemcpyAsync => {
            return CudaMemcpyAsync(parameters);
        }
        ProxyCommand::CudaRegisterFatBinary => {
            // error!("nvidia.rs: cudaRegisterFatBinary");

            let fatElfHeader = unsafe { &*(parameters.para2 as *const u8 as *const FatElfHeader) };
            // error!("fatElfHeader magic is :{:x}, version is :{:x}, header size is :{:x}, size is :{:x}", fatElfHeader.magic, fatElfHeader.version, fatElfHeader.header_size, fatElfHeader.size);
            let moduleKey = parameters.para3;
            match GetFatbinInfo(parameters.para2, fatElfHeader) {
                Ok(_) => {}
                Err(e) => {
                    return Err(e);
                }
            }

            let mut module: u64 = 0;
            let ret = unsafe {
                cuModuleLoadData(
                    &mut module as *mut _ as u64 as *mut CUmodule,
                    parameters.para2 as *const c_void,
                )
            };
            // TODO: try this https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaRegisterFatBinary(cuModuleLoadData): {}",
                    ret as u32
                );
            }

            // unsafe{ cudaDeviceSynchronize();}
            MODULES.lock().insert(moduleKey, module);
            let fatbinSize: usize = parameters.para1 as usize;
            let mut fatbinData = Vec::with_capacity(parameters.para1 as usize);
            unsafe { copy_nonoverlapping(parameters.para2 as *const u8, fatbinData.as_mut_ptr(), fatbinSize); }
            //MEM_MANAGER.lock().fatBinManager.fatBinVec.push(fatbinData);
            //MEM_MANAGER.lock().fatBinManager.fatBinHandleVec.push((moduleKey, module));
            //MEM_MANAGER.lock().fatBinManager.fatBinFuncVec.push(Vec::new());
            // error!("insert module: {:x} -> {:x}", moduleKey, module);
            return Ok(ret as u32);
        }
        ProxyCommand::CudaUnregisterFatBinary => {
            //error!("nvidia.rs: CudaUnregisterFatBinary");
            let moduleKey = parameters.para1;

            let module = match MODULES.lock().get(&moduleKey) {
                Some(module) => module.clone(),
                None => {
                    error!("CudaUnregisterFatBinary: no module be found with this fatcubinHandle: {:x}", moduleKey);
                    moduleKey.clone()
                }
            };

            let ret = unsafe { cuModuleUnload(module as *const u64 as CUmodule) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaUnregisterFatBinary(cuModuleUnload): {}",
                    ret as u32
                );
            }

            // delete the module
            //MODULES.lock().remove(&moduleKey);
            return Ok(ret as u32);
        }
        ProxyCommand::CudaRegisterFunction => {
            //error!("nvidia.rs: cudaRegisterFunction");
            let info = unsafe { &*(parameters.para1 as *const u8 as *const RegisterFunctionInfo) };
            let bytes = unsafe {
                std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize)
            };
            let deviceName = std::str::from_utf8(bytes).unwrap();

            let mut module = match MODULES.lock().get(&info.fatCubinHandle) {
                Some(module) => module.clone(),
                None => {
                    error!(
                        "CudaRegisterFunction: no module be found with this fatcubinHandle: {:x}",
                        info.fatCubinHandle
                    );
                    info.fatCubinHandle.clone()
                }
            };

            let mut hfunc: u64 = 0;
            let func_name = CString::new(deviceName).unwrap();
            let ret = unsafe {
                // cuda_driver_sys::
                cuModuleGetFunction(
                    &mut hfunc as *mut _ as u64 as *mut CUfunction,
                    *(&mut module as *mut _ as u64 as *mut CUmodule),
                    func_name.clone().as_ptr(),
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaRegisterFunction(cuModuleGetFunction): {}",
                    ret as u32
                );
            }

            //unsafe{ cudaDeviceSynchronize(); }
            FUNCTIONS.lock().insert(info.hostFun, hfunc);

            let kernelInfo = match KERNEL_INFOS.lock().get(&deviceName.to_string()) {
                Some(kernelInformations) => kernelInformations.clone(),
                None => {
                    //error!("No kernel infos found with this deviceName : {}", deviceName);
                    Arc::new(KernelInfo::default())
                }
            };
            let paramInfo = parameters.para3 as *const u8 as *mut ParamInfo;
            unsafe {
                (*paramInfo).paramNum = kernelInfo.paramNum;
                for i in 0..(*paramInfo).paramNum {
                    (*paramInfo).paramSizes[i] = kernelInfo.paramSizes[i];
                }
            }

            //let index = MEM_MANAGER.lock().fatBinManager.fatBinHandleVec.iter().position(|&r| r.0 == info.fatCubinHandle).unwrap();
            //MEM_MANAGER.lock().fatBinManager.fatBinFuncVec[index].push((info.hostFun, func_name));

            return Ok(ret as u32);
        }
        ProxyCommand::CudaRegisterVar => {
            //error!("nvidia.rs: cudaRegisterFunction");
            let info = unsafe { *(parameters.para1 as *const u8 as *const RegisterVarInfo) };
            let bytes = unsafe {
                std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize)
            };
            let deviceName = std::str::from_utf8(bytes).unwrap();

            let module = match MODULES.lock().get(&info.fatCubinHandle) {
                Some(module) => module.clone(),
                None => {
                    error!(
                        "CudaRegisterVar: no module be found with this fatcubinHandle: {}",
                        info.fatCubinHandle
                    );
                    info.fatCubinHandle.clone()
                }
            };

            let mut devicePtr: CUdeviceptr = 0;
            let mut dSize: usize = 0;
            let ownded_name = CString::new(deviceName).unwrap();
            let name = ownded_name.as_ptr();
            let ret = unsafe {
                cuModuleGetGlobal_v2(
                    &mut devicePtr,
                    &mut dSize,
                    (module as *const u64) as CUmodule,
                    name,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaRegisterVar(cuModuleGetGlobal_v2): {}",
                    ret as u32
                );
            }

            //GLOBALS.lock().insert(info.hostVar, devicePtr);
            // TODO: cudaMemcpyToSymbol may need to store this info, no need for pytorch for now

            return Ok(ret as u32);
        }
        ProxyCommand::CudaLaunchKernel => {
            //error!("nvidia.rs: cudaLaunchKernel");
            let info = unsafe { &*(parameters.para1 as *const u8 as *const LaunchKernelInfo) };
            let func = match FUNCTIONS.lock().get(&info.func) {
                Some(func) => func.clone(),
                None => {
                    //error!("no CUfunction has been found {:x}", info.func);
                    0
                }
            };

            let ret: CUresult = unsafe {
                cuLaunchKernel(
                    func as CUfunction,
                    info.gridDim.x,
                    info.gridDim.y,
                    info.gridDim.z,
                    info.blockDim.x,
                    info.blockDim.y,
                    info.blockDim.z,
                    info.sharedMem as u32,
                    info.stream as *mut CUstream_st,
                    info.args as *mut *mut c_void,
                    0 as *mut *mut c_void,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaLaunchKernel(cuLaunchKernel): {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamSynchronize => {
            //error!("nvidia.rs: cudaStreamSynchronize");
            let ret = unsafe { cudaStreamSynchronize(parameters.para1 as cudaStream_t) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamSynchronize: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamCreate => {
            //error!("nvidia.rs: cudaStreamCreate");
            let mut s: cudaStream_t = unsafe { *(parameters.para1 as *mut u64) as cudaStream_t };

            let ret = unsafe { cudaStreamCreate(&mut s) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamCreate: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut u64) = s as u64 };
            return Ok(ret as u32);
        }

        ProxyCommand::CudaStreamCreateWithFlags => {
            //error!("nvidia.rs: cudaStreamCreateWithFlags");
            let mut stream: cudaStream_t =
                unsafe { *(parameters.para1 as *mut u64) as cudaStream_t };

            let ret = unsafe { cudaStreamCreateWithFlags(&mut stream, parameters.para2 as u32) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamCreateWithFlags: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut u64) = stream as u64 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamCreateWithPriority => {
            //error!("nvidia.rs: CudaStreamCreateWithPriority");
            let mut stream: cudaStream_t =
                unsafe { *(parameters.para1 as *mut u64) as cudaStream_t };

            let ret = unsafe {
                cudaStreamCreateWithPriority(
                    &mut stream,
                    parameters.para2 as u32,
                    parameters.para3 as i32,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamCreateWithPriority: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut u64) = stream as u64 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamDestroy => {
            //error!("nvidia.rs: cudaStreamDestroy");
            let ret = unsafe { cudaStreamDestroy(parameters.para1 as cudaStream_t) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamDestroy: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamGetFlags => {
            //error!("nvidia.rs: CudaStreamGetFlags");
            let mut flags: u32 = 0;

            let ret = unsafe { cudaStreamGetFlags(parameters.para1 as cudaStream_t, &mut flags) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamGetFlags: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para2 as *mut u32) = flags };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamGetPriority => {
            // error!("nvidia.rs: CudaStreamGetPriority");
            let mut priority: i32 = 0;

            let ret =
                unsafe { cudaStreamGetPriority(parameters.para1 as cudaStream_t, &mut priority) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamGetPriority: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para2 as *mut i32) = priority };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamIsCapturing => {
            //error!("nvidia.rs: cudaStreamIsCapturing");
            let mut captureStatus: cudaStreamCaptureStatus =
                unsafe { *(parameters.para2 as *mut _) };

            let ret = unsafe {
                cudaStreamIsCapturing(parameters.para1 as cudaStream_t, &mut captureStatus)
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamIsCapturing: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para2 as *mut _) = captureStatus as u32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamQuery => {
            //error!("nvidia.rs: cudaStreamQuery");
            let ret = unsafe { cudaStreamQuery(parameters.para1 as cudaStream_t) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaStreamQuery: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaStreamWaitEvent => {
            //error!("nvidia.rs: cudaStreamWaitEvent");
            let ret = unsafe {
                cudaStreamWaitEvent(
                    parameters.para1 as cudaStream_t,
                    parameters.para2 as cudaEvent_t,
                    parameters.para3 as c_uint,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaStreamWaitEvent: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaThreadExchangeStreamCaptureMode => {
            //error!("nvidia.rs: cudaThreadExchangeStreamCaptureMode");
            let mut mode: cudaStreamCaptureMode =
                unsafe { *(parameters.para1 as *mut cudaStreamCaptureMode) };

            let ret = unsafe { cudaThreadExchangeStreamCaptureMode(&mut mode) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaThreadExchangeStreamCaptureMode: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = mode as u32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaEventCreate => {
            //error!("nvidia.rs: CudaEventCreate");
            let mut event: cudaEvent_t = unsafe { *(parameters.para1 as *mut u64) as cudaEvent_t };

            let ret = unsafe { cudaEventCreate(&mut event) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaEventCreate: {}", ret as u32);
            }

            unsafe { *(parameters.para1 as *mut u64) = event as u64 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaEventCreateWithFlags => {
            //error!("nvidia.rs: CudaEventCreateWithFlags");
            let mut event: cudaEvent_t = unsafe { *(parameters.para1 as *mut u64) as cudaEvent_t };

            let ret = unsafe { cudaEventCreateWithFlags(&mut event, parameters.para2 as c_uint) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by CudaEventCreateWithFlags: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut u64) = event as u64 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaEventDestroy => {
            //error!("nvidia.rs: CudaEventDestroy");
            let ret = unsafe { cudaEventDestroy(parameters.para1 as cudaEvent_t) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaEventDestroy: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaEventElapsedTime => {
            //error!("nvidia.rs: CudaEventElapsedTime");
            let mut time: f32 = 0.0;

            let ret = unsafe {
                cudaEventElapsedTime(
                    &mut time,
                    parameters.para2 as cudaEvent_t,
                    parameters.para3 as cudaEvent_t,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaEventElapsedTime: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = time };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaEventQuery => {
            //error!("nvidia.rs: CudaEventQuery");
            let ret = unsafe { cudaEventQuery(parameters.para1 as cudaEvent_t) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaEventQuery: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaEventRecord => {
            //error!("nvidia.rs: CudaEventRecord");
            let ret = unsafe {
                cudaEventRecord(
                    parameters.para1 as cudaEvent_t,
                    parameters.para2 as cudaStream_t,
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaEventRecord: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaEventSynchronize => {
            //error!("nvidia.rs: CudaEventSynchronize");
            let ret = unsafe { cudaEventSynchronize(parameters.para1 as cudaEvent_t) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaEventSynchronize: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaFuncGetAttributes => {
            //error!("nvidia.rs: CudaFuncGetAttributes");
            //error!("map is {:?}", FUNCTIONS.lock());
            let dev_func = match FUNCTIONS.lock().get(&parameters.para2) {
                Some(func) => func.clone(),
                None => {
                    //error!("CudaFuncGetAttributes not find for func: {:x}", parameters.para2);
                    0
                }
            };
            let mut attr: cudaFuncAttributes = Default::default();
            let mut tmpAttr: i32 = 0;
            let mut ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.sharedSizeBytes = tmpAttr as usize;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.constSizeBytes = tmpAttr as usize;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.localSizeBytes = tmpAttr as usize;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.maxThreadsPerBlock = tmpAttr as i32;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.numRegs = tmpAttr as i32;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.ptxVersion = tmpAttr as i32;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.binaryVersion = tmpAttr as i32;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.cacheModeCA = tmpAttr as i32;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.maxDynamicSharedSizeBytes = tmpAttr as i32;
            }
            ret = unsafe {
                cuFuncGetAttribute(
                    &mut tmpAttr,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    dev_func as CUfunction,
                )
            };
            if ret as i32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuFuncGetAttribute: {}",
                    ret as u32
                );
                return Ok(ret as u32);
            } else {
                attr.preferredShmemCarveout = tmpAttr as i32;
            }

            unsafe { *(parameters.para1 as *mut cudaFuncAttributes) = attr };
            return Ok(0 as u32);
        }
        ProxyCommand::CudaFuncSetAttribute => {
            //error!("nvidia.rs: CudaFuncSetAttribute");
            let dev_func = match FUNCTIONS.lock().get(&parameters.para1) {
                Some(func) => func.clone(),
                None => {
                    0
                }
            };
            let ret = unsafe {
                cuFuncSetAttribute(
                    dev_func as CUfunction,
                    parameters.para2 as u32,
                    parameters.para3 as i32,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaFuncSetAttribute(cuFuncSetAttribute): {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaFuncSetCacheConfig => {
            //error!("nvidia.rs: CudaFuncSetCacheConfig");
            let ret = unsafe {
                cudaFuncSetCacheConfig(
                    parameters.para1 as *const c_void,
                    *(&parameters.para2 as *const _ as *const cudaFuncCache),
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaFuncSetCacheConfig: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaFuncSetSharedMemConfig => {
            //error!("nvidia.rs: CudaFuncSetSharedMemConfig");
            let ret = unsafe {
                cudaFuncSetSharedMemConfig(
                    parameters.para1 as *const c_void,
                    *(&parameters.para2 as *const _ as *const cudaSharedMemConfig),
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaFuncSetSharedMemConfig: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaGetLastError => {
            //error!("nvidia.rs: cuModuleGetLoadingMode");
            let ret = unsafe { cudaGetLastError() };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaGetLastError: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CuModuleGetLoadingMode => {
            //error!("nvidia.rs: cuModuleGetLoadingMode");
            let mut loadingMode: CumoduleLoadingModeEnum = unsafe { *(parameters.para1 as *mut _) };

            let ret = unsafe { cuModuleGetLoadingMode(&mut loadingMode) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuModuleGetLoadingMode: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = loadingMode as u32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CuInit => {
            //error!("nvidia.rs: CuInit");
            let ret = unsafe { cuInit(parameters.para1 as c_uint) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cuInit: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CuDevicePrimaryCtxGetState => {
            //error!("nvidia.rs: cuDevicePrimaryCtxGetState");
            let mut flags: c_uint = Default::default();
            let mut active: c_int = Default::default();

            let ret = unsafe {
                cuDevicePrimaryCtxGetState(
                    parameters.para1 as CUdevice,
                    &mut flags as *mut c_uint,
                    &mut active as *mut c_int,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuDevicePrimaryCtxGetState: {}",
                    ret as u32
                );
            }

            unsafe {
                *(parameters.para2 as *mut _) = flags;
                *(parameters.para3 as *mut _) = active;
            };
            return Ok(ret as u32);
        }
        ProxyCommand::NvmlDeviceGetCountV2 => {
            //error!("nvidia.rs: nvmlDeviceGetCountV2");
            let mut deviceCount: c_int = Default::default();

            let ret = unsafe { cudaGetDeviceCount(&mut deviceCount) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cudaGetDeviceCount: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut _) = deviceCount as u32 };
            return Ok(ret as u32);
        }
        ProxyCommand::NvmlInitWithFlags => {
            //error!("nvidia.rs: nvmlInitWithFlags");
            let ret = unsafe { nvmlInitWithFlags(parameters.para1 as c_uint) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by nvmlInitWithFlags: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::NvmlInit => {
            //error!("nvidia.rs: NvmlInit");
            let ret = unsafe { nvmlInitWithFlags(0 as c_uint) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by NvmlInit(nvmlInitWithFlags): {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::NvmlInitV2 => {
            //error!("nvidia.rs: NvmlInitV2");
            let ret = unsafe { nvmlInit_v2() };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by nvmlInit_v2: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::NvmlShutdown => {
            //error!("nvidia.rs: NvmlShutdown");
            let ret = unsafe { nvmlShutdown() };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by nvmlShutdown: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasCreateV2 => {
            //error!("nvidia.rs: CublasCreateV2");
            let mut handle: cublasHandle_t =
                unsafe { *(parameters.para1 as *mut u64) as cublasHandle_t };

            let ret = unsafe { cublasCreate_v2(&mut handle) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cublasCreate_v2: {}", ret as u32);
            }

            unsafe { *(parameters.para1 as *mut u64) = handle as u64 };
            return Ok(ret as u32);
        }
        ProxyCommand::CublasDestroyV2 => {
            //error!("nvidia.rs: CublasDestroyV2");
            let ret = unsafe { cublasDestroy_v2(parameters.para1 as cublasHandle_t) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cublasDestroy_v2: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasSetStreamV2 => {
            //error!("nvidia.rs: CublasSetStreamV2");
            let ret = unsafe {
                cublasSetStream_v2(
                    parameters.para1 as cublasHandle_t,
                    parameters.para2 as cudaStream_t,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cublasSetStream_v2: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasSetWorkspaceV2 => {
            //error!("nvidia.rs: CublasSetWorkspaceV2");
            let ret = unsafe {
                cublasSetWorkspace_v2(
                    parameters.para1 as cublasHandle_t,
                    parameters.para2 as *mut c_void,
                    parameters.para3 as usize,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cublasSetWorkspace_v2: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasSetMathMode => {
            //error!("nvidia.rs: CublasSetMathMode");
            let ret = unsafe {
                cublasSetMathMode(parameters.para1 as cublasHandle_t, parameters.para2 as u32)
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cublasSetMathMode: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasSgemmStridedBatched => {
            //error!("nvidia.rs: CublasSgemmStridedBatched");
            let info =
                unsafe { *(parameters.para1 as *const u8 as *const SgemmStridedBatchedInfo) };
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };

            let ret = unsafe {
                cublasSgemmStridedBatched(
                    info.handle as cublasHandle_t,
                    info.transa,
                    info.transb,
                    info.m,
                    info.n,
                    info.k,
                    &alpha,
                    info.A,
                    info.lda,
                    info.strideA,
                    info.B,
                    info.ldb,
                    info.strideB,
                    &beta,
                    info.C,
                    info.ldc,
                    info.strideC,
                    info.batchCount,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cublasSgemmStridedBatched: {}",
                    ret as u32
                );
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasLtMatmul => {
            //error!("nvidia.rs: CublasLtMatmul");
            let info = unsafe { *(parameters.para1 as *const u8 as *const CublasLtMatmulInfo) };

            let alpha = unsafe { *(parameters.para2 as *const f64) };
            let beta = unsafe { *(parameters.para3 as *const f64) };
            let ret = unsafe {
                cublasLtMatmul(
                    info.lightHandle as cublasLtHandle_t,
                    info.computeDesc as cublasLtMatmulDesc_t,
                    &alpha,
                    info.A,
                    info.Adesc as cublasLtMatrixLayout_t,
                    info.B,
                    info.Bdesc as cublasLtMatrixLayout_t,
                    &beta,
                    info.C,
                    info.Cdesc as cublasLtMatrixLayout_t,
                    info.D,
                    info.Ddesc as cublasLtMatrixLayout_t,
                    info.algo as *const cublasLtMatmulAlgo_t,
                    info.workspace,
                    info.workspaceSizeInBytes,
                    info.stream as cudaStream_t,
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cublasLtMatmul: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasLtMatmulAlgoGetHeuristic => {
            //error!("nvidia.rs: CublasLtMatmulAlgoGetHeuristic");
            let info = unsafe {
                *(parameters.para1 as *const u8 as *const CublasLtMatmulAlgoGetHeuristicInfo)
            };
            let mut heuristicResultsArray: Vec<cublasLtMatmulHeuristicResult_t> =
                Vec::with_capacity(info.requestedAlgoCount as usize);
            unsafe {
                heuristicResultsArray.set_len(info.requestedAlgoCount as usize);
            };
            let mut returnAlgoCount: c_int = 0;

            let ret = unsafe {
                cublasLtMatmulAlgoGetHeuristic(
                    info.lightHandle as cublasLtHandle_t,
                    info.operationDesc as cublasLtMatmulDesc_t,
                    info.Adesc as cublasLtMatrixLayout_t,
                    info.Bdesc as cublasLtMatrixLayout_t,
                    info.Cdesc as cublasLtMatrixLayout_t,
                    info.Ddesc as cublasLtMatrixLayout_t,
                    info.preference as cublasLtMatmulPreference_t,
                    info.requestedAlgoCount,
                    &mut heuristicResultsArray[0] as *mut cublasLtMatmulHeuristicResult_t,
                    &mut returnAlgoCount,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cublasLtMatmulAlgoGetHeuristic: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para3 as *mut _) = returnAlgoCount };
            for i in 0..returnAlgoCount as u64 {
                unsafe {
                    (*((parameters.para2 + i) as *mut u8 as *mut cublasLtMatmulHeuristicResult_t)) =
                        heuristicResultsArray[i as usize]
                };
            }
            return Ok(ret as u32);
        }
        ProxyCommand::CublasGetMathMode => {
            //error!("nvidia.rs: CublasGetMathMode");
            let mut mode: u32 = 0;

            let ret = unsafe { cublasGetMathMode(parameters.para1 as cublasHandle_t, &mut mode) };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cublasGetMathMode: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para2 as *mut u32) = mode as u32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CudaMemset => {
            //error!("nvidia.rs: CudaMemset");
            let ret = unsafe {
                cudaMemset(
                    parameters.para1 as *const c_void,
                    parameters.para2 as c_int,
                    parameters.para3 as usize,
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaMemset: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaMemsetAsync => {
            //error!("nvidia.rs: cudaMemsetAsync");
            let ret = unsafe {
                cudaMemsetAsync(
                    parameters.para1 as *const c_void,
                    parameters.para2 as c_int,
                    parameters.para3 as usize,
                    parameters.para4 as cudaStream_t,
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cudaMemsetAsync: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags => {
            //error!("nvidia.rs: CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
            let dev_func = match FUNCTIONS.lock().get(&parameters.para2) {
                Some(func) => func.clone(),
                None => 0,
            };
            let mut numBlocks = 0;

            let ret = unsafe {
                cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &mut numBlocks,
                    dev_func as CUfunction,
                    parameters.para3 as c_int,
                    parameters.para4 as usize,
                    parameters.para5 as c_uint,
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: {}", ret as u32);
            }

            unsafe { *(parameters.para1 as *mut i32) = numBlocks as i32 };
            return Ok(ret as u32);
        }
        ProxyCommand::CuCtxGetCurrent => {
            //error!("nvidia.rs: CuCtxGetCurrent");
            let mut ctx: u64 = 0;

            let ret = unsafe { cuCtxGetCurrent(&mut ctx as *mut _ as *mut CUcontext) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cuCtxGetCurrent: {}", ret as u32);
            }

            unsafe { *(parameters.para1 as *mut u64) = ctx };
            return Ok(ret as u32);
        }
        ProxyCommand::CuModuleLoadData => {
            //error!("nvidia.rs: CuModuleLoadData");
            match GetParameterInfo(parameters.para2, parameters.para3) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let mut module: u64 = 0;

            let ret = unsafe {
                cuModuleLoadData(
                    &mut module as *mut _ as *mut CUmodule,
                    parameters.para2 as *const c_void,
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuModuleLoadData: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut u64) = module };
            return Ok(ret as u32);
        }
        ProxyCommand::CuModuleGetFunction => {
            // error!("nvidia.rs: CuModuleGetFunction");
            let bytes = unsafe {
                std::slice::from_raw_parts(parameters.para3 as *const u8, parameters.para4 as usize)
            };
            let funcName = std::str::from_utf8(bytes).unwrap();
            let mut hfunc: u64 = 0;

            let ret = unsafe {
                // cuda_driver_sys::
                cuModuleGetFunction(
                    &mut hfunc as *mut _ as u64 as *mut CUfunction,
                    parameters.para2 as CUmodule,
                    CString::new(funcName).unwrap().clone().as_ptr(),
                )
            };
            if ret as u32 != 0 {
                error!(
                    "nvidia.rs: error caused by cuModuleGetFunction: {}",
                    ret as u32
                );
            }

            unsafe { *(parameters.para1 as *mut u64) = hfunc };

            let kernelInfo = match KERNEL_INFOS.lock().get(&funcName.to_string()) {
                Some(kernelInformations) => kernelInformations.clone(),
                None => {
                    //error!("No kernel infos found with this funcName : {}",funcName);
                    Arc::new(KernelInfo::default())
                }
            };

            let paramInfo = parameters.para5 as *const u8 as *mut ParamInfo;
            unsafe {
                (*paramInfo).paramNum = kernelInfo.paramNum;
                for i in 0..(*paramInfo).paramNum {
                    (*paramInfo).paramSizes[i] = kernelInfo.paramSizes[i];
                }
            }

            // FUNCTIONS.lock().insert(info.hostFun, hfunc);
            return Ok(ret as u32);
        }
        ProxyCommand::CuModuleUnload => {
            // error!("nvidia.rs: CuModuleUnload");
            let ret = unsafe { cuModuleUnload(parameters.para1 as CUmodule) };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cuModuleUnload: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CuLaunchKernel => {
            //error!("nvidia.rs: CuLaunchKernel");
            let info = unsafe { &*(parameters.para1 as *const u8 as *const CuLaunchKernelInfo) };

            let ret: CUresult = unsafe {
                // cuda_driver_sys::
                cuLaunchKernel(
                    info.f as CUfunction,
                    info.gridDimX,
                    info.gridDimY,
                    info.gridDimZ,
                    info.blockDimX,
                    info.blockDimY,
                    info.blockDimZ,
                    info.sharedMemBytes as u32,
                    info.hStream as *mut CUstream_st,
                    info.kernelParams as *mut *mut ::std::os::raw::c_void,
                    0 as *mut *mut ::std::os::raw::c_void,
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cuLaunchKernel: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasSgemmV2 => {
            //error!("nvidia.rs: CublasSgemm_v2");
            let info = unsafe { *(parameters.para1 as *const u8 as *const CublasSgemmV2Info) };
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };

            let ret = unsafe {
                cublasSgemm_v2(
                    info.handle as cublasHandle_t,
                    info.transa,
                    info.transb,
                    info.m,
                    info.n,
                    info.k,
                    &alpha,
                    info.A,
                    info.lda,
                    info.B,
                    info.ldb,
                    &beta,
                    info.C,
                    info.ldc,
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cublasSgemm_v2: {}", ret as u32);
            }

            return Ok(ret as u32);
        } 
        ProxyCommand::CublasGemmEx => {
            //error!("nvidia.rs: CublasSgemm_v2");
            let info = unsafe { *(parameters.para1 as *const u8 as *const GemmExInfo) };
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };

            let ret = unsafe {
                cublasGemmEx(
                    info.handle as cublasHandle_t,
                    info.transa,
                    info.transb,
                    info.m,
                    info.n,
                    info.k,
                    &alpha as *const _ as u64,
                    info.A,
                    info.Atype,
                    info.lda,
                    info.B,
                    info.Btype,
                    info.ldb,
                    &beta as *const _ as u64,
                    info.C,
                    info.Ctype,
                    info.ldc,
                    info.computeType,
                    info.algo
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cublasGemmEx: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        ProxyCommand::CublasGemmStridedBatchedEx => {
            //error!("nvidia.rs: CublasSgemm_v2");
            let info = unsafe { *(parameters.para1 as *const u8 as *const GemmStridedBatchedExInfo) };
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };

            let ret = unsafe {
                cublasGemmStridedBatchedEx(
                    info.handle as cublasHandle_t,
                    info.transa,
                    info.transb,
                    info.m,
                    info.n,
                    info.k,
                    &alpha as *const _ as u64,
                    info.A,
                    info.Atype,
                    info.lda,
                    info.strideA,
                    info.B,
                    info.Btype,
                    info.ldb,
                    info.strideB,
                    &beta as *const _ as u64,
                    info.C,
                    info.Ctype,
                    info.ldc,
                    info.strideC,
                    info.batchCount,
                    info.computeType,
                    info.algo
                )
            };
            if ret as u32 != 0 {
                error!("nvidia.rs: error caused by cublasGemmStridedBatchedEx: {}", ret as u32);
            }

            return Ok(ret as u32);
        } 
        _ => { return Ok(0); },
        //_ => todo!()
        
    }
}

pub fn CudaMemcpy(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaMemcpy");
    // let func: extern "C" fn(u64, u64, u64, u64) -> u32 = unsafe { std::mem::transmute(handle) };
    //error!("nvidia.rs: CudaMemcpy: count:{}, len:{}", parameters.para4 as usize, parameters.para3 as usize);
    let kind = parameters.para5;

    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => todo!(),
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            let dst = parameters.para1;
            let ptr = parameters.para2 as *const Range;
            let len = parameters.para3 as usize;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                // let ret = func(dst + offset, r.start, r.len, kind);
                let ret = unsafe { cudaMemcpy(dst + offset, r.start, r.len, kind) };
                if ret != 0 {
                    error!("nvidia.rs: error caused by cudaMemcpy: {}", ret as u32);
                    return Ok(ret as u32);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0);
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            let ptr = parameters.para1 as *const Range;
            let len = parameters.para2 as usize;
            let src = parameters.para3;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                // let ret = func(r.start, src + offset, r.len, kind);
                let ret = unsafe { cudaMemcpy(r.start, src + offset, r.len, kind) };
                if ret != 0 {
                    error!("nvidia.rs: error caused by cudaMemcpy: {}", ret as u32);
                    return Ok(ret as u32);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0);
        }
        CUDA_MEMCPY_DEVICE_TO_DEVICE => {
            let dst = parameters.para1;
            let src = parameters.para3;
            let count = parameters.para4;
            // let ret = func(dst, src, count, kind);

            let ret = unsafe { cudaMemcpy(dst, src, count, kind) };
            if ret != 0 {
                error!("nvidia.rs: error caused by cudaMemcpy: {}", ret as u32);
            }

            return Ok(ret as u32);
        }
        _ => todo!(),
    }
}

pub fn CudaMemcpyAsync(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: cudaMemcpyAsync");
    //error!("nvidia.rs: CudaMemcpy: count:{}, len:{}", parameters.para4 as usize, parameters.para3 as usize);
    let kind = parameters.para5;
    let stream = parameters.para6 as cudaStream_t;
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => todo!(),
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            let dst = parameters.para1; // device
            let ptr = parameters.para2 as *const Range;
            // length of the vector
            let len = parameters.para3 as usize;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = unsafe { cudaMemcpyAsync(dst + offset, r.start, r.len, kind, stream) };
                if ret as u32 != 0 {
                    error!("nvidia.rs: error caused by CudaMemcpyAsync: {}", ret as u32);
                    return Ok(ret as u32);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0);
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            // dst is host(virtual address)
            let ptr = parameters.para1 as *const Range;
            let len = parameters.para2 as usize;

            let src = parameters.para3; //device
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = unsafe { cudaMemcpyAsync(r.start, src + offset, r.len, kind, stream) };

                if ret != 0 {
                    error!("nvidia.rs: error caused by CudaMemcpyAsync: {}", ret as u32);
                    return Ok(ret as u32);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0);
        }
        CUDA_MEMCPY_DEVICE_TO_DEVICE => {
            let dst = parameters.para1;
            let src = parameters.para3;
            let count = parameters.para4;

            let ret = unsafe { cudaMemcpyAsync(dst, src, count, kind, stream) };
            if ret != 0 {
                error!("nvidia.rs: error caused by CudaMemcpyAsync: {}", ret as u32);
            }
            return Ok(ret as u32);
        }
        _ => todo!(),
    }
}

fn InitNvidia(containerId: &str, qmsg: &QMsg) {
    // cuModuleLoadData requires libnvidia-ptxjitcompiler.so, and nvidia image will mount some host libraries into container,
    // the lib we want to use is locate in /usr/local/cuda/compat/, host version libraries will cause error.
    if let Msg::Proxy(msg) = qmsg.msg {
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
    } // else is impossible
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
                error!("nvidia.rs: error caused by cudaMemPrefetchAsync to host: {}", ret as u32);
            }
        }
        let ret2 = unsafe { cudaStreamSynchronize(0 as cudaStream_t) };
        if ret2 as u32 != 0 {
            error!("nvidia.rs: error caused by cudaMemPrefetchAsync to host: {}", ret2 as u32);
        }
    } else if QUARK_CONFIG.lock().CudaMemType == CudaMemType::MemPool {
        MEM_MANAGER.lock().offloadGPUMem();
        MEM_MANAGER.lock().offloadGPUFatbin();
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
                error!("nvidia.rs: error caused by cudaMemPrefetchAsync to gpu: {}", ret as u32);
            }
        }
        let ret2 = unsafe { cudaStreamSynchronize(0 as cudaStream_t) };
        if ret2 as u32 != 0 {
            error!("nvidia.rs: error caused by cudaMemPrefetchAsync to gpu: {}", ret2 as u32);
        }
    } else if QUARK_CONFIG.lock().CudaMemType == CudaMemType::MemPool {
        MEM_MANAGER.lock().restoreGPUFatbin();
        MEM_MANAGER.lock().restoreGPUMem();
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
