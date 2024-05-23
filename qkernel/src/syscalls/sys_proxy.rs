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

use core::sync::atomic::Ordering;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::qlib::common::*;
use crate::qlib::kernel::threadmgr::thread_group::CudaProcessCtx;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::kernel::TSC;
use crate::qlib::linux_def::{SysErr, PATH_MAX};
use crate::qlib::proxy::*;
use crate::syscalls::syscalls::*;
use crate::task::*;

// use std::ptr::null_mut;

lazy_static! {
    pub static ref PARAM_INFOS: Mutex<BTreeMap<u64, Arc<Vec<u16>>>> = Mutex::new(BTreeMap::new());
}

impl Drop for CudaProcessCtx {
    fn drop(&mut self) {
        if self.lock().enableGPU && Arc::strong_count(&self) == 1 {
            let mut parameters = ProxyParameters::default();
            let ret = HostSpace::Proxy(ProxyCommand::ExitWorkerThread, parameters); // not sure if its necessary
        }
    }
}
pub fn SysProxy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let commandId = args.arg0 as u64;
    let startTime = TSC.Rdtsc();

    defer!(if crate::SHARESPACE.config.read().PerfDebug {
        let gap = TSC.Rdtsc() - startTime;
        crate::qlib::kernel::threadmgr::task_exit::SYSPROXY_CALL_TIME[commandId as usize]
            .fetch_add(gap as u64, Ordering::SeqCst);
    });
    let cmd: ProxyCommand = unsafe { core::mem::transmute(commandId as u64) };
    let mut cudaProcessCtx = task.Thread().ThreadGroup().GetCudaCtx();
    cudaProcessCtx.lock().enableGPU = true;
    let mut parameters = ProxyParameters {
        para1: args.arg1,
        para2: args.arg2,
        para3: args.arg3,
        para4: args.arg4,
        para5: args.arg5,
        para6: 0,
        para7: 0,
    };
    let sys_ret ;
    match cmd {
        ProxyCommand::None => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        ProxyCommand::CudaDeviceReset
        | ProxyCommand::CudaDeviceSetLimit
        | ProxyCommand::CudaDeviceSetCacheConfig
        | ProxyCommand::CudaDeviceSetSharedMemConfig
        | ProxyCommand::CudaSetDeviceFlags
        | ProxyCommand::CudaPeekAtLastError
        | ProxyCommand::CudaFree
        | ProxyCommand::CudaDeviceSynchronize
        | ProxyCommand::CudaStreamSynchronize
        | ProxyCommand::CudaStreamDestroy
        | ProxyCommand::CudaStreamQuery
        | ProxyCommand::CudaStreamWaitEvent
        | ProxyCommand::CudaEventDestroy
        | ProxyCommand::CudaEventQuery
        | ProxyCommand::CudaEventRecord
        | ProxyCommand::CudaEventSynchronize
        | ProxyCommand::CudaMemset
        | ProxyCommand::CudaMemsetAsync
        | ProxyCommand::CudaFuncSetAttribute
        | ProxyCommand::CudaFuncSetCacheConfig
        | ProxyCommand::CudaFuncSetSharedMemConfig
        | ProxyCommand::CuModuleUnload
        | ProxyCommand::CudaGetLastError
        | ProxyCommand::CuInit
        | ProxyCommand::NvmlInitWithFlags
        | ProxyCommand::CudaUnregisterFatBinary
        | ProxyCommand::NvmlInit
        | ProxyCommand::NvmlInitV2
        | ProxyCommand::NvmlShutdown
        | ProxyCommand::CublasDestroyV2
        | ProxyCommand::CublasSetWorkspaceV2
        | ProxyCommand::CublasSetMathMode
        | ProxyCommand::CublasSetStreamV2 
        | ProxyCommand::NcclCommDestroy
        | ProxyCommand::NcclCommAbort 
        | ProxyCommand::NcclGroupStart
        | ProxyCommand::NcclGroupEnd => {
            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaSetDevice => {
            let ret = HostSpace::Proxy(cmd, parameters);
            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclGetVersion => {
            error!("ncclGetVersion_sysproxy");
            let mut version:u64 = 0;
            parameters.para1 = &mut version as *mut _ as u64;
            
            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&version, args.arg1)?;
            }
            sys_ret = Ok(ret);
            
        }
        ProxyCommand::NcclGetUniqueId => {
            error!("ncclGetUniqueId_sysproxy");
            let mut ncclUniqueId_:NcclUniqueId = NcclUniqueId::default();
            parameters.para1 = &mut ncclUniqueId_ as * mut _ as u64;
            
            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&ncclUniqueId_, args.arg1)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclCommInitRank => {
            error!("ncclCommInitRank_sysproxy");
            // let mut ncclComm_t_:NcclCommT = null_mut();
            let mut ncclComm_t_:u64 = 0;
            parameters.para1 = &mut ncclComm_t_ as *mut _ as u64;

            // let mut comm_id:ncclUniqueId = ncclUniqueId::default();
            // comm_id.internal = task.CopyInVec(parameters.para3, 128)?;
            let comm_id = task.CopyInObj::<NcclUniqueId>(parameters.para3)?;
            // error!("page fault? {}", comm_id.internal[0]);
            parameters.para3 = &comm_id as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

    

            if ret == 0 {
                task.CopyOutObj(&ncclComm_t_, args.arg1)?;
            }

            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclCommInitAll => {
            error!("ncclCommInitAll_sysproxy");
            // let mut ncclComm_t_s:Vec<u64> = Vec::with_capacity(args.arg2 as usize);
            // unsafe {
            //     ncclComm_t_s.set_len(args.arg2 as usize);
            // }
            let mut ncclComm_t_s: Vec<u64> = vec![0; args.arg2 as usize];
            let devlist:Vec<i32>=task.CopyInVec(parameters.para3, args.arg2 as usize)?;


            // parameters.para1 = &mut ncclComm_t_s[0] as *mut _ as u64;
            parameters.para1 = ncclComm_t_s.as_mut_ptr() as u64;
            parameters.para3 = &devlist[0] as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);
            error!("befor vec copy: sysproxy");
            if ret == 0 {
                task.CopyOutSlice(&ncclComm_t_s, args.arg1 as u64, args.arg2 as usize)?;
            }
            error!("after vec copy: sysproxy");
            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclCommInitRankConfig => {
            error!("ncclCommInitRankConfig_sysproxy");
            // let mut ncclComm_t_:NcclCommT = null_mut();
            let mut ncclComm_t_:u64 = 0;
            parameters.para1 = &mut ncclComm_t_ as *mut _ as u64;
            let comm_id: NcclUniqueId = task.CopyInObj::<NcclUniqueId>(parameters.para3)?;
            let config = task.CopyInObj::<NcclConfig>(parameters.para5)?;

            parameters.para3 = &comm_id as *const _ as u64;
            parameters.para5 = &config as *const _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&ncclComm_t_, args.arg1)?;
            }
            sys_ret = Ok(ret);

        }
        ProxyCommand::NcclCommCount => {
            error!("ncclCommCount_sysproxy");
            let mut count:u32 = 0;
            parameters.para2 = &mut count as *mut _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            if ret == 0 {
                task.CopyOutObj(&count, args.arg2)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclCommUserRank => {
            error!("ncclCommUserRank_sysproxy");
            let mut rank:u32 = 0;
            parameters.para2 = &mut rank as *mut _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            if ret == 0 {
                task.CopyOutObj(&rank, args.arg2)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclCommCuDevice => {
            error!("ncclCommCuDevice_sysproxy");
            let mut device:u32 = 0;
            parameters.para2 = &mut device as *mut _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            if ret == 0 {
                task.CopyOutObj(&device, args.arg2)?;
            }
            sys_ret = Ok(ret);
        }

        ProxyCommand::NcclCommGetAsyncError => {
            error!("ncclCommGetAsyncError_sysproxy");
            let mut result = NcclResultT::NcclSuccess;
            parameters.para2 = &mut result as *mut _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            if ret == 0 {
                task.CopyOutObj(&result, args.arg2)?;
            }
            sys_ret = Ok(ret);
        }


        ProxyCommand::NcclSend=> {
            error!("ncclSend_sysproxy");
            let sendinfo = task.CopyInObj::<NcclSendRecvInfo>(parameters.para2)?;
            parameters.para2 = &sendinfo as *const _ as u64;
            // let prs = task.V2P(src, count, false, false)?;
            let ret = HostSpace::Proxy(cmd, parameters);
            sys_ret = Ok(ret);
        }

        ProxyCommand::NcclRecv=> {
            error!("ncclRecv_sysproxy");
            let recvinfo = task.CopyInObj::<NcclSendRecvInfo>(parameters.para2)?;
            parameters.para2 = &recvinfo as *const _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            sys_ret = Ok(ret);
        }

        ProxyCommand::NcclAllReduce => {
            error!("ncclAllReduce_sysproxy");
            let sendinfo = task.CopyInObj::<NcclAllGatherReduceInfo>(parameters.para3)?;
            parameters.para3 = &sendinfo as *const _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclAllGather => {
            error!("ncclAllGather_sysproxy");
            let sendinfo = task.CopyInObj::<NcclAllGatherReduceInfo>(parameters.para3)?;
            parameters.para3 = &sendinfo as *const _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            sys_ret = Ok(ret);

        }
        ProxyCommand::NcclReduceScatter => {
            error!("ncclReduceScatter_sysproxy");
            let sendinfo = task.CopyInObj::<NcclAllGatherReduceInfo>(parameters.para3)?;
            parameters.para3 = &sendinfo as *const _ as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            sys_ret = Ok(ret);
        }
        ProxyCommand::NcclGetErrorString => {
            let mut errorString = String::from("");
            parameters.para2 = &mut errorString as *mut String as u64;
            let ret = HostSpace::Proxy(cmd, parameters);
            // error!("ncclGetErrorString_sysproxy: {}", errorString);
            if ret == 0 {
                task.CopyOutString(args.arg2, errorString.len(), &errorString)?;
            }
            sys_ret = Ok(ret);

        }
        ProxyCommand::CudaChooseDevice => {
            let mut device: i32 = 0;
            parameters.para1 = &mut device as *mut _ as u64;
            let deviceProperties = task.CopyInObj::<CudaDeviceProperties>(parameters.para2)?;
            // todo, check whether need to get the char array { name } , and the length
            parameters.para2 = &deviceProperties as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&device, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetAttribute => {
            let mut value: i32 = 0;
            parameters.para1 = &mut value as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&value, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetByPCIBusId => {
            let mut device: i32 = 0;
            parameters.para1 = &mut device as *mut _ as u64;
            let (PCIBusId, err) = task.CopyInString(parameters.para2, PATH_MAX);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }
            parameters.para2 = &(PCIBusId.as_bytes()[0]) as *const _ as u64; // address
            parameters.para3 = PCIBusId.as_bytes().len() as u64; // length

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&device, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetCacheConfig => {
            let mut CacheConfig: u32;
            unsafe {
                CacheConfig = *(parameters.para1 as *mut _);
            }
            parameters.para1 = &mut CacheConfig as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&CacheConfig, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetLimit => {
            let mut limit: usize = 0;
            parameters.para1 = &mut limit as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&limit, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetP2PAttribute => {
            let mut value: i32 = 0;
            parameters.para1 = &mut value as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&value, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetPCIBusId => {
            let (mut pciBusIdAddress, err) = task.CopyInString(parameters.para1, PATH_MAX);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }
            parameters.para1 = &mut pciBusIdAddress as *mut String as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutString(args.arg1, parameters.para2 as usize, &pciBusIdAddress)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetSharedMemConfig => {
            error!("CudaDeviceGetSharedMemConfig");
            let mut sharedMemConfig: u32 = unsafe { *(parameters.para1 as *mut u32) };
            parameters.para1 = &mut sharedMemConfig as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&sharedMemConfig, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaDeviceGetStreamPriorityRange => {
            let mut lowPriority: i32;
            let mut highPriority: i32;
            unsafe {
                lowPriority = *(parameters.para1 as *mut _);
                highPriority = *(parameters.para2 as *mut _);
            }
            parameters.para1 = &mut lowPriority as *mut _ as u64;
            parameters.para2 = &mut highPriority as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&lowPriority, args.arg1)?;
                task.CopyOutObj(&highPriority, args.arg2)?
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaSetValidDevices => {
            let mut data: Vec<i32> = task.CopyInVec(parameters.para1, parameters.para2 as usize)?;
            parameters.para1 = &mut data[0] as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaGetDevice => {
            let mut device: i32 = 0;
            parameters.para1 = &mut device as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&device, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaGetDeviceCount => {
            let mut deviceCount: i32 = 0;
            parameters.para1 = &mut deviceCount as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&deviceCount, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaGetDeviceFlags => {
            let mut deviceFlags: u32 = 0;
            parameters.para1 = &mut deviceFlags as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&deviceFlags, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaGetDeviceProperties => {
            let mut deviceProp: CudaDeviceProperties = CudaDeviceProperties::default();
            parameters.para1 = &mut deviceProp as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&deviceProp, args.arg1)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaGetErrorString => {
            let mut errorString = String::from("");
            parameters.para2 = &mut errorString as *mut String as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutString(args.arg2, errorString.len() + 1, &errorString)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaGetErrorName => {
            let mut errorName = String::from("");
            parameters.para2 = &mut errorName as *mut String as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutString(args.arg2, errorName.len() + 1, &errorName)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaMalloc => {
            let mut addr: u64 = 0;
            parameters.para1 = &mut addr as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&addr, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaMemcpy => {
            let ret = CudaMemcpy(
                task,
                parameters.para1,
                parameters.para2,
                parameters.para3,
                parameters.para4,
            )?;

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaMemcpyAsync => {
            let info = task.CopyInObj::<cudaMemcpyAsyncInfo>(parameters.para1)?;
            let ret = CudaMemcpyAsync(
                task,
                info.dst,
                info.src,
                info.count as u64,
                info.kind as CudaMemcpyKind,
                info.stream,
            )?;

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaRegisterFatBinary => {
            let data: Vec<u8> = task.CopyInVec(parameters.para2, parameters.para1 as usize)?;
            parameters.para2 = &data[0] as *const _ as u64;

            let (ptxlibPath, err) = task.CopyInString(parameters.para4, PATH_MAX);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }
            parameters.para4 = &(ptxlibPath.as_bytes()[0]) as *const _ as u64;
            parameters.para5 = ptxlibPath.as_bytes().len() as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaRegisterFunction => {
            let mut functionInfo = task.CopyInObj::<RegisterFunctionInfo>(parameters.para1)?;
            // error!("CudaRegisterFunction data {:x?}, parameters {:x?}", functionInfo, parameters);
            let (deviceName, err) = task.CopyInString(functionInfo.deviceName, PATH_MAX);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }
            functionInfo.deviceName = &(deviceName.as_bytes()[0]) as *const _ as u64;
            parameters.para1 = &functionInfo as *const _ as u64;
            parameters.para2 = deviceName.as_bytes().len() as u64;
            // error!("deviceName {}, data.deviceName {:x}, parameters {:x?}", deviceName, functionInfo.deviceName, parameters);

            let mut paramInfo = ParamInfo::default();
            parameters.para3 = &mut paramInfo as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);
            // error!("paramInfo {:x?}", paramInfo);

            let mut params_proxy: Vec<u16> = Vec::new();
            for i in 0..paramInfo.paramNum as usize {
                params_proxy.push(paramInfo.paramSizes[i]);
                // error!("i {}, paramInfo.paramSizes[i] {}", i, paramInfo.paramSizes[i]);
            }
            PARAM_INFOS
                .lock()
                .insert(functionInfo.hostFun, Arc::new(params_proxy));
            // error!("PARAM_INFOS {:x?}", PARAM_INFOS.lock());

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaRegisterVar => {
            let mut data = task.CopyInObj::<RegisterVarInfo>(parameters.para1)?; // still take the addresss
            let (deviceName, err) = task.CopyInString(data.deviceName, PATH_MAX);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }
            // get the deviceName string and assign the address of first byte to the data struct field
            data.deviceName = &(deviceName.as_bytes()[0]) as *const _ as u64;
            parameters.para1 = &data as *const _ as u64; // address
            parameters.para2 = deviceName.as_bytes().len() as u64; // device name length

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaLaunchKernel => {
            let mut kernelInfo = task.CopyInObj::<LaunchKernelInfo>(parameters.para1)?;
            let paramInfo = PARAM_INFOS.lock().get(&kernelInfo.func).unwrap().clone();
            // error!("LaunchKernelInfo data {:x?}, paramInfo {:x?}, parameters {:x?}", kernelInfo, paramInfo, parameters);

            let mut paramAddrs: Vec<u64> = task.CopyInVec(kernelInfo.args, paramInfo.len())?;
            let mut paramValues = Vec::new();
            for i in 0..paramInfo.len() {
                let valueBytes: Vec<u8> = task.CopyInVec(paramAddrs[i], (paramInfo[i]) as usize)?;
                // error!("valueBytes {:x?}", valueBytes);
                paramValues.push(valueBytes);
                paramAddrs[i] = &(paramValues[i][0]) as *const _ as u64;
                // error!("i {} paramAddrs[i] {:x} paramValues[i] {:x?}",i, paramAddrs[i], paramValues[i]);
            }
            // error!("paramAddrs after set {:x?}", paramAddrs);
            kernelInfo.args = &paramAddrs[0] as *const _ as u64;
            // error!("kernelInfo.args {:x?}", kernelInfo.args);

            parameters.para1 = &kernelInfo as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaStreamCreate => {
            let mut stream: u64 = unsafe { *(parameters.para1 as *mut _) };
            parameters.para1 = &mut stream as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&stream, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaStreamCreateWithFlags => {
            let mut stream: u64 = 0;
            parameters.para1 = &mut stream as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&stream, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaStreamCreateWithPriority => {
            let mut stream: u64 = 0;
            parameters.para1 = &mut stream as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&stream, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaStreamGetFlags => {
            let mut flags: u32 = 0;
            parameters.para2 = &mut flags as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&flags, args.arg2 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaStreamGetPriority => {
            let mut priority: i32 = 0;
            parameters.para2 = &mut priority as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&priority, args.arg2 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaStreamIsCapturing => {
            let mut pCaptureStatus: u32;
            unsafe {
                pCaptureStatus = *(parameters.para2 as *mut _);
            }
            parameters.para2 = &mut pCaptureStatus as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&pCaptureStatus, args.arg2 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaThreadExchangeStreamCaptureMode => {
            let mut mode: u32 = unsafe { *(parameters.para1 as *mut u32) };
            parameters.para1 = &mut mode as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&mode, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaEventCreate => {
            let mut event: u64 = unsafe { *(parameters.para1 as *mut _) };
            parameters.para1 = &mut event as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&event, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaEventCreateWithFlags => {
            let mut event: u64 = 0;
            parameters.para1 = &mut event as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&event, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaEventElapsedTime => {
            let mut time: f32 = 0.0;
            parameters.para1 = &mut time as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&time, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaFuncGetAttributes => {
            let mut attribute: CudaFuncAttributes = Default::default();
            parameters.para1 = &mut attribute as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&attribute, args.arg1)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CublasGetMathMode => {
            let mut mode: u32 = 0;
            parameters.para2 = &mut mode as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&mode, args.arg2 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags => {
            let info = task.CopyInObj::<cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInfo>(parameters.para1)?;
            let mut numBlocks: i32 = 0;
            parameters.para1 = &mut numBlocks as *mut _ as u64;
            parameters.para2 = info.func;
            parameters.para3 = info.blockSize as u64;
            parameters.para4 = info.dynamicSMemSize as u64;
            parameters.para5 = info.flags as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&info.numBlocks, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CublasSgemmV2 => {
            let cublasSgemm_v2Info = task.CopyInObj::<CublasSgemmV2Info>(parameters.para1)?;
            parameters.para1 = &cublasSgemm_v2Info as *const _ as u64;
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };

            parameters.para2 = &alpha as *const _ as u64;
            parameters.para3 = &beta as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CuCtxGetCurrent => {
            let mut ctx: u64 = 0;
            parameters.para1 = &mut ctx as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&ctx, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CuModuleLoadData => {
            let data: Vec<u8> = task.CopyInVec(parameters.para2, parameters.para3 as usize)?;
            parameters.para2 = &data[0] as *const _ as u64;
            let mut module: u64 = 0;
            parameters.para1 = &mut module as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&module, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CuModuleGetFunction => {
            let (funcName, err) = task.CopyInString(parameters.para3, PATH_MAX);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }
            parameters.para3 = &(funcName.as_bytes()[0]) as *const _ as u64;
            parameters.para4 = funcName.as_bytes().len() as u64;

            let mut hfunc: u64 = 0;
            parameters.para1 = &mut hfunc as *mut _ as u64;
            let mut paramInfo = ParamInfo::default();
            parameters.para5 = &mut paramInfo as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&hfunc, args.arg1 as u64)?;
            }

            let mut params_proxy: Vec<u16> = Vec::new();
            for i in 0..paramInfo.paramNum as usize {
                params_proxy.push(paramInfo.paramSizes[i]);
                // error!("i {}, paramInfo.paramSizes[i] {}", i, paramInfo.paramSizes[i]);
            }
            PARAM_INFOS
                .lock()
                .insert(hfunc.clone(), Arc::new(params_proxy));

            sys_ret = Ok(ret);
        }
        ProxyCommand::CuLaunchKernel => {
            let mut kernelInfo: CuLaunchKernelInfo = task.CopyInObj::<CuLaunchKernelInfo>(parameters.para1)?;
            let paramInfo = PARAM_INFOS.lock().get(&kernelInfo.f).unwrap().clone();
            // error!("cuLaunchKernelInfo data {:x?}, paramInfo {:x?}, parameters {:x?}", kernelInfo, paramInfo, parameters);

            let mut paramAddrs: Vec<u64> =
                task.CopyInVec(kernelInfo.kernelParams, paramInfo.len())?;
            // error!("cuLaunchKernelInfo paramAddrs {:x?}", paramAddrs);

            let mut paramValues = Vec::new();
            for i in 0..paramInfo.len() {
                let valueBytes: Vec<u8> = task.CopyInVec(paramAddrs[i], (paramInfo[i]) as usize)?;
                // error!("cuLaunchKernelInfo valueBytes {:x?}", valueBytes);

                paramValues.push(valueBytes);
                paramAddrs[i] = &(paramValues[i][0]) as *const _ as u64;
                // error!("cuLaunchKernelInfo i {} paramAddrs[i] {:x} paramValues[i] {:x?}",i, paramAddrs[i], paramValues[i]);
            }
            // error!("cuLaunchKernelInfo paramAddrs after set {:x?}", paramAddrs);
            kernelInfo.kernelParams = &paramAddrs[0] as *const _ as u64;
            // error!("cuLaunchKernelInfo kernelInfo.args {:x?}", kernelInfo.kernelParams);
            parameters.para1 = &kernelInfo as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CuModuleGetLoadingMode => {
            let mut mode: u32 = 0;
            parameters.para1 = &mut mode as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&mode, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CuDevicePrimaryCtxGetState => {
            let mut flags: u32 = 0;
            let mut active: i32 = 0;

            parameters.para2 = &mut flags as *mut _ as u64;
            parameters.para3 = &mut active as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&flags, args.arg2 as u64)?;
                task.CopyOutObj(&active, args.arg3 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::NvmlDeviceGetCountV2 => {
            let mut deviceCount: u32 = 0;
            parameters.para1 = &mut deviceCount as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&deviceCount, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }

        ProxyCommand::CublasCreateV2 => {
            let mut handle: u64 = unsafe { *(parameters.para1 as *mut _) };
            parameters.para1 = &mut handle as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutObj(&handle, args.arg1 as u64)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CublasSgemmStridedBatched => {
            let sgemmStridedBatchedInfo =
                task.CopyInObj::<SgemmStridedBatchedInfo>(parameters.para1)?;
            parameters.para1 = &sgemmStridedBatchedInfo as *const _ as u64;
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };
            parameters.para2 = &alpha as *const _ as u64;
            parameters.para3 = &beta as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CublasLtMatmul => {
            let cublasLtMatmulInfo = task.CopyInObj::<CublasLtMatmulInfo>(parameters.para1)?;
            parameters.para1 = &cublasLtMatmulInfo as *const _ as u64;
            let alpha = unsafe { *(parameters.para2 as *const f64) };
            let beta = unsafe { *(parameters.para3 as *const f64) };
            parameters.para2 = &alpha as *const _ as u64;
            parameters.para3 = &beta as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CublasLtMatmulAlgoGetHeuristic => {
            let cublasLtMatmulAlgoGetHeuristicInfo =
                task.CopyInObj::<CublasLtMatmulAlgoGetHeuristicInfo>(parameters.para1)?;
            parameters.para1 = &cublasLtMatmulAlgoGetHeuristicInfo as *const _ as u64;
            let mut heuristicResultsArray: Vec<CublasLtMatmulHeuristicResult> =
                Vec::with_capacity(cublasLtMatmulAlgoGetHeuristicInfo.requestedAlgoCount as usize);
            unsafe {
                heuristicResultsArray
                    .set_len(cublasLtMatmulAlgoGetHeuristicInfo.requestedAlgoCount as usize);
            };
            let mut returnAlgoCoun: i32 = 0;
            parameters.para2 = &mut heuristicResultsArray[0] as *mut _ as u64;
            parameters.para3 = &mut returnAlgoCoun as *mut _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            if ret == 0 {
                task.CopyOutSlice(&heuristicResultsArray, args.arg2, returnAlgoCoun as usize)?;
                task.CopyOutObj(&returnAlgoCoun, args.arg3)?;
            }
            sys_ret = Ok(ret);
        }
        ProxyCommand::CublasGemmEx => {
            let gemmExInfo = task.CopyInObj::<GemmExInfo>(parameters.para1)?;
            parameters.para1 = &gemmExInfo as *const _ as u64;
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };
            parameters.para2 = &alpha as *const _ as u64;
            parameters.para3 = &beta as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        }
        ProxyCommand::CublasGemmStridedBatchedEx => {
            let gemmStridedBatchedExInfo =
                task.CopyInObj::<GemmStridedBatchedExInfo>(parameters.para1)?;
            parameters.para1 = &gemmStridedBatchedExInfo as *const _ as u64;
            let alpha = unsafe { *(parameters.para2 as *const f32) };
            let beta = unsafe { *(parameters.para3 as *const f32) };
            parameters.para2 = &alpha as *const _ as u64;
            parameters.para3 = &beta as *const _ as u64;

            let ret = HostSpace::Proxy(cmd, parameters);

            sys_ret = Ok(ret);
        } //_ => todo!()
        _ => {
            sys_ret = Ok(0);
        }
    }
    return sys_ret;
}

pub fn CudaMemcpy(
    task: &Task,
    dst: u64,
    src: u64,
    count: u64,
    kind: CudaMemcpyKind,
) -> Result<i64> {
    //error!("CudaMemcpy: count:{}, kind{}", count, kind);
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => {
            // error!("CudaMemcpy get unexpected kind CUDA_MEMCPY_HOST_TO_HOST");
            return Ok(1);
        }
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            // src is the virtual addr
            let prs = task.V2P(src, count, false, false)?;

            let parameters = ProxyParameters {
                para1: dst,
                para2: &prs[0] as *const _ as u64,
                para3: prs.len() as u64,
                para4: count as u64,
                para5: kind,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(ProxyCommand::CudaMemcpy, parameters);

            return Ok(ret);
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            // dst is the virtual addr
            let prs = task.V2P(dst, count, true, false)?;

            let parameters = ProxyParameters {
                para1: &prs[0] as *const _ as u64,
                para2: prs.len() as u64,
                para3: src,
                para4: count as u64,
                para5: kind,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(ProxyCommand::CudaMemcpy, parameters);

            return Ok(ret);
        }
        CUDA_MEMCPY_DEVICE_TO_DEVICE => {
            let parameters = ProxyParameters {
                para1: dst,
                para2: 0,
                para3: src,
                para4: count as u64,
                para5: kind,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(ProxyCommand::CudaMemcpy, parameters);

            return Ok(ret);
        }
        _ => todo!(),
    }
}

fn CudaMemcpyAsync(
    task: &Task,
    dst: u64,
    src: u64,
    count: u64,
    kind: CudaMemcpyKind,
    stream: u64,
) -> Result<i64> {
    // error!("CudaMemcpyAsync: count:{}, kind{}", count, kind);
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => {
            error!("CudaMemcpy get unexpected kind CUDA_MEMCPY_HOST_TO_HOST");
            return Ok(1);
        }
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            // src is the virtual addr(src is host memory ), address and # of bytes
            let prs = task.V2P(src, count, false, false)?;

            let parameters = ProxyParameters {
                para1: dst,
                para2: &prs[0] as *const _ as u64,
                para3: prs.len() as u64,
                para4: count as u64,
                para5: kind,
                para6: stream,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(ProxyCommand::CudaMemcpyAsync, parameters);

            return Ok(ret);
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            // dst is the virtual addr(host memory)
            let prs = task.V2P(dst, count, true, false)?;

            let parameters = ProxyParameters {
                para1: &prs[0] as *const _ as u64,
                para2: prs.len() as u64,
                para3: src,
                para4: count as u64,
                para5: kind,
                para6: stream,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(ProxyCommand::CudaMemcpyAsync, parameters);

            return Ok(ret);
        }
        CUDA_MEMCPY_DEVICE_TO_DEVICE => {
            let parameters = ProxyParameters {
                para1: dst,
                para2: 0,
                para3: src,
                para4: count as u64,
                para5: kind,
                para6: stream,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(ProxyCommand::CudaMemcpyAsync, parameters);

            return Ok(ret);
        }
        _ => todo!(),
    }
}
