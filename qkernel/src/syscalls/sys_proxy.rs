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
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::qlib::addr::AccessType;
use crate::qlib::common::*;
use crate::qlib::kernel::fs::host::hostinodeop::MMappable;
use crate::qlib::kernel::memmgr::vma::{CudaHostMappable, CudaHostMappableInner};
use crate::qlib::kernel::memmgr::MLockMode;
use crate::qlib::kernel::memmgr::MMapOpts;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::kernel::TSC;
use crate::qlib::linux_def::{IoVec, MemoryDef, SysErr, PATH_MAX};
use crate::qlib::proxy::*;
use crate::syscalls::syscalls::*;
use crate::task::*;

lazy_static! {
    pub static ref PARAM_INFOS: Mutex<BTreeMap<u64, Arc<Vec<u16>>>> = Mutex::new(BTreeMap::new());
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
    let mut parameters = ProxyParameters {
        para1: args.arg1,
        para2: args.arg2,
        para3: args.arg3,
        para4: args.arg4,
        para5: args.arg5,
        para6: 0,
        para7: 0,
    };

    let sys_ret;
    match cmd {
        ProxyCommand::None => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        ProxyCommand::CudaSetDevice
        | ProxyCommand::CudaDeviceReset
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
        | ProxyCommand::CublasSetStreamV2 => {
            let ret = HostSpace::Proxy(cmd, parameters);

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
            let info = task
                .CopyInObj::<cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInfo>(
                    parameters.para1,
                )?;
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
            let mut kernelInfo = task.CopyInObj::<CuLaunchKernelInfo>(parameters.para1)?;
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
        ProxyCommand::CudaHostAlloc => {
            match CudaHostAlloc(task, parameters) {
                Ok(n) => return Ok(n),
                Err(e) => {
                    error!("CudaHostAlloc fail with error {:?}", e);

                    // todo: find right failure code
                    return Ok(2);
                }
            }
        }
        ProxyCommand::CudaFreeHost => {
            match CudeFreeHost(task, parameters) {
                Ok(n) => return Ok(n),
                Err(e) => {
                    error!("CudeFreeHost fail with error {:?}", e);

                    // todo: find right failure code
                    return Ok(2);
                }
            }
        }
    }

    return sys_ret;
}

pub fn CudaHostAlloc(task: &Task, mut parameters: ProxyParameters) -> Result<i64> {
    let size = parameters.para2 as usize;
    assert!(size > 0);
    let mut cnt = ((size as u64 + MemoryDef::PAGE_SIZE_2M - 1) / MemoryDef::PAGE_SIZE_2M) as usize;
    let mut iovs: Vec<IoVec> = Vec::new();
    iovs.resize(cnt, IoVec::NewFromAddr(0, 0));

    let arrAddr = &mut iovs[0] as *mut _ as u64;
    parameters.para4 = arrAddr;
    parameters.para5 = &mut cnt as *mut _ as u64;

    let mut hostAddr: u64 = 0;
    parameters.para6 = &mut hostAddr as *mut _ as u64;

    let ret = HostSpace::Proxy(ProxyCommand::CudaHostAlloc, parameters);
    if ret != 0 {
        return Ok(ret);
    }

    iovs.resize(cnt, IoVec::default());

    let inner = CudaHostMappableInner {
        hostAddr: hostAddr,
        iovs: iovs,
    };

    let hostm = CudaHostMappable(Arc::new(inner));
    let mappable = MMappable::CudaHost(hostm);

    let mut opts = MMapOpts {
        Length: size as u64,
        Addr: 0,
        Offset: 0,
        Fixed: false,
        Unmap: false,
        Map32Bit: false,
        Private: false,
        VDSO: false,
        Perms: AccessType::AnyAccess(),
        MaxPerms: AccessType::AnyAccess(),
        GrowsDown: false,
        Precommit: true,
        MLockMode: MLockMode::default(),
        Kernel: false,
        Mapping: None,
        Mappable: mappable,
        Hint: "".to_string(),
    };

    let addr = match task.mm.MMap(task, &mut opts) {
        Ok(addr) => addr,
        Err(e) => {
            error!("cudahostalloc fail with error {:?}", e);
            todo!("cudahostalloc fail");
        }
    };

    task.CopyOutObj(&addr, parameters.para1)?;
    task.mm.TlbShootdown();
    return Ok(0);
}

pub fn CudeFreeHost(task: &Task, mut parameters: ProxyParameters) -> Result<i64> {
    let vaddr = parameters.para1;
    let _rl = task.mm.MappingReadLock();
    match task.mm.GetVmaAndRangeLocked(vaddr) {
        None => return Ok(2),
        Some((v, r)) => {
            if r.Start() != vaddr {
                // the cudafreehost must be same as VMA start address
                return Ok(2);
            }

            let hm = match v.mappable {
                MMappable::CudaHost(hm) => hm,
                _ => {
                    return Ok(2);
                }
            };

            parameters.para2 = hm.hostAddr;
            parameters.para3 = &hm.iovs[0] as *const _ as u64;
            parameters.para4 = hm.iovs.len() as u64;
            let ret = HostSpace::Proxy(ProxyCommand::CudaFreeHost, parameters);
            if ret != 0 {
                return Ok(ret);
            }

            match task.mm.RemoveVMAsLocked(&r) {
                Err(_e) => return Ok(2),
                Ok(_) => (),
            }

            task.mm.TlbShootdown();

            return Ok(0);
        }
    }
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
            let prs = match task.mm.GetVmaAndRangeLocked(src) {
                None => {
                    return Ok(1);
                }
                Some((v, r)) => match v.mappable {
                    MMappable::CudaHost(hm) => {
                        if r.Start() <= src && src + count <= r.End() {
                            let mut prs = Vec::new();
                            prs.push(IoVec::NewFromAddr(hm.hostAddr, count as usize));
                            prs
                        } else {
                            return Ok(1);
                        }
                    }
                    _ => task.V2P(src, count, false, false)?,
                },
            };

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
            let prs = match task.mm.GetVmaAndRangeLocked(dst) {
                None => {
                    return Ok(1);
                }
                Some((v, r)) => match v.mappable {
                    MMappable::CudaHost(hm) => {
                        if r.Start() <= dst && dst + count <= r.End() {
                            let mut prs = Vec::new();
                            prs.push(IoVec::NewFromAddr(hm.hostAddr, count as usize));
                            prs
                        } else {
                            return Ok(1);
                        }
                    }
                    _ => task.V2P(dst, count, true, false)?,
                },
            };

            //let prs = task.V2P(dst, count, true, false)?;

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
            let prs = match task.mm.GetVmaAndRangeLocked(src) {
                None => {
                    return Ok(1);
                }
                Some((v, r)) => match v.mappable {
                    MMappable::CudaHost(hm) => {
                        if r.Start() <= src && src + count <= r.End() {
                            let mut prs = Vec::new();
                            prs.push(IoVec::NewFromAddr(hm.hostAddr, count as usize));
                            prs
                        } else {
                            return Ok(1);
                        }
                    }
                    _ => task.V2P(src, count, false, false)?,
                },
            };

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
            let prs = match task.mm.GetVmaAndRangeLocked(dst) {
                None => {
                    return Ok(1);
                }
                Some((v, r)) => match v.mappable {
                    MMappable::CudaHost(hm) => {
                        if r.Start() <= dst && dst + count <= r.End() {
                            let mut prs = Vec::new();
                            prs.push(IoVec::NewFromAddr(hm.hostAddr, count as usize));
                            prs
                        } else {
                            return Ok(1);
                        }
                    }
                    _ => task.V2P(dst, count, true, false)?,
                },
            };
            
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
