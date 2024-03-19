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

use alloc::string::String;
use spin::Mutex;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::qlib::common::*;
use crate::syscalls::syscalls::*;
use crate::task::*;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::linux_def::SysErr;
use crate::qlib::proxy::*;
use super::super::util::cstring::*;

lazy_static!{
    pub static ref PARAM_INFOS:Mutex<BTreeMap<u64, Arc<Vec<u16>>>> = Mutex::new(BTreeMap::new());
}

pub fn SysProxy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let commandId = args.arg0 as u64;
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

    match cmd {
        ProxyCommand::None => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        ProxyCommand::CudaChooseDevice => {
            let mut device: i32 = 0;

            parameters.para1 = &mut device as *mut _ as u64;

            let deviceProperties = task.CopyInObj::<CudaDeviceProperties>(parameters.para2)?;  

            parameters.para2 = &deviceProperties as * const _ as u64;  

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            // error!("device value after function call, is {}", device);

            if ret == 0 {
                task.CopyOutObj(&device, args.arg1 as u64)?; 
            }

            return Ok(ret);
        }
        ProxyCommand::CudaDeviceGetAttribute => {
            let mut value: i32 = 0;   
            parameters.para1 = &mut value as * mut _ as u64;

            // let attribute: u32 = parameters.para2 as u32;
            // error!("SysProxy CudaDeviceGetAttribute, query about attribute: {}, device: {}", attribute, parameters.para3);

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&value, args.arg1 as u64)?; 
            }

            return Ok(ret);
        }
        ProxyCommand::CudaDeviceGetByPCIBusId => {
            let mut device: i32 = 0;
            parameters.para1 = &mut device as *mut _ as u64;
            
            let PCIBusId = CString::ToString(task, parameters.para2)?;
       
            parameters.para2 = &(PCIBusId.as_bytes()[0]) as * const _ as u64;
          
            parameters.para3 = PCIBusId.as_bytes().len() as u64;

            // error!("PCIBusId {}",PCIBusId);

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&device, args.arg1 as u64)?; 
            }

            return Ok(ret);
        }
        ProxyCommand::CudaDeviceGetCacheConfig => {
            let mut CacheConfig:u32;

            unsafe{ 
                CacheConfig = *(parameters.para1 as *mut _) ;  
                // error!("value of parameter.para2 is: {}",*(parameters.para1 as *mut u32));         
            }

            parameters.para1 = &mut CacheConfig as *mut _ as u64;
               
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&CacheConfig, args.arg1 as u64)?; 
            }

            return Ok(ret);
        }
        ProxyCommand::CudaDeviceGetLimit => {

            let mut limit:usize = 0; 
            
            parameters.para1 = &mut limit as * mut _ as u64;

            // let attribute: u32 = parameters.para2 as u32;
            // error!("SysProxy CudaDeviceGetLimit, query about attribute: {}", attribute);

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            
            if ret == 0 {
                task.CopyOutObj(&limit, args.arg1 as u64)?; 
            }

            return Ok(ret);

        }
        ProxyCommand::CudaDeviceGetP2PAttribute => {
            let mut value:i32 = 0;
            parameters.para1 = &mut value as * mut _ as u64;

            // let attribute: u32 = parameters.para2 as u32;
            // error!("SysProxy CudaDeviceGetP2PAttribute, query about attribute: {}", attribute);

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&value, args.arg1 as u64)?;
            }
            return Ok(ret);

        }
        ProxyCommand::CudaDeviceGetPCIBusId => {
            let mut pciBusIdAddress = CString::ToString(task, parameters.para1)?;

            // error!("pciBusIdAddress: {}", pciBusIdAddress);

            parameters.para1 = &mut pciBusIdAddress as * mut String as u64;

            // error!("SysProxy CudaDeviceGetPCIBusId, query about device:{}", parameters.para3);

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

           if ret == 0 {
            task.CopyOutString(args.arg1, parameters.para2 as usize, &pciBusIdAddress)?;
           }

            return Ok(ret);
        }
        ProxyCommand::CudaDeviceGetSharedMemConfig => {
            let mut sharedMemConfig:u32 = unsafe { *(parameters.para1 as *mut _ )};
            // error!("value of sharedMemConfig is: {}", sharedMemConfig);

            parameters.para1 = &mut sharedMemConfig as *mut _ as u64;

            let ret = HostSpace::Proxy(
                cmd, 
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&sharedMemConfig, args.arg1 as u64)?;
            }

            return Ok(ret);

        }
        ProxyCommand::CudaDeviceGetStreamPriorityRange => {
            let mut lowPriority:i32;
            let mut highPriority:i32;

            unsafe{
                lowPriority = *(parameters.para1 as *mut _);
                highPriority = *(parameters.para2 as *mut _);
            }

            parameters.para1 = &mut lowPriority as *mut _ as u64;
            parameters.para2 = &mut highPriority as *mut _ as u64;

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&lowPriority, args.arg1)?;
                task.CopyOutObj(&highPriority, args.arg2)?
            }

            return Ok(ret);

        }
        ProxyCommand::CudaDeviceReset => {
            // error!("SysProxy CudaDeviceReset");
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        } 
        ProxyCommand::CudaDeviceSetCacheConfig => {
            // error!("SysProxy CudaDeviceSetCacheConfig cache configuration: {}", parameters.para1 as u32);

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaSetDevice |
        ProxyCommand::CudaDeviceSynchronize => {
            // error!("SysProxy CudaSetDevice");
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaSetDeviceFlags => {
            // error!("SysProxy CudaSetDeviceFlags");
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaGetDevice => {
            let mut device:i32 = 0;
            parameters.para1 = &mut device as *mut _ as u64;

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret ==0 {
                task.CopyOutObj(&device, args.arg1 as u64)?;
            }
            return Ok(ret);
        }
        ProxyCommand::CudaGetDeviceCount => {
            let mut deviceCount:i32 = 0 ; 
            
            parameters.para1 = &mut deviceCount as *mut _ as u64;

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
        
            if ret == 0 {
                task.CopyOutObj(&deviceCount, args.arg1 as u64)?; 
            }

            return Ok(ret);
        }
        ProxyCommand::CudaGetDeviceProperties => {

            let mut deviceProp:CudaDeviceProperties = CudaDeviceProperties::default();

            parameters.para1 = &mut deviceProp as * mut _ as u64;

            // error!("cudaGetDeviceProperties Device: {}",parameters.para2);

            let ret = HostSpace::Proxy(
                cmd, 
                parameters
            );

            if ret == 0 {
                task.CopyOutObj(&deviceProp, args.arg1)?;
            }

            return Ok(ret);

        }
        ProxyCommand::CudaMalloc => {
            let mut addr:u64 = 0;
            parameters.para1 = &mut addr as *mut _ as u64;

            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&addr, args.arg1 as u64)?;
                // task.CopyOutObj(&(paramInfo.addr), args.arg1 as u64)?;
            }

            return Ok(ret);
        }
         ProxyCommand::CudaFree => {
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0{
                // error!("cuda free memory at {:x}", parameters.para1);
            }

            return Ok(ret);
         }
        ProxyCommand::CudaMemcpy => {
            let ret = CudaMemcpy(
                task, 
                parameters.para1,
                parameters.para2,
                parameters.para3,
                parameters.para4
            )?;

            return Ok(ret);
            
        }
        ProxyCommand::CudaMemcpyAsync => {
            let ret = CudaMemcpyAsync(
                task,
                parameters.para1,
                parameters.para2,
                parameters.para3,
                parameters.para4,
                parameters.para5      
            )?;
            return Ok(ret);
        }
        ProxyCommand::CudaRegisterFatBinary => {
            let data: Vec<u8> = task.CopyInVec(parameters.para2, parameters.para1 as usize)?;
            parameters.para2 = &data[0] as *const _ as u64; 
            let ret = HostSpace::Proxy(
                ProxyCommand::CudaRegisterFatBinary,
                parameters,
            );
            return Ok(ret);
            
        }
        ProxyCommand::CudaUnregisterFatBinary => {
            // error!("fatCubinHandle from the cudaproxy is {:x}", parameters.para1 as u64);
            let ret = HostSpace::Proxy(
                 ProxyCommand::CudaUnregisterFatBinary,
                 parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaRegisterFunction => {
            let mut functionInfo = task.CopyInObj::<RegisterFunctionInfo>(parameters.para1)?;
            // error!("CudaRegisterFunction data {:x?}, parameters {:x?}", functionInfo, parameters);

            let deviceName = CString::ToString(task, functionInfo.deviceName)?;
          
            functionInfo.deviceName = &(deviceName.as_bytes()[0]) as * const _ as u64;
            parameters.para1 = &functionInfo as * const _ as u64;
            parameters.para2 = deviceName.as_bytes().len() as u64;
            // error!("deviceName {}, data.deviceName {:x}, parameters {:x?}", deviceName, functionInfo.deviceName, parameters);

            let mut paramInfo = ParamInfo::default();
            parameters.para3 = &mut paramInfo as *const _ as u64;

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaRegisterFunction,
                parameters,
            );

            // error!("paramInfo {:x?}", paramInfo);

            let mut params_proxy: Vec<u16>=Vec::new();
            for i in 0..paramInfo.paramNum as usize {
                params_proxy.push(paramInfo.paramSizes[i]);
                // error!("i {}, paramInfo.paramSizes[i] {}", i, paramInfo.paramSizes[i]);
            }

            PARAM_INFOS.lock().insert(functionInfo.hostFun, Arc::new(params_proxy));
            // error!("PARAM_INFOS {:x?}", PARAM_INFOS.lock());

            return Ok(ret);
        }
        ProxyCommand::CudaRegisterVar => {
            let mut data = task.CopyInObj::<RegisterVarInfo>(parameters.para1)?;   // still take the addresss 
            // error!("CudaRegisterVar data {:x?}, parameters {:x?}", data, parameters);

            let deviceName = CString::ToString(task, data.deviceName)?;
 
            data.deviceName = &(deviceName.as_bytes()[0]) as * const _ as u64;
            parameters.para1 = &data as * const _ as u64;

            parameters.para2 = deviceName.as_bytes().len() as u64; 
            // error!("deviceName {}, data.deviceName {:x}, parameters {:x?}", deviceName, data.deviceName, parameters);

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaRegisterVar,
                parameters,
            );

            return Ok(ret); 
            
        }
        ProxyCommand::CudaLaunchKernel => {
            let mut kernelInfo = task.CopyInObj::<LaunchKernelInfo>(parameters.para1)?;
            let paramInfo = PARAM_INFOS.lock().get(&kernelInfo.func).unwrap().clone();
            // error!("LaunchKernelInfo data {:x?}, paramInfo {:x?}, parameters {:x?}", kernelInfo, paramInfo, parameters);

            let mut paramAddrs:Vec<u64> = task.CopyInVec(kernelInfo.args, paramInfo.len())?;
            // error!("paramAddrs {:x?}", paramAddrs);

            let mut paramValues = Vec::new();
            for i in 0..paramInfo.len() {
                let valueBytes:Vec<u8> = task.CopyInVec(paramAddrs[i], (paramInfo[i]) as usize)?;
                // error!("valueBytes {:x?}", valueBytes);
                
                paramValues.push(valueBytes);
                paramAddrs[i] = &(paramValues[i][0]) as *const _ as u64;
                // error!("i {} paramAddrs[i] {:x} paramValues[i] {:x?}",i, paramAddrs[i], paramValues[i]);
            }
            // error!("paramAddrs after set {:x?}", paramAddrs);
            kernelInfo.args = &paramAddrs[0] as * const _ as u64;
            // error!("kernelInfo.args {:x?}", kernelInfo.args);

            parameters.para1 = &kernelInfo as * const _ as u64;
            let ret = HostSpace::Proxy(
                ProxyCommand::CudaLaunchKernel,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaStreamSynchronize => {
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaStreamCreate => {
            unsafe{
            let mut stream:u64 = *(parameters.para1 as *mut _);
            
            parameters.para1 = &mut stream as *mut _ as u64;
         
            // error!("stream content is :{:x}", stream);
            // error!("parameters.para1(address of stream) is:{:x}",parameters.para1);
            let ret = HostSpace::Proxy(
                    cmd,
                    parameters,
                );

            if ret == 0 {
                task.CopyOutObj(&stream, args.arg1 as u64)?; 
                }
            
            return Ok(ret);
            }
        }
        ProxyCommand::CudaStreamDestroy => {
            // error!("stream parameter from cudaproxy is : {:x}", parameters.para1);
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            return Ok(ret);
        }
        ProxyCommand::CudaStreamIsCapturing => {
            let mut pCaptureStatus:u32;
            
            unsafe{ 
                pCaptureStatus = *(parameters.para2 as *mut _) ; 
            }
            parameters.para2 = &mut pCaptureStatus as *mut _ as u64;
            // error!("content of the pCaptureStatus is: {}", pCaptureStatus );
               
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            
            if ret == 0 {
                task.CopyOutObj(&pCaptureStatus, args.arg2 as u64)?;
            }
                
            return Ok(ret);
            
        }
        ProxyCommand::CuModuleGetLoadingMode => {
            let mut mode:u32 = 0; 
            parameters.para1 = &mut mode as *mut _ as u64; 

            let ret = HostSpace::Proxy(
                 cmd,
                 parameters,
             );
             
            if ret == 0 {
                task.CopyOutObj(&mode, args.arg1 as u64)?;
            }
            return Ok(ret);

        }
        ProxyCommand::CudaGetLastError => {
            // error!("SysProxy CudaGetLastError");
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CuDevicePrimaryCtxGetState => {
            let mut flags:u32 = 0;
            let mut active:i32 = 0;

            parameters.para2 = &mut flags as *mut _ as u64;
            parameters.para3 = &mut active as *mut _ as u64;
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            // error!("flags {}, active {}", flags, active);

            if ret == 0 {
                task.CopyOutObj(&flags, args.arg2 as u64)?;
                task.CopyOutObj(&active, args.arg3 as u64)?;

            }
            return Ok(ret);
        }
        ProxyCommand::NvmlInitWithFlags => {
            // error!("SysProxy NvmlInitWithFlags");
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
            
        }
        _ => todo!()
    }
}

pub fn CudaMemcpy(task: &Task, dst: u64, src: u64, count: u64, kind: CudaMemcpyKind) -> Result<i64> {
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => {
            // error!("CudaMemcpy get unexpected kind CUDA_MEMCPY_HOST_TO_HOST");
            return Ok(1);
        }
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            
            let mut prs = Vec::new();
            task.V2P(src, count, &mut prs, true, false)?;

            let parameters = ProxyParameters {
                para1: dst,
                para2: &prs[0] as * const _ as u64,
                para3: prs.len() as u64,
                para4: count as u64,
                para5: kind,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaMemcpy,
                parameters,
            );

            return Ok(ret);
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            
            let mut prs = Vec::new();
            task.V2P(dst, count, &mut prs, true, false)?;

            let parameters = ProxyParameters {
                para1: &prs[0] as * const _ as u64,
                para2: prs.len() as u64,
                para3: src,
                para4: count as u64,
                para5: kind,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaMemcpy,
                parameters,
            );

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

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaMemcpy,
                parameters,
            );

            return Ok(ret);
        }
        _ => todo!()
    }
    
}

fn CudaMemcpyAsync(task: &Task, dst: u64, src: u64, count: u64, kind: CudaMemcpyKind, stream: u64) -> Result<i64> {
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => {
            // error!("CudaMemcpy get unexpected kind CUDA_MEMCPY_HOST_TO_HOST");
            return Ok(1);
        }
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            
            let mut prs = Vec::new();
            task.V2P(src, count, &mut prs, true, false)?;

            let parameters = ProxyParameters {
                para1: dst,
                para2: &prs[0] as * const _ as u64,
                para3: prs.len() as u64,
                para4: count as u64,
                para5: kind,
                para6: stream,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaMemcpyAsync,
                parameters,
            );

            return Ok(ret);
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            
            let mut prs = Vec::new();
            task.V2P(dst, count, &mut prs, true, false)?;

            let parameters = ProxyParameters {
                para1: &prs[0] as * const _ as u64,
                para2: prs.len() as u64,
                para3: src,
                para4: count as u64,
                para5: kind,
                para6: stream,
                ..Default::default()
            };

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaMemcpyAsync,
                parameters,
            );

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

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaMemcpyAsync,
                parameters,
            );

            return Ok(ret);
        }
        _ => todo!()
    }
}