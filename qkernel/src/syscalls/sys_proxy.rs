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

lazy_static! {
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
        ProxyCommand::CudaSetDevice |
        ProxyCommand::CudaDeviceSynchronize => {
            error!("hochan SysProxy CudaSetDevice");
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaMalloc => {
            let addr : u64 = 0;
            parameters.para1 = &addr as * const _ as u64;

            // todo hochan temporarely using ParamInfo to set allocated address from CudaMalloc. Need to figure out how to define parameters.para1 as mut
            let mut paramInfo = ParamInfo::default();
            parameters.para3 = &mut paramInfo as *const _ as u64;

            error!("hochan before CudaMalloc HostSpace::Proxy paramInfo {:x?}", paramInfo);    
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            error!("hochan after CudaMalloc HostSpace::Proxy paramInfo {:x?}", paramInfo);  

            if ret == 0 {
                // task.CopyOutObj(&addr, args.arg1 as u64)?;
                task.CopyOutObj(&(paramInfo.addr), args.arg1 as u64)?;
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
        ProxyCommand::CudaRegisterFatBinary => {
            let data: Vec<u8> = task.CopyInVec(parameters.para2, parameters.para1 as usize)?;
            parameters.para2 = &data[0] as *const _ as u64;
            let ret = HostSpace::Proxy(
                ProxyCommand::CudaRegisterFatBinary,
                parameters,
            );
            return Ok(ret);
            
        }
        ProxyCommand::CudaRegisterFunction => {
            let mut data = task.CopyInObj::<RegisterFunctionInfo>(parameters.para1)?;
            error!("hochan CudaRegisterFunction data {:x?}, parameters {:x?}", data, parameters);

            let deviceName = CString::ToString(task, data.deviceName)?;
            data.deviceName = &(deviceName.as_bytes()[0]) as * const _ as u64;
            parameters.para1 = &data as * const _ as u64;
            parameters.para2 = deviceName.as_bytes().len() as u64;
            error!("hochan deviceName {}, data.deviceName {:x}, parameters {:x?}", deviceName, data.deviceName, parameters);

            let mut paramInfo = ParamInfo::default();
            parameters.para3 = &mut paramInfo as *const _ as u64;

            let ret = HostSpace::Proxy(
                ProxyCommand::CudaRegisterFunction,
                parameters,
            );

            error!("hochan paramInfo {:x?}", paramInfo);

            let mut params_proxy: Vec<u16>=Vec::new();
            for i in 0..paramInfo.paramNum as usize {
                params_proxy.push(paramInfo.paramSizes[i]);
                error!("hochan i {}, paramInfo.paramSizes[i] {}", i, paramInfo.paramSizes[i]);
            }

            PARAM_INFOS.lock().insert(data.hostFun, Arc::new(params_proxy));
            error!("hochan PARAM_INFOS {:x?}", PARAM_INFOS.lock());

            return Ok(ret);
        }
        ProxyCommand::CudaLaunchKernel => {
            let mut data = task.CopyInObj::<LaunchKernelInfo>(parameters.para1)?;
            let paramInfo = PARAM_INFOS.lock().get(&data.func).unwrap().clone();
            error!("hochan LaunchKernelInfo data {:x?}, paramInfo {:x?}, parameters {:x?}", data, paramInfo, parameters);

            let mut paramAddrs:Vec<u64> = task.CopyInVec(data.args, paramInfo.len())?;
            error!("hochan paramAddrs {:x?}", paramAddrs);

            let mut paramValues = Vec::new();
            for i in 0..paramInfo.len() {
                let valueBytes:Vec<u8> = task.CopyInVec(paramAddrs[i], (paramInfo[i]) as usize)?;
                error!("hochan valueBytes {:x?}", valueBytes);
                
                paramValues.push(valueBytes);
                paramAddrs[i] = &(paramValues[i][0]) as *const _ as u64;
                error!("hochan i {} paramAddrs[i] {:x} paramValues[i] {:x?}",i, paramAddrs[i], paramValues[i]);
            }
            // paramAddrs[3] = 0x64;
            error!("hochan paramAddrs after set {:x?}", paramAddrs);
            data.args = &paramAddrs[0] as * const _ as u64;
            error!("hochan data.args {:x?}", data.args);

            // let v = task.CopyInObj::<u32>(paramAddrs[3])?;
            // error!("hochan v {}", v);

            parameters.para1 = &data as * const _ as u64;
            let ret = HostSpace::Proxy(
                ProxyCommand::CudaLaunchKernel,
                parameters,
            );
            return Ok(ret);
        }
        _ => todo!()
    }
}

pub fn CudaRegisterFunction(parameters:ProxyParameters )-> Result<i64>{
    error!("hochan CudaRegisterFunction");
    let ret = HostSpace::Proxy(
        ProxyCommand::CudaRegisterFunction,
        parameters,
    );
    return Ok(ret);
}

pub fn CudaMemcpy(task: &Task, dst: u64, src: u64, count: u64, kind: CudaMemcpyKind) -> Result<i64> {
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => {
            error!("CudaMemcpy get unexpected kind CUDA_MEMCPY_HOST_TO_HOST");
            return Ok(1);
        }
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            // src is the virtual addr
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
            // dst is the virtual addr
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