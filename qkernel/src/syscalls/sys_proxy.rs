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

use alloc::vec::Vec;

use crate::qlib::common::*;
use crate::syscalls::syscalls::*;
use crate::task::*;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::linux_def::SysErr;
use crate::qlib::proxy::*;

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
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );
            return Ok(ret);
        }
        ProxyCommand::CudaMalloc => {
            let addr : u64 = 0;
            parameters.para1 = &addr as * const _ as u64;
            let ret = HostSpace::Proxy(
                cmd,
                parameters,
            );

            if ret == 0 {
                task.CopyOutObj(&addr, args.arg1 as u64)?;
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
        _ => todo!()
    }
}

pub fn CudaMemcpy(task: &Task, dst: u64, src: u64, count: u64, kind: CudaMemcpyKind) -> Result<i64> {
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => todo!(),
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
        _ => todo!()
    }
    
}