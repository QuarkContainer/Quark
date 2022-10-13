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

use crate::qlib::proxy::*;
use crate::qlib::kernel::Kernel::HostSpace;
use super::super::qlib::common::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

// arg0: command id
// arg1: data in address
// arg2: data out address
pub fn SysProxy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let commandId = args.arg0 as u64;
    let addrIn = args.arg1 as u64;
    let addrOut = args.arg2 as u64;

    let cmd : Command = unsafe { core::mem::transmute(commandId as u64) };
    match cmd {
        Command::Cmd1 => {
            let dataIn: Cmd1In = task.CopyInObj(addrIn)?;
            let dataOut = Cmd1Out::default();
            let ret = HostSpace::Proxy(commandId, &dataIn as * const _ as u64, &dataOut as * const _ as u64);
            task.CopyOutObj(&dataOut, addrOut)?;
            return Ok(ret)
        }
        Command::Cmd2 => {}
    }

    return Ok(0)
}