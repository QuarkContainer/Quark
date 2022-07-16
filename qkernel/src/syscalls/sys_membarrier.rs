// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
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

use super::super::qlib::common::*;
use super::super::qlib::linux::membarrier::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::kernel::*;
use super::super::qlib::kernel::Kernel::HostSpace;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

// Membarrier implements syscall membarrier(2).
pub fn SysMembarrier(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let cmd = args.arg0 as i32;
    let flags = args.arg1 as u32;

    match cmd {
        MEMBARRIER_CMD_QUERY => {
            if flags != 0 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut supportedCommands = 0;
            if SHARESPACE.supportMemoryBarrier {
                supportedCommands = MEMBARRIER_CMD_GLOBAL
                    | MEMBARRIER_CMD_GLOBAL_EXPEDITED
                    | MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED
                    | MEMBARRIER_CMD_PRIVATE_EXPEDITED
                    | MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED;
            }
            return Ok(supportedCommands as _);
        }
        MEMBARRIER_CMD_GLOBAL
        | MEMBARRIER_CMD_GLOBAL_EXPEDITED
        | MEMBARRIER_CMD_PRIVATE_EXPEDITED => {
            if flags != 0 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            if !SHARESPACE.supportMemoryBarrier {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            if cmd == MEMBARRIER_CMD_PRIVATE_EXPEDITED && !task.mm.IsMembarrierPrivateEnabled() {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let ret = HostSpace::HostMemoryBarrier();
            if ret >= 0 {
                return Ok(ret)
            }
            return Err(Error::SysError(-ret as _))
        }
        MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED => {
            if flags != 0 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            if !SHARESPACE.supportMemoryBarrier {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            // no-op
            return Ok(0)
        }
        MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED => {
            if flags != 0 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            if !SHARESPACE.supportMemoryBarrier {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            task.mm.EnableMembarrierPrivate();
            return Ok(0)
        }
        MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ |
        MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ => {
            error!("membarrier doesn't support rseq cmd {}", cmd);
            return Err(Error::SysError(SysErr::EINVAL))
        }
        _ => {
            error!("membarrier doesn't support cmd {}", cmd);
            return Err(Error::SysError(SysErr::EINVAL))
        },
    }
}
