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

use super::super::loader::loader::*;
use super::super::memmgr::metadata::*;
use super::super::qlib::auth::cap_set::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::sys_seccomp::*;

// PR_* flags, from <linux/pcrtl.h> for prctl(2).

// PR_SET_PDEATHSIG sets the process' death signal.
pub const PR_SET_PDEATHSIG: i32 = 1;

// PR_GET_PDEATHSIG gets the process' death signal.
pub const PR_GET_PDEATHSIG: i32 = 2;

// PR_GET_DUMPABLE gets the process' dumpable flag.
pub const PR_GET_DUMPABLE: i32 = 3;

// PR_SET_DUMPABLE sets the process' dumpable flag.
pub const PR_SET_DUMPABLE: i32 = 4;

// PR_GET_KEEPCAPS gets the value of the keep capabilities flag.
pub const PR_GET_KEEPCAPS: i32 = 7;

// PR_SET_KEEPCAPS sets the value of the keep capabilities flag.
pub const PR_SET_KEEPCAPS: i32 = 8;

// PR_GET_TIMING gets the process' timing method.
pub const PR_GET_TIMING: i32 = 13;

// PR_SET_TIMING sets the process' timing method.
pub const PR_SET_TIMING: i32 = 14;

// PR_SET_NAME sets the process' name.
pub const PR_SET_NAME: i32 = 15;

// PR_GET_NAME gets the process' name.
pub const PR_GET_NAME: i32 = 16;

// PR_GET_SECCOMP gets a process' seccomp mode.
pub const PR_GET_SECCOMP: i32 = 21;

// PR_SET_SECCOMP sets a process' seccomp mode.
pub const PR_SET_SECCOMP: i32 = 22;

// PR_CAPBSET_READ gets the capability bounding set.
pub const PR_CAPBSET_READ: i32 = 23;

// PR_CAPBSET_DROP sets the capability bounding set.
pub const PR_CAPBSET_DROP: i32 = 24;

// PR_GET_TSC gets the value of the flag determining whether the
// timestamp counter can be read.
pub const PR_GET_TSC: i32 = 25;

// PR_SET_TSC sets the value of the flag determining whether the
// timestamp counter can be read.
pub const PR_SET_TSC: i32 = 26;

// PR_SET_TIMERSLACK sets the process' time slack.
pub const PR_SET_TIMERSLACK: i32 = 29;

// PR_GET_TIMERSLACK gets the process' time slack.
pub const PR_GET_TIMERSLACK: i32 = 30;

// PR_TASK_PERF_EVENTS_DISABLE disables all performance counters
// attached to the calling process.
pub const PR_TASK_PERF_EVENTS_DISABLE: i32 = 31;

// PR_TASK_PERF_EVENTS_ENABLE enables all performance counters attached
// to the calling process.
pub const PR_TASK_PERF_EVENTS_ENABLE: i32 = 32;

// PR_MCE_KILL sets the machine check memory corruption kill policy for
// the calling thread.
pub const PR_MCE_KILL: i32 = 33;

// PR_MCE_KILL_GET gets the machine check memory corruption kill policy
// for the calling thread.
pub const PR_MCE_KILL_GET: i32 = 34;

// PR_SET_MM modifies certain kernel memory map descriptor fields of
// the calling process. See prctl(2) for more information.
pub const PR_SET_MM: i32 = 35;

pub const PR_SET_MM_START_CODE: i32 = 1;
pub const PR_SET_MM_END_CODE: i32 = 2;
pub const PR_SET_MM_START_DATA: i32 = 3;
pub const PR_SET_MM_END_DATA: i32 = 4;
pub const PR_SET_MM_START_STACK: i32 = 5;
pub const PR_SET_MM_START_BRK: i32 = 6;
pub const PR_SET_MM_BRK: i32 = 7;
pub const PR_SET_MM_ARG_START: i32 = 8;
pub const PR_SET_MM_ARG_END: i32 = 9;
pub const PR_SET_MM_ENV_START: i32 = 10;
pub const PR_SET_MM_ENV_END: i32 = 11;
pub const PR_SET_MM_AUXV: i32 = 12;
// PR_SET_MM_EXE_FILE supersedes the /proc/pid/exe symbolic link with a
// new one pointing to a new executable file identified by the file
// descriptor provided in arg3 argument. See prctl(2) for more
// information.
pub const PR_SET_MM_EXE_FILE: i32 = 13;
pub const PR_SET_MM_MAP: i32 = 14;
pub const PR_SET_MM_MAP_SIZE: i32 = 15;

// PR_SET_CHILD_SUBREAPER sets the "child subreaper" attribute of the
// calling process.
pub const PR_SET_CHILD_SUBREAPER: i32 = 36;

// PR_GET_CHILD_SUBREAPER gets the "child subreaper" attribute of the
// calling process.
pub const PR_GET_CHILD_SUBREAPER: i32 = 37;

// PR_SET_NO_NEW_PRIVS sets the calling thread's no_new_privs bit.
pub const PR_SET_NO_NEW_PRIVS: i32 = 38;

// PR_GET_NO_NEW_PRIVS gets the calling thread's no_new_privs bit.
pub const PR_GET_NO_NEW_PRIVS: i32 = 39;

// PR_GET_TID_ADDRESS retrieves the clear_child_tid address.
pub const PR_GET_TID_ADDRESS: i32 = 40;

// PR_SET_THP_DISABLE sets the state of the "THP disable" flag for the
// calling thread.
pub const PR_SET_THP_DISABLE: i32 = 41;

// PR_GET_THP_DISABLE gets the state of the "THP disable" flag for the
// calling thread.
pub const PR_GET_THP_DISABLE: i32 = 42;

// PR_MPX_ENABLE_MANAGEMENT enables kernel management of Memory
// Protection eXtensions (MPX) bounds tables.
pub const PR_MPX_ENABLE_MANAGEMENT: i32 = 43;

// PR_MPX_DISABLE_MANAGEMENT disables kernel management of Memory
// Protection eXtensions (MPX) bounds tables.
pub const PR_MPX_DISABLE_MANAGEMENT: i32 = 44;

// From <asm/prctl.h>
// Flags are used in syscall arch_prctl(2).
pub const ARCH_SET_GS: i32 = 0x1001;
pub const ARCH_SET_FS: i32 = 0x1002;
pub const ARCH_GET_FS: i32 = 0x1003;
pub const ARCH_GET_GS: i32 = 0x1004;
pub const ARCH_SET_CPUID: i32 = 0x1012;

// Flags for prctl(PR_SET_DUMPABLE), defined in include/linux/sched/coredump.h.
pub const SUID_DUMP_DISABLE: i32 = 0;
pub const SUID_DUMP_USER: i32 = 1;
pub const SUID_DUMP_ROOT: i32 = 2;

pub fn SysPrctl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let option = args.arg0 as i32;

    let thread = task.Thread();
    match option {
        PR_SET_PDEATHSIG => {
            let sig = Signal(args.arg1 as i32);
            if sig.0 != 0 && !sig.IsValid() {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            thread.SetParentDeathSignal(sig);
            return Ok(0);
        }
        PR_GET_PDEATHSIG => {
            let addr = args.arg1 as u64;
            let sig = thread.ParentDeathSignal();

            task.CopyOutObj(&sig.0, addr)?;
            //*task.GetTypeMut(addr)? = sig.0;
            return Ok(0);
        }
        PR_GET_DUMPABLE => {
            let d = thread.MemoryManager().Dumpability();
            match d {
                NOT_DUMPABLE => return Ok(SUID_DUMP_DISABLE as i64),
                USER_DUMPABLE => return Ok(SUID_DUMP_USER as i64),
                ROOT_DUMPABLE => return Ok(SUID_DUMP_ROOT as i64),
                _ => {
                    panic!("Unknown dumpability {}", d)
                }
            }
        }
        PR_SET_DUMPABLE => {
            let d;
            let typ = args.arg1 as i32;
            match typ {
                SUID_DUMP_DISABLE => d = NOT_DUMPABLE,
                SUID_DUMP_USER => d = USER_DUMPABLE,
                _ => return Err(Error::SysError(SysErr::EINVAL)),
            }
            thread.MemoryManager().SetDumpability(d);
            return Ok(0);
        }
        PR_GET_KEEPCAPS => {
            if thread.Credentials().lock().KeepCaps {
                return Ok(1);
            }
            return Ok(0);
        }
        PR_SET_KEEPCAPS => {
            let val = args.arg1 as i32;
            // prctl(2): arg2 must be either 0 (permitted capabilities are cleared)
            // or 1 (permitted capabilities are kept).
            if val == 0 {
                thread.SetKeepCaps(false)
            } else if val == 1 {
                thread.SetKeepCaps(true)
            } else {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            return Ok(0);
        }
        PR_SET_NAME => {
            let addr = args.arg1 as u64;
            let (name, err) = task.CopyInString(addr, TASK_COMM_LEN - 1);
            match err {
                Ok(()) => (),
                Err(Error::SysError(SysErr::ENAMETOOLONG)) => (),
                Err(e) => return Err(e),
            }

            thread.SetName(&name);
        }
        PR_GET_NAME => {
            let addr = args.arg1 as u64;
            let mut buf: [u8; TASK_COMM_LEN] = [0; TASK_COMM_LEN];
            let name = thread.Name();
            let mut len = if name.len() > TASK_COMM_LEN {
                TASK_COMM_LEN
            } else {
                name.len()
            };

            buf[0..len].clone_from_slice(&name.as_bytes()[0..len]);
            if len < TASK_COMM_LEN {
                len += 1;
            }

            task.CopyOutSlice(&buf[0..len], addr, len)?;
        }
        PR_SET_MM => {
            if !thread
                .Credentials()
                .HasCapability(Capability::CAP_SYS_RESOURCE)
            {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let typ = args.arg1 as i32;
            match typ {
                PR_SET_MM_EXE_FILE => {
                    let fd = args.arg2 as i32;
                    let file = task.GetFile(fd)?;

                    // They trying to set exe to a non-file?
                    let inode = file.Dirent.Inode();
                    if !inode.StableAttr().IsFile() {
                        return Err(Error::SysError(SysErr::EBADF));
                    }

                    // Set the underlying executable.
                    task.mm.SetExecutable(&file.Dirent);
                }
                PR_SET_MM_AUXV
                | PR_SET_MM_START_CODE
                | PR_SET_MM_END_CODE
                | PR_SET_MM_START_DATA
                | PR_SET_MM_END_DATA
                | PR_SET_MM_START_STACK
                | PR_SET_MM_START_BRK
                | PR_SET_MM_BRK
                | PR_SET_MM_ARG_START
                | PR_SET_MM_ARG_END
                | PR_SET_MM_ENV_START
                | PR_SET_MM_ENV_END => {
                    info!("not implemented");
                    return Err(Error::SysError(SysErr::EINVAL));
                }
                _ => return Err(Error::SysError(SysErr::EINVAL)),
            }
        }
        PR_SET_NO_NEW_PRIVS => {
            if args.arg1 != 0 || args.arg2 != 0 || args.arg3 != 0 || args.arg4 != 0 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            return Ok(1);
        }
        PR_SET_SECCOMP => {
            if args.arg1 as i32 != SECCOMP_MODE_FILTER {
                // Unsupported mode.
                return Err(Error::SysError(SysErr::EINVAL));
            }

            panic!("SysPrctl::PR_SET_SECCOMP doesn't support.... ");
            //return seccomp(task, SECCOMP_SET_MODE_FILTER as u64, 0, args.arg2 as u64)
        }
        PR_GET_SECCOMP => {
            panic!("SysPrctl::PR_GET_SECCOMP doesn't support.... ");
            //return Err(Error::SysError(SysErr::ENOSYS))
        }
        PR_CAPBSET_READ => {
            let cap = args.arg1 as i32;
            if !Capability::Ok(cap) {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut rv = 0;
            let cred = thread.Credentials();
            if CapSetOf(cap as u64).0 & cred.lock().BoundingCaps.0 != 0 {
                rv = 1;
            }

            return Ok(rv);
        }
        PR_CAPBSET_DROP => {
            let cap = args.arg1 as i32;
            if !Capability::Ok(cap) {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            thread.DropBoundingCapability(cap as u64)?;
            return Ok(0);
        }
        PR_GET_TIMING
        | PR_SET_TIMING
        | PR_GET_TSC
        | PR_SET_TSC
        | PR_TASK_PERF_EVENTS_DISABLE
        | PR_TASK_PERF_EVENTS_ENABLE
        | PR_GET_TIMERSLACK
        | PR_SET_TIMERSLACK
        | PR_MCE_KILL
        | PR_MCE_KILL_GET
        | PR_GET_TID_ADDRESS
        | PR_SET_CHILD_SUBREAPER
        | PR_GET_CHILD_SUBREAPER
        | PR_GET_THP_DISABLE
        | PR_SET_THP_DISABLE
        | PR_MPX_ENABLE_MANAGEMENT
        | PR_MPX_DISABLE_MANAGEMENT => {
            info!("not implement...");
            return Err(Error::SysError(SysErr::EINVAL));
        }
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }

    return Ok(0);
}
