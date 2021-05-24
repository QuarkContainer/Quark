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

use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;

// We unconditionally report a single NUMA node. This also means that our
// "nodemask_t" is a single unsigned long (uint64).
pub const MAX_NODES : usize = 1;
pub const ALLOW_NODE_MASK : u64 =  (1 << MAX_NODES) - 1;

pub fn CopyInNodemask(task: &Task, addr: u64, maxnode: u32) -> Result<u64> {
    // "nodemask points to a bit mask of node IDs that contains up to maxnode
    // bits. The bit mask size is rounded to the next multiple of
    // sizeof(unsigned long), but the kernel will use bits only up to maxnode.
    // A NULL value of nodemask or a maxnode value of zero specifies the empty
    // set of nodes. If the value of maxnode is zero, the nodemask argument is
    // ignored." - set_mempolicy(2). Unfortunately, most of this is inaccurate
    // because of what appears to be a bug: mm/mempolicy.c:get_nodes() uses
    // maxnode-1, not maxnode, as the number of bits.
    if maxnode == 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let bits = maxnode - 1;
    if bits as u64 > MemoryDef::PAGE_SIZE * 8 { // also handles overflow from maxnode == 0
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if bits == 0 {
        return Ok(0)
    }

    // Copy in the whole nodemask.
    let numU64 = ((bits + 63) / 64) as usize;

    let val : &[u64] = task.GetSlice(addr, numU64)?;
    if val[0] & !ALLOW_NODE_MASK != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    for i in 1 .. numU64 {
        if val[i] != 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    }

    return Ok(val[0])
}

pub fn CopyOutNodemask(task: &Task, addr: u64, maxnode: u32, val: u64) -> Result<()> {
    // mm/mempolicy.c:copy_nodes_to_user() also uses maxnode-1 as the number of
    // bits.
    let bits = maxnode - 1;
    if bits as u64 > MemoryDef::PAGE_SIZE * 8 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    // Copy out the first unsigned long in the nodemask.
    *task.GetTypeMut(addr)? = val;

    // Zero out remaining unsigned longs in the nodemask.
    if bits > 64 {
        let mut remAddr = addr + 8;

        let remU64 = (bits - 65) / 64;
        for _i in 0.. remU64 as usize {
            *task.GetTypeMut(remAddr)? = 0;
            remAddr += 8;
        }
    }

    return Ok(())
}

// GetMempolicy implements the syscall get_mempolicy(2).
pub fn SysGetMempolicy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let mode = args.arg0 as u64;
    let nodemask = args.arg1 as u64;
    let maxnode = args.arg2 as u32;
    let addr = args.arg3 as u64;
    let flags = args.arg4 as i32;

    if flags & !(MPOL_F_NODE | MPOL_F_ADDR | MPOL_F_MEMS_ALLOWED) != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let nodeFlag = flags & MPOL_F_NODE != 0;
    let addrFlag = flags & MPOL_F_ADDR != 0;
    let memsAllowed = flags & MPOL_F_MEMS_ALLOWED != 0;

    // "EINVAL: The value specified by maxnode is less than the number of node
    // IDs supported by the system." - get_mempolicy(2)
    if nodemask != 0 && maxnode < MAX_NODES as u32 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    // "If flags specifies MPOL_F_MEMS_ALLOWED [...], the mode argument is
    // ignored and the set of nodes (memories) that the thread is allowed to
    // specify in subsequent calls to mbind(2) or set_mempolicy(2) (in the
    // absence of any mode flags) is returned in nodemask."
    if memsAllowed {
        // "It is not permitted to combine MPOL_F_MEMS_ALLOWED with either
        // MPOL_F_ADDR or MPOL_F_NODE."
        if nodeFlag || addrFlag {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        CopyOutNodemask(task, nodemask, maxnode, ALLOW_NODE_MASK)?;
        return Ok(0)
    }

    // "If flags specifies MPOL_F_ADDR, then information is returned about the
    // policy governing the memory address given in addr. ... If the mode
    // argument is not NULL, then get_mempolicy() will store the policy mode
    // and any optional mode flags of the requested NUMA policy in the location
    // pointed to by this argument. If nodemask is not NULL, then the nodemask
    // associated with the policy will be stored in the location pointed to by
    // this argument."
    let t = task.Thread();
    if addrFlag {
        let (mut policy, nodemaskVal) = t.MemoryManager().NumaPolicy(addr)?;

        if nodeFlag {
            // "If flags specifies both MPOL_F_NODE and MPOL_F_ADDR,
            // get_mempolicy() will return the node ID of the node on which the
            // address addr is allocated into the location pointed to by mode.
            // If no page has yet been allocated for the specified address,
            // get_mempolicy() will allocate a page as if the thread had
            // performed a read (load) access to that address, and return the
            // ID of the node where that page was allocated."
            *task.GetTypeMut(addr)? = 0 as u8;
            policy = MPOL_DEFAULT; // maxNodes == 1
        }

        if mode != 0 {
            let mode = task.GetTypeMut(mode)?;
            *mode = policy;
        }

        if nodemask != 0 {
            CopyOutNodemask(task, nodemask, maxnode, nodemaskVal)?;
        }

        return Ok(0)
    }

    // "EINVAL: ... flags specified MPOL_F_ADDR and addr is NULL, or flags did
    // not specify MPOL_F_ADDR and addr is not NULL." This is partially
    // inaccurate: if flags specifies MPOL_F_ADDR,
    // mm/mempolicy.c:do_get_mempolicy() doesn't special-case NULL; it will
    // just (usually) fail to find a VMA at address 0 and return EFAULT.
    if addr != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let (mut policy, nodemaskVal) = t.NumaPolicy();
    if nodeFlag {
        if policy & !MPOL_MODE_FLAGS != MPOL_INTERLEAVE {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        policy = MPOL_DEFAULT // maxNodes == 1
    }

    if mode != 0 {
        let mode = task.GetTypeMut(mode)?;
        *mode = policy;
    }

    if nodemask != 0 {
        CopyOutNodemask(task, nodemask, maxnode, nodemaskVal)?
    }

    return Ok(0)
}

pub fn CopyInMempolicyNodemask(task: &Task, modeWithFlags: i32, nodemask: u64, maxnode: u32) -> Result<(i32, u64)> {
    let flags = modeWithFlags & MPOL_MODE_FLAGS;
    let mut mode = modeWithFlags & !MPOL_MODE_FLAGS;

    if flags == MPOL_MODE_FLAGS {
        // Can't specify both mode flags simultaneously.
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if mode < 0 || mode >= MPOL_MAX {
        // Must specify a valid mode.
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let mut nodemaskVal : u64 = 0;
    if nodemask != 0 {
        nodemaskVal = CopyInNodemask(task, nodemask, maxnode)?;
    }

    match mode {
        MPOL_DEFAULT => {
            // "nodemask must be specified as NULL." - set_mempolicy(2). This is inaccurate;
            // Linux allows a nodemask to be specified, as long as it is empty.
            if nodemaskVal != 0 {
                return Err(Error::SysError(SysErr::EINVAL))
            }
        }
        MPOL_BIND | MPOL_INTERLEAVE => {
            // These require a non-empty nodemask.
            if nodemaskVal == 0 {
                return Err(Error::SysError(SysErr::EINVAL))
            }
        }
        MPOL_PREFERRED => {
            // This permits an empty nodemask, as long as no flags are set.
            if nodemaskVal == 0 && flags != 0 {
                return Err(Error::SysError(SysErr::EINVAL))
            }
        }
        MPOL_LOCAL => {
            // This requires an empty nodemask and no flags set ...
            if nodemaskVal != 0 && flags != 0 {
                return Err(Error::SysError(SysErr::EINVAL))
            }
            // ... and is implemented as MPOL_PREFERRED.
            mode = MPOL_PREFERRED
        }
        _ => {
            panic!("SysSetMempolicy unknow mode {}", mode);
        }
    }

    return Ok((mode | flags, nodemaskVal))
}

// SetMempolicy implements the syscall get_mempolicy(2).
pub fn SysSetMempolicy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let modeWithFlags = args.arg0 as i32;
    let nodemask = args.arg1 as u64;
    let maxnode = args.arg2 as u32;

    let (modeWithFlags, nodemaskVal) = CopyInMempolicyNodemask(task, modeWithFlags, nodemask, maxnode)?;

    task.Thread().SetNumaPolicy(modeWithFlags, nodemaskVal);
    return Ok(0)
}

// Mbind implements the syscall mbind(2).
pub fn SysMbind(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as u64;
    let mode = args.arg2 as i32;
    let nodemask = args.arg3 as u64;
    let maxnode = args.arg4 as u32;
    let flags = args.arg5 as i32;

    if flags & MPOL_MF_VALID != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let t = task.Thread();
    // "If MPOL_MF_MOVE_ALL is passed in flags ... [the] calling thread must be
    // privileged (CAP_SYS_NICE) to use this flag." - mbind(2)
    if flags & MPOL_MF_MOVE_ALL != 0 && !t.HasCapability(Capability::CAP_SYS_NICE) {
        return Err(Error::SysError(SysErr::EPERM))
    }

    let (mode, nodemaskVal) = CopyInMempolicyNodemask(task, mode, nodemask, maxnode)?;

    // Since we claim to have only a single node, all flags can be ignored
    // (since all pages must already be on that single node).
    t.MemoryManager().SetNumaPolicy(addr, length, mode, nodemaskVal)?;
    return Ok(0)
}
