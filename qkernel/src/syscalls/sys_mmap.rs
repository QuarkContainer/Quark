// Copyright (c) 2021 QuarkSoft LLC
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

use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::task::*;
use super::super::qlib::common::*;
use super::super::memmgr::*;
use super::super::memmgr::syscalls::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::addr::*;
use super::super::syscalls::syscalls::*;

pub fn SysMmap(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let len = args.arg1 as u64;
    let prot = args.arg2 as u64;
    let flags = args.arg3 as u64;
    let fd = args.arg4 as i32;
    let offset = args.arg5 as u64;

    let fixed = flags & MmapFlags::MAP_FIXED != 0;
    let private = flags & MmapFlags::MAP_PRIVATE != 0;
    let shared = flags & MmapFlags::MAP_SHARED != 0;
    let anon = flags & MmapFlags::MAP_ANONYMOUS != 0;
    let map32bit = flags & MmapFlags::MAP_32BIT != 0;

    // Require exactly one of MAP_PRIVATE and MAP_SHARED.
    if private == shared {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let mut opts = MMapOpts {
        Length: len,
        Addr: addr,
        Offset: offset,
        Fixed: fixed,
        Unmap: fixed,
        Map32Bit: map32bit,
        Private: private,
        VDSO: false,
        Perms: AccessType(prot),
        MaxPerms: AccessType::AnyAccess(),
        GrowsDown: flags & MmapFlags::MAP_GROWSDOWN != 0,
        Precommit: flags & MmapFlags::MAP_POPULATE != 0,
        MLockMode: 0,
        Kernel: false,
        Mapping: None,
        Mappable: None,
        Hint: "".to_string(),
    };

    if flags & MmapFlags::MAP_LOCKED != 0 {
        opts.MLockMode = MLOCK_EAGER;
    }

    if !anon {
        let file = task.GetFile(fd)?;
        let flags = file.Flags();

        // mmap unconditionally requires that the FD is readable.
        if !flags.Read {
            return Err(Error::SysError(SysErr::EACCES))
        }

        // MAP_SHARED requires that the FD be writable for PROT_WRITE.
        if shared && !flags.Write {
            opts.MaxPerms.ClearWrite();
        }

        opts.Mapping = Some(Arc::new(file.clone()));

        match file.Mappable() {
            Err(Error::ErrDevZeroMap) => {
                opts.Mappable = None;
                opts.Hint = "/dev/zero".to_string();
            }
            Err(e) => return Err(e),
            Ok(m) => opts.Mappable = Some(m)
        }
    }

    match task.mm.MMap(task, &mut opts) {
        Ok(addr) => Ok(addr as i64),
        Err(e) => Err(e),
    }
}

pub fn SysMprotect(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let len = args.arg1 as u64;
    let prot = args.arg2 as u64;

    let accessType = AccessType(prot);
    let growDown = prot & MmapProt::PROT_GROWSDOWN != 0;

    match task.mm.MProtect(addr, len, &accessType, growDown) {
        Err(e) => return Err(e),
        _ => return Ok(0),
    }
}

pub fn SysUnmap(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let len = args.arg1 as u64;

    match task.mm.MUnmap(task, addr, len) {
        Err(e) => return Err(e),
        _ => return Ok(0),
    }
}

pub fn SysBrk(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    match task.mm.Brk(task, addr) {
        Ok(addr) => Ok(addr as i64),
        Err(e) => Err(e),
    }
}

// Madvise implements linux syscall madvise(2).
pub fn SysMadvise(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as u64;
    let adv = args.arg2 as i32;

    // "The Linux implementation requires that the address addr be
    // page-aligned, and allows length to be zero." - madvise(2)
    if Addr(addr).RoundDown()?.0 != addr {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if length == 0 {
        return Ok(0)
    }

    //let length = Addr(length).RoundUp()?.0;

    match adv {
        MAdviseOp::MADV_DONTNEED => {
            //todo: DeCommit the memory
            return Ok(0)
        }
        MAdviseOp::MADV_HUGEPAGE | MAdviseOp::MADV_NOHUGEPAGE => {
            return Ok(0)
        }
        MAdviseOp::MADV_MERGEABLE | MAdviseOp::MADV_UNMERGEABLE => {
            return Ok(0)
        }
        MAdviseOp::MADV_DONTDUMP | MAdviseOp::MADV_DODUMP => {
            return Ok(0)
        }
        MAdviseOp::MADV_NORMAL | MAdviseOp::MADV_RANDOM | MAdviseOp::MADV_SEQUENTIAL | MAdviseOp::MADV_WILLNEED => {
            return Ok(0)
        }
        MAdviseOp::MADV_REMOVE | MAdviseOp::MADV_DOFORK | MAdviseOp::MADV_DONTFORK => {
            return Err(Error::SysError(SysErr::ENOSYS));
        }
        MAdviseOp::MADV_HWPOISON => {
            // Only privileged processes are allowed to poison pages.
            return Err(Error::SysError(SysErr::EPERM));
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }
}

// Mremap implements linux syscall mremap(2).
pub fn SysMremap(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldAddr = args.arg0 as u64;
    let oldSize = args.arg1 as u64;
    let newSize = args.arg2 as u64;
    let flags = args.arg3 as i32;
    let newAddr = args.arg4 as u64;

    if flags & !(MRemapType::MREMAP_MAYMOVE | MRemapType::MREMAP_FIXED) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mayMove = flags & MRemapType::MREMAP_MAYMOVE != 0;
    let fixed = flags & MRemapType::MREMAP_FIXED != 0;

    let moveMode: MRemapMoveMode;
    if !mayMove && !fixed {
        moveMode = MREMAP_NO_MOVE
    } else if mayMove && !fixed {
        moveMode = MREMAP_MAY_MOVE
    } else if mayMove && fixed {
        moveMode = MREMAP_MUST_MOVE
    } else { // !mayMove && fixed
        // "If MREMAP_FIXED is specified, then MREMAP_MAYMOVE must also be
		// specified." - mremap(2)
        return Err(Error::SysError(SysErr::EINVAL));
    }

    match task.mm.MRemap(task, oldAddr, oldSize, newSize, &MRemapOpts{
        Move: moveMode,
        NewAddr: newAddr
    })  {
        Ok(addr) => Ok(addr as i64),
        Err(e) => Err(e),
    }
}

// GetMempolicy implements the syscall get_mempolicy(2).
pub fn SysGetMempolicy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let mode = args.arg0 as u64;
    let nodemask = args.arg1 as u64;
    let maxnode = args.arg2 as u32;
    let addr = args.arg3 as u64;
    let flags = args.arg4 as u32;

    let memsAllowed = flags & MPOL_F_MEMS_ALLOWED != 0;
    let nodeFlag = flags & MPOL_F_NODE != 0;
    let addrFlag = flags & MPOL_F_ADDR != 0;

    if nodemask != 0 && maxnode < 1 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    // 'addr' provided iff 'addrFlag' set.
    if addrFlag == (addr==0) {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let t = task.Thread();

    // Default policy for the thread.
    if flags == 0 {
        let (policy, nodemaskVal) = t.NumaPolicy();
        if mode != 0 {
            let mode = task.GetTypeMut(mode)?;
            *mode = policy;
        }

        if nodemask != 0 {
            let nodemask = task.GetTypeMut(nodemask)?;
            *nodemask = nodemaskVal;
        }

        return Ok(0)
    }

    // Report all nodes available to caller.
    if memsAllowed {
        // MPOL_F_NODE and MPOL_F_ADDR not allowed with MPOL_F_MEMS_ALLOWED.
        if nodeFlag || addrFlag {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        // Report a single numa node.
        if nodemask != 0 {
            let nodemask = task.GetTypeMut(nodemask)?;
            *nodemask = 0x1 as u32;
        }

        return Ok(0)
    }

    if addrFlag {
        if nodeFlag {
            // Return the id for the node where 'addr' resides, via 'mode'.
            //
            // The real get_mempolicy(2) allocates the page referenced by 'addr'
            // by simulating a read, if it is unallocated before the call. It
            // then returns the node the page is allocated on through the mode
            // pointer.
            let byte : &mut u8= task.GetTypeMut(addr)?;
            *byte = 1;

            if mode != 0 {
                let mode = task.GetTypeMut(mode)?;
                *mode = 0;
            }
        } else {
            let (storedPolicy, _) = t.NumaPolicy();
            if mode != 0 {
                let mode = task.GetTypeMut(mode)?;
                *mode = storedPolicy;
            }
        }

        let (storedPolicy, _) = t.NumaPolicy();
        if nodeFlag && (storedPolicy as u32 & !MPOL_MODE_FLAGS == MPOL_INTERLEAVE) {
            // Policy for current thread is to interleave memory between
            // nodes. Return the next node we'll allocate on. Since we only have a
            // single node, this is always node 0.
            let mode = task.GetTypeMut(mode)?;
            *mode = 0;
        }

        return Ok(0)
    }

    return Err(Error::SysError(SysErr::EINVAL))
}

fn AllowNodesMask() -> u32 {
    let maxNodes = 1;
    return !((1 << maxNodes) - 1)
}

// SetMempolicy implements the syscall get_mempolicy(2).
pub fn SysSetMempolicy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let modeWithFlags = args.arg0 as i32;
    let nodemask = args.arg1 as u64;
    let maxnode = args.arg2 as u32;

    if nodemask != 0 && maxnode < 1 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if modeWithFlags & MPOL_MODE_FLAGS as i32 == MPOL_MODE_FLAGS as i32 {
        // Can't specify multiple modes simultaneously.
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let mode = modeWithFlags & !MPOL_MODE_FLAGS as i32;
    if mode < 0 || mode > MPOL_MAX as i32 {
        // Must specify a valid mode.
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let mut nodemaskVal : u32 = 0;
    // Nodemask may be empty for some policy modes.
    if nodemask != 0 && maxnode > 0 {
        nodemaskVal = *task.GetType(nodemask)?;
    }

    if (mode == MPOL_INTERLEAVE as i32 || mode == MPOL_BIND as i32) && nodemaskVal == 0 {
        // Mode requires a non-empty nodemask, but got an empty nodemask.
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if nodemaskVal & AllowNodesMask() != 0 {
        // Invalid node specified.
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let t = task.Thread();
    t.SetNumaPolicy(modeWithFlags as i32, nodemaskVal);

    return Ok(0)
}

// Mbind implements the syscall mbind(2).
pub fn SysMbind(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    if !task.Thread().HasCapability(Capability::CAP_SYS_NICE) {
        return Err(Error::SysError(SysErr::EPERM))
    }

    return Err(Error::SysError(SysErr::ENOSYS))
}
