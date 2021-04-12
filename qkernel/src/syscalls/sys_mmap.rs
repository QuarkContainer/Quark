// Copyright (c) 2021 Quark Container Authors
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
use super::super::fs::host::hostinodeop::*;

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
    } else if shared {
        let memfdIops = HostInodeOp::NewMemfdIops(len as i64)?;
        opts.Mappable = Some(memfdIops);
    }

    match task.mm.MMap(task, &mut opts) {
        Ok(addr) => {
            Ok(addr as i64)
        },
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
    //return Err(Error::SysError(SysErr::ENODATA));

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
