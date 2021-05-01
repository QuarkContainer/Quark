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
use super::super::memmgr::mm::*;
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
        MLockMode: MLockMode::default(),
        Kernel: false,
        Mapping: None,
        Mappable: None,
        Hint: "".to_string(),
    };

    if flags & MmapFlags::MAP_LOCKED != 0 {
        opts.MLockMode = MLockMode::MlockEager;
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
pub fn SysMadvise(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as i64;
    let adv = args.arg2 as i32;

    // "The Linux implementation requires that the address addr be
    // page-aligned, and allows length to be zero." - madvise(2)
    if Addr(addr).RoundDown()?.0 != addr {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if length == 0 {
        return Ok(0)
    }

    if length < 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let length = match Addr(length as u64).RoundUp() {
        Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
        Ok(l) => l.0,
    };

    match adv {
        MAdviseOp::MADV_DONTNEED => {
            task.mm.MAdvise(task, addr, length, adv)?;
        }
        MAdviseOp::MADV_HUGEPAGE | MAdviseOp::MADV_NOHUGEPAGE => {
            task.mm.MAdvise(task, addr, length, adv)?;
        }
        MAdviseOp::MADV_MERGEABLE | MAdviseOp::MADV_UNMERGEABLE => {
            task.mm.MAdvise(task, addr, length, adv)?;
        }
        MAdviseOp::MADV_DONTDUMP | MAdviseOp::MADV_DODUMP => {
            // Core dumping isn't implemented, so do nothing
        }
        MAdviseOp::MADV_NORMAL | MAdviseOp::MADV_RANDOM | MAdviseOp::MADV_SEQUENTIAL | MAdviseOp::MADV_WILLNEED => {
            task.mm.MAdvise(task, addr, length, adv)?;
        }
        MAdviseOp::MADV_DONTFORK => {
            task.mm.SetDontFork(task, addr, length, true)?;
        }
        MAdviseOp::MADV_DOFORK => {
            task.mm.SetDontFork(task, addr, length, false)?;
        }
        MAdviseOp::MADV_REMOVE => {
            // These "suggestions" have application-visible side effects, so we
            // have to indicate that we don't support them.
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

    return Ok(0)
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

// Mlock implements linux syscall mlock(2).
pub fn SysMlock(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as i64;
    if length < 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    task.mm.Mlock(task, addr, length as u64, MLockMode::MlockEager)?;
    return Ok(0)
}

// Mlock2 implements linux syscall mlock2(2).
pub fn SysMlock2(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as i64;
    let flags = args.arg2 as u32;

    if length < 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if flags & !MLOCK_ONFAULT != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let mut mode = MLockMode::MlockEager;
    if flags & MLOCK_ONFAULT != 0 {
        mode = MLockMode::MlockLazy;
    }

    task.mm.Mlock(task, addr, length as u64, mode)?;
    return Ok(0)
}

// Munlock implements linux syscall munlock(2).
pub fn SysMunlock(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as i64;

    if length < 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    task.mm.Mlock(task, addr, length as u64, MLockMode::MlockNone)?;
    return Ok(0)
}

pub fn SysMlockall(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let flags = args.arg0 as u32;

    if flags & !(LibcConst::MCL_CURRENT | LibcConst::MCL_FUTURE | LibcConst::MCL_ONFAULT) as u32 != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let mut mode = MLockMode::MlockEager;
    if flags & MLOCK_ONFAULT != 0 {
        mode = MLockMode::MlockLazy;
    }

    task.mm.MlockAll(task, &MLockAllOpts {
        Current: flags & LibcConst::MCL_CURRENT as u32 != 0,
        Future: flags & LibcConst::MCL_FUTURE as u32 != 0,
        Mode: mode,
    })?;

    return Ok(0)
}

pub fn SysMunlockall(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    task.mm.MlockAll(task, &MLockAllOpts {
        Current: true,
        Future: true,
        Mode: MLockMode::MlockNone,
    })?;

    return Ok(0)
}

// Msync implements Linux syscall msync(2).
pub fn SysMsync(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as u64;
    let flags = args.arg2 as i32;

    // "The flags argument should specify exactly one of MS_ASYNC and MS_SYNC,
    // and may additionally include the MS_INVALIDATE bit. ... However, Linux
    // permits a call to msync() that specifies neither of these flags, with
    // semantics that are (currently) equivalent to specifying MS_ASYNC." -
    // msync(2)
    if flags & !(LibcConst::MS_ASYNC | LibcConst::MS_SYNC | LibcConst::MS_INVALIDATE) as i32 != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let sync = flags & LibcConst::MS_SYNC as i32 != 0;
    let async = flags & LibcConst::MS_ASYNC as i32 != 0;
    if sync && async {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    task.mm.MSync(task, addr, length, &MSyncOpts {
        Sync: sync,
        Invalidate: flags & LibcConst::MS_INVALIDATE as i32 != 0,
    })?;

    return Ok(0)
}

// Mincore implements the syscall mincore(2).
pub fn SysMincore(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let length = args.arg1 as i64;
    let vec = args.arg2 as u64;

    if addr != Addr(addr).RoundDown().unwrap().0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if length < 0 {
        return Err(Error::SysError(SysErr::ENOMEM))
    }

    let length = length as u64;

    // "The length argument need not be a multiple of the page size, but since
    // residency information is returned for whole pages, length is effectively
    // rounded up to the next multiple of the page size." - mincore(2)
    let la = match Addr(length).RoundUp() {
        Ok(l) => l.0,
        Err(_) => return Err(Error::SysError(SysErr::ENOMEM))
    };

    let range = match Addr(addr).ToRange(la) {
        Err(_) => return Err(Error::SysError(SysErr::ENOMEM)),
        Ok(r) => r
    };

    let output = task.mm.MinCore(task, &range);
    task.CopyOutSlice(&output, vec, output.len())?;
    return Ok(0)
}