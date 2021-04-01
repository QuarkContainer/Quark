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

use alloc::vec::Vec;

use super::super::qlib::auth::id::*;
use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;

const MAX_NGROUPS: i32 = 65536;

// Getuid implements the Linux syscall getuid.
pub fn SysGetuid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let c = task.Thread().Credentials();
    let userns = c.lock().UserNamespace.clone();
    let ruid = c.lock().RealKUID.In(&userns).OrOverflow();
    return Ok(ruid.0 as i64)
}

// Geteuid implements the Linux syscall geteuid.
pub fn SysGeteuid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let c = task.Thread().Credentials();
    let userns = c.lock().UserNamespace.clone();
    let euid = c.lock().EffectiveKUID.In(&userns).OrOverflow();
    return Ok(euid.0 as i64)
}

// Getresuid implements the Linux syscall getresuid.
pub fn SysGetresuid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let ruidAddr = args.arg0 as u64;
    let euidAddr = args.arg1 as u64;
    let suidAddr = args.arg2 as u64;

    let c = task.Thread().Credentials();
    let userns = c.lock().UserNamespace.clone();
    let ruid = c.lock().RealKUID.In(&userns).OrOverflow();
    let euid = c.lock().EffectiveKUID.In(&userns).OrOverflow();
    let suid = c.lock().SavedKUID.In(&userns).OrOverflow();

    task.CopyOutObj(&ruid, ruidAddr)?;
    task.CopyOutObj(&euid, euidAddr)?;
    task.CopyOutObj(&suid, suidAddr)?;
    return Ok(0)
}

// Getgid implements the Linux syscall getgid.
pub fn SysGetgid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let c = task.Thread().Credentials();
    let userns = c.lock().UserNamespace.clone();
    let rgid = c.lock().RealKGID.In(&userns).OrOverflow();
    return Ok(rgid.0 as i64)
}

// Getegid implements the Linux syscall getegid.
pub fn SysGetegid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let c = task.Thread().Credentials();
    let userns = c.lock().UserNamespace.clone();
    let egid = c.lock().EffectiveKGID.In(&userns).OrOverflow();
    return Ok(egid.0 as i64)
}

// Getresgid implements the Linux syscall getresgid.
pub fn SysGetresgid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let rgidAddr = args.arg0 as u64;
    let egidAddr = args.arg1 as u64;
    let sgidAddr = args.arg2 as u64;

    let c = task.Thread().Credentials();
    let userns = c.lock().UserNamespace.clone();
    let rgid = c.lock().RealKGID.In(&userns).OrOverflow();
    let egid = c.lock().EffectiveKGID.In(&userns).OrOverflow();
    let sgid = c.lock().SavedKGID.In(&userns).OrOverflow();

    task.CopyOutObj(&rgid, rgidAddr)?;
    task.CopyOutObj(&egid, egidAddr)?;
    task.CopyOutObj(&sgid, sgidAddr)?;
    return Ok(0)
}

// Setuid implements the Linux syscall setuid.
pub fn SysSetuid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let uid = args.arg0 as u32;
    let thread = task.Thread();
    thread.SetUID(UID(uid))?;
    task.creds = thread.Creds();
    return Ok(0);
}

// Setuid implements the Linux syscall Setreuid.
pub fn SysSetreuid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let ruid = args.arg0 as u32;
    let euid = args.arg1 as u32;
    task.Thread().SetREUID(UID(ruid), UID(euid))?;
    task.creds = task.Thread().Creds();
    return Ok(0);
}

// Setuid implements the Linux syscall Setresuid.
pub fn SysSetresuid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let ruid = args.arg0 as u32;
    let euid = args.arg1 as u32;
    let suid = args.arg2 as u32;
    task.Thread().SetRESUID(UID(ruid), UID(euid), UID(suid))?;
    task.creds = task.Thread().Creds();
    return Ok(0);
}

// Setuid implements the Linux syscall Setgid.
pub fn SysSetgid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let gid = args.arg0 as u32;
    task.Thread().SetGID(GID(gid))?;
    task.creds = task.Thread().Creds();
    return Ok(0);
}

// Setuid implements the Linux syscall Setregid.
pub fn SysSetregid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let rgid = args.arg0 as u32;
    let egid = args.arg1 as u32;
    task.Thread().SetREGID(GID(rgid), GID(egid))?;
    task.creds = task.Thread().Creds();
    return Ok(0);
}

// Setuid implements the Linux syscall Setresgid.
pub fn SysSetresgid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let rgid = args.arg0 as u32;
    let egid = args.arg1 as u32;
    let sgid = args.arg2 as u32;
    task.Thread().SetRESGID(GID(rgid), GID(egid), GID(sgid))?;
    task.creds = task.Thread().Creds();
    return Ok(0);
}

// Getgroups implements the Linux syscall getgroups.
pub fn SysGetgroups(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let size = args.arg0 as i32;
    let addr = args.arg1 as u64;

    if size < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let creds = task.Thread().Credentials();
    let kgidslen = creds.lock().ExtraKGIDs.len();

    // "If size is zero, list is not modified, but the total number of
    // supplementary group IDs for the process is returned." - getgroups(2)
    if size == 0 {
        return Ok(kgidslen as i64);
    }

    if size < kgidslen as i32 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut gids = Vec::new();
    let userns = task.Thread().UserNamespace();
    for kgid in &creds.lock().ExtraKGIDs {
        gids.push(kgid.In(&userns).OrOverflow());
    }

    task.CopyOutSlice(&gids[..], addr, gids.len())?;

    return Ok(kgidslen as i64);
}

// Setgroups implements the Linux syscall setgroups.
pub fn SysSetgroups(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let size = args.arg0 as i32;
    let addr = args.arg1 as u64;

    if size < 0 || size > MAX_NGROUPS {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if size == 0 {
        let zero: [GID; 0] = [GID(0); 0];
        task.Thread().SetExtraGIDs(&zero)?;
        return Ok(0)
    }

    let gids: Vec<GID> = task.CopyIn(addr, size as usize)?;
    task.Thread().SetExtraGIDs(&gids[..])?;
    return Ok(0)
}