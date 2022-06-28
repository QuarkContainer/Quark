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

use super::super::qlib::auth::cap_set::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::super::threadmgr::thread::*;

pub fn LookupCaps(task: &Task, tid: ThreadID) -> Result<(CapSet, CapSet, CapSet)> {
    if tid < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let thread = if tid > 0 {
        match task.Thread().PIDNamespace().TaskWithID(tid) {
            None => return Err(Error::SysError(SysErr::ESRCH)),
            Some(t) => t,
        }
    } else {
        task.Thread()
    };

    let creds = thread.Credentials();
    let permitted = creds.lock().PermittedCaps;
    let inheritable = creds.lock().InheritableCaps;
    let effective = creds.lock().EffectiveCaps;

    return Ok((permitted, inheritable, effective));
}

// Capget implements Linux syscall capget.
pub fn SysCapget(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let hdrAddr = args.arg0 as u64;
    let dataAddr = args.arg1 as u64;

    let mut hdr = task.CopyInObj::<CapUserHeader>(hdrAddr)?;

    // hdr.Pid doesn't need to be valid if this capget() is a "version probe"
    // (hdr.Version is unrecognized and dataAddr is null), so we can't do the
    // lookup yet.
    match hdr.Version {
        LINUX_CAPABILITY_VERSION_1 => {
            if dataAddr == 0 {
                return Ok(0);
            }

            let (p, i, e) = LookupCaps(task, hdr.Pid)?;

            let data = CapUserData {
                Effective: e.0 as u32,
                Permitted: p.0 as u32,
                Inheritable: i.0 as u32,
            };

            //*task.GetTypeMut(dataAddr)? = data;
            task.CopyOutObj(&data, dataAddr)?;

            return Ok(0);
        }
        LINUX_CAPABILITY_VERSION_2 | LINUX_CAPABILITY_VERSION_3 => {
            if dataAddr == 0 {
                return Ok(0);
            }

            let (p, i, e) = LookupCaps(task, hdr.Pid)?;
            let data = [
                CapUserData {
                    Effective: e.0 as u32,
                    Permitted: p.0 as u32,
                    Inheritable: i.0 as u32,
                },
                CapUserData {
                    Effective: (e.0 >> 32) as u32,
                    Permitted: (p.0 >> 32) as u32,
                    Inheritable: (i.0 >> 32) as u32,
                },
            ];

            //*task.GetTypeMut(dataAddr)? = data;
            task.CopyOutObj(&data, dataAddr)?;
            return Ok(0);
        }
        _ => {
            hdr.Version = HIGHEST_CAPABILITY_VERSION;
            //*task.GetTypeMut(hdrAddr)? = hdr;
            task.CopyOutObj(&hdr, hdrAddr)?;

            if dataAddr != 0 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            return Ok(0);
        }
    }
}

// Capset implements Linux syscall capset.
pub fn SysCapSet(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let hdrAddr = args.arg0 as u64;
    let dataAddr = args.arg1 as u64;

    let mut hdr = task.CopyInObj::<CapUserHeader>(hdrAddr)?;

    match hdr.Version {
        LINUX_CAPABILITY_VERSION_1 => {
            let tid = hdr.Pid;
            if tid != 0 && tid != task.Thread().ThreadID() {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let data = task.CopyInObj::<CapUserData>(dataAddr)?;

            let p = CapSet(data.Permitted as u64 & ALL_CAP.0);
            let i = CapSet(data.Inheritable as u64 & ALL_CAP.0);
            let e = CapSet(data.Effective as u64 & ALL_CAP.0);
            task.Thread().SetCapabilitySets(p, i, e)?;
            return Ok(0);
        }
        LINUX_CAPABILITY_VERSION_2 | LINUX_CAPABILITY_VERSION_3 => {
            let tid = hdr.Pid;
            if tid != 0 && tid != task.Thread().ThreadID() {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let data: [CapUserData; 2] = task.CopyInObj(dataAddr)?;
            let p =
                CapSet((data[0].Permitted as u64 | (data[1].Permitted as u64) << 32) & ALL_CAP.0);
            let i = CapSet(
                (data[0].Inheritable as u64 | (data[1].Inheritable as u64) << 32) & ALL_CAP.0,
            );
            let e =
                CapSet((data[0].Effective as u64 | (data[1].Effective as u64) << 32) & ALL_CAP.0);
            task.Thread().SetCapabilitySets(p, i, e)?;
            return Ok(0);
        }
        _ => {
            hdr.Version = HIGHEST_CAPABILITY_VERSION;
            match task.CopyOutObj(&hdr, hdrAddr) {
                Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
                Ok(()) => (),
            };

            return Ok(0);
        }
    }
}
