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

use alloc::string::String;

use super::super::task::*;
use super::super::qlib::common::*;
use super::super::version::*;
use super::super::syscalls::syscalls::*;

// UTSLen is the maximum length of strings contained in fields of
// UtsName.
pub const UTS_LEN : usize = 64;

#[repr(C)]
pub struct UtsName {
    pub Sysname    : [u8; UTS_LEN + 1],
    pub Nodename   : [u8; UTS_LEN + 1],
    pub Release    : [u8; UTS_LEN + 1],
    pub Version    : [u8; UTS_LEN + 1],
    pub Machine    : [u8; UTS_LEN + 1],
    pub Domainname : [u8; UTS_LEN + 1],
}

impl Default for UtsName {
    fn default() -> Self {
        return Self {
            Sysname     : [0; UTS_LEN + 1],
            Nodename    : [0; UTS_LEN + 1],
            Release     : [0; UTS_LEN + 1],
            Version     : [0; UTS_LEN + 1],
            Machine     : [0; UTS_LEN + 1],
            Domainname  : [0; UTS_LEN + 1],
        }
    }
}

impl UtsName {
    pub fn ToString(&self) -> String {
        return format!("{{Sysname: {}, Nodename: {}, Release: {}, Version: {}, Machine: {}, Domainname: {}}}",
                       UtsNameString(&self.Sysname),
                       UtsNameString(&self.Nodename),
                       UtsNameString(&self.Release),
                       UtsNameString(&self.Version),
                       UtsNameString(&self.Machine),
                       UtsNameString(&self.Domainname)
        )
    }
}

// utsNameString converts a UtsName entry to a string without NULs.
pub fn UtsNameString(s: &[u8; UTS_LEN + 1]) -> String {
    // The NUL bytes will remain even in a cast to string. We must
    // explicitly strip them.
    let mut i = UTS_LEN;
    while s[i] == 0 {
        i -= 1;
    }

    let s = &s[..i+1];
    return String::from_utf8(s.to_vec()).unwrap();
}

pub fn SysUname(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let va = args.arg0 as u64;

    let version = &VERSION;
    let uts = task.Thread().UTSNamespace();

    let mut u = UtsName::default();

    u.Sysname[0..version.Sysname.len()].clone_from_slice(version.Sysname.as_bytes());
    u.Nodename[0..uts.HostName().len()].clone_from_slice(uts.HostName().as_bytes());
    u.Release[0..version.Release.len()].clone_from_slice(version.Release.as_bytes());
    u.Version[0..version.Version.len()].clone_from_slice(version.Version.as_bytes());
    u.Machine[0.."x86_64".len()].clone_from_slice("x86_64".as_bytes());
    u.Domainname[0..uts.DomainName().len()].clone_from_slice(uts.DomainName().as_bytes());

    let va : &mut UtsName = task.GetTypeMut(va)?;
    *va = u;

    /*info!("Sysname is {}", UtsNameString(&va.Sysname));
    info!("Nodename is {}", UtsNameString(&va.Nodename));
    info!("Release is {}", UtsNameString(&va.Release));
    info!("Version is {}", UtsNameString(&va.Version));
    info!("Machine is {}", UtsNameString(&va.Machine));
    info!("Domainname is {}", UtsNameString(&va.Domainname));*/

    return Ok(0);
}
