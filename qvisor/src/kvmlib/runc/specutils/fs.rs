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

use lazy_static::lazy_static;
use alloc::collections::btree_map::BTreeMap;

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::path::*;
use super::super::oci::*;
use super::specutils::*;

#[derive(Copy, Clone)]
pub struct Mapping {
    pub set: bool,
    pub val: u32,
}

lazy_static! {
    // optionsMap maps mount propagation-related OCI filesystem options to mount(2)
    // syscall flags.
    static ref OPTIONS_MAP : BTreeMap<&'static str, Mapping> = [
        ("acl",           Mapping{set: true, val: LibcConst::MS_POSIXACL as u32}),
        ("async",         Mapping{set: false, val: LibcConst::MS_SYNCHRONOUS as u32}),
        ("atime",         Mapping{set: false, val: LibcConst::MS_NOATIME as u32}),
        ("bind",          Mapping{set: true, val: LibcConst::MS_BIND as u32}),
        ("defaults",      Mapping{set: true, val: 0 as u32}),
        ("dev",           Mapping{set: false, val: LibcConst::MS_NODEV as u32}),
        ("diratime",      Mapping{set: false, val: LibcConst::MS_NODIRATIME as u32}),
        ("dirsync",       Mapping{set: true, val: LibcConst::MS_DIRSYNC as u32}),
        ("exec",          Mapping{set: false, val: LibcConst::MS_NOEXEC as u32}),
        ("noexec",        Mapping{set: true, val: LibcConst::MS_NOEXEC as u32}),
        ("iversion",      Mapping{set: true, val: LibcConst::MS_I_VERSION as u32}),
        ("loud",          Mapping{set: false, val: LibcConst::MS_SILENT as u32}),
        ("mand",          Mapping{set: true, val: LibcConst::MS_MANDLOCK as u32}),
        ("noacl",         Mapping{set: false, val: LibcConst::MS_POSIXACL as u32}),
        ("noatime",       Mapping{set: true, val: LibcConst::MS_NOATIME as u32}),
        ("nodev",         Mapping{set: true, val: LibcConst::MS_NODEV as u32}),
        ("nodiratime",    Mapping{set: true, val: LibcConst::MS_NODIRATIME as u32}),
        ("noiversion",    Mapping{set: false, val: LibcConst::MS_I_VERSION as u32}),
        ("nomand",        Mapping{set: false, val: LibcConst::MS_MANDLOCK as u32}),
        ("norelatime",    Mapping{set: false, val: LibcConst::MS_RELATIME as u32}),
        ("nostrictatime", Mapping{set: false, val: LibcConst::MS_STRICTATIME as u32}),
        ("nosuid",        Mapping{set: true, val: LibcConst::MS_NOSUID as u32}),
        ("rbind",         Mapping{set: true, val: (LibcConst::MS_BIND | LibcConst::MS_REC) as u32}),
        ("relatime",      Mapping{set: true, val: LibcConst::MS_RELATIME as u32}),
        ("remount",       Mapping{set: true, val: LibcConst::MS_REMOUNT as u32}),
        ("ro",            Mapping{set: true, val: LibcConst::MS_RDONLY as u32}),
        ("rw",            Mapping{set: false, val: LibcConst::MS_RDONLY as u32}),
        ("silent",        Mapping{set: true, val: LibcConst::MS_SILENT as u32}),
        ("strictatime",   Mapping{set: true, val: LibcConst::MS_STRICTATIME as u32}),
        ("suid",          Mapping{set: false, val: LibcConst::MS_NOSUID as u32}),
        ("sync",          Mapping{set: true, val: LibcConst::MS_SYNCHRONOUS as u32}),
    ].iter().cloned().collect();

    // propOptionsMap is similar to optionsMap, but it lists propagation options
    // that cannot be used together with other flags.
    static ref PROP_OPTIONS_MAP : BTreeMap<&'static str, Mapping> = [
        ("private",       Mapping{set: true, val: LibcConst::MS_PRIVATE as u32}),
        ("rprivate",      Mapping{set: true, val: LibcConst::MS_PRIVATE as u32 | LibcConst::MS_REC as u32}),
        ("slave",         Mapping{set: true, val: LibcConst::MS_SLAVE as u32}),
        ("rslave",        Mapping{set: true, val: LibcConst::MS_SLAVE as u32 | LibcConst::MS_REC as u32}),
        ("unbindable",    Mapping{set: true, val: LibcConst::MS_UNBINDABLE as u32}),
        ("runbindable",   Mapping{set: true, val: LibcConst::MS_UNBINDABLE as u32 | LibcConst::MS_REC as u32}),
    ].iter().cloned().collect();

    // invalidOptions list options not allowed.
    //   - shared: sandbox must be isolated from the host. Propagating mount changes
    //     from the sandbox to the host breaks the isolation.
    static ref INVALID_OPTIONS : [&'static str; 2] = ["shared", "rshared"];
}

// OptionsToFlags converts mount options to syscall flags.
pub fn OptionsToFlags(opts: &[&str]) -> u32 {
    return optionsToFlags(opts, &OPTIONS_MAP)
}

// PropOptionsToFlags converts propagation mount options to syscall flags.
// Propagation options cannot be set other with other options and must be
// handled separatedly.
pub fn PropOptionsToFlags(opts: &[&str]) -> u32 {
    return optionsToFlags(opts, &PROP_OPTIONS_MAP)
}

fn optionsToFlags(options: &[&str], source: &BTreeMap<&str, Mapping>) -> u32 {
    let mut rv : u32 = 0;

    for opt in options {
        match source.get(opt) {
            None => (),
            Some(m) => {
                if m.set {
                    rv |= m.val;
                } else {
                    rv ^= m.val;
                }
            }
        }
    }

    return rv;
}

// ValidateMount validates that spec mounts are correct.
pub fn ValidateMount(mnt: &Mount) -> Result<()> {
    if !IsAbs(&mnt.destination) {
        return Err(Error::Common(format!("Mount.Destination must be an absolute path: {:?}", mnt)));
    }

    if mnt.typ.as_str() == "bind" {
        for o in &mnt.options {
            let o : &str = o;
            if ContainsStr(&*INVALID_OPTIONS, o) {
                return Err(Error::Common(format!("mount option {:?} is not supported: {:?}", o, mnt)));
            }

            let ok1 = OPTIONS_MAP.contains_key(o);
            let ok2 = PROP_OPTIONS_MAP.contains_key(o);

            if !ok1 && !ok2 {
                return Err(Error::Common(format!("unknown mount option {:?}", o)));
            }
        }
    }

    return Ok(())
}

// ValidateRootfsPropagation validates that rootfs propagation options are
// correct.
pub fn ValidateRootfsPropagation(opt: &str) -> Result<()> {
    let flags = PropOptionsToFlags(&[opt]);

    if flags & (LibcConst::MS_SLAVE as u32 | LibcConst::MS_PRIVATE as u32) == 0 {
        return Err(Error::Common(format!("root mount propagation option must specify private or slave: {:?}", opt)));
    }

    return Ok(())
}