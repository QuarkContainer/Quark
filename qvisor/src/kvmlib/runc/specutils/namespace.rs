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
use std::fs::File;
use std::os::unix::io::AsRawFd;
use capabilities;

use super::super::super::qlib::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::path::*;
use super::super::super::vmspace::syscall::*;
use super::super::oci::*;
use super::super::runtime::sandbox_process::*;

// nsCloneFlag returns the clone flag that can be used to set a namespace of
// the given type.
pub fn nsCloneFlag(nst: LinuxNamespaceType) -> i32 {
    return nst as i32
}

// nsPath returns the path of the namespace for the current process and the
// given namespace.
pub fn NsPath(nst: LinuxNamespaceType) -> String {
    let base = "/proc/self/ns";

    match nst {
        LinuxNamespaceType::cgroup => Join(base, "cgroup"),
        LinuxNamespaceType::ipc => Join(base, "ipc"),
        LinuxNamespaceType::mount => Join(base, "mount"),
        LinuxNamespaceType::network => Join(base, "network"),
        LinuxNamespaceType::pid => Join(base, "pid"),
        LinuxNamespaceType::user => Join(base, "user"),
        LinuxNamespaceType::uts => Join(base, "uts"),
    }
}

// GetNS returns true and the namespace with the given type from the slice of
// namespaces in the spec.  It returns false if the slice does not contain a
// namespace with the type.
pub fn GetNS(nst: LinuxNamespaceType, s: &Spec) -> Option<LinuxNamespace> {
    if s.linux.is_none() {
        return None;
    }

    for ns in &s.linux.as_ref().unwrap().namespaces {
        if ns.typ == nst {
            return Some(LinuxNamespace {
                typ: ns.typ,
                path: ns.path.to_string(),
            })
        }
    }

    return None;
}

// FilterNS returns a slice of namespaces from the spec with types that match
// those in the `filter` slice.
pub fn FilterNS(filter: &[LinuxNamespaceType], s: &Spec) -> Vec<LinuxNamespace> {
    let mut out = Vec::new();
    if s.linux.is_none() {
        return out;
    }

    for nst in filter {
        match GetNS(*nst, s) {
            None => (),
            Some(ns) => out.push(ns),
        }
    }

    return out
}

// setNS sets the namespace of the given type.  It must be called with
// OSThreadLocked.
pub fn SetNS(fd: i32, nsType: u32) -> Result<()> {
    let nr = SysCallID::sys_setns as usize;

    let err = unsafe {
        syscall2(nr, fd as usize, nsType as usize) as i32
    };

    if err == 0 {
        return Ok(())
    }

    return Err(Error::SysError(-err))
}

// ApplyNS applies the namespace on the current thread and returns a function
// that will restore the namespace to the original value.
//
// Preconditions: Must be called with os thread locked.
pub fn ApplyNS(ns: &LinuxNamespace) -> Result<NSRestore> {
    info!("Applying namespace {:?} at path {:?}", ns.typ, ns.path);

    let newNs = File::open(&ns.path).map_err(|e|Error::IOError(format!("error opening {:?}: {:?}", ns.path, e)))?;

    // Store current namespace to restore back.
    let curPath = NsPath(ns.typ);
    let oldNs = File::open(&curPath).map_err(|e|Error::IOError(format!("error opening {:?}: {:?}", ns.path, e)))?;

    // Set namespace to the one requested and setup function to restore it back.
    let flag = nsCloneFlag(ns.typ);
    SetNS(newNs.as_raw_fd(), flag as u32).map_err(|e|Error::IOError(format!("error setting namespace of type {:?} and path {:?}: {:?}", ns.typ, ns.path, e)))?;

    return Ok(NSRestore {
        fd: oldNs.as_raw_fd(),
        flag: nsCloneFlag(ns.typ),
        typ: ns.typ,
    })
}

// StartInNS joins or creates the given namespaces and calls Process.Start before
// restoring the namespaces to the original values.
pub fn StartInNS(subProcess: &mut SandboxProcess, nss: &[LinuxNamespace]) -> Result<()> {
    let mut NSRestores = Vec::new();

    for ns in nss {
        if ns.path.as_str() == "" {
            subProcess.CloneFlags |= nsCloneFlag(ns.typ);
            continue;
        }

        // Join the given namespace, and restore the current namespace
        // before exiting.
        let restore = ApplyNS(ns)?;
        NSRestores.push(restore);
    }

    error!("should not reach here, need control sock");
    subProcess.Run(0);

    return Ok(())
}

// SetUIDGIDMappings sets the given uid/gid mappings from the spec on the process.
pub fn SetUIDGIDMappings(subProcess: &mut SandboxProcess, s: &Spec) {
    if s.linux.is_none() {
        return
    }

    let linux = s.linux.as_ref().unwrap();
    for idMap in &linux.uid_mappings {
        info!("Mapping host uid {:?} to container uid {:?} (size={:?})", idMap.host_id, idMap.container_id, idMap.size);
        subProcess.UidMappings.push(*idMap)
    }

    for idMap in &linux.gid_mappings {
        info!("Mapping host gid {:?} to container uid {:?} (size={:?})", idMap.host_id, idMap.container_id, idMap.size);
        subProcess.GidMappings.push(*idMap)
    }
}

// HasCapabilities returns true if the user has all capabilties in 'cs'.
pub fn HasCapabilities(cs: &[u32]) -> bool {
    let caps = match capabilities::Capabilities::from_current_proc() {
        Err(_e) => return false,
        Ok(c) => c,
    };

    for c in cs {
        if !(caps.check(capabilities::Capability::from(*c), capabilities::Flag::Effective)) {
            return false;
        }
    }

    return true;
}