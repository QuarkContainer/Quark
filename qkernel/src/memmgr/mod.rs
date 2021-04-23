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

pub mod vma;
pub mod mm;
pub mod arch;
pub mod pmamgr;
mod mapping;
pub mod memmap;
pub mod mapping_set;
pub mod pma;
pub mod syscalls;
pub mod metadata;
pub mod buf_allocator;
pub mod linked_list;

use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;

use super::fs::host::hostinodeop::*;
use super::qlib::common::*;
use super::qlib::addr::*;
use super::task::*;
use super::fs::file::*;
use self::mapping::*;

pub type MLockMode = i32;

// MLockNone specifies that a mapping has no memory locking behavior.
//
// This must be the zero value for MLockMode.
pub const MLOCK_NONE: MLockMode = 0;

// MLockEager specifies that a mapping is memory-locked, as by mlock() or
// similar. Pages in the mapping should be made, and kept, resident in
// physical memory as soon as possible.
//
// MLockEager is analogous to Linux's VM_LOCKED.
pub const MLOCK_EAGER: MLockMode = 1;

// MLockLazy specifies that a mapping is memory-locked, as by mlock() or
// similar. Pages in the mapping should be kept resident in physical memory
// once they have been made resident due to e.g. a page fault.
//
// As of this writing, MLockLazy does not cause memory-locking to be
// requested from the host; in fact, it has virtually no effect, except for
// interactions between mlocked pages and other syscalls.
//
// MLockLazy is analogous to Linux's VM_LOCKED | VM_LOCKONFAULT.
pub const MLOCK_LAZY: MLockMode = 2;

// MappingIdentity controls the lifetime of a Mappable, and provides
// information about the Mappable for /proc/[pid]/maps. It is distinct from
// Mappable because all Mappables that are coherent must compare equal to
// support the implementation of shared futexes, but different
// MappingIdentities may represent the same Mappable, in the same way that
// multiple fs.Files may represent the same fs.Inode. (This similarity is not
// coincidental; fs.File implements MappingIdentity, and some
// fs.InodeOperations implement Mappable.)
pub trait Mapping: Send + Sync {
    // MappedName returns the application-visible name shown in
    // /proc/[pid]/maps.
    fn MappedName(&self, task: &Task) -> String;

    // DeviceID returns the device number shown in /proc/[pid]/maps.
    fn DeviceID(&self) -> u64;

    // InodeID returns the inode number shown in /proc/[pid]/maps.
    fn InodeID(&self) -> u64;
}

pub struct MMapOpts {
    // Length is the length of the mapping.
    pub Length: u64,

    // Addr is the suggested address for the mapping.
    pub Addr: u64,

    pub Offset: u64,

    // Fixed specifies whether this is a fixed mapping (it must be located at
    // Addr).
    pub Fixed: bool,

    // Unmap specifies whether existing mappings in the range being mapped may
    // be replaced. If Unmap is true, Fixed must be true.
    pub Unmap: bool,

    // If Map32Bit is true, all addresses in the created mapping must fit in a
    // 32-bit integer. (Note that the "end address" of the mapping, i.e. the
    // address of the first byte *after* the mapping, need not fit in a 32-bit
    // integer.) Map32Bit is ignored if Fixed is true.
    pub Map32Bit: bool,

    // Perms is the set of permissions to the applied to this mapping.
    pub Perms: AccessType,

    // MaxPerms limits the set of permissions that may ever apply to this
    // mapping. If Mappable is not nil, all memmap.Translations returned by
    // Mappable.Translate must support all accesses in MaxPerms.
    //
    // Preconditions: MaxAccessType should be an effective AccessType, as
    // access cannot be limited beyond effective AccessTypes.
    pub MaxPerms: AccessType,

    // Private is true if writes to the mapping should be propagated to a copy
    // that is exclusive to the MemoryManager.
    pub Private: bool,

    pub VDSO: bool,

    // GrowsDown is true if the mapping should be automatically expanded
    // downward on guard page faults.
    pub GrowsDown: bool,

    // Precommit is true if the platform should eagerly commit resources to the
    // mapping (see platform.AddressSpace.MapFile).
    pub Precommit: bool,

    // MLockMode specifies the memory locking behavior of the mapping.
    pub MLockMode: MLockMode,

    //qkernel ocuppied kernel space
    pub Kernel: bool,

    pub Mapping: Option<Arc<Mapping>>,

    pub Mappable: Option<HostInodeOp>,

    pub Hint: String,
}

impl MMapOpts {
    pub fn NewAnonOptions(name: String) -> Result<Self> {
        return Ok(Self {
            Length: 0,
            Addr: 0,
            Offset: 0,
            Fixed: false,
            Unmap: false,
            Map32Bit: false,
            Perms: AccessType::Default(),
            MaxPerms: AccessType::Default(),
            Private: false,
            VDSO: false,
            GrowsDown: false,
            Precommit: false,
            MLockMode: MLockMode::default(),
            Kernel: false,
            Mapping: Some(NewAnonMapping(name.to_string())),
            Mappable: None,
            Hint: name.to_string(),
        })
    }

    pub fn NewFileOptions(file: &File) -> Result<Self> {
        return Ok(Self {
            Length: 0,
            Addr: 0,
            Offset: 0,
            Fixed: false,
            Unmap: false,
            Map32Bit: false,
            Perms: AccessType::Default(),
            MaxPerms: AccessType::Default(),
            Private: false,
            VDSO: false,
            GrowsDown: false,
            Precommit: false,
            MLockMode: MLockMode::default(),
            Kernel: false,
            Mapping: Some(Arc::new(file.clone())),
            Mappable: Some(file.Mappable()?),
            Hint: "".to_string(),
        });
    }
}

pub fn NewSharedAnonMapping() -> Arc<Mapping> {
    let m = SpecialMapping::New("/dev/zero (deleted)".to_string());
    return Arc::new(m)
}

pub fn NewAnonMapping(name: String) -> Arc<Mapping> {
    let m = SpecialMapping::New(name);
    return Arc::new(m)
}
