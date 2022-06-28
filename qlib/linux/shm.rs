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

use core::u64;

use super::ipc::*;
use super::time::*;

// shmat(2) flags. Source: include/uapi/linux/shm.h
pub const SHM_RDONLY: u16 = 0o010000; // Read-only access.
pub const SHM_RND: u16 = 0o020000; // Round attach address to SHMLBA boundary.
pub const SHM_REMAP: u16 = 0o040000; // Take-over region on attach.
pub const SHM_EXEC: u16 = 0o0100000; // Execution access.

// IPCPerm.Mode upper byte flags. Source: include/linux/shm.h
pub const SHM_DEST: u16 = 0o01000; // Segment will be destroyed on last detach.
pub const SHM_LOCKED: u16 = 0o02000; // Segment will not be swapped.
pub const SHM_HUGETLB: u16 = 0o04000; // Segment will use huge TLB pages.
pub const SHM_NORESERVE: u16 = 0o010000; // Don't check for reservations.

// Additional Linux-only flags for shmctl(2). Source: include/uapi/linux/shm.h
pub const SHM_LOCK: i32 = 11;
pub const SHM_UNLOCK: i32 = 12;
pub const SHM_STAT: i32 = 13;
pub const SHM_INFO: i32 = 14;

// SHM defaults as specified by linux. Source: include/uapi/linux/shm.h
pub const SHMMIN: u64 = 1;
pub const SHMMNI: u64 = 4096;
pub const SHMMAX: u64 = u64::MAX - 1 << 24;
pub const SHMALL: u64 = u64::MAX - 1 << 24;
pub const SHMSEG: u64 = 4096;

// ShmidDS is equivalent to struct shmid64_ds. Source:
// include/uapi/asm-generic/shmbuf.h
#[repr(C)]
#[derive(Clone, Debug, Default, Copy)]
pub struct ShmidDS {
    pub ShmPerm: IPCPerm,
    pub ShmSegsz: u64,
    pub ShmAtime: TimeT,
    pub ShmDtime: TimeT,
    pub ShmCtime: TimeT,
    pub ShmCpid: i32,
    pub ShmLpid: i32,
    pub ShmNattach: i64,

    pub Unused4: i64,
    pub Unused5: i64,
}

// ShmParams is equivalent to struct shminfo. Source: include/uapi/linux/shm.h
#[repr(C)]
#[derive(Clone, Debug, Default, Copy)]
pub struct ShmParams {
    pub ShmMax: u64,
    pub ShmMin: u64,
    pub ShmMni: u64,
    pub ShmSeg: u64,
    pub ShmAll: u64,
}

// ShmInfo is equivalent to struct shm_info. Source: include/uapi/linux/shm.h
#[repr(C)]
#[derive(Clone, Debug, Default, Copy)]
pub struct ShmInfo {
    pub UsedIDs: i32,
    // Number of currently existing segments.
    //pad               : i32,
    pub ShmTot: u64,
    // Total number of shared memory pages.
    pub ShmRss: u64,
    // Number of resident shared memory pages.
    pub ShmSwp: u64,
    // Number of swapped shared memory pages.
    pub SwapAttempts: u64,
    // Unused since Linux 2.4.
    pub SwapSuccesses: u64,
    // Unused since Linux 2.4.
}
