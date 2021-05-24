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

use super::futex::*;
use alloc::collections::btree_map::BTreeMap;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref INIT_RLIMITS : BTreeMap<i32, RLimit> = [
        (RLIMIT_CPU,        RLimit {Cur: RLIM_INFINITY,             Max: RLIM_INFINITY}),
        (RLIMIT_FSIZE,      RLimit {Cur: RLIM_INFINITY,             Max: RLIM_INFINITY}),
        (RLIMIT_DATA,       RLimit {Cur: RLIM_INFINITY,             Max: RLIM_INFINITY}),
        (RLIMIT_STACK,      RLimit {Cur: DEFAULT_STACK_SOFT_LIMIT,  Max: RLIM_INFINITY}),
        (RLIMIT_CORE,       RLimit {Cur: 0,                         Max: RLIM_INFINITY}),
        (RLIMIT_RSS,        RLimit {Cur: RLIM_INFINITY,             Max: RLIM_INFINITY}),
        (RLIMIT_NPROC,      RLimit {Cur: DEFAULT_NPROC_LIMIT,       Max: DEFAULT_NPROC_LIMIT}),
        (RLIMIT_NOFILE,     RLimit {Cur: DEFAULT_NOFILE_SOFT_LIMIT, Max: DEFAULT_NOFILE_HARD_LIMIT}),
        (RLIMIT_MEMLOCK,    RLimit {Cur: DEFAULT_MEMLOCK_LIMIT,     Max: DEFAULT_MEMLOCK_LIMIT}),
        (RLIMIT_AS,         RLimit {Cur: RLIM_INFINITY,             Max: RLIM_INFINITY}),
        (RLIMIT_LOCKS,      RLimit {Cur: RLIM_INFINITY,             Max: RLIM_INFINITY}),
        (RLIMIT_SIGPENDING, RLimit {Cur: 0,                         Max: 0}),
        (RLIMIT_MSGQUEUE,   RLimit {Cur: DEFAULT_MSGQUEUE_LIMIT,    Max: DEFAULT_MSGQUEUE_LIMIT}),
        (RLIMIT_NICE,       RLimit {Cur: 0,                         Max: 0}),
        (RLIMIT_RTPRIO,     RLimit {Cur: 0,                         Max: 0}),
        (RLIMIT_RTTIME,     RLimit {Cur: RLIM_INFINITY,             Max: RLIM_INFINITY}),
    ].iter().cloned().collect();
}

// RLimit corresponds to Linux's struct rlimit.
#[derive(Clone, Copy, Debug)]
pub struct RLimit {
    // Cur specifies the soft limit.
    pub Cur: u64,
    // Max specifies the hard limit.
    pub Max: u64,
}

pub const RLIMIT_CPU: i32 = 0;
pub const RLIMIT_FSIZE: i32 = 1;
pub const RLIMIT_DATA: i32 = 2;
pub const RLIMIT_STACK: i32 = 3;
pub const RLIMIT_CORE: i32 = 4;
pub const RLIMIT_RSS: i32 = 5;
pub const RLIMIT_NPROC: i32 = 6;
pub const RLIMIT_NOFILE: i32 = 7;
pub const RLIMIT_MEMLOCK: i32 = 8;
pub const RLIMIT_AS: i32 = 9;
pub const RLIMIT_LOCKS: i32 = 10;
pub const RLIMIT_SIGPENDING: i32 = 11;
pub const RLIMIT_MSGQUEUE: i32 = 12;
pub const RLIMIT_NICE: i32 = 13;
pub const RLIMIT_RTPRIO: i32 = 14;
pub const RLIMIT_RTTIME: i32 = 15;

// RLimInfinity is RLIM_INFINITY on Linux.
pub const RLIM_INFINITY: u64 = !0;

// DefaultStackSoftLimit is called _STK_LIM in Linux.
pub const DEFAULT_STACK_SOFT_LIMIT: u64 = 8 * 1024 * 1024;

// DefaultNprocLimit is defined in kernel/fork.c:set_max_threads, and
// called MAX_THREADS / 2 in Linux.
pub const DEFAULT_NPROC_LIMIT: u64 = FUTEX_TID_MASK as u64 / 2;

// DefaultNofileSoftLimit is called INR_OPEN_CUR in Linux.
pub const DEFAULT_NOFILE_SOFT_LIMIT: u64 = 1024;

// DefaultNofileHardLimit is called INR_OPEN_MAX in Linux.
pub const DEFAULT_NOFILE_HARD_LIMIT: u64 = 4096;

// DefaultMemlockLimit is called MLOCK_LIMIT in Linux.
pub const DEFAULT_MEMLOCK_LIMIT: u64 = 64 * 1024;

// DefaultMsgqueueLimit is called MQ_BYTES_MAX in Linux.
pub const DEFAULT_MSGQUEUE_LIMIT: u64 = 819200;