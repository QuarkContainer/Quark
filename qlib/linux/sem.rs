// Copyright (c) 2021 QuarkSoft LLC
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

use super::ipc::*;
use super::time::*;

// semctl Command Definitions. Source: include/uapi/linux/sem.h
pub const GETPID: i32 = 11;
pub const GETVAL: i32 = 12;
pub const GETALL: i32 = 13;
pub const GETNCNT: i32 = 14;
pub const GETZCNT: i32 = 15;
pub const SETVAL: i32 = 16;
pub const SETALL: i32 = 17;

// ipcs ctl cmds. Source: include/uapi/linux/sem.h
pub const SEM_STAT: i32 = 18;
pub const SEM_INFO: i32 = 19;
pub const SEM_STAT_ANY: i32 = 20;

pub const SEM_UNDO: i32 = 0x1000;

// SemidDS is equivalent to struct semid64_ds.
pub struct SemidDS {
    pub SemPerm: IPCPerm,
    pub SemOTime: TimeT,
    pub SemCTime: TimeT,
    pub SemNSems: u64,
    pub unused3: u64,
    pub unused4: u64,
}

// Sembuf is equivalent to struct sembuf.
pub struct Sembuf {
    pub SemNum: u16,
    pub SemOp: i16,
    pub SemFlag: i16,
}
