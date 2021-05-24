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


// Commands from linux/fcntl.h.
pub const F_DUPFD         : i32 = 0;
pub const F_GETFD         : i32 = 1;
pub const F_SETFD         : i32 = 2;
pub const F_GETFL         : i32 = 3;
pub const F_SETFL         : i32 = 4;
pub const F_SETLK         : i32 = 6;
pub const F_SETLKW        : i32 = 7;
pub const F_SETOWN        : i32 = 8;
pub const F_GETOWN        : i32 = 9;
pub const F_SETOWN_EX     : i32 = 15;
pub const F_GETOWN_EX     : i32 = 16;
pub const F_DUPFD_CLOEXEC : i32 = 1024 + 6;
pub const F_SETPIPE_SZ    : i32 = 1024 + 7;
pub const F_GETPIPE_SZ    : i32 = 1024 + 8;

// Commands for F_SETLK.
pub const F_RDLCK : i32 = 0;
pub const F_WRLCK : i32 = 1;
pub const F_UNLCK : i32 = 2;

// Flags for fcntl.
pub const FD_CLOEXEC : i32 = 1;

// Flock is the lock structure for F_SETLK.
#[repr(C)]
pub struct Flock {
    pub Type   : i16,
    pub Whence : i16,
    pub Start  : i64,
    pub Len    : i64,
    pub Pid    : i32,
}

// Flags for F_SETOWN_EX and F_GETOWN_EX.
pub const F_OWNER_TID  :i32 = 0;
pub const F_OWNER_PID  :i32 = 1;
pub const F_OWNER_PGRP :i32 = 2;

// FOwnerEx is the owner structure for F_SETOWN_EX and F_GETOWN_EX.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct FOwnerEx {
    pub Type : i32,
    pub PID  : i32,
}