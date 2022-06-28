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

// Control commands used with semctl, shmctl, and msgctl. Source:
// include/uapi/linux/ipc.h.
pub const IPC_RMID: i32 = 0;
pub const IPC_SET: i32 = 1;
pub const IPC_STAT: i32 = 2;
pub const IPC_INFO: i32 = 3;

// resource get request flags. Source: include/uapi/linux/ipc.h
pub const IPC_CREAT: i16 = 0o01000;
pub const IPC_EXCL: i16 = 0o002000;
pub const IPC_NOWAIT: i16 = 0o004000;

pub const IPC_PRIVATE: i32 = 0;

// In Linux, amd64 does not enable CONFIG_ARCH_WANT_IPC_PARSE_VERSION, so SysV
// IPC unconditionally uses the "new" 64-bit structures that are needed for
// features like 32-bit UIDs.

// IPCPerm is equivalent to struct ipc64_perm.
#[repr(C)]
#[derive(Clone, Debug, Default, Copy)]
pub struct IPCPerm {
    pub Key: u32,
    pub UID: u32,
    pub GID: u32,
    pub CUID: u32,
    pub CGID: u32,
    pub Mode: u16,
    pub pad1: u16,
    pub Seq: u16,
    pub pad2: u16,
    pub pad3: u32,
    pub unused1: u64,
    pub unused2: u64,
}
