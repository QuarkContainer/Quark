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

use alloc::vec::Vec;

use super::ipc::*;
use super::time::*;

// Linux-specific control commands. Source: include/uapi/linux/msg.h
pub const MSG_STAT: i32 = 11;
pub const MSG_INFO: i32 = 12;
pub const MSG_STAT_ANY: i32 = 13;

// msgrcv(2) options. Source: include/uapi/linux/msg.h
pub const MSG_NOERROR: i32 = 0o10000; // No error if message is too big.
pub const MSG_EXCEPT: i32 = 0o20000; // Receive any message except of specified type.
pub const MSG_COPY: i32 = 0o40000; // Copy (not remove) all queue messages.

// System-wide limits for message queues. Source: include/uapi/linux/msg.h
pub const MSGMNI: usize = 32000; // Maximum number of message queue identifiers.
pub const MSGMAX: usize = 8192; // Maximum size of message (bytes).
pub const MSGMNB: usize = 16384; // Default max size of a message queue.

// System-wide limits. Unused. Source: include/uapi/linux/msg.h
pub const MSGPOOL: usize = MSGMNI * MSGMNB / 1024;
pub const MSGTQL: usize = MSGMNB;
pub const MSGMAP: usize = MSGMNB;
pub const MSGSSZ: usize = 16;

// MSGSEG is simplified due to the inexistance of a ternary operator.
pub const MSGSEG: usize = 0xffff;

// MsqidDS is equivelant to struct msqid64_ds. Source:
// include/uapi/asm-generic/shmbuf.h
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MsqidDS {
    pub MsgPerm: IPCPerm, // IPC permissions.
    pub MsgStime: TimeT,  // Last msgsnd time.
    pub MsgRtime: TimeT,  // Last msgrcv time.
    pub MsgCtime: TimeT,  // Last change time.
    pub MsgCbytes: u64,   // Current number of bytes on the queue.
    pub MsgQnum: u64,     // Number of messages in the queue.
    pub MsgQbytes: u64,   // Max number of bytes in the queue.
    pub MsgLspid: i32,    // PID of last msgsnd.
    pub MsgLrpid: i32,    // PID of last msgrcv.
    pub unused4: u64,
    pub unused5: u64,
}

// MsgInfo is equivelant to struct msginfo. Source: include/uapi/linux/msg.h
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MsgInfo {
    pub MsgPool: i32,
    pub MsgMap: i32,
    pub MsgMax: i32,
    pub MsgMnb: i32,
    pub MsgMni: i32,
    pub MsgSsz: i32,
    pub MsgTql: i32,
    pub MsgSeg: u16,
}

pub struct MsgBuf {
    pub Type: i64,
    pub Text: Vec<u8>,
}
