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

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use lazy_static::lazy_static;

use super::loader::*;
use super::auth::id::*;

lazy_static! {
    static ref MSG_ID : AtomicU64 = AtomicU64::new(1);
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ControlMsg {
    pub msgId: u64,
    pub payload: Payload,
}

impl ControlMsg {
    pub fn New(payLoad: Payload) -> Self {
        return Self {
            msgId: MSG_ID.fetch_add(1, Ordering::SeqCst),
            payload: payLoad
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct WaitPid {
    pub pid: i32,
    pub clearStatus: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq)]
pub enum SignalDeliveryMode {
    // DeliverToProcess delivers the signal to the container process with
    // the specified PID. If PID is 0, then the container init process is
    // signaled.
    DeliverToProcess,

    // DeliverToAllProcesses delivers the signal to all processes in the
    // container. PID must be 0.
    DeliverToAllProcesses,

    // DeliverToForegroundProcessGroup delivers the signal to the
    // foreground process group in the same TTY session as the specified
    // process. If PID is 0, then the signal is delivered to the foreground
    // process group for the TTY for the init process.
    DeliverToForegroundProcessGroup,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct SignalArgs {
    // Signo is the signal to send to the process.
    pub Signo: i32,

    // PID is the process ID in the given container that will be signaled.
    // If 0, the root container will be signalled.
    pub PID: i32,

    // Mode is the signal delivery mode.
    pub Mode: SignalDeliveryMode,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Payload {
    RootContainerStart(RootProcessStart),
    ExecProcess(Process),
    Pause,
    Unpause,
    WaitContainer,
    WaitPid(WaitPid),
    Ps(String),
    Signal(SignalArgs),
    ContainerDestroy,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RootProcessStart {
    pub cid: String
}

#[derive(Serialize, Deserialize, Debug)]
pub enum UCallResp {
    UCallRespErr(String),
    RootContainerStartResp,
    ExecProcessResp(i32),
    PauseResp,
    UnpauseResp,
    PsResp(Vec<ProcessInfo>),
    WaitContainerResp(u32),
    WaitPidResp(u32),
    SignalResp,
    ContainerDestroyResp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProcessInfo {
    pub UID: KUID,
    pub PID: i32,
    // Parent PID
    pub PPID: i32,
    // Processor utilization
    pub Utilization: i32,
    // Start time
    pub STime: i64,
    // CPU time
    pub Time: i64,
    // Executable shortname (e.g. "sh" for /bin/sh)
    pub Cmd: String,
}
