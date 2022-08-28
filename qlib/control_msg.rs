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

use super::auth::id::*;
use super::loader::*;
use super::singleton::*;

type Cid = String;

pub static MSG_ID: Singleton<AtomicU64> = Singleton::<AtomicU64>::New();

pub unsafe fn InitSingleton() {
    MSG_ID.Init(AtomicU64::new(1));
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ControlMsg {
    pub msgId: u64,
    pub payload: Payload,
}

impl ControlMsg {
    pub fn New(payLoad: Payload) -> Self {
        return Self {
            msgId: MSG_ID.fetch_add(1, Ordering::SeqCst),
            payload: payLoad,
        };
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WaitAll {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WaitPid {
    pub cid: String,
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

/// SignalArgs is payload for Signal control msg to quark sandbox,
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignalArgs {
    // CID is the container ID that will be signaled
    pub CID: String,

    // Signo is the signal to send to the process.
    pub Signo: i32,

    // PID is the process ID in the given container that will be signaled.
    // If 0, the root process of the container will be signalled.
    pub PID: i32,

    // Mode is the signal delivery mode.
    pub Mode: SignalDeliveryMode,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateArgs {
    // cid is the id for the new container
    pub cid: String,

    // fd to be sent to the new container, usually tty replica
    pub fds: Vec<i32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StartArgs {
    pub process: Process,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Payload {
    RootContainerStart(RootProcessStart),
    ExecProcess(Process),
    Pause,
    Unpause,
    WaitContainer(Cid),
    WaitPid(WaitPid),
    Ps(Cid),
    Signal(SignalArgs),
    ContainerDestroy(Cid),
    CreateSubContainer(CreateArgs),
    StartSubContainer(StartArgs),
    WaitAll,
}

impl Default for Payload {
    fn default() -> Self {
        return Self::WaitAll
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RootContainerStart {
    pub cid: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RootProcessStart {
    pub cid: String,
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
    CreateSubContainerResp,
    StartSubContainerResp,
    WaitAllResp(WaitAllResp),
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

#[derive(Serialize, Deserialize, Debug)]
pub struct WaitAllResp {
    pub cid: String,
    pub execId: String,
    pub status: i32,
}
