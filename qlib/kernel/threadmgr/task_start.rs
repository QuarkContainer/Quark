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

//use super::super::syscalls::util::KLoadBinary;
use super::super::super::auth::*;
use super::super::kernel::cpuset::*;
use super::super::kernel::fd_table::*;
use super::super::kernel::fs_context::*;
use super::super::kernel::ipc_namespace::*;
use super::super::kernel::kernel::*;
use super::super::kernel::uts_namespace::*;
use super::super::memmgr::mm::*;
use super::super::SignalDef::*;
use super::task_block::*;
use super::thread::*;
use super::thread_group::*;

pub struct TaskConfig {
    pub TaskId: u64,

    // Kernel is the owning Kernel.
    pub Kernel: Kernel,

    // Parent is the new task's parent. Parent may be nil.
    pub Parent: Option<Thread>,

    // If InheritParent is not nil, use InheritParent's parent as the new
    // task's parent.
    pub InheritParent: Option<Thread>,

    // ThreadGroup is the ThreadGroup the new task belongs to.
    pub ThreadGroup: ThreadGroup,

    // SignalMask is the new task's initial signal mask.
    pub SignalMask: SignalSet,

    pub MemoryMgr: MemoryManager,

    // FSContext is the FSContext of the new task. A reference must be held on
    // FSContext, which is transferred to TaskSet.NewTask whether or not it
    // succeeds.
    pub FSContext: FSContext,

    // Fdtble is the FDMap of the new task. A reference must be held on FDMap,
    // which is transferred to TaskSet.NewTask whether or not it succeeds.
    pub Fdtbl: FDTable,

    // Credentials is the Credentials of the new task.
    pub Credentials: Credentials,

    // Niceness is the niceness of the new task.
    pub Niceness: i32,

    // If NetworkNamespaced is true, the new task should observe a non-root
    // network namespace.
    pub NetworkNamespaced: bool,

    // AllowedCPUMask contains the cpus that this task can run on.
    pub AllowedCPUMask: CPUSet,

    // UTSNamespace is the UTSNamespace of the new task.
    pub UTSNamespace: UTSNamespace,

    // IPCNamespace is the IPCNamespace of the new task.
    pub IPCNamespace: IPCNamespace,

    pub Blocker: Blocker,

    pub ContainerID: String,
}
