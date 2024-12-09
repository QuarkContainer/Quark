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

use crate::qlib::mutex::*;
use alloc::collections::btree_set::BTreeSet;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::sync::Weak;
use core::cmp::*;
use core::ops::Deref;

use super::super::super::auth::*;
use super::super::super::linux_def::*;
use super::super::super::usage::io::*;
use super::super::kernel::cpuset::*;
use super::super::kernel::fd_table::*;
use super::super::kernel::fs_context::*;
use super::super::kernel::ipc_namespace::*;
use super::super::kernel::kernel::*;
use super::super::kernel::time::*;
use super::super::kernel::uts_namespace::*;
use super::super::kernel::waiter::queue::*;
use super::super::kernel::waiter::waitgroup::*;
use super::super::memmgr::mm::*;
use super::super::threadmgr::task_block::*;
use super::super::threadmgr::task_exit::*;
use super::super::threadmgr::task_sched::*;
use super::super::threadmgr::task_stop::*;
use super::super::SignalDef::*;
use super::pid_namespace::*;
use super::thread_group::*;
use super::threads::*;

pub type UniqueID = u64;
pub type ThreadID = i32;
pub type SessionID = i32;
pub type ProcessGroupID = i32;

pub const ROBUST_LIST_LEN: u64 = 0x18;

//#[derive(Default)]
pub struct ThreadInternal {
    pub id: ThreadID,

    // Name is the thread name set by the prctl(PR_SET_NAME) system call.
    pub name: String,

    pub taskId: u64,
    //the task of the Task Stack

    pub blocker: Blocker,

    pub k: Kernel,

    pub memoryMgr: MemoryManager,

    pub fsc: FSContext,

    pub fdTbl: FDTable,

    // If vforkParent is not nil, it is the task that created this task with
    // vfork() or clone(CLONE_VFORK), and should have its vforkStop ended when
    // this TaskContext is released.
    //
    // vforkParent is protected by the TaskSet mutex.
    pub vforkParent: Option<Thread>,

    pub creds: Credentials,

    pub utsns: UTSNamespace,
    pub ipcns: IPCNamespace,

    pub SignalQueue: Queue,

    // tg is the thread group that this task belongs to. The tg pointer is
    // immutable.
    pub tg: ThreadGroup,

    // parent is the task's parent. parent may be nil.
    //
    // parent is protected by the TaskSet mutex.
    pub parent: Option<Thread>,

    // children is this task's children.
    //
    // children is protected by the TaskSet mutex.
    pub children: BTreeSet<Thread>,

    // If childPIDNamespace is not nil, all new tasks created by this task will
    // be members of childPIDNamespace rather than this one. (As a corollary,
    // this task becomes unable to create sibling tasks in the same thread
    // group.)
    //
    // childPIDNamespace is exclusive to the task goroutine.
    pub childPIDNamespace: Option<PIDNamespace>,

    // haveSyscallReturn is true if tc.Arch().Return() represents a value
    // returned by a syscall (or set by ptrace after a syscall).
    //
    // haveSyscallReturn is exclusive to the task goroutine.
    pub SysCallReturn: Option<u64>,

    // sched contains the current scheduling state of the task.
    //
    // sched is protected by scedSeq. sched is owned by the task
    //pub scedSeq: SeqCount,
    pub sched: TaskSchedInfo,

    // yieldCount is the number of times the task goroutine has called
    // Task.InterruptibleSleepStart, Task.UninterruptibleSleepStart, or
    // Task.Yield(), voluntarily ceasing execution.
    //
    // yieldCount is accessed using atomic memory operations. yieldCount is
    // owned by the task goroutine.
    pub yieldCount: u64,

    // pendingSignals is the set of pending signals that may be handled only by
    // this task.
    //
    // pendingSignals is protected by (taskNode.)tg.signalHandlers.mu
    // (hereafter "the signal mutex"); see comment on
    // ThreadGroup.signalHandlers.
    pub pendingSignals: PendingSignals,

    // signalMask is the set of signals whose delivery is currently blocked.
    //
    // signalMask is accessed using atomic memory operations, and is protected
    // by the signal mutex (such that reading signalMask is safe if either the
    // signal mutex is locked or if atomic memory operations are used, while
    // writing signalMask requires both). signalMask is owned by the task
    // goroutine.
    pub signalMask: SignalSet,

    // If the task goroutine is currently executing Task.sigtimedwait,
    // realSignalMask is the previous value of signalMask, which has temporarily
    // been replaced by Task.sigtimedwait. Otherwise, realSignalMask is 0.
    //
    // realSignalMask is exclusive to the task goroutine.
    pub realSignalMask: SignalSet,

    // If haveSavedSignalMask is true, savedSignalMask is the signal mask that
    // should be applied after the task has either delivered one signal to a
    // user handler or is about to resume execution in the untrusted
    // application.
    //
    // Both haveSavedSignalMask and savedSignalMask are exclusive to the task
    // goroutine.
    pub haveSavedSignalMask: bool,
    pub savedSignalMask: SignalSet,

    // signalStack is the alternate signal stack used by signal handlers for
    // which the SA_ONSTACK flag is set.
    //
    // signalStack is exclusive to the task goroutine.
    pub signalStack: SignalStack,

    // If groupStopPending is true, the task should participate in a group
    // stop in the interrupt path.
    //
    // groupStopPending is analogous to JOBCTL_STOP_PENDING in Linux.
    //
    // groupStopPending is protected by the signal mutex.
    pub groupStopPending: bool,

    // If groupStopAcknowledged is true, the task has already acknowledged that
    // it is entering the most recent group stop that has been initiated on its
    // thread group.
    //
    // groupStopAcknowledged is analogous to !JOBCTL_STOP_CONSUME in Linux.
    //
    // groupStopAcknowledged is protected by the signal mutex.
    pub groupStopAcknowledged: bool,

    // If trapStopPending is true, the task goroutine should enter a
    // PTRACE_INTERRUPT-induced stop from the interrupt path.
    //
    // trapStopPending is analogous to JOBCTL_TRAP_STOP in Linux, except that
    // Linux also sets JOBCTL_TRAP_STOP when a ptraced task detects
    // JOBCTL_STOP_PENDING.
    //
    // trapStopPending is protected by the signal mutex.
    pub trapStopPending: bool,

    // If trapNotifyPending is true, this task is PTRACE_SEIZEd, and a group
    // stop has begun or ended since the last time the task entered a
    // ptrace-stop from the group-stop path.
    //
    // trapNotifyPending is analogous to JOBCTL_TRAP_NOTIFY in Linux.
    //
    // trapNotifyPending is protected by the signal mutex.
    pub trapNotifyPending: bool,

    //pub containerID: String,

    // This is mostly a fake cpumask just for sched_set/getaffinity as we
    // don't really control the affinity.
    //
    // Invariant: allowedCPUMask.Size() ==
    // sched.CPUMaskSize(Kernel.applicationCores).
    //
    // allowedCPUMask is protected by mu.
    pub allowedCPUMask: CPUSet,

    // cpu is the fake cpu number returned by getcpu(2). cpu is ignored
    // entirely if Kernel.useHostCores is true.
    //
    // cpu is accessed using atomic memory operations.
    pub cpu: i32,

    // This is used to keep track of changes made to a process' priority/niceness.
    // It is mostly used to provide some reasonable return value from
    // getpriority(2) after a call to setpriority(2) has been made.
    // We currently do not actually modify a process' scheduling priority.
    // NOTE: This represents the userspace view of priority (nice).
    // This means that the value should be in the range [-20, 19].
    //
    // niceness is protected by mu.
    pub niceness: i32,

    // This is used to track the numa policy for the current thread. This can be
    // modified through a set_mempolicy(2) syscall. Since we always report a
    // single numa node, all policies are no-ops. We only track this information
    // so that we can return reasonable values if the application calls
    // get_mempolicy(2) after setting a non-default policy. Note that in the
    // real syscall, nodemask can be longer than 4 bytes, but we always report a
    // single node so never need to save more than a single bit.
    //
    // numaPolicy and numaNodeMask are protected by mu.
    pub numaPolicy: i32,
    pub numaNodeMask: u64,

    // If netns is true, the task is in a non-root network namespace. Network
    // namespaces aren't currently implemented in full; being in a network
    // namespace simply prevents the task from observing any network devices
    // (including loopback) or using abstract socket addresses (see unix(7)).
    //
    // netns is protected by mu. netns is owned by the task goroutine.
    pub netns: bool,

    // parentDeathSignal is sent to this task's thread group when its parent exits.
    //
    // parentDeathSignal is protected by mu.
    pub parentDeathSignal: Signal,

    // If stop is not nil, it is the internally-initiated condition that
    // currently prevents the task goroutine from running.
    //
    // stop is protected by the signal mutex.
    pub stop: Option<Arc<TaskStop>>,

    // stopCount is the number of active external stops (calls to
    // Task.BeginExternalStop that have not been paired with a call to
    // Task.EndExternalStop), plus 1 if stop is not nil. Hence stopCount is
    // non-zero if the task goroutine should stop.
    pub stopCount: WaitGroup,

    // exitStatus is the task's exit status.
    //
    // exitStatus is protected by the signal mutex.
    pub exitStatus: ExitStatus,

    // exitState is the task's progress through the exit path.
    //
    // exitState is protected by the TaskSet mutex. exitState is owned by the
    // task goroutine.
    pub exitState: TaskExitState,

    // exitTracerNotified is true if the exit path has either signaled the
    // task's tracer to indicate the exit, or determined that no such signal is
    // needed. exitTracerNotified can only be true if exitState is
    // TaskExitZombie or TaskExitDead.
    //
    // exitTracerNotified is protected by the TaskSet mutex.
    pub exitTracerNotified: bool,

    // exitTracerAcked is true if exitTracerNotified is true and either the
    // task's tracer has acknowledged the exit notification, or the exit path
    // has determined that no such notification is needed.
    //
    // exitTracerAcked is protected by the TaskSet mutex.
    pub exitTracerAcked: bool,

    // exitParentNotified is true if the exit path has either signaled the
    // task's parent to indicate the exit, or determined that no such signal is
    // needed. exitParentNotified can only be true if exitState is
    // TaskExitZombie or TaskExitDead.
    //
    // exitParentNotified is protected by the TaskSet mutex.
    pub exitParentNotified: bool,

    // exitParentAcked is true if exitParentNotified is true and either the
    // task's parent has acknowledged the exit notification, or the exit path
    // has determined that no such acknowledgment is needed.
    //
    // exitParentAcked is protected by the TaskSet mutex.
    pub exitParentAcked: bool,

    // startTime is the real time at which the task started. It is set when
    // a Task is created or invokes execve(2).
    //
    // startTime is protected by mu.
    pub startTime: Time,

    // containerID has no equivalent in Linux; it's used by runsc to track all
    // tasks that belong to a given containers since cgroups aren't implemented.
    // It's inherited by the children, is immutable, and may be empty.
    //
    // NOTE: cgroups can be used to track this when implemented.
    pub containerID: String,

    pub ioUsage: IO,

    pub robust_list_head: u64,
}

impl ThreadInternal {
    pub fn IsChrooted(&self) -> bool {
        let realRoot = self
            .k
            .mounts
            .read()
            .get(&self.containerID)
            .unwrap()
            .root
            .clone();
        let root = self.fsc.RootDirectory();
        return realRoot == root;
    }

    pub fn SetRet(&mut self, ret: u64) {
        self.SysCallReturn = Some(ret)
    }
}

#[derive(Clone, Default)]
pub struct ThreadWeak {
    pub uid: UniqueID,
    pub data: Weak<QMutex<ThreadInternal>>,
}

impl ThreadWeak {
    pub fn Upgrade(&self) -> Option<Thread> {
        let t = match self.data.upgrade() {
            None => return None,
            Some(t) => t,
        };

        return Some(Thread {
            uid: self.uid,
            data: t,
        });
    }
}

//#[derive(Default)]
pub struct Thread {
    pub uid: UniqueID,
    pub data: Arc<QMutex<ThreadInternal>>,
}

impl Clone for Thread {
    fn clone(&self) -> Self {
        return Self {
            uid: self.Uid(),
            data: self.data.clone(),
        };
    }
}

impl Deref for Thread {
    type Target = Arc<QMutex<ThreadInternal>>;

    fn deref(&self) -> &Arc<QMutex<ThreadInternal>> {
        &self.data
    }
}

impl Ord for Thread {
    fn cmp(&self, other: &Self) -> Ordering {
        self.Uid().cmp(&other.Uid())
    }
}

impl PartialOrd for Thread {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Thread {
    fn eq(&self, other: &Self) -> bool {
        return self.Uid() == other.Uid();
    }
}

impl Eq for Thread {}

impl Thread {
    // debug api
    pub fn IDs(&self) -> (i32, i32) {
        let tg = self.ThreadGroup();
        let pidns = tg.PIDNamespace();
        return (pidns.IDOfThreadGroup(&tg), pidns.IDOfTask(&self));
    }

    pub fn RefCount(&self) -> usize {
        return Arc::strong_count(&self.data);
    }

    pub fn Creds(&self) -> Credentials {
        return self.lock().creds.clone();
    }

    pub fn UTSNamespace(&self) -> UTSNamespace {
        return self.lock().utsns.clone();
    }

    pub fn MemoryManager(&self) -> MemoryManager {
        return self.lock().memoryMgr.clone();
    }

    pub fn Downgrade(&self) -> ThreadWeak {
        return ThreadWeak {
            uid: self.uid,
            data: Arc::downgrade(&self.data),
        };
    }

    pub fn Kernel(&self) -> Kernel {
        return self.lock().k.clone();
    }

    pub fn Uid(&self) -> UniqueID {
        return self.uid;
    }

    pub fn ThreadGroup(&self) -> ThreadGroup {
        return self.lock().tg.clone();
    }

    pub fn PIDNamespace(&self) -> PIDNamespace {
        let tg = self.lock().tg.clone();
        return tg.lock().pidns.clone();
    }

    pub fn TaskSet(&self) -> TaskSet {
        let pidns = self.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        return owner;
    }

    pub fn Parent(&self) -> Option<Thread> {
        let taskSet = self.TaskSet();
        let _r = taskSet.ReadLock();
        match &self.lock().parent {
            None => None,
            Some(ref p) => Some(p.clone()),
        }
    }

    pub fn ThreadID(&self) -> ThreadID {
        let ns = self.PIDNamespace();
        return ns.IDOfTask(self);
    }

    pub fn StartTime(&self) -> Time {
        return self.lock().startTime;
    }

    pub fn ContainerID(&self) -> String {
        return self.lock().containerID.to_string();
    }
}
