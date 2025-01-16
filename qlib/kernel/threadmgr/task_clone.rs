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

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::sync::Arc;
use core::ptr;
use core::sync::atomic::AtomicUsize;

use crate::qlib::kernel::arch::tee::is_cc_active;
use super::super::super::super::kernel_def::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::task_mgr::*;
use super::super::arch::__arch::context::MAX_ADDR64;
use super::super::arch::__arch::arch_def::Context;
use super::super::kernel::ipc_namespace::*;
use super::super::threadmgr::task_start::*;
use super::super::threadmgr::thread::*;
use super::super::SignalDef::*;
use super::super::*;
use super::super::task::*;
use super::task_block::*;
use super::task_stop::*;

pub fn IsValidSegmentBase(addr: u64) -> bool {
    return addr < MAX_ADDR64;
}

const DEFAULT_STACK_SIZE: usize = MemoryDef::DEFAULT_STACK_SIZE as usize;
const DEFAULT_STACK_PAGES: u64 = DEFAULT_STACK_SIZE as u64 / (4 * 1024);
const DEFAULT_STACK_MAST: u64 = !(DEFAULT_STACK_SIZE as u64 - 1);

#[derive(Debug, Copy, Clone, Default)]
pub struct SharingOptions {
    // If NewAddressSpace is true, the task should have an independent virtual
    // address space.
    pub NewAddressSpace: bool,

    // If NewSignalHandlers is true, the task should use an independent set of
    // signal handlers.
    pub NewSignalHandlers: bool,

    // If NewThreadGroup is true, the task should be the leader of its own
    // thread group. TerminationSignal is the signal that the thread group
    // will send to its parent when it exits. If NewThreadGroup is false,
    // TerminationSignal is ignored.
    pub NewThreadGroup: bool,
    pub TerminationSignal: Signal,

    // If NewPIDNamespace is true:
    //
    // - In the context of Task.Clone, the new task should be the init task
    // (TID 1) in a new PID namespace.
    //
    // - In the context of Task.Unshare, the task should create a new PID
    // namespace, and all subsequent clones of the task should be members of
    // the new PID namespace.
    pub NewPIDNamespace: bool,

    // If NewUserNamespace is true, the task should have an independent user
    // namespace.
    pub NewUserNamespace: bool,

    // If NewNetworkNamespace is true, the task should have an independent
    // network namespace. (Note that network namespaces are not really
    // implemented; see comment on Task.netns for details.)
    pub NewNetworkNamespace: bool,

    // If NewFiles is true, the task should use an independent file descriptor
    // table.
    pub NewFiles: bool,

    // If NewFSContext is true, the task should have an independent FSContext.
    pub NewFSContext: bool,

    // If NewUTSNamespace is true, the task should have an independent UTS
    // namespace.
    pub NewUTSNamespace: bool,

    // If NewIPCNamespace is true, the task should have an independent IPC
    // namespace.
    pub NewIPCNamespace: bool,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct CloneOptions {
    // SharingOptions defines the set of resources that the new task will share
    // with its parent.
    pub sharingOption: SharingOptions,

    // Stack is the initial stack pointer of the new task. If Stack is 0, the
    // new task will start with the same stack pointer as its parent.
    pub Stack: u64,

    // If SetTLS is true, set the new task's TLS (thread-local storage)
    // descriptor to TLS. If SetTLS is false, TLS is ignored.
    pub SetTLS: bool,
    pub TLS: u64,

    // If ChildClearTID is true, when the child exits, 0 is written to the
    // address ChildTID in the child's memory, and if the write is successful a
    // futex wake on the same address is performed.
    //
    // If ChildSetTID is true, the child's thread ID (in the child's PID
    // namespace) is written to address ChildTID in the child's memory. (As in
    // Linux, failed writes are silently ignored.)
    pub ChildClearTID: bool,
    pub ChildSetTID: bool,
    pub ChildTID: u64,

    // If ParentSetTID is true, the child's thread ID (in the parent's PID
    // namespace) is written to address ParentTID in the parent's memory. (As
    // in Linux, failed writes are silently ignored.)
    //
    // Older versions of the clone(2) man page state that CLONE_PARENT_SETTID
    // causes the child's thread ID to be written to ptid in both the parent
    // and child's memory, but this is a documentation error fixed by
    // 87ab04792ced ("clone.2: Fix description of CLONE_PARENT_SETTID").
    pub ParentSetTID: bool,
    pub ParentTID: u64,

    // If Vfork is true, place the parent in vforkStop until the cloned task
    // releases its TaskContext.
    pub Vfork: bool,

    // If Untraced is true, do not report PTRACE_EVENT_CLONE/FORK/VFORK for
    // this clone(), and do not ptrace-attach the caller's tracer to the new
    // task. (PTRACE_EVENT_VFORK_DONE will still be reported if appropriate).
    pub Untraced: bool,

    // If InheritTracer is true, ptrace-attach the caller's tracer to the new
    // task, even if no PTRACE_EVENT_CLONE/FORK/VFORK event would be reported
    // for it. If both Untraced and InheritTracer are true, no event will be
    // reported, but tracer inheritance will still occur.
    pub InheritTracer: bool,
}

impl CloneOptions {
    const EXIT_SIGNAL_MASK: i32 = 0xff;

    pub fn New(
        flags: u64,
        cStack: u64,
        pTid: u64,
        cTid: u64,
        tls: u64,
        hasChildPIdNamespace: bool,
    ) -> Result<Self> {
        let flags = flags as i32;
        let opts = CloneOptions {
            sharingOption: SharingOptions {
                NewAddressSpace: flags & CloneOp::CLONE_VM == 0,
                NewSignalHandlers: flags & CloneOp::CLONE_SIGHAND == 0,
                NewThreadGroup: flags & CloneOp::CLONE_THREAD == 0,
                TerminationSignal: Signal((flags & Self::EXIT_SIGNAL_MASK) as i32),

                NewPIDNamespace: flags & CloneOp::CLONE_NEWPID != 0,
                NewUserNamespace: flags & CloneOp::CLONE_NEWUSER != 0,
                NewNetworkNamespace: flags & CloneOp::CLONE_NEWNET != 0,
                NewFiles: flags & CloneOp::CLONE_FILES == 0,
                NewFSContext: flags & CloneOp::CLONE_FS == 0,
                NewUTSNamespace: flags & CloneOp::CLONE_NEWUTS != 0,
                NewIPCNamespace: flags & CloneOp::CLONE_NEWIPC != 0,
            },

            Stack: cStack,
            SetTLS: flags & CloneOp::CLONE_SETTLS != 0,
            TLS: tls,
            ChildClearTID: flags & CloneOp::CLONE_CHILD_CLEARTID != 0,
            ChildSetTID: flags & CloneOp::CLONE_CHILD_SETTID != 0,
            ChildTID: cTid,
            ParentSetTID: flags & CloneOp::CLONE_PARENT_SETTID != 0,
            ParentTID: pTid,
            Vfork: flags & CloneOp::CLONE_VFORK != 0,
            Untraced: flags & CloneOp::CLONE_UNTRACED != 0,
            InheritTracer: flags & CloneOp::CLONE_PTRACE != 0,
        };

        if opts.sharingOption.NewUserNamespace {
            // todo: handle NewUserNamespace
            //panic!("doesn't support new usernamespace ...");
            error!("doesn't support new usernamespace ...");
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // Since signal actions may refer to application signal handlers by virtual
        // address, any set of signal handlers must refer to the same address
        // space.
        if !opts.sharingOption.NewSignalHandlers && opts.sharingOption.NewAddressSpace {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // In order for the behavior of thread-group-directed signals to be sane,
        // all tasks in a thread group must share signal handlers.
        if !opts.sharingOption.NewThreadGroup && opts.sharingOption.NewSignalHandlers {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if !opts.sharingOption.NewThreadGroup
            && (opts.sharingOption.NewPIDNamespace || hasChildPIdNamespace)
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        // The two different ways of specifying a new PID namespace are
        // incompatible.
        if opts.sharingOption.NewPIDNamespace && hasChildPIdNamespace {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if opts.sharingOption.NewUserNamespace
            && (!opts.sharingOption.NewThreadGroup || !opts.sharingOption.NewFSContext)
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        return Ok(opts);
    }
}

impl Thread {
    pub fn Clone(&self, opts: &CloneOptions, taskid: u64) -> Result<Self> {
        let pidns = self.PIDNamespace();
        let ts = pidns.Owner();
        let _wl = ts.WriteLock();

        let t = self.lock();
        let creds = t.creds.clone();
        let mut userns = creds.lock().UserNamespace.clone();

        if opts.sharingOption.NewUserNamespace {
            if t.IsChrooted() {
                return Err(Error::SysError(SysErr::EPERM));
            }

            userns = creds.NewChildUserNamespace()?;
        }

        if opts.sharingOption.NewPIDNamespace
            || opts.sharingOption.NewNetworkNamespace
            || opts.sharingOption.NewUTSNamespace
                && !creds.HasCapabilityIn(Capability::CAP_SYS_ADMIN, &userns)
        {
            return Err(Error::SysError(SysErr::EPERM));
        }

        let mut utsns = t.utsns.clone();
        if opts.sharingOption.NewUTSNamespace {
            let tmp = utsns.Fork(&userns);
            utsns = tmp;
        }

        let mut ipcns = t.ipcns.clone();
        if opts.sharingOption.NewIPCNamespace {
            ipcns = IPCNamespace::New(&userns);
        }

        let mut memoryMgr = t.memoryMgr.clone();
        if opts.sharingOption.NewAddressSpace {
            let newMM = memoryMgr.Fork()?;
            memoryMgr = newMM;
        }

        let vforkParent = if opts.Vfork { Some(self.clone()) } else { None };

        let mut fsc = t.fsc.clone();
        if opts.sharingOption.NewFSContext {
            let temp = fsc.Fork();
            fsc = temp;
        }

        let mut fdTbl = t.fdTbl.clone();
        if opts.sharingOption.NewFiles {
            let newFDTbl = fdTbl.Fork(i32::MAX);
            fdTbl = newFDTbl;
        }

        let pidns = t.tg.PIDNamespace();

        if t.childPIDNamespace.is_some() {
            panic!("doesn't support childPIDNamespace********************");
            //pidns = t.childPIDNamespace.clone().unwrap();
        } else if opts.sharingOption.NewPIDNamespace {
            panic!("doesn't support NewPIDNamespace********************");
            //pidns = pidns.NewChild(&userns);
        }

        let mut tg = t.tg.clone();
        if opts.sharingOption.NewThreadGroup {
            let mut sh = tg.lock().signalHandlers.clone();
            if opts.sharingOption.NewSignalHandlers {
                sh = sh.Fork();
            }

            let kernel = t.k.clone();
            let limit = tg.lock().limits.clone();
            let cid = tg.lock().containerID.clone();
            tg = kernel.newThreadGroup(
                &pidns,
                &sh,
                opts.sharingOption.TerminationSignal.clone(),
                &limit.GetCopy(),
                &cid,
                &None,
            );
        }

        let mut cfg = TaskConfig {
            TaskId: taskid,
            Kernel: t.k.clone(),
            Parent: None,
            InheritParent: None,
            ThreadGroup: tg.clone(),
            SignalMask: t.signalMask.clone(),
            MemoryMgr: memoryMgr,
            FSContext: fsc,
            Fdtbl: fdTbl,
            Credentials: creds.clone(),
            Niceness: t.niceness,
            NetworkNamespaced: false,
            AllowedCPUMask: t.allowedCPUMask.Copy(),
            UTSNamespace: utsns,
            IPCNamespace: ipcns,
            Blocker: Blocker::New(taskid),
            ContainerID: t.containerID.to_string(),
        };

        if opts.sharingOption.NewThreadGroup {
            cfg.Parent = Some(self.clone());
        } else {
            cfg.InheritParent = Some(self.clone())
        }

        if opts.sharingOption.NewNetworkNamespace {
            cfg.NetworkNamespaced = true;
        }

        let pidns = tg.PIDNamespace();
        let ts = pidns.lock().owner.clone();

        let name = t.name.to_string();
        core::mem::drop(t);
        let kernel = self.lock().k.clone();
        let nt = ts.NewTask(&cfg, false, &kernel)?;

        nt.lock().name = name;

        if userns != creds.lock().UserNamespace.clone() {
            nt.SetUserNamespace(&userns)
                .expect("Task.Clone: SetUserNamespace failed: ")
        }

        if opts.Vfork {
            nt.lock().vforkParent = vforkParent;
            self.MaybeBeginVforkStop(&nt);
        }

        return Ok(nt);
    }

    pub fn MaybeBeginVforkStop(&self, child: &Thread) {
        let tg = self.ThreadGroup();
        let _owner = tg.PIDNamespace().Owner();
        //let _r = owner.ReadLock();

        let lock = tg.lock().signalLock.clone();
        let _s = lock.lock();

        {
            let mut threadLocked = self.lock();
            if threadLocked.killedLocked() {
                threadLocked.vforkParent = None;
                return;
            }
        }

        let vforkParent = child.lock().vforkParent.clone();
        if vforkParent == Some(self.clone()) {
            self.lock().beginInternalStopLocked(&Arc::new(VforkStop {}))
        }
    }

    pub fn UnstopVforkParent(&self) {
        let tg = self.ThreadGroup();
        let owner = tg.PIDNamespace().Owner();
        let _r = owner.ReadLock();

        let p = self.lock().vforkParent.take();
        match p {
            None => (),
            Some(p) => {
                let ptg = p.ThreadGroup();

                let lock = ptg.lock().signalLock.clone();
                let _s = lock.lock();

                let stop = p.lock().stop.clone();
                if stop.is_some() && stop.unwrap().Type() == TaskStopType::VFORKSTOP {
                    p.lock().endInternalStopLocked();
                }
            }
        }
    }
}

impl Task {
    pub fn SetClearTID(&mut self, addr: u64) {
        self.tidInfo.clear_child_tid = Some(addr);
    }

    pub fn Clone(&self, flags: u64, cStack: u64, pTid: u64, cTid: u64, tls: u64) -> Result<i32> {
        let opts = CloneOptions::New(flags, cStack, pTid, cTid, tls, false)?;

        if opts.SetTLS && !IsValidSegmentBase(opts.TLS) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        let mut userSp = cStack;
        if opts.sharingOption.NewAddressSpace || cStack == 0 {
            userSp = Self::Current().GetPtRegs().get_stack_pointer();
        }

        info!("Clone opts is {:x?}", &opts);

        let (pid, childTask) = self.CloneVM(&opts, userSp)?; //, cStack as * const u8);
        if opts.ParentSetTID {
            self.CopyOutObj(&pid, pTid)?;
        }

        let cTask = unsafe { &mut (*childTask) };

        if opts.ChildClearTID == true {
            cTask.SetClearTID(cTid);
        }

        if opts.ChildSetTID == true {
            // can't use the GetTypeMut as it is used with current pagetable.
            //*Task::GetTask(cTask.taskId).GetTypeMut(cTid)? = pid;

            cTask.CopyOutObjManual(&pid, cTid)?;
        }

        if opts.SetTLS {
            cTask.context.set_tls(tls);
        }

        taskMgr::NewTask(TaskId::New(cTask.taskId));
        return Ok(pid);
    }

    pub fn CloneVM(&self, opts: &CloneOptions, userSp: u64) -> Result<(i32, *mut Self)> {
        //let pid = self.GetProcessId();
        let cPid;

        let s_ptr = KERNEL_STACK_ALLOCATOR.Allocate().unwrap() as *mut u8;
        let taskPtr = s_ptr as *mut Self;

        let task = Task::Current();
        let thread = task.Thread();

        let mut taskId = s_ptr as u64;
        if is_cc_active() {
            let tw_size  = core::mem::size_of::<TaskWrapper>();
            let tw_ptr = unsafe {
                crate::GLOBAL_ALLOCATOR.AllocSharedBuf(tw_size, 2)
            };

            let t_wp = TaskWrapper::New(s_ptr as u64);
            let t_wp_ptr = tw_ptr as *mut TaskWrapper;
            unsafe {
                ptr::write(
                    t_wp_ptr,
                    t_wp
                );
            }
            taskId = t_wp_ptr as u64;
        }
        


        let nt = thread.Clone(&opts, taskId as u64)?;

        unsafe {
            let mm = nt.lock().memoryMgr.clone();
            let creds = nt.lock().creds.clone();
            let utsns = nt.lock().utsns.clone();
            let ipcns = nt.lock().ipcns.clone();
            let fsContext = nt.lock().fsc.clone();
            let fdTbl = nt.lock().fdTbl.clone();
            let blocker = nt.lock().blocker.clone();
            let sched = nt.lock().sched.clone();

            let tg = nt.lock().tg.clone();
            tg.lock().liveThreads.Add(1);
            let pidns = tg.PIDNamespace();
            let ntid = pidns.IDOfTask(&nt);

            let futexMgr = if opts.sharingOption.NewAddressSpace {
                task.futexMgr.Fork()
            } else {
                task.futexMgr.clone()
            };

            cPid = ntid;

            let signalStack = if opts.sharingOption.NewAddressSpace || opts.Vfork {
                self.CloneSignalStack()
            } else {
                SignalStack::default()
            };

            let ioUsage = nt.lock().ioUsage.clone();

            ptr::write_volatile(
                taskPtr,
                Self {
                    context: Context::New(),
                    taskId: taskId as u64,
                    mm: mm,
                    tidInfo: Default::default(),
                    isWaitThread: false,
                    signalStack: signalStack,
                    mountNS: task.mountNS.clone(),
                    // Arc::new(QMutex::new(Default::default())),
                    creds: creds,
                    utsns: utsns,
                    ipcns: ipcns,
                    fsContext: fsContext,
                    fdTbl: fdTbl,
                    blocker: blocker,
                    //Blocker::New(s_ptr as u64),
                    thread: Some(nt.clone()),
                    haveSyscallReturn: false,
                    syscallRestartBlock: None,
                    futexMgr: futexMgr,
                    ioUsage: ioUsage,
                    sched: sched,
                    exiting: false,
                    perfcounters: None, //Some(THREAD_COUNTS.lock().NewCounters()),
                    savefpsate: false,
                    archfpstate:  Some(Default::default()),
                    queueId: AtomicUsize::new(0),
                    guard: Guard::default(),
                },
            );
        }

        let curr = Self::Current();
        let new = unsafe { &mut *taskPtr };

        //new.PerfGoto(PerfType::Blocked);
        //new.PerfGoto(PerfType::User);
        CreateCloneTask(curr, new, userSp);

        return Ok((cPid, taskPtr));
    }

    pub fn UnshareFdTable(&mut self, maxFd: i32) {
        let newfdtbl = self.fdTbl.Fork(maxFd);
        self.fdTbl = newfdtbl;
    }

    pub fn Unshare(&mut self, opts: &SharingOptions) -> Result<()> {
        // In Linux unshare(2), NewThreadGroup implies NewSignalHandlers and
        // NewSignalHandlers implies NewAddressSpace. All three flags are no-ops if
        // t is the only task using its MM, which due to clone(2)'s rules imply
        // that it is also the only task using its signal handlers / in its thread
        // group, and cause EINVAL to be returned otherwise.
        //
        // Since we don't count the number of tasks using each address space or set
        // of signal handlers, we reject NewSignalHandlers and NewAddressSpace
        // altogether, and interpret NewThreadGroup as requiring that t be the only
        // member of its thread group. This seems to be logically coherent, in the
        // sense that clone(2) allows a task to share signal handlers and address
        // spaces with tasks in other thread groups.
        if opts.NewAddressSpace || opts.NewSignalHandlers {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let t = self.Thread();
        let tg = t.lock().tg.clone();
        let signallock = tg.lock().signalLock.clone();
        if opts.NewThreadGroup {
            let _s = signallock.lock();
            if tg.lock().tasksCount != 1 {
                return Err(Error::SysError(SysErr::EINVAL));
                // This isn't racy because we're the only living task, and therefore
                // the only task capable of creating new ones, in our thread group.
            }
        }

        if opts.NewUserNamespace {
            if self.IsChrooted() {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let creds = t.Credentials();
            let newUserNs = creds.NewChildUserNamespace()?;
            t.SetUserNamespace(&newUserNs)?;
            self.creds = creds;
        }

        let creds = self.creds.clone();
        let haveCapSysAdmin = t.HasCapability(Capability::CAP_SYS_ADMIN);
        if opts.NewPIDNamespace {
            if !haveCapSysAdmin {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let userns = creds.lock().UserNamespace.clone();
            let pidns = tg.PIDNamespace();
            t.lock().childPIDNamespace = Some(pidns.NewChild(&userns));
        }

        let mut tlock = t.lock();
        if opts.NewNetworkNamespace {
            if !haveCapSysAdmin {
                return Err(Error::SysError(SysErr::EPERM));
            }

            tlock.netns = true;
        }

        if opts.NewUTSNamespace {
            if !haveCapSysAdmin {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let userns = creds.lock().UserNamespace.clone();
            let utsns = self.utsns.clone();
            self.utsns = utsns.Fork(&userns);
            tlock.utsns = self.utsns.clone();
        }

        if opts.NewIPCNamespace {
            if !haveCapSysAdmin {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let userns = creds.lock().UserNamespace.clone();
            self.ipcns = IPCNamespace::New(&userns);
            tlock.ipcns = self.ipcns.clone();
        }

        if opts.NewFiles {
            let fdtbl = self.fdTbl.clone();
            self.fdTbl = fdtbl.Fork(i32::MAX);
            tlock.fdTbl = self.fdTbl.clone();
        }

        if opts.NewFSContext {
            let fsc = self.fsContext.clone();
            self.fsContext = fsc.Fork();
            tlock.fsc = self.fsContext.clone();
        }

        return Ok(());
    }
}

pub fn CreateCloneTask(fromTask: &Task, toTask: &mut Task, userSp: u64) {
    let mut from = fromTask.GetKernelSp();
    let fromSp = fromTask.GetPtRegs() as *const _ as u64;

    let mut to = toTask.GetKernelSp();
    let toPtRegs = toTask.GetPtRegs();

    unsafe {
        while from >= fromSp {
            *(to as *mut u64) = *(from as *const u64);
            from -= 8;
            to -= 8;
        }

        toTask.SetReady(1);
        toTask.context.set_tls(fromTask.context.get_tls());
        toTask.context.set_sp(toTask.GetPtRegs() as *const _ as u64 - 8);
        toTask.context.set_para(userSp);
        toTask.savefpsate = true;
        toTask.archfpstate = Some(Box::new(
            fromTask.archfpstate.as_ref().unwrap().Fork(),
        ));
        // 1. set sys_clone return value to 0 to indicate child.
        // 2. set child pc (return addr of current call frame)
        //      to child_clone function
        #[cfg(target_arch = "x86_64")]
        {
            toPtRegs.rax = 0;
			*(toTask.context.rsp as *mut u64) = child_clone as u64;
        }
        #[cfg(target_arch = "aarch64")]
        {
            toPtRegs.regs[0] = 0;
            toTask.context.set_pc(child_clone as u64);
        }
        toPtRegs.set_stack_pointer(userSp);
    }
}

pub struct VforkStop {}

impl TaskStop for VforkStop {
    fn Type(&self) -> TaskStopType {
        return TaskStopType::VFORKSTOP;
    }

    fn Killable(&self) -> bool {
        return true;
    }
}
