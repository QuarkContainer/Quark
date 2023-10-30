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
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

pub use xmas_elf::header::HeaderPt2;
pub use xmas_elf::program::{Flags, ProgramHeader, ProgramHeader64};
pub use xmas_elf::sections::Rela;
pub use xmas_elf::symbol_table::{Entry, Entry64};
pub use xmas_elf::{P32, P64};

use super::super::asm::*;
use super::super::kernel::cpuset::*;
use super::super::loader::loader::*;
use super::super::memmgr::mm::*;
use super::super::qlib::common::*;
use super::super::qlib::linux::rusage::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::path::*;
use super::super::qlib::LoadAddr;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::super::taskMgr::*;
use super::super::threadmgr::task_clone::*;
use super::super::threadmgr::task_exec::*;
use super::super::threadmgr::task_exit::*;
use super::super::threadmgr::task_sched::*;
use super::super::threadmgr::thread::*;
use super::super::vcpu::*;
use super::super::SignalDef::*;
use super::super::SHARESPACE;
use super::sys_rusage::*;
use arch::__arch::arch_def::ArchFPState;

#[derive(Default, Debug)]
pub struct ElfInfo {
    pub interpreter: String,
    pub entry: u64,
    pub start: u64,
    pub end: u64,
    pub phdrAddr: u64,
    pub phdrSize: usize,
    pub phdrNum: usize,
    pub addrs: Vec<LoadAddr>,
}

// Getppid implements linux syscall getppid(2).
pub fn SysGetPpid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let t = task.Thread();

    let parent = match t.Parent() {
        None => return Ok(0),
        Some(p) => p,
    };

    let pidns = t.PIDNamespace();
    let ptg = parent.ThreadGroup();
    let pid = pidns.IDOfThreadGroup(&ptg);
    return Ok(pid as i64);
}

// Getpid implements linux syscall getpid(2).
pub fn SysGetPid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let pid = task.Thread().ThreadGroup().ID();
    return Ok(pid as i64);
}

// Gettid implements linux syscall gettid(2).
pub fn SysGetTid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let tid = task.Thread().ThreadID();
    return Ok(tid as i64);
}

pub fn SysSetRobustList(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let robust_list_head = args.arg0 as u64;
    let robust_list_len = args.arg1 as i64;

    if robust_list_len as u64 != ROBUST_LIST_LEN {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let thread = task.Thread();
    thread.lock().robust_list_head = robust_list_head;
    return Ok(0);
}

pub fn SysGetRobustList(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tid = args.arg0 as i32;
    let headAddr = args.arg1 as u64;
    let lenAddr = args.arg2 as u64;

    let thread = if tid == 0 {
        task.Thread()
    } else {
        match task.Thread().PIDNamespace().TaskWithID(tid) {
            None => return Err(Error::SysError(SysErr::ESRCH)),
            Some(t) => t,
        }
    };

    // todo: check whether the current thread has permission to get the RobustList from target thread
    //*task.GetTypeMut::<u64>(headAddr)? = thread.lock().robust_list_head;
    //*task.GetTypeMut::<i64>(lenAddr)? = ROBUST_LIST_LEN as i64;

    task.CopyOutObj(&(thread.lock().robust_list_head as u64), headAddr)?;
    task.CopyOutObj(&(ROBUST_LIST_LEN as i64), lenAddr)?;

    return Ok(0);
}

// ExecMaxTotalSize is the maximum length of all argv and envv entries.
//
// N.B. The behavior here is different than Linux. Linux provides a limit on
// individual arguments of 32 pages, and an aggregate limit of at least 32 pages
// but otherwise bounded by min(stack size / 4, 8 MB * 3 / 4). We don't implement
// any behavior based on the stack size, and instead provide a fixed hard-limit of
// 2 MB (which should work well given that 8 MB stack limits are common).
const EXEC_MAX_TOTAL_SIZE: usize = 2 * 1024 * 1024;

// ExecMaxElemSize is the maximum length of a single argv or envv entry.
const EXEC_MAX_ELEM_SIZE: usize = 32 * MemoryDef::PAGE_SIZE as usize;

pub fn ExecvFilleName(task: &mut Task, dirfd: i32, filename: &str, flags: i32) -> Result<String> {
    if flags & !(ATType::AT_EMPTY_PATH | ATType::AT_SYMLINK_NOFOLLOW) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let atEmptyPath = flags & ATType::AT_EMPTY_PATH != 0;

    let resolveFinal = flags & ATType::AT_SYMLINK_NOFOLLOW != 0;

    let cwd;
    if dirfd == ATType::AT_FDCWD || IsAbs(filename) {
        let fscontex = task.fsContext.clone();
        cwd = fscontex.lock().cwd.clone();
    } else {
        let f = task.GetFile(dirfd)?;
        cwd = f.Dirent.clone();
        let inode = cwd.Inode();
        if atEmptyPath && filename.len() == 0 {
            inode.CheckPermission(
                task,
                &PermMask {
                    read: true,
                    execute: true,
                    ..Default::default()
                },
            )?;
            return Ok(f.Dirent.MyFullName());
        } else {
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }
        }
    };

    let fscontex = task.fsContext.clone();
    let root = fscontex.lock().root.clone();
    let mut remainingTraversals = 40;
    let d = task.mountNS.FindDirent(
        task,
        &root,
        Some(cwd),
        filename,
        &mut remainingTraversals,
        resolveFinal,
    )?;

    return Ok(d.MyFullName());
}

pub fn Execvat(
    task: &mut Task,
    dirfd: i32,
    filenameAddr: u64,
    argvAddr: u64,
    envvAddr: u64,
    flags: i32,
) -> Result<i64> {
    let (entry, usersp, kernelsp) = {
        let (fileName, err) = task.CopyInString(filenameAddr, PATH_MAX);
        match err {
            Err(e) => return Err(e),
            _ => (),
        }

        let mut argv =
            task.CopyInVector(argvAddr, EXEC_MAX_ELEM_SIZE, EXEC_MAX_TOTAL_SIZE as i32)?;
        let envv = task.CopyInVector(envvAddr, EXEC_MAX_ELEM_SIZE, EXEC_MAX_TOTAL_SIZE as i32)?;

        if argv.len() == 0 {
            argv.push(fileName.clone())
        };
        let mut cmd = format!("");
        for arg in &argv {
            cmd += &arg;
            cmd += " ";
        }

        let mut envs = format!("");
        for env in &envv {
            envs += &env;
            envs += " ";
        }
        info!("in the execve: the cmd is {} \n envs is {:?}", &cmd, &envs);

        let fileName = ExecvFilleName(task, dirfd, &fileName, flags)?;

        {
            let t = task.Thread().clone();
            let tg = t.lock().tg.clone();
            let pidns = tg.PIDNamespace();
            let owner = pidns.lock().owner.clone();
            let signallock = tg.lock().signalLock.clone();
            {
                let ol = owner.WriteLock();
                let sl = signallock.lock();

                let exiting = tg.lock().exiting;
                let execing = tg.lock().execing.Upgrade();

                if exiting || execing.is_some() {
                    // We lost to a racing group-exit, kill, or exec from another thread
                    // and should just exit.
                    return Err(Error::SysError(SysErr::EINTR));
                }

                // Cancel any racing group stops.
                tg.lock().endGroupStopLocked(false);

                // If the task has any siblings, they have to exit before the exec can
                // continue.
                tg.lock().execing = t.Downgrade();

                let taskCnt = tg.lock().tasks.len();
                if taskCnt != 1 {
                    // "[All] other threads except the thread group leader report death as
                    // if they exited via _exit(2) with exit code 0." - ptrace(2)
                    let tasks: Vec<_> = tg.lock().tasks.iter().cloned().collect();
                    for sibling in &tasks {
                        if t != sibling.clone() {
                            sibling.lock().killLocked();
                        }
                    }
                    // The last sibling to exit will wake t.
                    t.lock().beginInternalStopLocked(&Arc::new(ExecStop {}));

                    core::mem::drop(sl);
                    core::mem::drop(ol);
                    task.DoStop();
                }
            }

            let mut its = Vec::new();
            {
                let _l = owner.WriteLock();
                tg.lock().execing = ThreadWeak::default();
                if t.lock().killed() {
                    //return (*runInterrupt)(nil)
                    return Err(Error::SysError(SysErr::EINTR));
                }

                t.promoteLocked();

                // "POSIX timers are not preserved (timer_create(2))." - execve(2). Handle
                // this first since POSIX timers are protected by the signal mutex, which
                // we're about to change. Note that we have to stop and destroy timers
                // without holding any mutexes to avoid circular lock ordering.
                {
                    let _s = signallock.lock();

                    for (_, it) in &tg.lock().timers {
                        its.push(it.clone());
                    }

                    tg.lock().timers.clear();
                }
            }

            for it in its {
                it.DestroyTimer();
            }

            {
                let _l = owner.WriteLock();
                let sh = tg.lock().signalHandlers.clone();
                // "During an execve(2), the dispositions of handled signals are reset to
                // the default; the dispositions of ignored signals are left unchanged. ...
                // [The] signal mask is preserved across execve(2). ... [The] pending
                // signal set is preserved across an execve(2)." - signal(7)
                //
                // Details:
                //
                // - If the thread group is sharing its signal handlers with another thread
                // group via CLONE_SIGHAND, execve forces the signal handlers to be copied
                // (see Linux's fs/exec.c:de_thread). We're not reference-counting signal
                // handlers, so we always make a copy.
                //
                // - "Disposition" only means sigaction::sa_handler/sa_sigaction; flags,
                // restorer (if present), and mask are always reset. (See Linux's
                // fs/exec.c:setup_new_exec => kernel/signal.c:flush_signal_handlers.)
                tg.lock().signalHandlers = sh.CopyForExec();
                // "Any alternate signal stack is not preserved (sigaltstack(2))." - execve(2)
                t.lock().signalStack = SignalStack::default();
                task.signalStack = SignalStack::default();
                // "The termination signal is reset to SIGCHLD (see clone(2))."
                tg.lock().terminationSignal = Signal(Signal::SIGCHLD);
                // execed indicates that the process can no longer join a process group
                // in some scenarios (namely, the parent call setpgid(2) on the child).
                // See the JoinProcessGroup function in sessions.go for more context.
                tg.lock().execed = true;
            }

            let fdtbl = t.lock().fdTbl.clone();
            fdtbl.RemoveCloseOnExec();

            t.ExitRobustList(task);

            t.lock().updateCredsForExecLocked();

            t.UnstopVforkParent();

            SetTLS(0);
            task.tidInfo = TidInfo::default();
            task.context.set_tls(0);
            task.archfpstate = Some(Box::new(ArchFPState::default()));

            let newMM = MemoryManager::Init(false);
            let oldMM = task.mm.clone();
            *newMM.metadata.lock() = oldMM.metadata.lock().Fork();
            newMM.SetVcpu(GetVcpuId());
            task.mm = newMM.clone();
            task.futexMgr = task.futexMgr.Fork();
            task.Thread().lock().memoryMgr = newMM;
            if !SHARESPACE.config.read().KernelPagetable {
                task.SwitchPageTable();
            }

            // make the old mm exist before switch pagetable
            core::mem::drop(oldMM);
        }

        let extraAxv = Vec::new();
        Load(task, &fileName, &mut argv, &envv, &extraAxv)?
    };

    //need to clean object on stack before enter_user as the stack will be destroyed
    task.AccountTaskEnter(SchedState::RunningApp);

    EnterUser(entry, usersp, kernelsp);
}

// Execveat implements linux syscall execveat(2).
pub fn SysExecveat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let filenameAddr = args.arg1 as u64;
    let argvAddr = args.arg2 as u64;
    let envvAddr = args.arg3 as u64;
    let flags = args.arg4 as i32;

    return Execvat(task, dirfd, filenameAddr, argvAddr, envvAddr, flags);
}

// Execve implements linux syscall execve(2).
pub fn SysExecve(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let filenameAddr = args.arg0 as u64;
    let argvAddr = args.arg1 as u64;
    let envvAddr = args.arg2 as u64;

    return Execvat(task, ATType::AT_FDCWD, filenameAddr, argvAddr, envvAddr, 0);
}

pub fn SysExit(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let exitcode = args.arg0 as i32;

    let exitStatus = ExitStatus::New(exitcode as i32, 0);

    task.Thread().PrepareExit(exitStatus);
    return Err(Error::SysCallRetCtrl(TaskRunState::RunThreadExit));
}

pub fn SysExitThreadGroup(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let exitcode = args.arg0 as i32;

    let exitStatus = ExitStatus::New(exitcode as i32, 0);

    task.Thread().PrepareGroupExit(exitStatus);
    return Err(Error::SysCallRetCtrl(TaskRunState::RunExit));
}

// Clone implements linux syscall clone(2).
// sys_clone has so many flavors. We implement the default one in linux 3.11
// x86_64:
//    sys_clone(clone_flags, newsp, parent_tidptr, child_tidptr, tls_val)
pub fn SysClone(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let flags = args.arg0;
    let cStack = args.arg1;
    let pTid = args.arg2;
    let cTid = args.arg3;
    let tls = args.arg4;

    let pid = task.Clone(flags, cStack, pTid, cTid, tls)?;
    return Ok(pid as i64);
}

// Fork implements Linux syscall fork(2).
pub fn SysFork(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let pid = task.Clone(Signal::SIGCHLD as u64, 0, 0, 0, 0)?;
    return Ok(pid as i64);
}

pub fn SysVfork(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let pid = task.Clone(
        LibcConst::CLONE_VM | LibcConst::CLONE_VFORK | Signal::SIGCHLD as u64,
        0,
        0,
        0,
        0,
    )?;
    return Ok(pid as i64);
}

// Wait4 implements linux syscall wait4(2).
pub fn SysWait4(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    //error!("SysWait4 {:x?}", args);
    let pid = args.arg0;
    let status = args.arg1;
    let option = args.arg2 as u32;
    let rusage = args.arg3;

    return wait4(task, pid as i32, status, option, rusage);
}

// Waitid implements linux syscall waitid(2).
pub fn SysWaitid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    //error!("SysWaitid {:x?}", args);
    let idtype = args.arg0 as i32;
    let id = args.arg1 as i32;
    let infop = args.arg2;
    let options = args.arg3 as u32;
    let rusageAddr = args.arg4;

    if options
        & !(WaitOption::WNOHANG
            | WaitOption::WEXITED
            | WaitOption::WSTOPPED
            | WaitOption::WCONTINUED
            | WaitOption::WNOWAIT
            | WaitOption::WNOTHREAD
            | WaitOption::WALL
            | WaitOption::WCLONE)
        != 0
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if options & (WaitOption::WEXITED | WaitOption::WSTOPPED | WaitOption::WCONTINUED) == 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut wopts = WaitOptions {
        Events: TaskEvent::TRACE_STOP as EventMask,
        ConsumeEvent: options & WaitOption::WNOWAIT == 0,
        ..Default::default()
    };

    match idtype {
        IDType::P_ALL | IDType::P_PID => {
            wopts.SpecificTID = id;
        }
        IDType::P_PGID => {
            wopts.SpecificPGID = id;
        }
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }

    parseCommonWaitOptions(&mut wopts, options)?;
    if options & WaitOption::WEXITED != 0 {
        wopts.Events |= EVENT_EXIT;
    }

    if options & WaitOption::WSTOPPED != 0 {
        wopts.Events |= EVENT_CHILD_GROUP_STOP;
    }

    let wr = match task.Wait(&wopts) {
        Err(Error::ErrNoWaitableEvent) => {
            // "If WNOHANG was specified in options and there were no children
            // in a waitable state, then waitid() returns 0 immediately and the
            // state of the siginfo_t structure pointed to by infop is
            // unspecified." - waitid(2). But Linux's waitid actually zeroes
            // out the fields it would set for a successful waitid in this case
            // as well.
            if infop != 0 {
                let si = SignalInfo::default();
                task.CopyOutObj(&si, infop)?;
            }

            return Ok(0);
        }
        Err(e) => return Err(e),
        Ok(wr) => wr,
    };

    if rusageAddr != 0 {
        let ru = GetUsage(task, RUSAGE_BOTH)?;
        task.CopyOutObj(&ru, rusageAddr)?;
    }

    if infop == 0 {
        return Ok(0);
    }

    let mut si = SignalInfo {
        Signo: Signal::SIGCHLD,
        ..Default::default()
    };
    let sigChld = si.SigChld();
    sigChld.pid = wr.TID;
    sigChld.uid = wr.UID.0;

    // TODO: convert kernel.ExitStatus to functions and make
    // WaitResult.Status a linux.WaitStatus

    let mut siCode = 0;
    let s = WaitStatus(wr.Status);
    if s.Exited() {
        siCode = SignalInfo::CLD_EXITED;
        sigChld.status = s.ExitStatus();
    } else if s.Signaled() {
        siCode = SignalInfo::CLD_KILLED;
        sigChld.status = s.Signal();
    } else if s.CoreDump() {
        siCode = SignalInfo::CLD_DUMPED;
        sigChld.status = s.Signal();
    } else if s.Stopped() {
        if wr.Event == EVENT_TRACEE_STOP {
            siCode = SignalInfo::CLD_TRAPPED;
            sigChld.status = s.TrapCause();
        } else {
            siCode = SignalInfo::CLD_TRAPPED;
            sigChld.status = s.StopSignal();
        }
    } else if s.Continued() {
        siCode = SignalInfo::CLD_CONTINUED;
        sigChld.status = Signal::SIGCONT;
    } else {
        info!("waitid got incomprehensible wait status {:b}", s.0)
    }

    si.Code = siCode;

    task.CopyOutObj(&si, infop)?;
    return Ok(0);
}

pub fn wait4(task: &Task, pid: i32, statusAddr: u64, options: u32, rusageAddr: u64) -> Result<i64> {
    if options
        & !(WaitOption::WNOHANG
            | WaitOption::WUNTRACED
            | WaitOption::WCONTINUED
            | WaitOption::WNOTHREAD
            | WaitOption::WALL
            | WaitOption::WCLONE)
        != 0
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut wopts = WaitOptions {
        Events: (TaskEvent::EXIT | TaskEvent::TRACE_STOP) as u64,
        ConsumeEvent: true,
        ..Default::default()
    };

    // There are four cases to consider:
    //
    // pid < -1    any child process whose process group ID is equal to the absolute value of pid
    // pid == -1   any child process
    // pid == 0    any child process whose process group ID is equal to that of the calling process
    // pid > 0     the child whose process ID is equal to the value of pid
    if pid < -1 {
        wopts.SpecificPGID = -pid;
    } else if pid == -1 {
        // Any process is the default.
    } else if pid == 0 {
        let pg = task.Thread().ThreadGroup().ProcessGroup();
        let pidns = task.Thread().PIDNamespace();
        wopts.SpecificPGID = pidns.IDOfProcessGroup(&pg.unwrap());
    } else {
        wopts.SpecificTID = pid;
    }

    parseCommonWaitOptions(&mut wopts, options)?;

    if options & WaitOption::WUNTRACED != 0 {
        wopts.Events |= EVENT_CHILD_GROUP_STOP;
    }

    let wr = match task.Wait(&wopts) {
        Err(Error::ErrNoWaitableEvent) => return Ok(0),
        Err(e) => return Err(e),
        Ok(wr) => wr,
    };

    //error!("wait4 status is {:x?}", wr.Status);
    if statusAddr != 0 {
        //task.CopyInObject(statusAddr, &wr.Status as * const _ as u64, 4)?;
        task.CopyOutObj(&wr.Status, statusAddr)?;
    }

    if rusageAddr != 0 {
        let ru = GetUsage(task, RUSAGE_BOTH)?;
        task.CopyOutObj(&ru, rusageAddr)?;
    }

    return Ok(wr.TID as i64);
}

pub fn parseCommonWaitOptions(wopts: &mut WaitOptions, options: u32) -> Result<()> {
    let tmp = options & (WaitOption::WCLONE | WaitOption::WALL);
    if tmp == 0 {
        wopts.NonCloneTasks = true;
    } else if tmp == WaitOption::WCLONE {
        wopts.CloneTasks = true;
    } else if tmp == WaitOption::WALL {
        wopts.CloneTasks = true;
        wopts.NonCloneTasks = true;
    } else {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if options & WaitOption::WCONTINUED != 0 {
        wopts.Events |= TaskEvent::GROUP_CONTINUE as u64;
    }

    if options & WaitOption::WNOHANG == 0 {
        wopts.BlockInterruptErr = Some(Error::SysError(SysErr::ERESTARTSYS));
    }

    if options & WaitOption::WNOTHREAD == 0 {
        wopts.SiblingChildren = true;
    }

    return Ok(());
}

// SetTidAddress implements linux syscall set_tid_address(2).
pub fn SysSetTidAddr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    task.SetClearTID(addr);
    let tid = task.Thread().ThreadID();
    return Ok(tid as i64);
}

// Unshare implements linux syscall unshare(2).
pub fn SysUnshare(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let flags = args.arg0 as i32;

    let mut opts = SharingOptions {
        NewAddressSpace: flags & CloneOp::CLONE_VM == CloneOp::CLONE_VM,
        NewSignalHandlers: flags & CloneOp::CLONE_SIGHAND == CloneOp::CLONE_SIGHAND,
        NewThreadGroup: flags & CloneOp::CLONE_THREAD == CloneOp::CLONE_THREAD,
        NewPIDNamespace: flags & CloneOp::CLONE_NEWPID == CloneOp::CLONE_NEWPID,
        NewUserNamespace: flags & CloneOp::CLONE_NEWUSER == CloneOp::CLONE_NEWUSER,
        NewNetworkNamespace: flags & CloneOp::CLONE_NEWNET == CloneOp::CLONE_NEWNET,
        NewFiles: flags & CloneOp::CLONE_FILES == CloneOp::CLONE_FILES,
        NewFSContext: flags & CloneOp::CLONE_FS == CloneOp::CLONE_FS,
        NewUTSNamespace: flags & CloneOp::CLONE_NEWUTS == CloneOp::CLONE_NEWUTS,
        NewIPCNamespace: flags & CloneOp::CLONE_NEWIPC == CloneOp::CLONE_NEWIPC,
        ..Default::default()
    };

    // "CLONE_NEWPID automatically implies CLONE_THREAD as well." - unshare(2)
    if opts.NewPIDNamespace {
        opts.NewThreadGroup = true;
    }

    // "... specifying CLONE_NEWUSER automatically implies CLONE_THREAD. Since
    // Linux 3.9, CLONE_NEWUSER also automatically implies CLONE_FS."
    if opts.NewUserNamespace {
        opts.NewThreadGroup = true;
        opts.NewFSContext = true;
        panic!("Doesn't support create new usernamespace...");
    }

    task.Unshare(&opts)?;
    return Ok(0);
}

// SchedYield implements linux syscall sched_yield(2).
pub fn SysScheduleYield(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    Yield();

    return Ok(0);
}

// SchedSetaffinity implements linux syscall sched_setaffinity(2).
pub fn SysSchedSetaffinity(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tid = args.arg0 as i32;
    let mut size = args.arg1 as usize;
    let maskAddr = args.arg2 as u64;

    let t = if tid == 0 {
        task.Thread().clone()
    } else {
        let pidns = task.Thread().PIDNamespace();
        match pidns.TaskWithID(tid) {
            None => return Err(Error::SysError(SysErr::ESRCH)),
            Some(t) => t,
        }
    };

    let mut mask = CPUSet::New(task.Thread().lock().k.ApplicationCores() as usize);
    if size > mask.Size() {
        size = mask.Size();
    }

    let arr = task.CopyInVec::<u8>(maskAddr, size)?;
    for i in 0..size {
        mask.0[i] = arr[i];
    }

    t.SetCPUMask(mask)?;
    return Ok(0);
}

// SchedGetaffinity implements linux syscall sched_getaffinity(2).
pub fn SysSchedGetaffinity(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tid = args.arg0 as i32;
    let size = args.arg1 as usize;
    let maskAddr = args.arg2 as u64;

    // This limitation is because linux stores the cpumask
    // in an array of "unsigned long" so the buffer needs to
    // be a multiple of the word size.
    if size & (8 - 1) > 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let t = if tid == 0 {
        task.Thread()
    } else {
        let pidns = task.Thread().PIDNamespace();
        match pidns.TaskWithID(tid) {
            None => return Err(Error::SysError(SysErr::ESRCH)),
            Some(t) => t,
        }
    };

    let mask = t.CPUMask();
    // The buffer needs to be big enough to hold a cpumask with
    // all possible cpus.
    if size < mask.Size() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // info!("SysSchedGetaffinity cpu count is {}", mask.NumCPUs());

    task.CopyOutSlice(&mask.0[..], maskAddr, mask.0.len())?;
    // NOTE: The syscall interface is slightly different than the glibc
    // interface. The raw sched_getaffinity syscall returns the number of
    // bytes used to represent a cpu mask.
    return Ok(mask.Size() as i64);
}

// Getcpu implements linux syscall getcpu(2).
pub fn SysGetcpu(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let cpu = args.arg0 as u64;
    let node = args.arg1 as u64;
    // third argument to this system call is nowadays unused.

    if cpu != 0 {
        let id = task.CPU();
        //*task.GetTypeMut(cpu)? = id;
        task.CopyOutObj(&id, cpu)?;
    }

    if node != 0 {
        let val: u32 = 0;
        //*task.GetTypeMut(node)? = val;
        task.CopyOutObj(&val, node)?;
    }

    return Ok(0);
}

// Setpgid implements the linux syscall setpgid(2).
pub fn SysSetpgid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    // Note that throughout this function, pgid is interpreted with respect
    // to t's namespace, not with respect to the selected ThreadGroup's
    // namespace (which may be different).
    let pid = args.arg0 as i32;
    let mut pgid = args.arg1 as i32;

    // "If pid is zero, then the process ID of the calling process is used."
    let mut tg = task.Thread().ThreadGroup();
    let pidns = tg.PIDNamespace();
    if pid != 0 {
        let ot = match pidns.TaskWithID(pid) {
            None => return Err(Error::SysError(SysErr::ESRCH)),
            Some(t) => t,
        };

        tg = ot.ThreadGroup();
        let tgLeader = tg.Leader().unwrap();
        if tgLeader != ot {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // Setpgid only operates on child threadgroups.
        if tg != task.Thread().ThreadGroup()
            && (tgLeader.Parent().is_none()
                || tgLeader.Parent().unwrap().ThreadGroup() != task.Thread().ThreadGroup())
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }

    // "If pgid is zero, then the PGID of the process specified by pid is made
    // the same as its process ID."
    let defaultPGID = pidns.IDOfThreadGroup(&tg);
    if pgid == 0 {
        pgid = defaultPGID;
    } else if pgid < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // If the pgid is the same as the group, then create a new one. Otherwise,
    // we attempt to join an existing process group.
    if pgid == defaultPGID {
        // For convenience, errors line up with Linux syscall API.
        match tg.CreateProcessGroup() {
            Err(e) => {
                let pg = tg.ProcessGroup().unwrap();
                if pidns.IDOfProcessGroup(&pg) == defaultPGID {
                    return Ok(0);
                }

                return Err(e);
            }
            _ => (),
        }
    } else {
        let localtg = task.Thread().ThreadGroup();
        match tg.JoinProcessGroup(&pidns, pgid, tg != localtg) {
            Err(e) => {
                let pg = tg.ProcessGroup().unwrap();
                if pidns.IDOfProcessGroup(&pg) == pgid {
                    return Ok(0);
                }
                return Err(e);
            }
            Ok(_) => (),
        }
    }

    // Success.
    return Ok(0);
}

// Getpgrp implements the linux syscall getpgrp(2).
pub fn SysGetpgrp(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let pidns = task.Thread().PIDNamespace();
    let tg = task.Thread().ThreadGroup();
    let pg = tg.ProcessGroup().unwrap();
    let pgid = pidns.IDOfProcessGroup(&pg);

    return Ok(pgid as i64);
}

// Getpgid implements the linux syscall getpgid(2).
pub fn SysGetpgid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tid = args.arg0 as i32;
    if tid == 0 {
        return SysGetpgrp(task, args);
    }

    let pidns = task.Thread().PIDNamespace();
    let target = match pidns.TaskWithID(tid) {
        None => return Err(Error::SysError(SysErr::ESRCH)),
        Some(t) => t,
    };

    let tg = target.ThreadGroup();
    let pg = tg.ProcessGroup().unwrap();
    let id = pidns.IDOfProcessGroup(&pg);

    return Ok(id as i64);
}

// Setsid implements the linux syscall setsid(2).
pub fn SysSetsid(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let tg = task.Thread().ThreadGroup();
    tg.CreateSessoin()?;
    return Ok(0);
}

// Getsid implements the linux syscall getsid(2).
pub fn SysGetsid(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tid = args.arg0 as i32;

    let pidns = task.Thread().PIDNamespace();
    if tid == 0 {
        let tg = task.Thread().ThreadGroup();
        let session = tg.Session().unwrap();
        return Ok(pidns.IDOfSession(&session) as i64);
    }

    let target = match pidns.TaskWithID(tid) {
        None => return Err(Error::SysError(SysErr::ESRCH)),
        Some(t) => t,
    };

    let tg = target.ThreadGroup();
    let session = tg.Session().unwrap();
    return Ok(pidns.IDOfSession(&session) as i64);
}

// Getpriority pretends to implement the linux syscall getpriority(2).
//
// This is a stub; real priorities require a full scheduler.
pub fn SysGetpriority(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let which = args.arg0 as i32;
    let who = args.arg1 as i32;

    match which as u64 {
        LibcConst::PRIO_PROCESS => {
            let t = if who == 0 {
                task.Thread()
            } else {
                let pidns = task.Thread().PIDNamespace();
                match pidns.TaskWithID(who) {
                    None => return Err(Error::SysError(SysErr::ESRCH)),
                    Some(t) => t,
                }
            };

            return Ok((20 - t.Niceness()) as i64);
        }
        LibcConst::PRIO_USER | LibcConst::PRIO_PGRP => {
            // PRIO_USER and PRIO_PGRP have no further implementation yet.
            return Ok(0);
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }
}

// Setpriority pretends to implement the linux syscall setpriority(2).
//
// This is a stub; real priorities require a full scheduler.
pub fn SysSetpriority(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let which = args.arg0 as i32;
    let who = args.arg1 as i32;
    let mut niceval = args.arg2 as i32;

    // In the kernel's implementation, values outside the range
    // of [-20, 19] are truncated to these minimum and maximum
    // values.
    if niceval < -20
    /* min niceval */
    {
        niceval = -20
    } else if niceval > 19
    /* max niceval */
    {
        niceval = 19
    }

    match which as u64 {
        LibcConst::PRIO_PROCESS => {
            let t = if who == 0 {
                task.Thread().clone()
            } else {
                let pidns = task.Thread().PIDNamespace();
                match pidns.TaskWithID(who) {
                    None => return Err(Error::SysError(SysErr::ESRCH)),
                    Some(t) => t,
                }
            };

            t.SetNiceness(niceval);
            return Ok(0);
        }
        LibcConst::PRIO_USER | LibcConst::PRIO_PGRP => {
            // PRIO_USER and PRIO_PGRP have no further implementation yet.
            return Ok(0);
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }
}
