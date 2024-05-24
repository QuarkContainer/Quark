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

use super::super::fs::flags::*;
use super::super::kernel::fd_table::*;
use super::super::kernel::signalfd::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::super::threadmgr::pid_namespace::*;
use super::super::threadmgr::thread::*;
use super::super::SignalDef::*;
use super::sys_poll::*;

// "For a process to have permission to send a signal it must
// - either be privileged (CAP_KILL), or
// - the real or effective user ID of the sending process must be equal to the
// real or saved set-user-ID of the target process.
//
// In the case of SIGCONT it suffices when the sending and receiving processes
// belong to the same session." - kill(2)
//
// Equivalent to kernel/signal.c:check_kill_permission.
fn mayKill(t: &Thread, target: &Thread, sig: Signal) -> bool {
    // kernel/signal.c:check_kill_permission also allows a signal if the
    // sending and receiving tasks share a thread group, which is not
    // mentioned in kill(2) since kill does not allow task-level
    // granularity in signal sending.
    let tg = t.ThreadGroup();
    let ttg = target.ThreadGroup();
    if tg == ttg {
        return true;
    }

    if t.HasCapabilityIn(Capability::CAP_KILL, &target.UserNamespace()) {
        return true;
    }

    let creds = t.Credentials();
    let tcreds = target.Credentials();

    let EffectiveKUID = creds.lock().EffectiveKUID;
    let tSavedKUID = tcreds.lock().SavedKUID;
    let tRealKUID = tcreds.lock().RealKUID;
    let RealKUID = creds.lock().RealKUID;
    if EffectiveKUID == tSavedKUID
        || EffectiveKUID == tRealKUID
        || RealKUID == tSavedKUID
        || RealKUID == tRealKUID
    {
        return true;
    }

    let session = tg.Session();
    let tsession = ttg.Session();
    if sig.0 == Signal::SIGCONT && session == tsession {
        return true;
    }

    return false;
}

pub fn SysRtSigaction(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let signum = args.arg0 as u64;
    let sigAction = args.arg1 as u64;
    let oldSigAction = args.arg2 as u64;
    let sigsetsize = args.arg3 as u32;

    if sigsetsize != SIGNAL_SET_SIZE as u32 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut newactptr: Option<SigAct> = None;
    if sigAction != 0 {
        let sigAction1 = task.CopyInObj::<SigAct>(sigAction)?;
        newactptr = Some(sigAction1);
    }

    let t = task.Thread();
    let tg = t.ThreadGroup();
    let oldact = tg.SetSignalAct(Signal(signum as i32), &newactptr)?;

    if oldSigAction != 0 {
        //*task.GetTypeMut(oldSigAction)? = oldact;
        task.CopyOutObj(&oldact, oldSigAction)?;
    }

    return Ok(0);
}

pub fn SysRtSigreturn(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return task.SignalReturn(true);
}

pub fn SysRtSigProcMask(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let how = args.arg0 as u64;
    let setaddr = args.arg1 as u64;
    let oldaddr = args.arg2 as u64;
    let sigsetsize = args.arg3 as u64;

    if sigsetsize != 8 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let t = task.Thread();
    let oldMask = t.SignalMask().0;
    if setaddr != 0 {
        let mask: u64 = task.CopyInObj(setaddr)?;

        match how {
            SigHow::SIG_BLOCK => t.SetSignalMask(SignalSet(oldMask | mask)),
            SigHow::SIG_UNBLOCK => t.SetSignalMask(SignalSet(oldMask & !mask)),
            SigHow::SIG_SETMASK => t.SetSignalMask(SignalSet(mask)),
            _ => return Err(Error::SysError(SysErr::EINVAL)),
        }
    }

    if oldaddr != 0 {
        task.CopyOutObj(&oldMask, oldaddr)?;
    }

    return Ok(0);
}

pub fn SysSigaltstack(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let setaddr = args.arg0 as u64;
    let oldaddr = args.arg1 as u64;

    let alt = task.SignalStack();
    if oldaddr != 0 {
        task.CopyOutObj(&alt, oldaddr)?;
    }

    if setaddr != 0 {
        let alt = task.CopyInObj::<SignalStack>(setaddr)?;
        // The signal stack cannot be changed if the task is currently
        // on the stack. This is enforced at the lowest level because
        // these semantics apply to changing the signal stack via a
        // ucontext during a signal handler.
        if !task.SetSignalStack(alt) {
            return Err(Error::SysError(SysErr::EPERM));
        }
    }

    return Ok(0);
}

pub fn SysPause(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    task.blocker.BlockInterrupt()?;
    return Ok(0);
}

pub fn SysRtSigsuspend(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let sigset = args.arg0 as u64;

    let mask = SignalSet(task.CopyInObj::<u64>(sigset)?);
    let mask = SignalSet(mask.0 & !UnblockableSignals().0);

    let thread = task.Thread();
    let oldmask = thread.SignalMask();
    thread.SetSignalMask(mask);
    thread.SetSavedSignalMask(oldmask);

    match task.blocker.block(false, None) {
        Ok(_) => Ok(0),
        Err(_) => Err(Error::SysError(SysErr::EINTR)),
    }
}

pub fn SysRtSigpending(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    let pending = task.Thread().PendingSignals();
    task.CopyOutObj(&pending.0, addr)?;
    return Ok(0);
}

// RtSigtimedwait implements linux syscall rt_sigtimedwait(2).
pub fn SysRtSigtimedwait(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let sigset = args.arg0 as u64;
    let siginfo = args.arg1 as u64;
    let timespec = args.arg2 as u64;
    let sigsetsize = args.arg3 as u32;

    let mask = CopyInSigSet(task, sigset, sigsetsize as usize)?;

    let timeout = if timespec != 0 {
        CopyTimespecIntoDuration(task, timespec)?
    } else {
        core::i64::MAX
    };

    let thread = task.Thread();
    let mut si = thread.Sigtimedwait(mask, timeout)?;

    if siginfo != 0 {
        si.FixSignalCodeForUser();
        task.CopyOutObj::<SignalInfo>(&*si, siginfo)?;
    }

    return Ok(si.Signo as i64);
}

// RtSigqueueinfo implements linux syscall rt_sigqueueinfo(2).
pub fn SysRtSigqueueinfo(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pid = args.arg0 as i32;
    let sig = args.arg1 as i32;
    let infoAddr = args.arg2 as u64;

    // Copy in the info.
    //
    // We must ensure that the Signo is set (Linux overrides this in the
    // same way), and that the code is in the allowed set. This same logic
    // appears below in RtSigtgqueueinfo and should be kept in sync.
    let mut info: SignalInfo = task.CopyInObj(infoAddr)?;
    info.Signo = sig;

    let t = task.Thread();
    let pidns = t.PIDNamespace();
    // This must loop to handle the race with execve described in Kill.
    loop {
        // Deliver to the given task's thread group.
        let target = match pidns.TaskWithID(pid) {
            None => return Err(Error::SysError(SysErr::ESRCH)),
            Some(t) => t,
        };

        // If the sender is not the receiver, it can't use si_codes used by the
        // kernel or SI_TKILL.
        if (info.Code >= 0 || info.Code == SignalInfo::SIGNAL_INFO_TKILL) && target != t {
            return Err(Error::SysError(SysErr::EPERM));
        }

        if !mayKill(&t, &target, Signal(sig)) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        match target.SendGroupSignal(&info) {
            Err(Error::SysError(SysErr::ESRCH)) => continue,
            Err(e) => return Err(e),
            Ok(_) => return Ok(0),
        }
    }
}

// RtTgsigqueueinfo implements linux syscall rt_tgsigqueueinfo(2).
pub fn SysRtTgsigqueueinfo(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tgid = args.arg0 as i32;
    let tid = args.arg1 as i32;
    let sig = args.arg2 as i32;
    let infoAddr = args.arg3 as u64;

    // N.B. Inconsistent with man page, linux actually rejects calls with
    // tgid/tid <=0 by EINVAL. This isn't the same for all signal calls.
    if tgid <= 0 || tid <= 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // Copy in the info.
    //
    // We must ensure that the Signo is set (Linux overrides this in the
    // same way), and that the code is in the allowed set. This same logic
    // appears below in RtSigtgqueueinfo and should be kept in sync.
    let mut info: SignalInfo = task.CopyInObj(infoAddr)?;
    info.Signo = sig;

    let t = task.Thread();
    let pidns = t.PIDNamespace();

    // Deliver to the given task.
    let targetTG = match pidns.ThreadGroupWithID(tgid) {
        None => return Err(Error::SysError(SysErr::ESRCH)),
        Some(tg) => tg,
    };

    let target = match pidns.TaskWithID(tid) {
        None => return Err(Error::SysError(SysErr::ESRCH)),
        Some(t) => t,
    };

    if target.ThreadGroup() != targetTG {
        return Err(Error::SysError(SysErr::ESRCH));
    }

    // If the sender is not the receiver, it can't use si_codes used by the
    // kernel or SI_TKILL.
    if (info.Code >= 0 || info.Code == SignalInfo::SIGNAL_INFO_TKILL) && target != t {
        return Err(Error::SysError(SysErr::EPERM));
    }

    if !mayKill(&t, &target, Signal(sig)) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    target.SendSignal(&info)?;
    return Ok(0);
}

pub fn SysKill(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pid = args.arg0 as i32;
    let sig = args.arg1 as i32;

    let t = task.Thread();
    let pidns = t.PIDNamespace();
    let creds = t.Credentials();
    if pid > 0 {
        // "If pid is positive, then signal sig is sent to the process with the
        // ID specified by pid." - kill(2)
        // This loops to handle races with execve where target dies between
        // TaskWithID and SendGroupSignal. Compare Linux's
        // kernel/signal.c:kill_pid_info().
        loop {
            let target = match pidns.TaskWithID(pid) {
                None => return Err(Error::SysError(SysErr::ESRCH)),
                Some(t) => t,
            };

            if !mayKill(&t, &target, Signal(sig)) {
                return Err(Error::SysError(SysErr::EPERM));
            }

            let mut info = SignalInfo {
                Signo: sig,
                Code: SignalInfo::SIGNAL_INFO_USER,
                ..Default::default()
            };

            let sigRt = info.SigRt();
            sigRt.pid = pidns.IDOfTask(&t);
            let tuserns = target.UserNamespace();
            sigRt.uid = creds.lock().RealKUID.In(&tuserns).OrOverflow().0;
            match target.SendGroupSignal(&info) {
                Err(Error::SysError(SysErr::ESRCH)) => (),
                Ok(()) => return Ok(0),
                Err(e) => return Err(e),
            }
        }
    } else if pid == -1 {
        // "If pid equals -1, then sig is sent to every process for which the
        // calling process has permission to send signals, except for process 1
        // (init), but see below. ... POSIX.1-2001 requires that kill(-1,sig)
        // send sig to all processes that the calling process may send signals
        // to, except possibly for some implementation-defined system
        // processes. Linux allows a process to signal itself, but on Linux the
        // call kill(-1,sig) does not signal the calling process."

        let mut lastErr: Error = Error::None;
        let mut delivered = 0;

        let tgs = pidns.ThreadGroups();
        for tg in tgs {
            if tg == t.ThreadGroup() {
                continue;
            }

            if pidns.IDOfThreadGroup(&tg) == INIT_TID {
                continue;
            }

            // If pid == -1, the returned error is the last non-EPERM error
            // from any call to group_send_sig_info.
            let leader = tg.Leader().unwrap();
            if !mayKill(&t, &leader, Signal(sig)) {
                continue;
            }

            // Here and below, whether or not kill returns an error may
            // depend on the iteration order. We at least implement the
            // semantics documented by the man page: "On success (at least
            // one signal was sent), zero is returned."
            let mut info = SignalInfo {
                Signo: sig,
                Code: SignalInfo::SIGNAL_INFO_USER,
                ..Default::default()
            };

            let sigRt = info.SigRt();
            sigRt.pid = pidns.IDOfTask(&t);
            let tuserns = leader.UserNamespace();
            sigRt.uid = creds.lock().RealKUID.In(&tuserns).OrOverflow().0;
            match tg.SendSignal(&info) {
                // ESRCH is ignored because it means the task
                // exited while we were iterating.  This is a
                // race which would not normally exist on
                // Linux, so we suppress it.
                Err(Error::SysError(SysErr::ESRCH)) => continue,
                Ok(()) => (),
                Err(e) => {
                    lastErr = e;
                }
            }

            delivered += 1;
        }

        if delivered > 0 {
            if lastErr == Error::None {
                return Ok(0);
            } else {
                return Err(lastErr);
            }
        }

        return Err(Error::SysError(SysErr::ESRCH));
    } else {
        // "If pid equals 0, then sig is sent to every process in the process
        // group of the calling process."
        //
        // "If pid is less than -1, then sig is sent to every process
        // in the process group whose ID is -pid."
        let mut pgid = -pid;
        if pgid == 0 {
            let tg = t.ThreadGroup();
            let pg = tg.ProcessGroup().unwrap();
            pgid = pidns.IDOfProcessGroup(&pg);
        }

        // If pid != -1 (i.e. signalling a process group), the returned error
        // is the last error from any call to group_send_sig_info.
        let tgs = pidns.ThreadGroups();

        if tgs.len() == 0 {
            return Err(Error::SysError(SysErr::ESRCH));
        }

        let mut lastErr = Err(Error::SysError(SysErr::ESRCH));
        let mut cnt = 0;
        for tg in tgs {
            let pg = tg.ProcessGroup().unwrap();
            if pidns.IDOfProcessGroup(&pg) == pgid {
                cnt += 1;
                let leader = match tg.Leader() {
                    Some(l) => l,
                    None => {
                        // sometime the tgleader is null from bazel test, hard to repro
                        // todo: root cause it and fix it
                        error!("SysKill: get non tglead for pgid {}", pgid);
                        return Err(Error::SysError(SysErr::EINVAL));
                    }
                };
                if !mayKill(&t, &leader, Signal(sig)) {
                    lastErr = Err(Error::SysError(SysErr::EPERM));
                    continue;
                }

                let mut info = SignalInfo {
                    Signo: sig,
                    Code: SignalInfo::SIGNAL_INFO_USER,
                    ..Default::default()
                };

                let sigRt = info.SigRt();
                sigRt.pid = pidns.IDOfTask(&t);
                let tuserns = leader.UserNamespace();
                sigRt.uid = creds.lock().RealKUID.In(&tuserns).OrOverflow().0;
                // See note above regarding ESRCH race above.
                match tg.SendSignal(&info) {
                    // ESRCH is ignored because it means the task
                    // exited while we were iterating.  This is a
                    // race which would not normally exist on
                    // Linux, so we suppress it.
                    Err(Error::SysError(SysErr::ESRCH)) => (),
                    Ok(()) => (),
                    Err(e) => {
                        lastErr = Err(e);
                    }
                }
            }
        }

        if cnt > 0 && lastErr == Err(Error::SysError(SysErr::ESRCH)) {
            return Ok(0);
        }

        return lastErr;
    }
}

pub fn TkillSignal(sender: &Thread, receiver: &Thread, sig: Signal) -> SignalInfo {
    let info = SignalInfo {
        Signo: sig.0,
        Code: SignalInfo::SIGNAL_INFO_TKILL,
        ..Default::default()
    };

    let kill = info.Kill();

    let senderTG = sender.ThreadGroup();
    let pidns = receiver.PIDNamespace();
    kill.pid = pidns.IDOfThreadGroup(&senderTG);

    let credential = sender.Credentials();
    let realKUID = credential.lock().RealKUID;
    let userns = receiver.UserNamespace();
    kill.uid = realKUID.In(&userns).OrOverflow().0 as i32;

    return info;
}

// Tkill implements linux syscall tkill(2).
pub fn SysTkill(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tid = args.arg0 as i32;
    let sig = args.arg1 as i32;

    // N.B. Inconsistent with man page, linux actually rejects calls with
    // tid <=0 by EINVAL. This isn't the same for all signal calls.
    if tid <= 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let t = task.Thread();
    let target = match t.PIDNamespace().TaskWithID(tid) {
        None => return Err(Error::SysError(SysErr::ESRCH)),
        Some(t) => t,
    };

    if !mayKill(&t, &target, Signal(sig)) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    let sigInfo = TkillSignal(&t, &target, Signal(sig));
    target.SendSignal(&sigInfo)?;
    return Ok(0);
}

pub fn SysTgkill(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tgid = args.arg0 as i32;
    let tid = args.arg1 as i32;
    let sig = args.arg2 as i32;

    // N.B. Inconsistent with man page, linux actually rejects calls with
    // tgid/tid <=0 by EINVAL. This isn't the same for all signal calls.
    if tgid <= 0 || tid <= 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let t = task.Thread();
    let pidns = t.PIDNamespace();
    let targetTG = match pidns.ThreadGroupWithID(tgid) {
        None => return Err(Error::SysError(SysErr::ESRCH)),
        Some(tg) => tg.clone(),
    };

    let target = match pidns.TaskWithID(tid) {
        None => return Err(Error::SysError(SysErr::ESRCH)),
        Some(t) => t,
    };

    if target.ThreadGroup() != targetTG {
        return Err(Error::SysError(SysErr::ESRCH));
    }

    if !mayKill(&t, &target, Signal(sig)) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    let sigInfo = TkillSignal(&t, &target, Signal(sig));
    target.SendSignal(&sigInfo)?;
    return Ok(0);
}

pub fn SysRestartSyscall(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let r = task.TakeSyscallRestartBlock();
    match r {
        None => {
            //what will happen if there is restart in signal handling
            return Err(Error::SysError(SysErr::EINTR));
        }
        Some(r) => {
            return r.Restart(task);
        }
    }
}

pub fn SharedSignalfd(
    task: &Task,
    fd: i32,
    sigset: u64,
    signsetsize: u32,
    flags: i32,
) -> Result<i64> {
    // Copy in the signal mask.
    let mask = CopyInSigSet(task, sigset, signsetsize as usize)?;

    // Always check for valid flags, even if not creating.
    if flags & !(SFD_NONBLOCK | SFD_CLOEXEC) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // Is this a change to an existing signalfd?
    //
    // The spec indicates that this should adjust the mask.
    if fd != -1 {
        let file = task.GetFile(fd)?;
        let fops = file.FileOp.clone();
        match fops.SignalOperation() {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(fops) => {
                fops.SetMask(mask);
                return Ok(0);
            }
        }
    }

    // Create a new file.
    let file = SignalOperation::NewSignalFile(task, mask);
    file.SetFlags(
        task,
        SettableFileFlags {
            NonBlocking: flags & SFD_NONBLOCK != 0,
            ..Default::default()
        },
    );

    let fd = task.NewFDFrom(
        0,
        &file,
        &FDFlags {
            CloseOnExec: flags & SFD_CLOEXEC != 0,
        },
    )?;

    return Ok(fd as i64);
}

pub fn SysSignalfd(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let sigset = args.arg1 as u64;
    let sigsetsize = args.arg2 as u32;

    return SharedSignalfd(task, fd, sigset, sigsetsize, 0);
}

pub fn SysSignalfd4(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let sigset = args.arg1 as u64;
    let sigsetsize = args.arg2 as u32;
    let flags = args.arg3 as i32;

    return SharedSignalfd(task, fd, sigset, sigsetsize, flags);
}
