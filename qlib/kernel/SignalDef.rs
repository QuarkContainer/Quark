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
use alloc::collections::linked_list::LinkedList;
use core::fmt;

use super::super::common::*;
use super::super::linux_def::*;
use super::kernel::posixtimer::*;
use super::task::*;

#[cfg(target_arch = "x86_64")]
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
//copy from https://elixir.bootlin.com/linux/latest/source/arch/x86/include/uapi/asm/ptrace.h#L18
pub struct PtRegs {
    /*
     * C ABI says these regs are callee-preserved. They aren't saved on kernel entry
     * unless syscall needs a complete, fully filled "struct pt_regs".
     */
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbp: u64,
    pub rbx: u64,

    /* These regs are callee-clobbered. Always saved on kernel entry. */
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,

    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    /*
     * On syscall entry, this is syscall#. On CPU exception, this is error code.
     * On hw interrupt, it's IRQ number:
     */
    pub orig_rax: u64,
    /* Return frame for iretq */
    pub rip: u64,
    pub cs: u64,
    pub eflags: u64,
    pub rsp: u64,
    pub ss: u64,
    /* top of stack page */
}

#[cfg(target_arch = "x86_64")]
impl PtRegs {
    pub fn Set(&mut self, ctx: &SigContext) {
        self.r15 = ctx.r15;
        self.r14 = ctx.r14;
        self.r13 = ctx.r13;
        self.r12 = ctx.r12;
        self.rbp = ctx.rbp;
        self.rbx = ctx.rbx;
        self.r11 = ctx.r11;
        self.r10 = ctx.r10;
        self.r9 = ctx.r9;
        self.r8 = ctx.r8;
        self.rax = ctx.rax;
        self.rcx = ctx.rcx;
        self.rdx = ctx.rdx;
        self.rsi = ctx.rsi;
        self.rdi = ctx.rdi;
        self.orig_rax = ctx.rax;
        self.rip = ctx.rip;
        self.cs = ctx.cs as u64;
        self.eflags = ctx.eflags;
        self.rsp = ctx.rsp;
        self.ss = ctx.ss as u64;
    }

    pub fn get_stack_pointer(&self) -> u64 {
        return self.rsp;
    }

    pub fn set_stack_pointer(&mut self, sp: u64) {
        self.rsp = sp;
    }
}

#[cfg(target_arch = "aarch64")]
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct PtRegs {
    pub regs: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub pstate: u64,
    pub orig_x0: u64,
    pub __pad: u64,
}

#[cfg(target_arch = "aarch64")]
impl PtRegs {
    pub fn Set(&mut self, ctx: &SigContext) {
        self.regs = ctx.regs.clone();
        self.sp = ctx.sp;
        self.pc = ctx.pc;
        self.pstate = ctx.pstate;
        self.orig_x0 = ctx.regs[0];
    }

    pub fn get_stack_pointer(&self) -> u64 {
        return self.sp;
    }

    pub fn set_stack_pointer(&mut self, sp: u64) {
        self.sp = sp;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigRetInfo {
    pub sigInfoAddr: u64,
    pub sigCtxAddr: u64,
    pub ret: u64,
}

/* kill() */
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct Kill {
    pub pid: i32,
    pub uid: i32,
}

/* POSIX.1b timers */
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigTimer {
    pub tid: i32,
    pub overrun: i32,
    pub sigval: u64,
    pub sysPrivate: i32,
}

/* POSIX.1b signals */
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigRt {
    pub pid: i32,
    pub uid: u32,
    pub sigval: u64,
}

/* SIGCHLD */
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigChld {
    pub pid: i32,
    //child
    pub uid: u32,
    //sender's uid
    pub status: i32,
    //Exit code
    pub uTime: i32,
    pub sTime: i32,
}

/* SIGILL, SIGFPE, SIGSEGV, SIGBUS */
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigFault {
    pub addr: u64,
    pub lsb: u16,
}

/* SIGPOLL */
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigPoll {
    pub band: u64,
    pub fd: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SignalInfo {
    pub Signo: i32,
    // Signal number
    pub Errno: i32,
    // Errno value
    pub Code: i32,
    // Signal code
    pub _r: u32,

    pub fields: [u8; 128 - 16],
}

impl<'a> Default for SignalInfo {
    fn default() -> Self {
        return Self {
            Signo: 0,
            Errno: 0,
            Code: 0,
            _r: 0,
            fields: [0; 128 - 16],
        };
    }
}

impl core::fmt::Debug for SignalInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SignalInfo")
            .field("Signo", &self.Signo)
            .field("Errno", &self.Errno)
            .field("Code", &self.Code)
            .finish()
    }
}

impl SignalInfo {
    pub fn SignalInfoPriv(sig: Signal) -> Self {
        return Self {
            Signo: sig.0,
            Code: Self::SIGNAL_INFO_KERNEL,
            ..Default::default()
        };
    }

    // FixSignalCodeForUser fixes up si_code.
    //
    // The si_code we get from Linux may contain the kernel-specific code in the
    // top 16 bits if it's positive (e.g., from ptrace). Linux's
    // copy_siginfo_to_user does
    //     err |= __put_user((short)from->si_code, &to->si_code);
    // to mask out those bits and we need to do the same.
    pub fn FixSignalCodeForUser(&mut self) {
        if self.Code > 0 {
            self.Code &= 0xffff;
        }
    }

    pub fn Kill(&self) -> &mut Kill {
        let addr = &self.fields[0] as *const _ as u64;
        return unsafe { &mut *(addr as *mut Kill) };
    }

    pub fn SigTimer(&mut self) -> &mut SigTimer {
        let addr = &self.fields[0] as *const _ as u64;
        return unsafe { &mut *(addr as *mut SigTimer) };
    }

    pub fn SigRt(&mut self) -> &mut SigRt {
        let addr = &self.fields[0] as *const _ as u64;
        return unsafe { &mut *(addr as *mut SigRt) };
    }

    pub fn SigChld(&mut self) -> &mut SigChld {
        let addr = &self.fields[0] as *const _ as u64;
        return unsafe { &mut *(addr as *mut SigChld) };
    }

    pub fn SigFault(&self) -> &mut SigFault {
        let addr = &self.fields[0] as *const _ as u64;
        return unsafe { &mut *(addr as *mut SigFault) };
    }

    pub fn SigPoll(&self) -> &mut SigPoll {
        let addr = &self.fields[0] as *const _ as u64;
        return unsafe { &mut *(addr as *mut SigPoll) };
    }

    // SignalInfoUser (properly SI_USER) indicates that a signal was sent from
    // a kill() or raise() syscall.
    pub const SIGNAL_INFO_USER: i32 = 0;

    // SignalInfoKernel (properly SI_KERNEL) indicates that the signal was sent
    // by the kernel.
    pub const SIGNAL_INFO_KERNEL: i32 = 0x80;

    // SignalInfoTimer (properly SI_TIMER) indicates that the signal was sent
    // by an expired timer.
    pub const SIGNAL_INFO_TIMER: i32 = -2;

    // SignalInfoTkill (properly SI_TKILL) indicates that the signal was sent
    // from a tkill() or tgkill() syscall.
    pub const SIGNAL_INFO_TKILL: i32 = -6;

    // CLD_* codes are only meaningful for SIGCHLD.

    // CLD_EXITED indicates that a task exited.
    pub const CLD_EXITED: i32 = 1;

    // CLD_KILLED indicates that a task was killed by a signal.
    pub const CLD_KILLED: i32 = 2;

    // CLD_DUMPED indicates that a task was killed by a signal and then dumped
    // core.
    pub const CLD_DUMPED: i32 = 3;

    // CLD_TRAPPED indicates that a task was stopped by ptrace.
    pub const CLD_TRAPPED: i32 = 4;

    // CLD_STOPPED indicates that a thread group completed a group stop.
    pub const CLD_STOPPED: i32 = 5;

    // CLD_CONTINUED indicates that a group-stopped thread group was continued.
    pub const CLD_CONTINUED: i32 = 6;

    // SYS_* codes are only meaningful for SIGSYS.

    // SYS_SECCOMP indicates that a signal originates from seccomp.
    pub const SYS_SECCOMP: i32 = 1;

    // TRAP_* codes are only meaningful for SIGTRAP.

    // TRAP_BRKPT indicates a breakpoint trap.
    pub const TRAP_BRKPT: i32 = 1;
}

pub const UC_FP_XSTATE: u64 = 1;
pub const UC_SIGCONTEXT_SS: u64 = 2;
pub const UC_STRICT_RESTORE_SS: u64 = 4;

// https://elixir.bootlin.com/linux/latest/source/include/uapi/asm-generic/ucontext.h#L5
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct UContext {
    pub Flags: u64,
    pub Link: u64,
    pub Stack: SignalStack,
    pub MContext: SigContext,
    pub Sigset: u64,
}

impl UContext {
    pub fn New(ptRegs: &PtRegs, oldMask: u64, cr2: u64, fpstate: u64, alt: &SignalStack) -> Self {
        return Self {
            Flags: 2,
            Link: 0,
            Stack: alt.clone(),
            MContext: SigContext::New(ptRegs, oldMask, cr2, fpstate),
            Sigset: 0,
        };
    }
}

// https://elixir.bootlin.com/linux/latest/source/arch/x86/include/uapi/asm/sigcontext.h#L284
#[cfg(target_arch = "x86_64")]
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigContext {
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rbp: u64,
    pub rbx: u64,
    pub rdx: u64,
    pub rax: u64,
    pub rcx: u64,
    pub rsp: u64,
    pub rip: u64,
    pub eflags: u64,
    pub cs: u16,
    pub gs: u16,
    // always 0 on amd64.
    pub fs: u16,
    // always 0 on amd64.
    pub ss: u16,
    // only restored if _UC_STRICT_RESTORE_SS (unsupported).
    pub err: u64,
    pub trapno: u64,
    pub oldmask: u64,
    pub cr2: u64,
    // Pointer to a struct _fpstate.
    pub fpstate: u64,
    pub reserved: [u64; 8],
}

#[cfg(target_arch = "aarch64")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SigContext {
	pub fault_address: u64,
	/* AArch64 registers */
	pub regs: [u64; 31],
	pub sp: u64,
	pub pc: u64,
	pub pstate: u64,
	/* 4K reserved for FP/SIMD state and future expansion */
	pub __reserved: [u8; 4096],
}

#[cfg(target_arch = "x86_64")]
impl SigContext {
    pub fn New(ptRegs: &PtRegs, oldMask: u64, cr2: u64, fpstate: u64) -> Self {
        return Self {
            r8: ptRegs.r8,
            r9: ptRegs.r9,
            r10: ptRegs.r10,
            r11: ptRegs.r11,
            r12: ptRegs.r12,
            r13: ptRegs.r13,
            r14: ptRegs.r14,
            r15: ptRegs.r15,
            rdi: ptRegs.rdi,
            rsi: ptRegs.rsi,
            rbp: ptRegs.rbp,
            rbx: ptRegs.rbx,
            rdx: ptRegs.rdx,
            rax: ptRegs.rax,
            rcx: ptRegs.rcx,
            rsp: ptRegs.rsp,
            rip: ptRegs.rip,
            eflags: ptRegs.eflags,
            cs: ptRegs.cs as u16,
            gs: 0,
            fs: 0,
            ss: ptRegs.ss as u16,
            err: 0,
            trapno: 0,
            oldmask: oldMask,
            cr2: cr2,
            fpstate: fpstate,
            ..Default::default()
        };
    }
}

#[cfg(target_arch = "aarch64")]
impl Default for SigContext {
    fn default() -> Self {
        Self {
            __reserved: [0u8;4096],
            ..Default::default()
        }
    }
}

//TODO  how to set sigcontext
#[cfg(target_arch = "aarch64")]
impl SigContext {
    pub fn New(ptRegs: &PtRegs, oldMask: u64, cr2: u64, fpstate: u64) -> Self {
        return Self {
            fault_address: 0,
            regs: ptRegs.regs.clone(),
            pc: ptRegs.pc,
            sp: ptRegs.sp,
            pstate: ptRegs.pstate,
            ..Default::default()
        };
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SigFlag(pub u64);

impl SigFlag {
    pub const SIGNAL_FLAG_NO_CLD_STOP: u64 = 0x00000001;
    pub const SIGNAL_FLAG_NO_CLD_WAIT: u64 = 0x00000002;
    pub const SIGNAL_FLAG_SIG_INFO: u64 = 0x00000004;
    pub const SIGNAL_FLAG_RESTORER: u64 = 0x04000000;
    pub const SIGNAL_FLAG_ON_STACK: u64 = 0x08000000;
    pub const SIGNAL_FLAG_RESTART: u64 = 0x10000000;
    pub const SIGNAL_FLAG_INTERRUPT: u64 = 0x20000000;
    pub const SIGNAL_FLAG_NO_DEFER: u64 = 0x40000000;
    pub const SIGNAL_FLAG_RESET_HANDLER: u64 = 0x80000000;

    pub fn IsNoCldStop(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_NO_CLD_STOP != 0;
    }

    pub fn IsNoCldWait(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_NO_CLD_WAIT != 0;
    }

    pub fn IsSigInfo(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_SIG_INFO != 0;
    }

    pub fn IsNoDefer(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_NO_DEFER != 0;
    }

    pub fn IsRestart(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_RESTART != 0;
    }

    pub fn IsResetHandler(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_RESET_HANDLER != 0;
    }

    pub fn IsOnStack(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_ON_STACK != 0;
    }

    pub fn HasRestorer(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_RESTORER != 0;
    }

    pub fn IsNoChildStop(&self) -> bool {
        return self.0 & Self::SIGNAL_FLAG_NO_CLD_STOP != 0;
    }
}

// https://github.com/lattera/glibc/blob/master/sysdeps/unix/sysv/linux/kernel_sigaction.h
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct SigAct {
    pub handler: u64,
    pub flags: SigFlag,
    pub restorer: u64,
    pub mask: u64,
}

impl fmt::Debug for SigAct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SigAction {{ \n\
        handler: {:x}, \n\
        flag : {:x}, \n \
        flags::HasRestorer: {}, \n \
        flags::IsOnStack: {}, \n \
        flags::IsRestart: {}, \n \
        flags::IsResetHandler: {}, \n \
        flags::IsNoDefer: {}, \n \
        flags::IsSigInfo: {}, \n \
        restorer : {:x},  \n\
        mask: {:x},  \n}}",
            self.handler,
            self.flags.0,
            self.flags.HasRestorer(),
            self.flags.IsOnStack(),
            self.flags.IsRestart(),
            self.flags.IsResetHandler(),
            self.flags.IsNoDefer(),
            self.flags.IsSigInfo(),
            self.restorer,
            self.mask
        )
    }
}

impl SigAct {
    // SignalActDefault is SIG_DFL and specifies that the default behavior for
    // a signal should be taken.
    pub const SIGNAL_ACT_DEFAULT: u64 = 0;

    // SignalActIgnore is SIG_IGN and specifies that a signal should be
    // ignored.
    pub const SIGNAL_ACT_IGNORE: u64 = 1;
}

pub const UNMASKABLE_MASK: u64 = 1 << (Signal::SIGKILL - 1) | 1 << (Signal::SIGSTOP - 1);

#[derive(Clone, Copy, Debug)]
pub struct SignalSet(pub u64);

impl Default for SignalSet {
    fn default() -> Self {
        return Self(0);
    }
}

impl SignalSet {
    pub fn New(sig: Signal) -> Self {
        return SignalSet(1 << sig.Index());
    }

    pub fn Add(&mut self, sig: Signal) {
        self.0 |= 1 << sig.Index()
    }

    pub fn Remove(&mut self, sig: Signal) {
        self.0 &= !(1 << sig.0)
    }

    pub fn TailingZero(&self) -> usize {
        for i in 0..64 {
            let idx = 64 - i - 1;
            if self.0 & (1 << idx) != 0 {
                return idx;
            }
        }

        return 64;
    }

    pub fn MakeSignalSet(sigs: &[Signal]) -> Self {
        let mut res = Self::default();
        for sig in sigs {
            res.Add(*sig)
        }

        return res;
    }

    pub fn ForEachSignal(&self, mut f: impl FnMut(Signal)) {
        for i in 0..64 {
            if self.0 & (1 << i) != 0 {
                f(Signal(i as i32 + 1))
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SignalQueue {
    signals: LinkedList<PendingSignal>,
}

impl SignalQueue {
    pub const RT_SIG_CAP: usize = 32;

    pub fn Len(&mut self) -> u64 {
        return self.signals.len() as u64;
    }

    pub fn Enque(&mut self, info: Box<SignalInfo>, timer: Option<IntervalTimer>) -> bool {
        if self.signals.len() == Self::RT_SIG_CAP {
            return false;
        }

        self.signals.push_back(PendingSignal {
            sigInfo: info,
            timer: timer,
        });

        return true;
    }

    pub fn Deque(&mut self) -> Option<PendingSignal> {
        return self.signals.pop_front();
    }

    pub fn Clear(&mut self) {
        self.signals.clear();
    }
}

pub const SIGNAL_COUNT: usize = 64;
pub const STD_SIGNAL_COUNT: usize = 31; // 1 ~ 31
pub const RT_SIGNAL_COUNT: usize = 33; // 32 ~ 64
pub const RT_SIGNAL_START: usize = 32; // 32 ~ 64

#[derive(Debug, Clone, Default)]
pub struct PendingSignal {
    pub sigInfo: Box<SignalInfo>,
    pub timer: Option<IntervalTimer>,
}

pub struct PendingSignals {
    pub stdSignals: [Option<PendingSignal>; STD_SIGNAL_COUNT],
    pub rtSignals: [SignalQueue; RT_SIGNAL_COUNT],
    pub pendingSet: SignalSet,
}

impl fmt::Debug for PendingSignals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingSignals")
            .field("stdSignals", &self.stdSignals)
            .field("rtSignals0", &self.rtSignals[0])
            .field("rtSignals2", &self.rtSignals[32])
            .field("pendingSet", &self.pendingSet)
            .finish()
    }
}

impl Default for PendingSignals {
    fn default() -> Self {
        return Self {
            stdSignals: Default::default(),
            rtSignals: [
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
                SignalQueue::default(),
            ],
            pendingSet: Default::default(),
        };
    }
}

impl PendingSignals {
    pub fn Enque(&mut self, info: Box<SignalInfo>, timer: Option<IntervalTimer>) -> Result<bool> {
        let sig = Signal(info.Signo);
        if sig.IsStandard() {
            match &self.stdSignals[sig.Index()] {
                None => (),
                _ => return Ok(false),
            }

            self.stdSignals[sig.Index()] = Some(PendingSignal {
                sigInfo: info,
                timer: timer,
            });
            self.pendingSet.Add(sig);

            return Ok(true);
        } else if sig.IsRealtime() {
            let q = &mut self.rtSignals[sig.Index() - 31];
            self.pendingSet.Add(sig);
            return Ok(q.Enque(info, timer));
        } else {
            return Err(Error::InvalidInput);
        }
    }

    pub fn HasSignal(&self, mask: SignalSet) -> bool {
        let set = SignalSet(self.pendingSet.0 & !(mask.0));

        if set.0 == 0 {
            return false;
        }

        return true;
    }

    pub fn Deque(&mut self, mask: SignalSet) -> Option<Box<SignalInfo>> {
        let set = SignalSet(self.pendingSet.0 & !(mask.0));

        if set.0 == 0 {
            return None;
        }

        let lastOne = set.TailingZero();

        if lastOne < STD_SIGNAL_COUNT {
            self.pendingSet.0 &= !(1 << lastOne);
            let ps = self.stdSignals[lastOne].take();
            if let Some(ps) = ps {
                let mut sigInfo = ps.sigInfo;
                match ps.timer {
                    None => (),
                    Some(timer) => timer.lock().updateDequeuedSignalLocked(&mut sigInfo),
                }

                return Some(sigInfo);
            } else {
                return None;
            }
        }

        if self.rtSignals[lastOne + 1 - RT_SIGNAL_START].Len() == 1 {
            self.pendingSet.0 &= !(1 << lastOne);
        }

        let ps = self.rtSignals[lastOne + 1 - RT_SIGNAL_START].Deque();
        if let Some(ps) = ps {
            let mut sigInfo = ps.sigInfo;
            match ps.timer {
                None => (),
                Some(timer) => timer.lock().updateDequeuedSignalLocked(&mut sigInfo),
            }

            return Some(sigInfo);
        } else {
            return None;
        }
    }

    pub fn Discard(&mut self, sig: Signal) {
        self.pendingSet.0 &= !(1 << sig.Index());

        if sig.0 <= STD_SIGNAL_COUNT as i32 {
            self.stdSignals[sig.Index()] = None;
            return;
        }

        self.rtSignals[sig.0 as usize - RT_SIGNAL_START].Clear()
    }
}

#[derive(Default, Debug)]
pub struct SignalStruct {
    pendingSignals: PendingSignals,
    signalMask: SignalSet,
    realSignalMask: SignalSet,
    //sigtimedwait
    groupStopPending: bool,
    groupStopAck: bool,
    trapStopPending: bool,
}

// https://elixir.bootlin.com/linux/latest/source/arch/x86/include/uapi/asm/signal.h#L132
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SignalStack {
    pub addr: u64,
    pub flags: u32,
    pub size: u64,
}

impl Default for SignalStack {
    fn default() -> Self {
        return Self {
            addr: 0,
            flags: Self::FLAG_DISABLE,
            size: 0,
        };
    }
}

impl SignalStack {
    pub const FLAG_ON_STACK: u32 = 1;
    pub const FLAG_DISABLE: u32 = 2;

    pub fn Contains(&self, sp: u64) -> bool {
        return self.addr < sp && sp <= self.addr + self.size;
    }

    pub fn SetOnStack(&mut self) {
        self.flags |= Self::FLAG_ON_STACK;
    }

    pub fn IsEnable(&self) -> bool {
        return self.flags & Self::FLAG_DISABLE == 0;
    }

    pub fn Top(&self) -> u64 {
        return self.addr + self.size;
    }
}

pub struct SigHow {}

impl SigHow {
    pub const SIG_BLOCK: u64 = 0;
    pub const SIG_UNBLOCK: u64 = 1;
    pub const SIG_SETMASK: u64 = 2;
}

pub fn SignalInfoPriv(sig: i32) -> SignalInfo {
    return SignalInfo {
        Signo: sig,
        Code: SignalInfo::SIGNAL_INFO_KERNEL,
        ..Default::default()
    };
}

// Sigevent represents struct sigevent.
#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct Sigevent {
    pub Value: u64,
    pub Signo: i32,
    pub Notify: i32,
    pub Tid: i32,

    // struct sigevent here contains 48-byte union _sigev_un. However, only
    // member _tid is significant to the kernel.
    pub UnRemainder1: [u8; 32],
    pub UnRemainder: [u8; 12],
}

pub const SIGEV_SIGNAL: i32 = 0;
pub const SIGEV_NONE: i32 = 1;
pub const SIGEV_THREAD: i32 = 2;
pub const SIGEV_THREAD_ID: i32 = 4;

// copyInSigSetWithSize copies in a structure as below
//
//   struct {
//           const sigset_t *ss;     /* Pointer to signal set */
//           size_t          ss_len; /* Size (in bytes) of object pointed to by 'ss' */
//   };
//
// and returns sigset_addr and size.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct SigMask {
    pub addr: u64,
    pub len: usize,
}

pub fn CopyInSigSetWithSize(task: &Task, addr: u64) -> Result<(u64, usize)> {
    let mask: SigMask = task.CopyInObj(addr)?;
    return Ok((mask.addr, mask.len));
}

pub const SIGNAL_SET_SIZE: usize = 8;

pub fn UnblockableSignals() -> SignalSet {
    return SignalSet::MakeSignalSet(&[Signal(Signal::SIGKILL), Signal(Signal::SIGSTOP)]);
}

pub fn CopyInSigSet(task: &Task, sigSetAddr: u64, size: usize) -> Result<SignalSet> {
    if size != SIGNAL_SET_SIZE {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mask: u64 = task.CopyInObj(sigSetAddr)?;
    return Ok(SignalSet(mask & !UnblockableSignals().0));
}
