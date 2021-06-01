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

use super::super::linux_def::*;

// SignalMaximum is the highest valid signal number.
pub const SIGNAL_MAXIMUM: i32 = 64;

// FirstStdSignal is the lowest standard signal number.
pub const FIRST_STD_SIGNAL: i32 = 1;

// LastStdSignal is the highest standard signal number.
pub const LAST_STD_SIGNAL: i32 = 31;

// FirstRTSignal is the lowest real-time signal number.
//
// 32 (SIGCANCEL) and 33 (SIGSETXID) are used internally by glibc.
pub const FIRST_RTSIGNAL: i32 = 32;

// LastRTSignal is the highest real-time signal number.
pub const LAST_RTSIGNAL: i32 = 64;

pub const SIGHUP: Signal = Signal(1);
pub const SIGINT: Signal = Signal(2);
pub const SIGQUIT: Signal = Signal(3);
pub const SIGILL: Signal = Signal(4);
pub const SIGTRAP: Signal = Signal(5);
pub const SIGABRT: Signal = Signal(6);
pub const SIGIOT: Signal = Signal(6);
pub const SIGBUS: Signal = Signal(7);
pub const SIGFPE: Signal = Signal(8);
pub const SIGKILL: Signal = Signal(9);
pub const SIGUSR1: Signal = Signal(10);
pub const SIGSEGV: Signal = Signal(11);
pub const SIGUSR2: Signal = Signal(12);
pub const SIGPIPE: Signal = Signal(13);
pub const SIGALRM: Signal = Signal(14);
pub const SIGTERM: Signal = Signal(15);
pub const SIGSTKFLT: Signal = Signal(16);
pub const SIGCHLD: Signal = Signal(17);
pub const SIGCLD: Signal = Signal(17);
pub const SIGCONT: Signal = Signal(18);
pub const SIGSTOP: Signal = Signal(19);
pub const SIGTSTP: Signal = Signal(20);
pub const SIGTTIN: Signal = Signal(21);
pub const SIGTTOU: Signal = Signal(22);
pub const SIGURG: Signal = Signal(23);
pub const SIGXCPU: Signal = Signal(24);
pub const SIGXFSZ: Signal = Signal(25);
pub const SIGVTALRM: Signal = Signal(26);
pub const SIGPROF: Signal = Signal(27);
pub const SIGWINCH: Signal = Signal(28);
pub const SIGIO: Signal = Signal(29);
pub const SIGPOLL: Signal = Signal(29);
pub const SIGPWR: Signal = Signal(30);
pub const SIGSYS: Signal = Signal(31);
pub const SIGUNUSED: Signal = Signal(31);

#[derive(Copy, Clone, Default)]
pub struct SignalSet(pub u64);

impl SignalSet {
    pub fn MakeSignalSet(sigs: &[&Signal]) -> Self {
        let mut res: u64 = 0;
        for s in sigs {
            res |= s.Mask();
        }

        return Self(res)
    }

    pub fn SignalSetOf(sigs: &Signal) -> Self {
        let res = sigs.Mask();
        return Self(res)
    }

    pub fn ForEachSignal(&self, f: fn(sig: &Signal)) {
        for i in 0..64 {
            if (1 << i) & self.0 != 0 {
                f(&Signal(i + 1))
            }
        }
    }
}

// Sigevent represents struct sigevent.
pub struct SigEvent {
    pub Value: u64,
    pub Signo: i32,
    pub Notify: i32,
    pub Tid: i32,
    pub UnRemainder: [u8; 44],
}

// 'how' values for rt_sigprocmask(2).
// SIG_BLOCK blocks the signals in the set.
pub const SIG_BLOCK: u64 = 0;

// SIG_UNBLOCK blocks the signals in the set.
pub const SIG_UNBLOCK: u64 = 1;


// Signal actions for rt_sigaction(2), from uapi/asm-generic/signal-defs.h.
// SIG_SETMASK sets the signal mask to set.
pub const SIG_SETMASK: u64 = 2;

// SIG_DFL performs the default action.
pub const SIG_DFL: u64 = 0;

// SIG_IGN ignores the signal.
pub const SIG_IGN: u64 = 1;


// Signal action flags for rt_sigaction(2), from uapi/asm-generic/signal.h
pub const SA_NOCLDSTOP: u64 = 0x00000001;
pub const SA_NOCLDWAIT: u64 = 0x00000002;
pub const SA_SIGINFO: u64 = 0x00000004;
pub const SA_RESTORER: u64 = 0x04000000;
pub const SA_ONSTACK: u64 = 0x08000000;
pub const SA_RESTART: u64 = 0x10000000;
pub const SA_NODEFER: u64 = 0x40000000;
pub const SA_RESETHAND: u64 = 0x80000000;
pub const SA_NOMASK: u64 = SA_NODEFER;
pub const SA_ONESHOT: u64 = SA_RESETHAND;


// Signal info types.
pub const SI_MASK: u64 = 0xffff0000;
pub const SI_KILL: u64 = 0 << 16;
pub const SI_TIMER: u64 = 1 << 16;
pub const SI_POLL: u64 = 2 << 16;
pub const SI_FAULT: u64 = 3 << 16;
pub const SI_CHLD: u64 = 4 << 16;
pub const SI_RT: u64 = 5 << 16;
pub const SI_MESGQ: u64 = 6 << 16;
pub const SI_SYS: u64 = 7 << 16;

// SIGPOLL si_codes.
// POLL_IN indicates that data input available.
pub const POLL_IN: u64 = SI_POLL | 1;

// POLL_OUT indicates that output buffers available.
pub const POLL_OUT: u64 = SI_POLL | 2;

// POLL_MSG indicates that an input message available.
pub const POLL_MSG: u64 = SI_POLL | 3;

// POLL_ERR indicates that there was an i/o error.
pub const POLL_ERR: u64 = SI_POLL | 4;

// POLL_PRI indicates that a high priority input available.
pub const POLL_PRI: u64 = SI_POLL | 5;

// POLL_HUP indicates that a device disconnected.
pub const POLL_HUP: u64 = SI_POLL | 6;

// Possible values for Sigevent.Notify, aka struct sigevent::sigev_notify.
pub const SIGEV_SIGNAL: u64 = 0;
pub const SIGEV_NONE: u64 = 1;
pub const SIGEV_THREAD: u64 = 2;
pub const SIGEV_THREAD_ID: u64 = 4;