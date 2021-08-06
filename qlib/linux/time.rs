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

use super::super::common::*;
use super::super::linux_def::*;

// ClockTick is the length of time represented by a single clock tick, as
// used by times(2) and /proc/[pid]/stat.
pub const CLOCK_TICK: i64 = SECOND / CLOCKS_PER_SEC;

// CLOCKS_PER_SEC is the number of ClockTicks per second.
//
// Linux defines this to be 100 on most architectures, irrespective of
// CONFIG_HZ. Userspace obtains the value through sysconf(_SC_CLK_TCK),
// which uses the AT_CLKTCK entry in the auxiliary vector if one is
// provided, and assumes 100 otherwise (glibc:
// sysdeps/posix/sysconf.c:__sysconf() =>
// sysdeps/unix/sysv/linux/getclktck.c, elf/dl-support.c:_dl_aux_init()).
//
// Not to be confused with POSIX CLOCKS_PER_SEC, as used by clock(3); "XSI
// requires that [POSIX] CLOCKS_PER_SEC equals 1000000 independent of the
// actual resolution" - clock(3).
pub const CLOCKS_PER_SEC: i64 = 100;

// CPU clock types for use with clock_gettime(2) et al.
//
// The 29 most significant bits of a 32 bit clock ID are either a PID or a FD.
//
// Bits 1 and 0 give the type: PROF=0, VIRT=1, SCHED=2, or FD=3.
//
// Bit 2 indicates whether a cpu clock refers to a thread or a process.
pub const CPUCLOCK_PROF: i32 = 0;
pub const CPUCLOCK_VIRT: i32 = 1;
pub const CPUCLOCK_SCHED: i32 = 2;
pub const CPUCLOCK_MAX: i32 = 3;
pub const CLOCKFD: i32 = CPUCLOCK_MAX;

pub const CPUCLOCK_CLOCK_MASK: i32 = 3;
pub const CPUCLOCK_PERTHREAD_MASK: i32 = 4;

// Clock identifiers for use with clock_gettime(2), clock_getres(2),
// clock_nanosleep(2).
pub const CLOCK_REALTIME: i32 = 0;
pub const CLOCK_MONOTONIC: i32 = 1;
pub const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
pub const CLOCK_THREAD_CPUTIME_ID: i32 = 3;
pub const CLOCK_MONOTONIC_RAW: i32 = 4;
pub const CLOCK_REALTIME_COARSE: i32 = 5;
pub const CLOCK_MONOTONIC_COARSE: i32 = 6;
pub const CLOCK_BOOTTIME: i32 = 7;
pub const CLOCK_REALTIME_ALARM: i32 = 8;
pub const CLOCK_BOOTTIME_ALARM: i32 = 9;

// Flags for clock_nanosleep(2).
pub const TIMER_ABSTIME: i32 = 1;

// Flags for timerfd syscalls (timerfd_create(2), timerfd_settime(2)).

// TFD_CLOEXEC is a timerfd_create flag.
pub const TFD_CLOEXEC: i32 = Flags::O_CLOEXEC;

// TFD_NONBLOCK is a timerfd_create flag.
pub const TFD_NONBLOCK: i32 = Flags::O_NONBLOCK;

// TFD_TIMER_ABSTIME is a timerfd_settime flag.
pub const TFD_TIMER_ABSTIME: i32 = 1;

// The safe number of seconds you can represent by int64.
pub const MAX_SEC_IN_DURATION: i64 = core::i64::MAX / SECOND;

pub type TimeT = i64;

// NsecToTimeT translates nanoseconds to TimeT (seconds).
pub fn NsecToTimeT(nsec: i64) -> TimeT {
    return nsec / 1_000_000_000;
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

impl Timespec {
    const E9: i64 = 1_000_000_000;
    pub fn FromNs(ns: i64) -> Self {
        let mut sec = ns / Self::E9;
        let mut nsec = ns % Self::E9;

        if nsec < 0 {
            nsec += Self::E9;
            sec -= 1;
        }

        return Self {
            tv_sec: sec,
            tv_nsec: nsec,
        }
    }

    pub fn ToNs(&self) -> Result<i64> {
        if self.tv_sec < 0 || self.tv_nsec < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }
        return Ok(self.tv_sec * Self::E9 + self.tv_nsec)
    }

    pub fn ToDuration(&self) -> Result<i64> {
        if self.tv_sec > MAX_SEC_IN_DURATION {
            return Ok(core::i64::MAX);
        }

        return self.ToNs();
    }

    pub fn IsValid(&self) -> bool {
        //return self.tv_nsec == Utime::UTIME_OMIT || self.tv_nsec == Utime::UTIME_NOW || self.tv_nsec < Self::E9;
        return !(self.tv_sec < 0 || self.tv_nsec <0 || self.tv_nsec >= Self::E9)
    }
}

pub const SIZE_OF_TIMEVAL : usize = 16;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Timeval {
    pub Sec: i64,
    pub Usec: i64,
}

impl Timeval {
    const E9: i64 = 1_000_000_000;
    const E6: i64 = 1_000_000;
    const E3: i64 = 1_000;

    pub fn FromNs(ns: i64) -> Self {
        let ns = ns + 999;
        let mut usec = ns % Self::E9 / Self::E3;
        let mut sec = ns / Self::E9;

        if usec < 0 {
            usec += Self::E6;
            sec -= 1;
        }

        return Self {
            Sec: sec,
            Usec: usec,
        }
    }

    // ToDuration returns the safe nanosecond representation as a time.Duration.
    pub fn ToDuration(&self) -> Duration {
        if self.Sec > MAX_SEC_IN_DURATION {
            return core::i64::MAX
        }

        return self.Sec * Self::E9 + self.Usec * Self::E3
    }
}

// Itimerspec represents struct itimerspec in <time.h>.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Itimerspec {
    pub Interval: Timespec,
    pub Value: Timespec,
}

// ItimerVal mimics the following struct in <sys/time.h>
//   struct itimerval {
//     struct timeval it_interval; /* next value */
//     struct timeval it_value;    /* current value */
//   };
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct ItimerVal {
    pub Interval: Timeval,
    pub Value: Timeval,
}

// ClockT represents type clock_t.
pub type ClockT = i64;

// ClockTFromDuration converts time.Duration to clock_t.
pub fn ClockTFromDuration(d: Duration) -> ClockT {
    return d / CLOCK_TICK
}

// Tms represents struct tms, used by times(2).
#[derive(Debug, Clone, Copy)]
pub struct Tms {
    pub UTime: ClockT,
    pub STime: ClockT,
    pub CUTime: ClockT,
    pub CSTime: ClockT,
}

// TimerID represents type timer_t, which identifies a POSIX per-process
// interval timer.
pub type TimerID = i32;

pub type Duration = i64;

pub const MIN: i64 = core::i64::MIN;
pub const MAX: i64 = core::i64::MAX;
pub const NANOSECOND: i64 = 1;
pub const MICROSECOND: i64 = 1000 * NANOSECOND;
pub const MILLISECOND: i64 = 1000 * MICROSECOND;
pub const SECOND: i64 = 1000 * MILLISECOND;
pub const MINUTE: i64 = 60 * SECOND;
pub const HOUR: i64 = 60 * MINUTE;

// itimer types for getitimer(2) and setitimer(2), from
// include/uapi/linux/time.h.
pub const ITIMER_REAL: i32 = 0;
pub const ITIMER_VIRTUAL: i32 = 1;
pub const ITIMER_PROF: i32 = 2;