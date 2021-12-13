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

extern crate alloc;

// Segment indices and Selectors.
pub const SEG_KCODE      : u16 = 1;
pub const SEG_KDATA      : u16 = 2;
pub const SEG_UDATA      : u16 = 3;
pub const SEG_UCODE64    : u16 = 4;
pub const SEG_TSS        : u16 = 5;
pub const SEG_TSS_HI     : u16 = 6;

pub const KCODE     : u16 = SEG_KCODE << 3;
pub const KDATA     : u16 = SEG_KDATA << 3;
pub const UDATA     : u16 = (SEG_UDATA << 3) | 3;
pub const UCODE64   : u16 = (SEG_UCODE64 << 3) | 3;
pub const TSS       : u16 = SEG_TSS << 3;

pub const CR0_PE : u64 = 1 << 0;
pub const CR0_MP : u64 = 1 << 1;
pub const CR0_EM : u64 = 1 << 2;
pub const CR0_TS : u64 = 1 << 3;
pub const CR0_ET : u64 = 1 << 4;
pub const CR0_NE : u64 = 1 << 5;
pub const CR0_WP : u64 = 1 << 16;
pub const CR0_AM : u64 = 1 << 18;
pub const CR0_NW : u64 = 1 << 29;
pub const CR0_CD : u64 = 1 << 30;
pub const CR0_PG : u64 = 1 << 31;

pub const CR4_PSE        : u64 = 1 << 4;
pub const CR4_PAE        : u64 = 1 << 5;
pub const CR4_PGE        : u64 = 1 << 7;
pub const CR4_OSFXSR     : u64 = 1 << 9;
pub const CR4_OSXMMEXCPT : u64 = 1 << 10;
pub const CR4_UMIP       : u64 = 1 << 11;
pub const CR4_FSGSBASE   : u64 = 1 << 16;
pub const CR4_PCIDE      : u64 = 1 << 17;
pub const CR4_OSXSAVE    : u64 = 1 << 18;
pub const CR4_SMEP       : u64 = 1 << 20;
pub const CR4_SMAP       : u64 = 1 << 21;

pub const RFLAGS_CF : u64 = 1 << 0;
pub const RFLAGS_RESERVED : u64 = 1 << 1;
pub const RFLAGS_PF : u64 = 1 << 2;
pub const RFLAGS_AF : u64 = 1 << 4;
pub const RFLAGS_ZF : u64 = 1 << 6;
pub const RFLAGS_SF : u64 = 1 << 7;

pub const RFLAGS_STEP     : u64 = 1 << 8;
pub const RFLAGS_IF       : u64 = 1 << 9;
pub const RFLAGS_DF       : u64 = 1 << 10;
pub const RFLAGS_OF       : u64 = 1 << 11;
pub const RFLAGS_IOPL     : u64 = 3 << 12;
pub const RFLAGS_NT       : u64 = 1 << 14;
pub const RFLAGS_RF       : u64 = 1 << 16;
pub const RFLAGS_VM       : u64 = 1 << 17;
pub const RFLAGS_AC       : u64 = 1 << 18;
pub const RFLAGS_VIF       : u64 = 1 << 19;
pub const RFLAGS_VIP       : u64 = 1 << 20;
pub const RFLAGS_ID       : u64 = 1 << 21;

pub const EFER_SCE : u64 = 0x001;
pub const EFER_LME : u64 = 0x100;
pub const EFER_LMA : u64 = 0x400;
pub const EFER_NX  : u64 = 0x800;

pub const MSR_STAR          : u64 = 0xc0000081;
pub const MSR_LSTAR         : u64 = 0xc0000082;
pub const MSR_CSTAR         : u64 = 0xc0000083;
pub const MSR_SYSCALL_MASK  : u64 = 0xc0000084;
pub const MSR_PLATFORM_INFO : u64 = 0xce;
pub const MSR_MISC_FEATURES : u64 = 0x140;

pub const PLATFORM_INFO_CPUID_FAULT : u64 = 1 << 31;

pub const MISC_FEATURE_CPUID_TRAP : u64 = 0x1;

// KernelFlagsSet should always be set in the kernel.
pub const KERNEL_FLAGS_SET : u64 = RFLAGS_RESERVED;

// UserFlagsSet are always set in userspace.
pub const USER_FLAGS_SET : u64 = RFLAGS_RESERVED | RFLAGS_IF;

// KernelFlagsClear should always be clear in the kernel.
pub const KERNEL_FLAGS_CLEAR : u64 = RFLAGS_STEP | RFLAGS_IF | RFLAGS_IOPL | RFLAGS_AC | RFLAGS_NT;

// UserFlagsClear are always cleared in userspace.
pub const USER_FLAGS_CLEAR : u64 = RFLAGS_NT | RFLAGS_IOPL;

// eflagsRestorable is the mask for the set of EFLAGS that may be changed by
// SignalReturn. eflagsRestorable is analogous to Linux's FIX_EFLAGS.
pub const RFLAGS_RESTORABLE : u64 = RFLAGS_AC | RFLAGS_OF | RFLAGS_DF | RFLAGS_STEP | RFLAGS_SF | RFLAGS_ZF | RFLAGS_AF | RFLAGS_PF | RFLAGS_CF | RFLAGS_RF;

use alloc::string::String;

use super::linux_def::*;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub enum TaskRunState {
    RunApp,
    RunInterrupt,
    RunExit,
    RunExitNotify,
    RunThreadExit,
    RunThreadExitNotify,
    RunExitDone,
    RunNoneReachAble,
    // can't reach this state
    RunSyscallRet,
}


// Error represents an error in the netstack error space. Using a special type
// ensures that errors outside of this space are not accidentally introduced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TcpipErr {
    pub msg: &'static str,
    pub ignoreStats: bool,
    pub sysErr: i32,
}

impl TcpipErr {
    pub fn New(msg: &'static str, ignoreStats: bool, sysErr: i32) -> Self {
        return Self {
            msg,
            ignoreStats,
            sysErr,
        }
    }

    pub const ERR_UNKNOWN_PROTOCOL     : Self = Self {msg: "unknown protocol", ignoreStats: false, sysErr: SysErr::EINVAL};
    pub const ERR_UNKNOWN_NICID          : Self = Self {msg: "unknown nic id", ignoreStats: false, sysErr: SysErr::EINVAL};
    pub const ERR_UNKNOWN_DEVICE         : Self = Self {msg: "unknown device", ignoreStats: false, sysErr: SysErr::ENODEV};
    pub const ERR_UNKNOWN_PROTOCOL_OPTION : Self = Self {msg: "unknown option for protocol", ignoreStats: false, sysErr: SysErr::ENOPROTOOPT};
    pub const ERR_DUPLICATE_NICID        : Self = Self {msg: "duplicate nic id", ignoreStats: false, sysErr: SysErr::EEXIST};
    pub const ERR_DUPLICATE_ADDRESS      : Self = Self {msg: "duplicate address", ignoreStats: false, sysErr: SysErr::EEXIST};
    pub const ERR_NO_ROUTE               : Self = Self {msg: "no route", ignoreStats: false, sysErr: SysErr::EHOSTUNREACH};
    pub const ERR_BAD_LINK_ENDPOINT       : Self = Self {msg: "bad link layer endpoint", ignoreStats: false, sysErr: SysErr::EINVAL};
    pub const ERR_ALREADY_BOUND          : Self = Self {msg: "endpoint already bound", ignoreStats: true, sysErr: SysErr::EINVAL};
    pub const ERR_INVALID_ENDPOINT_STATE  : Self = Self {msg: "endpoint is in invalid state", ignoreStats: false, sysErr: SysErr::EINVAL};
    pub const ERR_ALREADY_CONNECTING     : Self = Self {msg: "endpoint is already connecting", ignoreStats: true, sysErr: SysErr::EALREADY};
    pub const ERR_ALREADY_CONNECTED      : Self = Self {msg: "endpoint is already connected", ignoreStats: true, sysErr: SysErr::EISCONN};
    pub const ERR_NO_PORT_AVAILABLE       : Self = Self {msg: "no ports are available", ignoreStats: false, sysErr: SysErr::EAGAIN};
    pub const ERR_PORT_IN_USE             : Self = Self {msg: "port is in use", ignoreStats: false, sysErr: SysErr::EADDRINUSE};
    pub const ERR_BAD_LOCAL_ADDRESS       : Self = Self {msg: "bad local address", ignoreStats: false, sysErr: SysErr::EADDRNOTAVAIL};
    pub const ERR_CLOSED_FOR_SEND         : Self = Self {msg: "endpoint is closed for send", ignoreStats: false, sysErr: SysErr::EPIPE};
    pub const ERR_CLOSED_FOR_RECEIVE      : Self = Self {msg: "endpoint is closed for receive", ignoreStats: false, sysErr: SysErr::NONE};
    pub const ERR_WOULD_BLOCK            : Self = Self {msg: "operation would block", ignoreStats: true, sysErr: SysErr::EWOULDBLOCK};
    pub const ERR_CONNECTION_REFUSED     : Self = Self {msg: "connection was refused", ignoreStats: false, sysErr: SysErr::ECONNREFUSED};
    pub const ERR_TIMEOUT               : Self = Self {msg: "operation timed out", ignoreStats: false, sysErr: SysErr::ETIMEDOUT};
    pub const ERR_ABORTED               : Self = Self {msg: "operation aborted", ignoreStats: false, sysErr: SysErr::EPIPE};
    pub const ERR_CONNECT_STARTED        : Self = Self {msg: "connection attempt started", ignoreStats: true, sysErr: SysErr::EINPROGRESS};
    pub const ERR_DESTINATION_REQUIRED   : Self = Self {msg: "destination address is required", ignoreStats: false, sysErr: SysErr::EDESTADDRREQ};
    pub const ERR_NOT_SUPPORTED          : Self = Self {msg: "operation not supported", ignoreStats: false, sysErr: SysErr::EOPNOTSUPP};
    pub const ERR_QUEUE_SIZE_NOT_SUPPORTED : Self = Self {msg: "queue size querying not supported", ignoreStats: false, sysErr: SysErr::ENOTTY};
    pub const ERR_NOT_CONNECTED          : Self = Self {msg: "endpoint not connected", ignoreStats: false, sysErr: SysErr::ENOTCONN};
    pub const ERR_CONNECTION_RESET       : Self = Self {msg: "connection reset by peer", ignoreStats: false, sysErr: SysErr::ECONNRESET};
    pub const ERR_CONNECTION_ABORTED     : Self = Self {msg: "connection aborted", ignoreStats: false, sysErr: SysErr::ECONNABORTED};
    pub const ERR_NO_SUCH_FILE            : Self = Self {msg: "no such file", ignoreStats: false, sysErr: SysErr::ENOENT};
    pub const ERR_INVALID_OPTION_VALUE    : Self = Self {msg: "invalid option value specified", ignoreStats: false, sysErr: SysErr::EINVAL};
    pub const ERR_NO_LINK_ADDRESS         : Self = Self {msg: "no remote link address", ignoreStats: false, sysErr: SysErr::EHOSTDOWN};
    pub const ERR_BAD_ADDRESS            : Self = Self {msg: "bad address", ignoreStats: false, sysErr: SysErr::EFAULT};
    pub const ERR_NETWORK_UNREACHABLE    : Self = Self {msg: "network is unreachable", ignoreStats: false, sysErr: SysErr::ENETUNREACH};
    pub const ERR_MESSAGE_TOO_LONG        : Self = Self {msg: "message too long", ignoreStats: false, sysErr: SysErr::EMSGSIZE};
    pub const ERR_NO_BUFFER_SPACE         : Self = Self {msg: "no buffer space available", ignoreStats: false, sysErr: SysErr::ENOBUFS};
    pub const ERR_BROADCAST_DISABLED     : Self = Self {msg: "broadcast socket option disabled", ignoreStats: false, sysErr: SysErr::EACCES};
    pub const ERR_NOT_PERMITTED          : Self = Self {msg: "operation not permitted", ignoreStats: false, sysErr: SysErr::EPERM};
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    None,
    Timeout,
    PipeClosed,
    TcpipErr(TcpipErr),
    SysCallNotImplement,
    Common(String),
    CreateMMap(String),
    UnallignedAddress,
    UnallignedSize,
    NoEnoughMemory,
    AddressNotInRange,
    RootPageIdxNoExist,
    IOError(String),
    NoEnoughSpace,
    RangeUnavailable,
    Overflow,
    WrongELFFormat,
    ELFLoadError(&'static str),
    InterpreterFileErr,
    MMampError,
    UnmatchRegion,
    AddressDoesMatch,
    Locked,
    ZeroCount,
    QueueFull,
    NoData,
    NoUringReq,
    NoneIdx,
    AddressNotMap(u64),
    InvalidInput,
    NotExist,
    Signal,
    Exit,
    SysError(i32),
    FileMapError,
    NoEnoughData,
    EOF,
    ChanClose,

    // mem map of /dev/zero
    ErrDevZeroMap,

    //todo handle this.
    ErrClosedForReceive,
    ErrConnectionReset,

    //this is for chmod operation, fchmod doesn't work.
    CHMOD,

    SysCallRetCtrlWithRet(TaskRunState, u64),
    SysCallRetCtrl(TaskRunState),

    //link should be resolved via Readlink()
    ErrResolveViaReadlink,

    // ERESTARTSYS is returned by an interrupted syscall to indicate that it
    // should be converted to EINTR if interrupted by a signal delivered to a
    // user handler without SA_RESTART set, and restarted otherwise.
    ERESTARTSYS,
    // ERESTARTNOINTR is returned by an interrupted syscall to indicate that it
    // should always be restarted.
    ERESTARTNOINTR,
    // ERESTARTNOHAND is returned by an interrupted syscall to indicate that it
    // should be converted to EINTR if interrupted by a signal delivered to a
    // user handler, and restarted otherwise.
    ERESTARTNOHAND,
    // ERESTART_RESTARTBLOCK is returned by an interrupted syscall to indicate
    // that it should be restarted using a custom function. The interrupted
    // syscall must register a custom restart function by calling
    // Task.SetRestartSyscallFn.
    ERESTARTRESTARTBLOCK,
    // ErrWouldBlock is an internal error used to indicate that an operation
    // cannot be satisfied immediately, and should be retried at a later
    // time, possibly when the caller has received a notification that the
    // operation may be able to complete. It is used by implementations of
    // the kio.File interface.
    ErrWouldBlock,
    //request was interrupted
    ErrInterrupted,
    // ErrExceedsFileSizeLimit is returned if a request would exceed the
    // file's size limit.
    ErrExceedsFileSizeLimit,

    // ErrNoWaitableEvent is returned by non-blocking Task.Waits (e.g.
    // waitpid(WNOHANG)) that find no waitable events, but determine that waitable
    // events may exist in the future. (In contrast, if a non-blocking or blocking
    // Wait determines that there are no tasks that can produce a waitable event,
    // Task.Wait returns ECHILD.)
    ErrNoWaitableEvent,
}

impl Error {
    pub fn SystemErr(err: i32) -> Self {
        return Self::SysError(err)
    }

    pub fn Message(e: String) -> Self {
        return Self::Common(e);
    }

    pub fn MapRes(res: i32) -> Result<()> {
        if res == 0 {
            return Ok(())
        }

        if res < 0 {
            return Err(Error::SysError(-res))
        }

        panic!("MapRes get res {}", res);
    }
}

impl Default for Error {
    fn default() -> Self { Error::None }
}

pub fn ConvertIntr(err: Error, intr: Error) -> Error {
    if err == Error::ErrInterrupted {
        return intr
    }

    return err
}

pub trait RefMgr: Send {
    //ret: (the page's ref count, the pma's ref count)
    fn Ref(&self, addr: u64) -> Result<u64>;
    fn Deref(&self, addr: u64) -> Result<u64>;
    fn GetRef(&self, addr: u64) -> Result<u64>;
}

pub trait Allocator: RefMgr {
    fn AllocPage(&self, incrRef: bool) -> Result<u64>;
    fn FreePage(&self, addr: u64) -> Result<()>;
}

