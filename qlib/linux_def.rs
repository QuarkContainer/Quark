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

use alloc::slice;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use super::super::kernel_def::*;
use crate::qlib::mem::list_allocator::GuestHostSharedAllocator;
use crate::GUEST_HOST_SHARED_ALLOCATOR;

pub struct Xattr {}

impl Xattr {
    pub const XATTR_NAME_MAX: usize = 255;
    pub const XATTR_SIZE_MAX: usize = 65536;
    pub const XATTR_LIST_MAX: usize = 65536;

    pub const XATTR_CREATE: u32 = 1;
    pub const XATTR_REPLACE: u32 = 2;

    pub const XATTR_TRUSTED_PREFIX: &'static str = "trusted.";
    pub const XATTR_TRUSTED_PREFIX_LEN: usize = Self::XATTR_TRUSTED_PREFIX.len();

    pub const XATTR_USER_PREFIX: &'static str = "user.";
    pub const XATTR_USER_PREFIX_LEN: usize = Self::XATTR_USER_PREFIX.len();
}

pub struct InotifyEvent {}

impl InotifyEvent {
    // Inotify events observable by userspace. These directly correspond to
    // filesystem operations and there may only be a single of them per inotify
    // event read from an inotify fd.

    // IN_ACCESS indicates a file was accessed.
    pub const IN_ACCESS: u32 = 0x00000001;
    // IN_MODIFY indicates a file was modified.
    pub const IN_MODIFY: u32 = 0x00000002;
    // IN_ATTRIB indicates a watch target's metadata changed.
    pub const IN_ATTRIB: u32 = 0x00000004;
    // IN_CLOSE_WRITE indicates a writable file was closed.
    pub const IN_CLOSE_WRITE: u32 = 0x00000008;
    // IN_CLOSE_NOWRITE indicates a non-writable file was closed.
    pub const IN_CLOSE_NOWRITE: u32 = 0x00000010;
    // IN_OPEN indicates a file was opened.
    pub const IN_OPEN: u32 = 0x00000020;
    // IN_MOVED_FROM indicates a file was moved from X.
    pub const IN_MOVED_FROM: u32 = 0x00000040;
    // IN_MOVED_TO indicates a file was moved to Y.
    pub const IN_MOVED_TO: u32 = 0x00000080;
    // IN_CREATE indicates a file was created in a watched directory.
    pub const IN_CREATE: u32 = 0x00000100;
    // IN_DELETE indicates a file was deleted in a watched directory.
    pub const IN_DELETE: u32 = 0x00000200;
    // IN_DELETE_SELF indicates a watch target itself was deleted.
    pub const IN_DELETE_SELF: u32 = 0x00000400;
    // IN_MOVE_SELF indicates a watch target itself was moved.
    pub const IN_MOVE_SELF: u32 = 0x00000800;
    // IN_ALL_EVENTS is a mask for all observable userspace events.
    pub const IN_ALL_EVENTS: u32 = 0x00000fff;

    // Inotify control events. These may be present in their own events, or ORed
    // with other observable events.

    // IN_UNMOUNT indicates the backing filesystem was unmounted.
    pub const IN_UNMOUNT: u32 = 0x00002000;
    // IN_Q_OVERFLOW indicates the event queued overflowed.
    pub const IN_Q_OVERFLOW: u32 = 0x00004000;
    // IN_IGNORED indicates a watch was removed, either implicitly or through
    // inotify_rm_watch(2).
    pub const IN_IGNORED: u32 = 0x00008000;
    // IN_ISDIR indicates the subject of an event was a directory.
    pub const IN_ISDIR: u32 = 0x40000000;

    // Feature flags for inotify_add_watch(2).
    // IN_ONLYDIR indicates that a path should be watched only if it's a
    // directory.
    pub const IN_ONLYDIR: u32 = 0x01000000;
    // IN_DONT_FOLLOW indicates that the watch path shouldn't be resolved if
    // it's a symlink.
    pub const IN_DONT_FOLLOW: u32 = 0x02000000;
    // IN_EXCL_UNLINK indicates events to this watch from unlinked objects
    // should be filtered out.
    pub const IN_EXCL_UNLINK: u32 = 0x04000000;
    // IN_MASK_ADD indicates the provided mask should be ORed into any existing
    // watch on the provided path.
    pub const IN_MASK_ADD: u32 = 0x20000000;
    // IN_ONESHOT indicates the watch should be removed after one event.
    pub const IN_ONESHOT: u32 = 0x80000000;

    // IN_CLOEXEC is an alias for O_CLOEXEC. It indicates that the inotify
    // fd should be closed on exec(2) and friends.
    pub const IN_CLOEXEC: u32 = 0x00080000;
    // IN_NONBLOCK is an alias for O_NONBLOCK. It indicates I/O syscall on the
    // inotify fd should not block.
    pub const IN_NONBLOCK: u32 = 0x00000800;

    // ALL_INOTIFY_BITS contains all the bits for all possible inotify events. It's
    // defined in the Linux source at "include/linux/inotify.h".
    pub const ALL_INOTIFY_BITS: u32 = Self::IN_ACCESS
        | Self::IN_MODIFY
        | Self::IN_ATTRIB
        | Self::IN_CLOSE_WRITE
        | Self::IN_CLOSE_NOWRITE
        | Self::IN_OPEN
        | Self::IN_MOVED_FROM
        | Self::IN_MOVED_TO
        | Self::IN_CREATE
        | Self::IN_DELETE
        | Self::IN_DELETE_SELF
        | Self::IN_MOVE_SELF
        | Self::IN_UNMOUNT
        | Self::IN_Q_OVERFLOW
        | Self::IN_IGNORED
        | Self::IN_ONLYDIR
        | Self::IN_DONT_FOLLOW
        | Self::IN_EXCL_UNLINK
        | Self::IN_MASK_ADD
        | Self::IN_ISDIR
        | Self::IN_ONESHOT;
}

// Scheduling policies, exposed by sched_getscheduler(2)/sched_setscheduler(2).
pub struct Sched {}

impl Sched {
    pub const SCHED_NORMAL: i32 = 0;
    pub const SCHED_FIFO: i32 = 1;
    pub const SCHED_RR: i32 = 2;
    pub const SCHED_BATCH: i32 = 3;
    pub const SCHED_IDLE: i32 = 5;
    pub const SCHED_DEADLINE: i32 = 6;
    pub const SCHED_MICROQ: i32 = 16;

    // SCHED_RESET_ON_FORK is a flag that indicates that the process is
    // reverted back to SCHED_NORMAL on fork.
    pub const SCHED_RESET_ON_FORK: i32 = 0x40000000;

    pub const PRIO_PGRP: i32 = 0x1;
    pub const PRIO_PROCESS: i32 = 0x0;
    pub const PRIO_USER: i32 = 0x2;
}

// UNIX_PATH_MAX is the maximum length of the path in an AF_UNIX socket.
//
// From uapi/linux/un.h.
pub const UNIX_PATH_MAX: usize = 108;

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct TcpSockAddr {
    pub data: [u8; UNIX_PATH_MAX + 2],
}

impl Default for TcpSockAddr {
    fn default() -> Self {
        return Self {
            data: [0; UNIX_PATH_MAX + 2],
        };
    }
}

impl TcpSockAddr {
    pub fn Addr(&self) -> u64 {
        return &self.data[0] as *const _ as u64;
    }

    pub fn NewFromInet(addr: SockAddrInet) -> Self {
        let mut ret = Self::default();
        let ptr = unsafe { &mut *(&mut ret.data[0] as *mut _ as u64 as *mut SockAddrInet) };

        *ptr = addr;
        return ret;
    }

    pub fn Dup(&self) -> Self {
        return Self {
            data: self.data.clone(),
        };
    }
}

pub struct QOrdering {}
impl QOrdering {
    pub const RELAXED: Ordering = Ordering::Relaxed;
    pub const RELEASE: Ordering = Ordering::Release;
    pub const ACQUIRE: Ordering = Ordering::Acquire;
    pub const ACQ_REL: Ordering = Ordering::AcqRel;
    pub const SEQ_CST: Ordering = Ordering::SeqCst;
    /*pub const RELAXED :Ordering = Ordering::SeqCst;
    pub const RELEASE :Ordering = Ordering::SeqCst;
    pub const ACQUIRE :Ordering = Ordering::SeqCst;
    pub const ACQ_REL :Ordering = Ordering::SeqCst;
    pub const SEQ_CST :Ordering = Ordering::SeqCst;*/
}

pub const MLOCK_ONFAULT: u32 = 0x01;

pub const SIOCGIFMEM: u64 = 0x891f;
pub const SIOCGIFPFLAGS: u64 = 0x8935;
pub const SIOCGMIIPHY: u64 = 0x8947;
pub const SIOCGMIIREG: u64 = 0x8948;

//flags for getrandom(2)
pub const _GRND_NONBLOCK: i32 = 0x1;
pub const _GRND_RANDOM: i32 = 0x2;

// Policies for get_mempolicy(2)/set_mempolicy(2).
pub const MPOL_DEFAULT: i32 = 0;
pub const MPOL_PREFERRED: i32 = 1;
pub const MPOL_BIND: i32 = 2;
pub const MPOL_INTERLEAVE: i32 = 3;
pub const MPOL_LOCAL: i32 = 4;
pub const MPOL_MAX: i32 = 5;

// Flags for get_mempolicy(2).
pub const MPOL_F_NODE: i32 = 1 << 0;
pub const MPOL_F_ADDR: i32 = 1 << 1;
pub const MPOL_F_MEMS_ALLOWED: i32 = 1 << 2;

// Flags for set_mempolicy(2).
pub const MPOL_F_RELATIVE_NODES: i32 = 1 << 14;
pub const MPOL_F_STATIC_NODES: i32 = 1 << 15;

pub const MPOL_MODE_FLAGS: i32 = MPOL_F_STATIC_NODES | MPOL_F_RELATIVE_NODES;

// Flags for mbind(2).
pub const MPOL_MF_STRICT: i32 = 1 << 0;
pub const MPOL_MF_MOVE: i32 = 1 << 1;
pub const MPOL_MF_MOVE_ALL: i32 = 1 << 2;
pub const MPOL_MF_VALID: i32 = MPOL_MF_STRICT | MPOL_MF_MOVE | MPOL_MF_MOVE_ALL;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct LibcSysinfo {
    pub uptime: i64,
    pub loads: [u64; 3],
    pub totalram: u64,
    pub freeram: u64,
    pub sharedram: u64,
    pub bufferram: u64,
    pub totalswap: u64,
    pub freeswap: u64,
    pub procs: u16,
    pub pad: [u8; 6],
    pub totalhigh: u64,
    pub freehigh: u64,
    pub mem_unit: u32,
    //pub _f: [i8; 0],
}

#[cfg(target_arch = "x86_64")]
#[repr(packed)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct EpollEvent {
    pub Events: u32,
    pub Data: u64,
}

#[cfg(target_arch = "aarch64")]
#[repr(packed)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct EpollEvent {
    pub Events: u32,
    _pad: i32,
    pub Data: u64,
}

impl EpollEvent {
    #[cfg(target_arch = "x86_64")]
    pub fn new(events: u32, data: u64) -> Self {
        return Self {
            Events: events,
            Data: data,
        };
    }
    #[cfg(target_arch = "aarch64")]
    pub fn new(events: u32, data: u64) -> Self {
        return Self {
            Events: events,
            _pad: 0,
            Data: data,
        };
    }
}

pub struct MRemapType {}

impl MRemapType {
    pub const MREMAP_MAYMOVE: i32 = 1 << 0;
    pub const MREMAP_FIXED: i32 = 1 << 1;
}

pub struct SignaCode {}

impl SignaCode {
    // SI_USER is sent by kill, sigsend, raise.
    pub const SI_USER: i32 = 0;

    // SI_KERNEL is sent by the kernel from somewhere.
    pub const SI_KERNEL: i32 = 0x80;

    // SI_QUEUE is sent by sigqueue.
    pub const SI_QUEUE: i32 = -1;

    // SI_TIMER is sent by timer expiration.
    pub const SI_TIMER: i32 = -2;

    // SI_MESGQ is sent by real time mesq state change.
    pub const SI_MESGQ: i32 = -3;

    // SI_ASYNCIO is sent by AIO completion.
    pub const SI_ASYNCIO: i32 = -4;

    // SI_SIGIO is sent by queued SIGIO.
    pub const SI_SIGIO: i32 = -5;

    // SI_TKILL is sent by tkill system call.
    pub const SI_TKILL: i32 = -6;

    // SI_DETHREAD is sent by execve() killing subsidiary threads.
    pub const SI_DETHREAD: i32 = -7;

    // SI_ASYNCNL is sent by glibc async name lookup completion.
    pub const SI_ASYNCNL: i32 = -60;
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Signal(pub i32);

impl Signal {
    pub const SIGHUP: i32 = 1; //Term
    pub const SIGINT: i32 = 2; //Term
    pub const SIGQUIT: i32 = 3; //Core
    pub const SIGILL: i32 = 4; //Core
    pub const SIGTRAP: i32 = 5; //Core
    pub const SIGABRT: i32 = 6; //core
    pub const SIGIOT: i32 = 6; //Core
    pub const SIGBUS: i32 = 7; //Core
    pub const SIGFPE: i32 = 8; //Core
    pub const SIGKILL: i32 = 9; //Term
    pub const SIGUSR1: i32 = 10; //Term
    pub const SIGSEGV: i32 = 11; //Core
    pub const SIGUSR2: i32 = 12; //Term
    pub const SIGPIPE: i32 = 13; //Term
    pub const SIGALRM: i32 = 14; //Term
    pub const SIGTERM: i32 = 15; //Term
    pub const SIGSTKFLT: i32 = 16; //Term
    pub const SIGCHLD: i32 = 17; //Ignore
    pub const SIGCLD: i32 = 17; //ignore
    pub const SIGCONT: i32 = 18; //Cont
    pub const SIGSTOP: i32 = 19; //Stop
    pub const SIGTSTP: i32 = 20; //Stop
    pub const SIGTTIN: i32 = 21; //Stop
    pub const SIGTTOU: i32 = 22; //Stop
    pub const SIGURG: i32 = 23; //Ignore
    pub const SIGXCPU: i32 = 24; //Core
    pub const SIGXFSZ: i32 = 25; //Core
    pub const SIGVTALRM: i32 = 26; //Term
    pub const SIGPROF: i32 = 27; //Term
    pub const SIGWINCH: i32 = 28; //Ignore
    pub const SIGIO: i32 = 29; //Term
    pub const SIGPOLL: i32 = 29; //Term
    pub const SIGPWR: i32 = 30; //Term
    pub const SIGSYS: i32 = 31; //Core
    pub const SIGUNUSED: i32 = 31; //Core

    pub const SIGNAL_MAX: i32 = 64;
    pub const FIRST_STD_SIGNAL: i32 = 1;
    pub const LAST_STD_SIGNAL: i32 = 31;
    pub const FIRST_RT_SIGNAL: i32 = 32;
    pub const LAST_RT_SIGNAL: i32 = 64;

    pub fn IsValid(&self) -> bool {
        return 0 < self.0 && self.0 <= Self::SIGNAL_MAX;
    }

    pub fn Maskable(&self) -> bool {
        return self.0 != Self::SIGKILL && self.0 != Self::SIGSTOP;
    }

    pub fn IsStandard(&self) -> bool {
        return self.0 <= Self::LAST_STD_SIGNAL;
    }

    pub fn IsRealtime(&self) -> bool {
        return self.0 >= Self::FIRST_RT_SIGNAL;
    }

    pub fn Index(&self) -> usize {
        return (self.0 - 1) as usize;
    }

    pub fn Mask(&self) -> u64 {
        1 << self.Index()
    }
}

// EventMask represents io events as used in the poll() syscall.
pub type EventMask = u64;

pub const EVENTMASK_ALL: EventMask = 0xFFFF;

// Events that waiters can wait on. The meaning is the same as those in the
// poll() syscall.
pub const EVENT_IN: EventMask = 0x01; // POLLIN
pub const EVENT_PRI: EventMask = 0x02; // POLLPRI
pub const EVENT_OUT: EventMask = 0x04; // POLLOUT
pub const EVENT_ERR: EventMask = 0x08; // POLLERR
pub const EVENT_HUP: EventMask = 0x10; // POLLHUP
pub const EVENT_RD_NORM: EventMask = 0x0040; // POLLRDNORM
pub const EVENT_WR_NORM: EventMask = 0x0100; // POLLWRNORM
pub const EVENT_INTERNAL: EventMask = 0x1000;

// Quark event, when application shutdown the connection, it is used for wait the uring to drain the writing buffer
pub const EVENT_PENDING_SHUTDOWN: EventMask = 0x20;

pub const ALL_EVENTS: EventMask = 0x1f | EVENT_RD_NORM | EVENT_WR_NORM | EVENT_PENDING_SHUTDOWN;
pub const EVENT_READ: EventMask = EVENT_IN | EVENT_HUP | EVENT_ERR | EVENT_RD_NORM;
pub const EVENT_WRITE: EventMask = EVENT_OUT | EVENT_HUP | EVENT_ERR | EVENT_WR_NORM;
pub const READABLE_EVENT: EventMask = EVENT_IN | EVENT_RD_NORM;
pub const WRITEABLE_EVENT: EventMask = EVENT_OUT | EVENT_WR_NORM;

pub struct SocketSize {}

impl SocketSize {
    pub const SIZEOF_INT32: usize = 4;
    pub const SIZEOF_SOCKADDR_INET4: usize = 0x10;
    pub const SIZEOF_SOCKADDR_INET6: usize = 0x1c;
    pub const SIZEOF_SOCKADDR_ANY: usize = 0x70;
    pub const SIZEOF_SOCKADDR_UNIX: usize = 0x6e;
    pub const SIZEOF_SOCKADDR_LINKLAYER: usize = 0x14;
    pub const SIZEOF_SOCKADDR_NETLINK: usize = 0xc;
    pub const SIZEOF_LINGER: usize = 0x8;
    pub const SIZEOF_IPMREQ: usize = 0x8;
    pub const SIZEOF_IPMREQN: usize = 0xc;
    pub const SIZEOF_IPV6_MREQ: usize = 0x14;
    pub const SIZEOF_MSGHDR: usize = 0x38;
    pub const SIZEOF_CMSGHDR: usize = 0x10;
    pub const SIZEOF_INET4_PKTINFO: usize = 0xc;
    pub const SIZEOF_INET6_PKTINFO: usize = 0x14;
    pub const SIZEOF_IPV6_MTUINFO: usize = 0x20;
    pub const SIZEOF_ICMPV6_FILTER: usize = 0x20;
    pub const SIZEOF_UCRED: usize = 0xc;
    pub const SIZEOF_TCPINFO: usize = 0x68;
    pub const SIZEOF_TIMEVAL: usize = 0x10;
}

pub struct StatxFlags {}

impl StatxFlags {
    pub const AT_NO_AUTOMOUNT: u32 = 0x800;
    pub const AT_STATX_SYNC_TYPE: u32 = 0x6000;
    pub const AT_STATX_SYNC_AS_STAT: u32 = 0x0000;
    pub const AT_STATX_FORCE_SYNC: u32 = 0x2000;
    pub const AT_STATX_DONT_SYNC: u32 = 0x4000;
}

pub struct StatxMask {}

impl StatxMask {
    pub const STATX_TYPE: u32 = 0x00000001;
    pub const STATX_MODE: u32 = 0x00000002;
    pub const STATX_NLINK: u32 = 0x00000004;
    pub const STATX_UID: u32 = 0x00000008;
    pub const STATX_GID: u32 = 0x00000010;
    pub const STATX_ATIME: u32 = 0x00000020;
    pub const STATX_MTIME: u32 = 0x00000040;
    pub const STATX_CTIME: u32 = 0x00000080;
    pub const STATX_INO: u32 = 0x00000100;
    pub const STATX_SIZE: u32 = 0x00000200;
    pub const STATX_BLOCKS: u32 = 0x00000400;
    pub const STATX_BASIC_STATS: u32 = 0x000007ff;
    pub const STATX_BTIME: u32 = 0x00000800;
    pub const STATX_ALL: u32 = 0x00000fff;
    pub const STATX__RESERVED: u32 = 0x80000000;
}

pub struct StatxBitmask {}

impl StatxBitmask {
    pub const STATX_ATTR_COMPRESSED: u32 = 0x00000004;
    pub const STATX_ATTR_IMMUTABLE: u32 = 0x00000010;
    pub const STATX_ATTR_APPEND: u32 = 0x00000020;
    pub const STATX_ATTR_NODUMP: u32 = 0x00000040;
    pub const STATX_ATTR_ENCRYPTED: u32 = 0x00000800;
    pub const STATX_ATTR_AUTOMOUNT: u32 = 0x00001000;
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct StatxTimestamp {
    pub tv_sec: i64,
    pub tv_nsec: u32,
    pub __statx_timestamp_pad1: i32,
}

impl StatxTimestamp {
    const E9: i64 = 1_000_000_000;

    pub fn FromNs(nsec: i64) -> Self {
        return Self {
            tv_sec: nsec / Self::E9,
            tv_nsec: (nsec % Self::E9) as u32,
            __statx_timestamp_pad1: 0,
        };
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Statx {
    pub stx_mask: u32,
    pub stx_blksize: u32,
    pub stx_attributes: u64,
    pub stx_nlink: u32,
    pub stx_uid: u32,
    pub stx_gid: u32,
    pub stx_mode: u16,
    pub __statx_pad1: [u16; 1],
    pub stx_ino: u64,
    pub stx_size: u64,
    pub stx_blocks: u64,
    pub stx_attributes_mask: u64,
    pub stx_atime: StatxTimestamp,
    pub stx_btime: StatxTimestamp,
    pub stx_ctime: StatxTimestamp,
    pub stx_mtime: StatxTimestamp,
    pub stx_rdev_major: u32,
    pub stx_rdev_minor: u32,
    pub stx_dev_major: u32,
    pub stx_dev_minor: u32,
    pub __statx_pad2: [u64; 14],
}

// Statfs is struct statfs, from uapi/asm-generic/statfs.h.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct LibcStatfs {
    // Type is one of the filesystem magic values, defined above.
    pub Type: u64,

    // BlockSize is the data block size.
    pub BlockSize: i64,

    // Blocks is the number of data blocks in use.
    pub Blocks: u64,

    // BlocksFree is the number of free blocks.
    pub BlocksFree: u64,

    // BlocksAvailable is the number of blocks free for use by
    // unprivileged users.
    pub BlocksAvailable: u64,

    // Files is the number of used file nodes on the filesystem.
    pub Files: u64,

    // FileFress is the number of free file nodes on the filesystem.
    pub FilesFree: u64,

    // FSID is the filesystem ID.
    pub FSID: [i32; 2],

    // NameLength is the maximum file name length.
    pub NameLength: u64,

    // FragmentSize is equivalent to BlockSize.
    pub FragmentSize: i64,

    // Flags is the set of filesystem mount flags.
    pub Flags: u64,

    // Spare is unused.
    pub Spare: [u64; 4],
}

// Filesystem types used in statfs(2).
// See linux/magic.h.
pub struct FSMagic {}

impl FSMagic {
    pub const ANON_INODE_FS_MAGIC: u64 = 0x09041934;
    pub const DEVPTS_SUPER_MAGIC: u64 = 0x00001cd1;
    pub const EXT_SUPER_MAGIC: u64 = 0xef53;
    pub const OVERLAYFS_SUPER_MAGIC: u64 = 0x794c7630;
    pub const PIPEFS_MAGIC: u64 = 0x50495045;
    pub const PROC_SUPER_MAGIC: u64 = 0x9fa0;
    pub const RAMFS_MAGIC: u64 = 0x09041934;
    pub const SOCKFS_MAGIC: u64 = 0x534F434B;
    pub const SYSFS_MAGIC: u64 = 0x62656572;
    pub const TMPFS_MAGIC: u64 = 0x01021994;
    pub const V9FS_MAGIC: u64 = 0x01021997;
}

pub struct MfdType {}

impl MfdType {
    pub const MFD_CLOEXEC: u32 = 0x0001;
    pub const MFD_ALLOW_SEALING: u32 = 0x0002;
}

pub const MAX_SYMLINK_TRAVERSALS: u32 = 40;
pub const NAME_MAX: usize = 255;
pub const PATH_MAX: usize = 4096;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Utime {
    pub Actime: i64,
    pub Modtime: i64,
}

impl Utime {
    pub const UTIME_NOW: i64 = ((1 << 30) - 1);
    pub const UTIME_OMIT: i64 = ((1 << 30) - 2);
}

pub struct Capability {}

impl Capability {
    pub const CAP_CHOWN: u64 = 0;
    pub const CAP_DAC_OVERRIDE: u64 = 1;
    pub const CAP_DAC_READ_SEARCH: u64 = 2;
    pub const CAP_FOWNER: u64 = 3;
    pub const CAP_FSETID: u64 = 4;
    pub const CAP_KILL: u64 = 5;
    pub const CAP_SETGID: u64 = 6;
    pub const CAP_SETUID: u64 = 7;
    pub const CAP_SETPCAP: u64 = 8;
    pub const CAP_LINUX_IMMUTABLE: u64 = 9;
    pub const CAP_NET_BIND_SERVICE: u64 = 10;
    pub const CAP_NET_BROADCAST: u64 = 11;
    pub const CAP_NET_ADMIN: u64 = 12;
    pub const CAP_NET_RAW: u64 = 13;
    pub const CAP_IPC_LOCK: u64 = 14;
    pub const CAP_IPC_OWNER: u64 = 15;
    pub const CAP_SYS_MODULE: u64 = 16;
    pub const CAP_SYS_RAWIO: u64 = 17;
    pub const CAP_SYS_CHROOT: u64 = 18;
    pub const CAP_SYS_PTRACE: u64 = 19;
    pub const CAP_SYS_PACCT: u64 = 20;
    pub const CAP_SYS_ADMIN: u64 = 21;
    pub const CAP_SYS_BOOT: u64 = 22;
    pub const CAP_SYS_NICE: u64 = 23;
    pub const CAP_SYS_RESOURCE: u64 = 24;
    pub const CAP_SYS_TIME: u64 = 25;
    pub const CAP_SYS_TTY_CONFIG: u64 = 26;
    pub const CAP_MKNOD: u64 = 27;
    pub const CAP_LEASE: u64 = 28;
    pub const CAP_AUDIT_WRITE: u64 = 29;
    pub const CAP_AUDIT_CONTROL: u64 = 30;
    pub const CAP_SETFCAP: u64 = 31;
    pub const CAP_MAC_OVERRIDE: u64 = 32;
    pub const CAP_MAC_ADMIN: u64 = 33;
    pub const CAP_SYSLOG: u64 = 34;
    pub const CAP_WAKE_ALARM: u64 = 35;
    pub const CAP_BLOCK_SUSPEND: u64 = 36;
    pub const CAP_AUDIT_READ: u64 = 37;

    pub const CAP_LAST_CAP: u64 = Self::CAP_AUDIT_READ;

    pub fn Ok(cap: i32) -> bool {
        return cap >= 0 && cap <= Self::CAP_LAST_CAP as i32;
    }
}

// LINUX_CAPABILITY_VERSION_1 causes the data pointer to be
// interpreted as a pointer to a single cap_user_data_t. Since capability
// sets are 64 bits and the "capability sets" in cap_user_data_t are 32
// bits only, this causes the upper 32 bits to be implicitly 0.
pub const LINUX_CAPABILITY_VERSION_1: u32 = 0x19980330;

// LINUX_CAPABILITY_VERSION_2 and LINUX_CAPABILITY_VERSION_3 cause the
// data pointer to be interpreted as a pointer to an array of 2
// cap_user_data_t, using the second to store the 32 MSB of each capability
// set. Versions 2 and 3 are identical, but Linux printk's a warning on use
// of version 2 due to a userspace API defect.
pub const LINUX_CAPABILITY_VERSION_2: u32 = 0x20071026;
pub const LINUX_CAPABILITY_VERSION_3: u32 = 0x20080522;

// HIGHEST_CAPABILITY_VERSION is the highest supported
// LINUX_CAPABILITY_VERSION_* version.
pub const HIGHEST_CAPABILITY_VERSION: u32 = LINUX_CAPABILITY_VERSION_3;

// CapUserHeader is equivalent to Linux's cap_user_header_t.
#[derive(Clone, Copy, Default)]
pub struct CapUserHeader {
    pub Version: u32,
    pub Pid: i32,
}

// CapUserData is equivalent to Linux's cap_user_data_t.
#[derive(Clone, Copy, Default)]
pub struct CapUserData {
    pub Effective: u32,
    pub Permitted: u32,
    pub Inheritable: u32,
}

pub struct ATType {}

impl ATType {
    pub const AT_REMOVEDIR: i32 = 0x200;
    pub const AT_SYMLINK_FOLLOW: i32 = 0x400;
    pub const AT_EMPTY_PATH: i32 = 0x1000;
    pub const AT_FDCWD: i32 = -100;

    // Constants for fstatat(2)
    pub const AT_SYMLINK_NOFOLLOW: i32 = 0x100;
}

// Values for linux_dirent64.d_type.
pub struct DType {}

impl DType {
    pub const DT_UNKNOWN: u8 = 0;
    pub const DT_FIFO: u8 = 1;
    pub const DT_CHR: u8 = 2;
    pub const DT_DIR: u8 = 4;
    pub const DT_BLK: u8 = 6;
    pub const DT_REG: u8 = 8;
    pub const DT_LNK: u8 = 10;
    pub const DT_SOCK: u8 = 12;
    pub const DT_WHT: u8 = 14;

    pub fn ModeType(dtType: u8) -> u16 {
        match dtType {
            DType::DT_SOCK => ModeType::S_IFSOCK,
            DType::DT_LNK => ModeType::S_IFLNK,
            DType::DT_REG => ModeType::S_IFREG,
            DType::DT_BLK => ModeType::S_IFBLK,
            DType::DT_DIR => ModeType::S_IFDIR,
            DType::DT_CHR => ModeType::S_IFCHR,
            DType::DT_FIFO => ModeType::S_IFIFO,
            t => {
                error!("unknow DTtype {}", t);
                ModeType::S_IFREG
            }
        }
    }
}

// mode_t
pub struct ModeType {}

impl ModeType {
    pub const S_IFMT: u16 = 0o170000;
    pub const S_IFSOCK: u16 = 0o140000;
    pub const S_IFLNK: u16 = 0o120000;
    pub const S_IFREG: u16 = 0o0100000;
    pub const S_IFBLK: u16 = 0o060000;
    pub const S_IFDIR: u16 = 0o040000;
    pub const S_IFCHR: u16 = 0o020000;
    pub const S_IFIFO: u16 = 0o010000;

    pub const FILE_TYPE_MASK: u16 = Self::S_IFMT;
    pub const MODE_SOCKET: u16 = Self::S_IFSOCK;
    pub const MODE_SYMLINK: u16 = Self::S_IFLNK;
    pub const MODE_REGULAR: u16 = Self::S_IFREG;
    pub const MODE_BLOCK_DEVICE: u16 = Self::S_IFBLK;
    pub const MODE_DIRECTORY: u16 = Self::S_IFDIR;
    pub const MODE_CHARACTER_DEVICE: u16 = Self::S_IFCHR;
    pub const MODE_NAMED_PIPE: u16 = Self::S_IFIFO;

    pub const S_ISUID: u32 = 0o04000;
    pub const S_ISGID: u32 = 0o02000;
    pub const S_ISVTX: u32 = 0o01000;

    pub const MODE_SET_UID: u32 = Self::S_ISUID;
    pub const MODE_SET_GID: u32 = Self::S_ISGID;
    pub const MODE_STICKY: u32 = Self::S_ISVTX;

    pub const MODE_USER_ALL: u16 = 0o0700;
    pub const MODE_USER_READ: u16 = 0o0400;
    pub const MODE_USER_WRITE: u16 = 0o0200;
    pub const MODE_USER_EXEC: u16 = 0o0100;
    pub const MODE_GROUP_ALL: u16 = 0o0070;
    pub const MODE_GROUP_READ: u16 = 0o0040;
    pub const MODE_GROUP_WRITE: u16 = 0o0020;
    pub const MODE_GROUP_EXEC: u16 = 0o0010;
    pub const MODE_OTHER_ALL: u16 = 0o0007;
    pub const MODE_OTHER_READ: u16 = 0o0004;
    pub const MODE_OTHER_WRITE: u16 = 0o0002;
    pub const MODE_OTHER_EXEC: u16 = 0o0001;
    pub const PERMISSIONS_MASK: u16 = 0o0777;
}

#[derive(Debug, Default, Copy, Clone)]
pub struct FileMode(pub u16);

impl FileMode {
    pub fn Permission(&self) -> Self {
        return Self(self.0 & ModeType::PERMISSIONS_MASK);
    }

    pub fn FileType(&self) -> Self {
        return Self(self.0 & ModeType::FILE_TYPE_MASK);
    }

    pub fn ExtraBits(&self) -> Self {
        return Self(self.0 & !(ModeType::PERMISSIONS_MASK | ModeType::FILE_TYPE_MASK));
    }

    pub fn OtherRead(self) -> bool {
        return self.0 & ModeType::MODE_OTHER_READ != 0;
    }

    pub fn OtherWrite(self) -> bool {
        return self.0 & ModeType::MODE_OTHER_WRITE != 0;
    }

    pub fn OtherExec(self) -> bool {
        return self.0 & ModeType::MODE_OTHER_EXEC != 0;
    }

    pub fn Sticky(self) -> bool {
        return self.0 as u32 & ModeType::MODE_STICKY == ModeType::MODE_STICKY;
    }

    pub fn SetUID(self) -> bool {
        return self.0 as u32 & ModeType::MODE_SET_UID == ModeType::MODE_SET_UID;
    }

    pub fn SetGID(self) -> bool {
        return self.0 as u32 & ModeType::MODE_SET_GID == ModeType::MODE_SET_GID;
    }

    pub fn DirentType(&self) -> u8 {
        match self.FileType().0 {
            ModeType::S_IFSOCK => return DType::DT_SOCK,
            ModeType::S_IFLNK => return DType::DT_LNK,
            ModeType::S_IFREG => return DType::DT_REG,
            ModeType::S_IFBLK => return DType::DT_BLK,
            ModeType::S_IFDIR => return DType::DT_DIR,
            ModeType::S_IFCHR => return DType::DT_CHR,
            ModeType::S_IFIFO => return DType::DT_FIFO,
            _ => return DType::DT_UNKNOWN,
        }
    }

    pub fn Perms(mode: u16) -> PermMask {
        return PermMask {
            read: (mode & ModeType::MODE_OTHER_READ) != 0,
            write: (mode & ModeType::MODE_OTHER_WRITE) != 0,
            execute: (mode & ModeType::MODE_OTHER_EXEC) != 0,
        };
    }

    const MODE_SET_UID: u16 = ModeType::S_ISUID as u16;
    const MODE_SET_GID: u16 = ModeType::S_ISGID as u16;
    const MODE_STICKY: u16 = ModeType::S_ISVTX as u16;

    pub fn FilePerms(&self) -> FilePermissions {
        let perm = self.Permission().0;
        return FilePermissions {
            Other: Self::Perms(perm),
            Group: Self::Perms(perm >> 3),
            User: Self::Perms(perm >> 6),
            Sticky: self.0 & Self::MODE_STICKY == Self::MODE_STICKY,
            SetUid: self.0 & Self::MODE_SET_UID == Self::MODE_SET_UID,
            SetGid: self.0 & Self::MODE_SET_GID == Self::MODE_SET_GID,
        };
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct FilePermissions {
    pub User: PermMask,
    pub Group: PermMask,
    pub Other: PermMask,

    pub Sticky: bool,
    pub SetUid: bool,
    pub SetGid: bool,
}

impl FilePermissions {
    pub fn FromMode(mode: FileMode) -> Self {
        let mut fp = Self::default();

        let perm = mode.Permission();

        fp.Other = PermMask::FromMode(perm);
        fp.Group = PermMask::FromMode(FileMode(perm.0 >> 3));
        fp.User = PermMask::FromMode(FileMode(perm.0 >> 6));
        fp.Sticky = mode.Sticky();
        fp.SetUid = mode.SetUID();
        fp.SetGid = mode.SetGID();

        return fp;
    }

    pub fn LinuxMode(&self) -> u32 {
        let mut m = (self.User.Mode() << 6) | (self.Group.Mode() << 3) | self.Other.Mode();
        if self.SetUid {
            m |= ModeType::S_ISUID;
        }

        if self.SetUid {
            m |= ModeType::S_ISGID;
        }

        if self.Sticky {
            m |= ModeType::S_ISVTX;
        }

        return m;
    }

    pub fn AnyExec(&self) -> bool {
        return self.User.execute | self.Group.execute | self.Other.execute;
    }

    pub fn AnyWrite(&self) -> bool {
        return self.User.write | self.Group.write | self.Other.write;
    }

    pub fn AnyRead(&self) -> bool {
        return self.User.read | self.Group.read | self.Other.read;
    }
}

pub enum MSyncType {
    MsAsync,
    MsSync,
    MsInvalidate,
}

impl MSyncType {
    pub fn MSyncFlags(&self) -> i32 {
        match self {
            Self::MsAsync => return LibcConst::MS_ASYNC as i32,
            Self::MsSync => return LibcConst::MS_SYNC as i32,
            Self::MsInvalidate => return LibcConst::MS_INVALIDATE as i32,
        }
    }
}

pub struct LibcConst {}

impl LibcConst {
    pub const AF_ALG: u64 = 0x26;
    pub const AF_APPLETALK: u64 = 0x5;
    pub const AF_ASH: u64 = 0x12;
    pub const AF_ATMPVC: u64 = 0x8;
    pub const AF_ATMSVC: u64 = 0x14;
    pub const AF_AX25: u64 = 0x3;
    pub const AF_BLUETOOTH: u64 = 0x1f;
    pub const AF_BRIDGE: u64 = 0x7;
    pub const AF_CAIF: u64 = 0x25;
    pub const AF_CAN: u64 = 0x1d;
    pub const AF_DECNET: u64 = 0xc; //AF_DECnet
    pub const AF_ECONET: u64 = 0x13;
    pub const AF_FILE: u64 = 0x1;
    pub const AF_IEEE802154: u64 = 0x24;
    pub const AF_INET: u64 = 0x2;
    pub const AF_INET6: u64 = 0xa;
    pub const AF_IPX: u64 = 0x4;
    pub const AF_IRDA: u64 = 0x17;
    pub const AF_ISDN: u64 = 0x22;
    pub const AF_IUCV: u64 = 0x20;
    pub const AF_KEY: u64 = 0xf;
    pub const AF_LLC: u64 = 0x1a;
    pub const AF_LOCAL: u64 = 0x1;
    pub const AF_MAX: u64 = 0x27;
    pub const AF_NETBEUI: u64 = 0xd;
    pub const AF_NETLINK: u64 = 0x10;
    pub const AF_NETROM: u64 = 0x6;
    pub const AF_PACKET: u64 = 0x11;
    pub const AF_PHONET: u64 = 0x23;
    pub const AF_PPPOX: u64 = 0x18;
    pub const AF_RDS: u64 = 0x15;
    pub const AF_ROSE: u64 = 0xb;
    pub const AF_ROUTE: u64 = 0x10;
    pub const AF_RXRPC: u64 = 0x21;
    pub const AF_SECURITY: u64 = 0xe;
    pub const AF_SNA: u64 = 0x16;
    pub const AF_TIPC: u64 = 0x1e;
    pub const AF_UNIX: u64 = 0x1;
    pub const AF_UNSPEC: u64 = 0x0;
    pub const AF_WANPIPE: u64 = 0x19;
    pub const AF_X25: u64 = 0x9;
    pub const AF_UNSPECADAPT: u64 = 0x108;
    pub const AF_UNSPECAPPLETLK: u64 = 0x8;
    pub const AF_UNSPECARCNET: u64 = 0x7;
    pub const AF_UNSPECASH: u64 = 0x30d;
    pub const AF_UNSPECATM: u64 = 0x13;
    pub const AF_UNSPECAX25: u64 = 0x3;
    pub const AF_UNSPECBIF: u64 = 0x307;
    pub const AF_UNSPECCHAOS: u64 = 0x5;
    pub const AF_UNSPECCISCO: u64 = 0x201;
    pub const AF_UNSPECCSLIP: u64 = 0x101;
    pub const AF_UNSPECCSLIP6: u64 = 0x103;
    pub const AF_UNSPECDDCMP: u64 = 0x205;
    pub const AF_UNSPECDLCI: u64 = 0xf;
    pub const AF_UNSPECECONET: u64 = 0x30e;
    pub const AF_UNSPECEETHER: u64 = 0x2;
    pub const AF_UNSPECETHER: u64 = 0x1;
    pub const AF_UNSPECEUI64: u64 = 0x1b;
    pub const AF_UNSPECFCAL: u64 = 0x311;
    pub const AF_UNSPECFCFABRIC: u64 = 0x313;
    pub const AF_UNSPECFCPL: u64 = 0x312;
    pub const AF_UNSPECFCPP: u64 = 0x310;
    pub const AF_UNSPECFDDI: u64 = 0x306;
    pub const AF_UNSPECFRAD: u64 = 0x302;
    pub const AF_UNSPECHDLC: u64 = 0x201;
    pub const AF_UNSPECHIPPI: u64 = 0x30c;
    pub const AF_UNSPECHWX25: u64 = 0x110;
    pub const AF_UNSPECIEEE1394: u64 = 0x18;
    pub const AF_UNSPECIEEE802: u64 = 0x6;
    pub const AF_UNSPECIEEE80211: u64 = 0x321;
    pub const AF_UNSPECIEEE80211_PRISM: u64 = 0x322;
    pub const AF_UNSPECIEEE80211_RADIOTAP: u64 = 0x323;
    pub const AF_UNSPECIEEE802154: u64 = 0x324;
    pub const AF_UNSPECIEEE802154_PHY: u64 = 0x325;
    pub const AF_UNSPECIEEE802_TR: u64 = 0x320;
    pub const AF_UNSPECINFINIBAND: u64 = 0x20;
    pub const AF_UNSPECIPDDP: u64 = 0x309;
    pub const AF_UNSPECIPGRE: u64 = 0x30a;
    pub const AF_UNSPECIRDA: u64 = 0x30f;
    pub const AF_UNSPECLAPB: u64 = 0x204;
    pub const AF_UNSPECLOCALTLK: u64 = 0x305;
    pub const AF_UNSPECLOOPBACK: u64 = 0x304;
    pub const AF_UNSPECMETRICOM: u64 = 0x17;
    pub const AF_UNSPECNETROM: u64 = 0x0;
    pub const AF_UNSPECNONE: u64 = 0xfffe;
    pub const AF_UNSPECPIMREG: u64 = 0x30b;
    pub const AF_UNSPECPPP: u64 = 0x200;
    pub const AF_UNSPECPRONET: u64 = 0x4;
    pub const AF_UNSPECRAWHDLC: u64 = 0x206;
    pub const AF_UNSPECROSE: u64 = 0x10e;
    pub const AF_UNSPECRSRVD: u64 = 0x104;
    pub const AF_UNSPECSIT: u64 = 0x308;
    pub const AF_UNSPECSKIP: u64 = 0x303;
    pub const AF_UNSPECSLIP: u64 = 0x100;
    pub const AF_UNSPECSLIP6: u64 = 0x102;
    pub const AF_UNSPECTUNNEL: u64 = 0x300;
    pub const AF_UNSPECTUNNEL6: u64 = 0x301;
    pub const AF_UNSPECVOID: u64 = 0xffff;
    pub const AF_UNSPECX25: u64 = 0x10f;
    pub const AF_UNSPECA: u64 = 0x10;
    pub const AF_UNSPECABS: u64 = 0x20;
    pub const AF_UNSPECADD: u64 = 0x0;
    pub const AF_UNSPECALU: u64 = 0x4;
    pub const AF_UNSPECAND: u64 = 0x50;
    pub const AF_UNSPECB: u64 = 0x10;
    pub const AF_UNSPECDIV: u64 = 0x30;
    pub const AF_UNSPECH: u64 = 0x8;
    pub const AF_UNSPECIMM: u64 = 0x0;
    pub const AF_UNSPECIND: u64 = 0x40;
    pub const AF_UNSPECJA: u64 = 0x0;
    pub const AF_UNSPECJEQ: u64 = 0x10;
    pub const AF_UNSPECJGE: u64 = 0x30;
    pub const AF_UNSPECJGT: u64 = 0x20;
    pub const AF_UNSPECJMP: u64 = 0x5;
    pub const AF_UNSPECJSET: u64 = 0x40;
    pub const AF_UNSPECK: u64 = 0x0;
    pub const AF_UNSPECLD: u64 = 0x0;
    pub const AF_UNSPECLDX: u64 = 0x1;
    pub const AF_UNSPECLEN: u64 = 0x80;
    pub const AF_UNSPECLSH: u64 = 0x60;
    pub const AF_UNSPECMAJOR_VERSION: u64 = 0x1;
    pub const AF_UNSPECMAXINSNS: u64 = 0x1000;
    pub const AF_UNSPECMEM: u64 = 0x60;
    pub const AF_UNSPECMEMWORDS: u64 = 0x10;
    pub const AF_UNSPECMINOR_VERSION: u64 = 0x1;
    pub const AF_UNSPECMISC: u64 = 0x7;
    pub const AF_UNSPECMSH: u64 = 0xa0;
    pub const AF_UNSPECMUL: u64 = 0x20;
    pub const AF_UNSPECNEG: u64 = 0x80;
    pub const AF_UNSPECOR: u64 = 0x40;
    pub const AF_UNSPECRET: u64 = 0x6;
    pub const AF_UNSPECRSH: u64 = 0x70;
    pub const AF_UNSPECST: u64 = 0x2;
    pub const AF_UNSPECSTX: u64 = 0x3;
    pub const AF_UNSPECSUB: u64 = 0x10;
    pub const AF_UNSPECTAX: u64 = 0x0;
    pub const AF_UNSPECTXA: u64 = 0x80;
    pub const AF_UNSPECW: u64 = 0x0;
    pub const AF_UNSPECX: u64 = 0x8;
    pub const CLONE_CHILD_CLEARTID: u64 = 0x200000;
    pub const CLONE_CHILD_SETTID: u64 = 0x1000000;
    pub const CLONE_DETACHED: u64 = 0x400000;
    pub const CLONE_FILES: u64 = 0x400;
    pub const CLONE_FS: u64 = 0x200;
    pub const CLONE_IO: u64 = 0x80000000;
    pub const CLONE_NEWIPC: u64 = 0x8000000;
    pub const CLONE_NEWNET: u64 = 0x40000000;
    pub const CLONE_NEWNS: u64 = 0x20000;
    pub const CLONE_NEWPID: u64 = 0x20000000;
    pub const CLONE_NEWUSER: u64 = 0x10000000;
    pub const CLONE_NEWUTS: u64 = 0x4000000;
    pub const CLONE_PARENT: u64 = 0x8000;
    pub const CLONE_PARENT_SETTID: u64 = 0x100000;
    pub const CLONE_PTRACE: u64 = 0x2000;
    pub const CLONE_SETTLS: u64 = 0x80000;
    pub const CLONE_SIGHAND: u64 = 0x800;
    pub const CLONE_SYSVSEM: u64 = 0x40000;
    pub const CLONE_THREAD: u64 = 0x10000;
    pub const CLONE_UNTRACED: u64 = 0x800000;
    pub const CLONE_VFORK: u64 = 0x4000;
    pub const CLONE_VM: u64 = 0x100;
    pub const DT_BLK: u64 = 0x6;
    pub const DT_CHR: u64 = 0x2;
    pub const DT_DIR: u64 = 0x4;
    pub const DT_FIFO: u64 = 0x1;
    pub const DT_LNK: u64 = 0xa;
    pub const DT_REG: u64 = 0x8;
    pub const DT_SOCK: u64 = 0xc;
    pub const DT_UNKNOWN: u64 = 0x0;
    pub const DT_WHT: u64 = 0xe;
    pub const EPOLLERR: u64 = 0x8;
    pub const EPOLLET: i64 = -0x80000000;
    pub const EPOLLHUP: u64 = 0x10;
    pub const EPOLLIN: u64 = 0x1;
    pub const EPOLLMSG: u64 = 0x400;
    pub const EPOLLONESHOT: u64 = 0x40000000;
    pub const EPOLLOUT: u64 = 0x4;
    pub const EPOLLPRI: u64 = 0x2;
    pub const EPOLLRDBAND: u64 = 0x80;
    pub const EPOLLRDHUP: u64 = 0x2000;
    pub const EPOLLRDNORM: u64 = 0x40;
    pub const EPOLLWRBAND: u64 = 0x200;
    pub const EPOLLWRNORM: u64 = 0x100;
    pub const EPOLL_CLOEXEC: u64 = 0x80000;
    pub const EPOLL_CTL_ADD: u64 = 0x1;
    pub const EPOLL_CTL_DEL: u64 = 0x2;
    pub const EPOLL_CTL_MOD: u64 = 0x3;
    pub const EPOLL_NONBLOCK: u64 = 0x800;
    pub const ETH_P_1588: u64 = 0x88f7;
    pub const ETH_P_8021Q: u64 = 0x8100;
    pub const ETH_P_802_2: u64 = 0x4;
    pub const ETH_P_802_3: u64 = 0x1;
    pub const ETH_P_AARP: u64 = 0x80f3;
    pub const ETH_P_ALL: u64 = 0x3;
    pub const ETH_P_AOE: u64 = 0x88a2;
    pub const ETH_P_ARCNET: u64 = 0x1a;
    pub const ETH_P_ARP: u64 = 0x806;
    pub const ETH_P_ATALK: u64 = 0x809b;
    pub const ETH_P_ATMFATE: u64 = 0x8884;
    pub const ETH_P_ATMMPOA: u64 = 0x884c;
    pub const ETH_P_AX25: u64 = 0x2;
    pub const ETH_P_BPQ: u64 = 0x8ff;
    pub const ETH_P_CAIF: u64 = 0xf7;
    pub const ETH_P_CAN: u64 = 0xc;
    pub const ETH_P_CONTROL: u64 = 0x16;
    pub const ETH_P_CUST: u64 = 0x6006;
    pub const ETH_P_DDCMP: u64 = 0x6;
    pub const ETH_P_DEC: u64 = 0x6000;
    pub const ETH_P_DIAG: u64 = 0x6005;
    pub const ETH_P_DNA_DL: u64 = 0x6001;
    pub const ETH_P_DNA_RC: u64 = 0x6002;
    pub const ETH_P_DNA_RT: u64 = 0x6003;
    pub const ETH_P_DSA: u64 = 0x1b;
    pub const ETH_P_ECONET: u64 = 0x18;
    pub const ETH_P_EDSA: u64 = 0xdada;
    pub const ETH_P_FCOE: u64 = 0x8906;
    pub const ETH_P_FIP: u64 = 0x8914;
    pub const ETH_P_HDLC: u64 = 0x19;
    pub const ETH_P_IEEE802154: u64 = 0xf6;
    pub const ETH_P_IEEEPUP: u64 = 0xa00;
    pub const ETH_P_IEEEPUPAT: u64 = 0xa01;
    pub const ETH_P_IP: u64 = 0x800;
    pub const ETH_P_IPV6: u64 = 0x86dd;
    pub const ETH_P_IPX: u64 = 0x8137;
    pub const ETH_P_IRDA: u64 = 0x17;
    pub const ETH_P_LAT: u64 = 0x6004;
    pub const ETH_P_LINK_CTL: u64 = 0x886c;
    pub const ETH_P_LOCALTALK: u64 = 0x9;
    pub const ETH_P_LOOP: u64 = 0x60;
    pub const ETH_P_MOBITEX: u64 = 0x15;
    pub const ETH_P_MPLS_MC: u64 = 0x8848;
    pub const ETH_P_MPLS_UC: u64 = 0x8847;
    pub const ETH_P_PAE: u64 = 0x888e;
    pub const ETH_P_PAUSE: u64 = 0x8808;
    pub const ETH_P_PHONET: u64 = 0xf5;
    pub const ETH_P_PPPTALK: u64 = 0x10;
    pub const ETH_P_PPP_DISC: u64 = 0x8863;
    pub const ETH_P_PPP_MP: u64 = 0x8;
    pub const ETH_P_PPP_SES: u64 = 0x8864;
    pub const ETH_P_PUP: u64 = 0x200;
    pub const ETH_P_PUPAT: u64 = 0x201;
    pub const ETH_P_RARP: u64 = 0x8035;
    pub const ETH_P_SCA: u64 = 0x6007;
    pub const ETH_P_SLOW: u64 = 0x8809;
    pub const ETH_P_SNAP: u64 = 0x5;
    pub const ETH_P_TEB: u64 = 0x6558;
    pub const ETH_P_TIPC: u64 = 0x88ca;
    pub const ETH_P_TRAILER: u64 = 0x1c;
    pub const ETH_P_TR_802_2: u64 = 0x11;
    pub const ETH_P_WAN_PPP: u64 = 0x7;
    pub const ETH_P_WCCP: u64 = 0x883e;
    pub const ETH_P_X25: u64 = 0x805;
    pub const FD_CLOEXEC: u64 = 0x1;
    pub const FD_SETSIZE: u64 = 0x400;
    pub const F_DUPFD: u64 = 0x0;
    pub const F_DUPFD_CLOEXEC: u64 = 0x406;
    pub const F_EXLCK: u64 = 0x4;
    pub const F_GETFD: u64 = 0x1;
    pub const F_GETFL: u64 = 0x3;
    pub const F_GETLEASE: u64 = 0x401;
    pub const F_GETLK: u64 = 0x5;
    pub const F_GETLK64: u64 = 0x5;
    pub const F_GETOWN: u64 = 0x9;
    pub const F_GETOWN_EX: u64 = 0x10;
    pub const F_GETPIPE_SZ: u64 = 0x408;
    pub const F_GETSIG: u64 = 0xb;
    pub const F_LOCK: u64 = 0x1;
    pub const F_NOTIFY: u64 = 0x402;
    pub const F_OK: u64 = 0x0;
    pub const F_RDLCK: u64 = 0x0;
    pub const F_SETFD: u64 = 0x2;
    pub const F_SETFL: u64 = 0x4;
    pub const F_SETLEASE: u64 = 0x400;
    pub const F_SETLK: u64 = 0x6;
    pub const F_SETLK64: u64 = 0x6;
    pub const F_SETLKW: u64 = 0x7;
    pub const F_SETLKW64: u64 = 0x7;
    pub const F_SETOWN: u64 = 0x8;
    pub const F_SETOWN_EX: u64 = 0xf;
    pub const F_SETPIPE_SZ: u64 = 0x407;
    pub const F_SETSIG: u64 = 0xa;
    pub const F_SHLCK: u64 = 0x8;
    pub const F_TEST: u64 = 0x3;
    pub const F_TLOCK: u64 = 0x2;
    pub const F_ULOCK: u64 = 0x0;
    pub const F_UNLCK: u64 = 0x2;
    pub const F_WRLCK: u64 = 0x1;
    pub const ICMPV6_FILTER: u64 = 0x1;
    pub const IFA_F_DADFAILED: u64 = 0x8;
    pub const IFA_F_DEPRECATED: u64 = 0x20;
    pub const IFA_F_HOMEADDRESS: u64 = 0x10;
    pub const IFA_F_NODAD: u64 = 0x2;
    pub const IFA_F_OPTIMISTIC: u64 = 0x4;
    pub const IFA_F_PERMANENT: u64 = 0x80;
    pub const IFA_F_SECONDARY: u64 = 0x1;
    pub const IFA_F_TEMPORARY: u64 = 0x1;
    pub const IFA_F_TENTATIVE: u64 = 0x40;
    pub const IFA_MAX: u64 = 0x7;
    pub const IFF_ALLMULTI: u64 = 0x200;
    pub const IFF_AUTOMEDIA: u64 = 0x4000;
    pub const IFF_BROADCAST: u64 = 0x2;
    pub const IFF_DEBUG: u64 = 0x4;
    pub const IFF_DYNAMIC: u64 = 0x8000;
    pub const IFF_LOOPBACK: u64 = 0x8;
    pub const IFF_MASTER: u64 = 0x400;
    pub const IFF_MULTICAST: u64 = 0x1000;
    pub const IFF_NOARP: u64 = 0x80;
    pub const IFF_NOTRAILERS: u64 = 0x20;
    pub const IFF_NO_PI: u64 = 0x1000;
    pub const IFF_ONE_QUEUE: u64 = 0x2000;
    pub const IFF_POINTOPOINT: u64 = 0x10;
    pub const IFF_PORTSEL: u64 = 0x2000;
    pub const IFF_PROMISC: u64 = 0x100;
    pub const IFF_RUNNING: u64 = 0x40;
    pub const IFF_SLAVE: u64 = 0x800;
    pub const IFF_TAP: u64 = 0x2;
    pub const IFF_TUN: u64 = 0x1;
    pub const IFF_TUN_EXCL: u64 = 0x8000;
    pub const IFF_UP: u64 = 0x1;
    pub const IFF_VNET_HDR: u64 = 0x4000;
    pub const IFNAMSIZ: u64 = 0x10;
    pub const IN_ACCESS: u64 = 0x1;
    pub const IN_ALL_EVENTS: u64 = 0xfff;
    pub const IN_ATTRIB: u64 = 0x4;
    pub const IN_CLASSA_HOST: u64 = 0xffffff;
    pub const IN_CLASSA_MAX: u64 = 0x80;
    pub const IN_CLASSA_NET: u64 = 0xff000000;
    pub const IN_CLASSA_NSHIFT: u64 = 0x18;
    pub const IN_CLASSB_HOST: u64 = 0xffff;
    pub const IN_CLASSB_MAX: u64 = 0x10000;
    pub const IN_CLASSB_NET: u64 = 0xffff0000;
    pub const IN_CLASSB_NSHIFT: u64 = 0x10;
    pub const IN_CLASSC_HOST: u64 = 0xff;
    pub const IN_CLASSC_NET: u64 = 0xffffff00;
    pub const IN_CLASSC_NSHIFT: u64 = 0x8;
    pub const IN_CLOEXEC: u64 = 0x80000;
    pub const IN_CLOSE: u64 = 0x18;
    pub const IN_CLOSE_NOWRITE: u64 = 0x10;
    pub const IN_CLOSE_WRITE: u64 = 0x8;
    pub const IN_CREATE: u64 = 0x100;
    pub const IN_DELETE: u64 = 0x200;
    pub const IN_DELETE_SELF: u64 = 0x400;
    pub const IN_DONT_FOLLOW: u64 = 0x2000000;
    pub const IN_EXCL_UNLINK: u64 = 0x4000000;
    pub const IN_IGNORED: u64 = 0x8000;
    pub const IN_ISDIR: u64 = 0x40000000;
    pub const IN_LOOPBACKNET: u64 = 0x7f;
    pub const IN_MASK_ADD: u64 = 0x20000000;
    pub const IN_MODIFY: u64 = 0x2;
    pub const IN_MOVE: u64 = 0xc0;
    pub const IN_MOVED_FROM: u64 = 0x40;
    pub const IN_MOVED_TO: u64 = 0x80;
    pub const IN_MOVE_SELF: u64 = 0x800;
    pub const IN_NONBLOCK: u64 = 0x800;
    pub const IN_ONESHOT: u64 = 0x80000000;
    pub const IN_ONLYDIR: u64 = 0x1000000;
    pub const IN_OPEN: u64 = 0x20;
    pub const IN_Q_OVERFLOW: u64 = 0x4000;
    pub const IN_UNMOUNT: u64 = 0x2000;
    pub const IPPROTO_AH: u64 = 0x33;
    pub const IPPROTO_COMP: u64 = 0x6c;
    pub const IPPROTO_DCCP: u64 = 0x21;
    pub const IPPROTO_DSTOPTS: u64 = 0x3c;
    pub const IPPROTO_EGP: u64 = 0x8;
    pub const IPPROTO_ENCAP: u64 = 0x62;
    pub const IPPROTO_ESP: u64 = 0x32;
    pub const IPPROTO_FRAGMENT: u64 = 0x2c;
    pub const IPPROTO_GRE: u64 = 0x2f;
    pub const IPPROTO_HOPOPTS: u64 = 0x0;
    pub const IPPROTO_ICMP: u64 = 0x1;
    pub const IPPROTO_ICMPV6: u64 = 0x3a;
    pub const IPPROTO_IDP: u64 = 0x16;
    pub const IPPROTO_IGMP: u64 = 0x2;
    pub const IPPROTO_IP: u64 = 0x0;
    pub const IPPROTO_IPIP: u64 = 0x4;
    pub const IPPROTO_IPV6: u64 = 0x29;
    pub const IPPROTO_MTP: u64 = 0x5c;
    pub const IPPROTO_NONE: u64 = 0x3b;
    pub const IPPROTO_PIM: u64 = 0x67;
    pub const IPPROTO_PUP: u64 = 0xc;
    pub const IPPROTO_RAW: u64 = 0xff;
    pub const IPPROTO_ROUTING: u64 = 0x2b;
    pub const IPPROTO_RSVP: u64 = 0x2e;
    pub const IPPROTO_SCTP: u64 = 0x84;
    pub const IPPROTO_TCP: u64 = 0x6;
    pub const IPPROTO_TP: u64 = 0x1d;
    pub const IPPROTO_UDP: u64 = 0x11;
    pub const IPPROTO_UDPLITE: u64 = 0x88;
    pub const IPV6_2292DSTOPTS: u64 = 0x4;
    pub const IPV6_2292HOPLIMIT: u64 = 0x8;
    pub const IPV6_2292HOPOPTS: u64 = 0x3;
    pub const IPV6_2292PKTINFO: u64 = 0x2;
    pub const IPV6_2292PKTOPTIONS: u64 = 0x6;
    pub const IPV6_2292RTHDR: u64 = 0x5;
    pub const IPV6_ADDRFORM: u64 = 0x1;
    pub const IPV6_ADD_MEMBERSHIP: u64 = 0x14;
    pub const IPV6_AUTHHDR: u64 = 0xa;
    pub const IPV6_CHECKSUM: u64 = 0x7;
    pub const IPV6_DROP_MEMBERSHIP: u64 = 0x15;
    pub const IPV6_DSTOPTS: u64 = 0x3b;
    pub const IPV6_HOPLIMIT: u64 = 0x34;
    pub const IPV6_HOPOPTS: u64 = 0x36;
    pub const IPV6_IPSEC_POLICY: u64 = 0x22;
    pub const IPV6_JOIN_ANYCAST: u64 = 0x1b;
    pub const IPV6_JOIN_GROUP: u64 = 0x14;
    pub const IPV6_LEAVE_ANYCAST: u64 = 0x1c;
    pub const IPV6_LEAVE_GROUP: u64 = 0x15;
    pub const IPV6_MTU: u64 = 0x18;
    pub const IPV6_MTU_DISCOVER: u64 = 0x17;
    pub const IPV6_MULTICAST_HOPS: u64 = 0x12;
    pub const IPV6_MULTICAST_IF: u64 = 0x11;
    pub const IPV6_MULTICAST_LOOP: u64 = 0x13;
    pub const IPV6_NEXTHOP: u64 = 0x9;
    pub const IPV6_PKTINFO: u64 = 0x32;
    pub const IPV6_PMTUDISC_DO: u64 = 0x2;
    pub const IPV6_PMTUDISC_DONT: u64 = 0x0;
    pub const IPV6_PMTUDISC_PROBE: u64 = 0x3;
    pub const IPV6_PMTUDISC_WANT: u64 = 0x1;
    pub const IPV6_RECVDSTOPTS: u64 = 0x3a;
    pub const IPV6_RECVERR: u64 = 0x19;
    pub const IPV6_RECVHOPLIMIT: u64 = 0x33;
    pub const IPV6_RECVHOPOPTS: u64 = 0x35;
    pub const IPV6_RECVPKTINFO: u64 = 0x31;
    pub const IPV6_RECVRTHDR: u64 = 0x38;
    pub const IPV6_RECVTCLASS: u64 = 0x42;
    pub const IPV6_ROUTER_ALERT: u64 = 0x16;
    pub const IPV6_RTHDR: u64 = 0x39;
    pub const IPV6_RTHDRDSTOPTS: u64 = 0x37;
    pub const IPV6_RTHDR_LOOSE: u64 = 0x0;
    pub const IPV6_RTHDR_STRICT: u64 = 0x1;
    pub const IPV6_RTHDR_TYPE_0: u64 = 0x0;
    pub const IPV6_RXDSTOPTS: u64 = 0x3b;
    pub const IPV6_RXHOPOPTS: u64 = 0x36;
    pub const IPV6_TCLASS: u64 = 0x43;
    pub const IPV6_UNICAST_HOPS: u64 = 0x10;
    pub const IPV6_V6ONLY: u64 = 0x1a;
    pub const IPV6_XFRM_POLICY: u64 = 0x23;
    pub const IP_ADD_MEMBERSHIP: u64 = 0x23;
    pub const IP_ADD_SOURCE_MEMBERSHIP: u64 = 0x27;
    pub const IP_BLOCK_SOURCE: u64 = 0x26;
    pub const IP_DEFAULT_MULTICAST_LOOP: u64 = 0x1;
    pub const IP_DEFAULT_MULTICAST_TTL: u64 = 0x1;
    pub const IP_DF: u64 = 0x4000;
    pub const IP_DROP_MEMBERSHIP: u64 = 0x24;
    pub const IP_DROP_SOURCE_MEMBERSHIP: u64 = 0x28;
    pub const IP_FREEBIND: u64 = 0xf;
    pub const IP_HDRINCL: u64 = 0x3;
    pub const IP_IPSEC_POLICY: u64 = 0x10;
    pub const IP_MAXPACKET: u64 = 0xffff;
    pub const IP_MAX_MEMBERSHIPS: u64 = 0x14;
    pub const IP_MF: u64 = 0x2000;
    pub const IP_MINTTL: u64 = 0x15;
    pub const IP_MSFILTER: u64 = 0x29;
    pub const IP_MSS: u64 = 0x240;
    pub const IP_MTU: u64 = 0xe;
    pub const IP_MTU_DISCOVER: u64 = 0xa;
    pub const IP_MULTICAST_IF: u64 = 0x20;
    pub const IP_MULTICAST_LOOP: u64 = 0x22;
    pub const IP_MULTICAST_TTL: u64 = 0x21;
    pub const IP_OFFMASK: u64 = 0x1fff;
    pub const IP_OPTIONS: u64 = 0x4;
    pub const IP_ORIGDSTADDR: u64 = 0x14;
    pub const IP_PASSSEC: u64 = 0x12;
    pub const IP_PKTINFO: u64 = 0x8;
    pub const IP_PKTOPTIONS: u64 = 0x9;
    pub const IP_PMTUDISC: u64 = 0xa;
    pub const IP_PMTUDISC_DO: u64 = 0x2;
    pub const IP_PMTUDISC_DONT: u64 = 0x0;
    pub const IP_PMTUDISC_PROBE: u64 = 0x3;
    pub const IP_PMTUDISC_WANT: u64 = 0x1;
    pub const IP_RECVERR: u64 = 0xb;
    pub const IP_RECVOPTS: u64 = 0x6;
    pub const IP_RECVORIGDSTADDR: u64 = 0x14;
    pub const IP_RECVRETOPTS: u64 = 0x7;
    pub const IP_RECVTOS: u64 = 0xd;
    pub const IP_RECVTTL: u64 = 0xc;
    pub const IP_RETOPTS: u64 = 0x7;
    pub const IP_RF: u64 = 0x8000;
    pub const IP_ROUTER_ALERT: u64 = 0x5;
    pub const IP_TOS: u64 = 0x1;
    pub const IP_TRANSPARENT: u64 = 0x13;
    pub const IP_TTL: u64 = 0x2;
    pub const IP_UNBLOCK_SOURCE: u64 = 0x25;
    pub const IP_XFRM_POLICY: u64 = 0x11;
    pub const LINUX_REBOOT_CMD_CAD_OFF: u64 = 0x0;
    pub const LINUX_REBOOT_CMD_CAD_ON: u64 = 0x89abcdef;
    pub const LINUX_REBOOT_CMD_HALT: u64 = 0xcdef0123;
    pub const LINUX_REBOOT_CMD_KEXEC: u64 = 0x45584543;
    pub const LINUX_REBOOT_CMD_POWER_OFF: u64 = 0x4321fedc;
    pub const LINUX_REBOOT_CMD_RESTART: u64 = 0x1234567;
    pub const LINUX_REBOOT_CMD_RESTART2: u64 = 0xa1b2c3d4;
    pub const LINUX_REBOOT_CMD_SW_SUSPEND: u64 = 0xd000fce2;
    pub const LINUX_REBOOT_MAGIC1: u64 = 0xfee1dead;
    pub const LINUX_REBOOT_MAGIC2: u64 = 0x28121969;
    pub const LOCK_EX: u64 = 0x2;
    pub const LOCK_NB: u64 = 0x4;
    pub const LOCK_SH: u64 = 0x1;
    pub const LOCK_UN: u64 = 0x8;
    pub const MADV_DOFORK: u64 = 0xb;
    pub const MADV_DONTFORK: u64 = 0xa;
    pub const MADV_DONTNEED: u64 = 0x4;
    pub const MADV_HUGEPAGE: u64 = 0xe;
    pub const MADV_HWPOISON: u64 = 0x64;
    pub const MADV_MERGEABLE: u64 = 0xc;
    pub const MADV_NOHUGEPAGE: u64 = 0xf;
    pub const MADV_NORMAL: u64 = 0x0;
    pub const MADV_RANDOM: u64 = 0x1;
    pub const MADV_REMOVE: u64 = 0x9;
    pub const MADV_SEQUENTIAL: u64 = 0x2;
    pub const MADV_UNMERGEABLE: u64 = 0xd;
    pub const MADV_WILLNEED: u64 = 0x3;
    pub const MAP_32BIT: u64 = 0x40;
    pub const MAP_ANON: u64 = 0x20;
    pub const MAP_ANONYMOUS: u64 = 0x20;
    pub const MAP_DENYWRITE: u64 = 0x800;
    pub const MAP_EXECUTABLE: u64 = 0x1000;
    pub const MAP_FILE: u64 = 0x0;
    pub const MAP_FIXED: u64 = 0x10;
    pub const MAP_GROWSDOWN: u64 = 0x100;
    pub const MAP_HUGETLB: u64 = 0x40000;
    pub const MAP_LOCKED: u64 = 0x2000;
    pub const MAP_NONBLOCK: u64 = 0x10000;
    pub const MAP_NORESERVE: u64 = 0x4000;
    pub const MAP_POPULATE: u64 = 0x8000;
    pub const MAP_PRIVATE: u64 = 0x2;
    pub const MAP_SHARED: u64 = 0x1;
    pub const MAP_STACK: u64 = 0x20000;
    pub const MAP_TYPE: u64 = 0xf;
    pub const MCL_CURRENT: u64 = 0x1;
    pub const MCL_FUTURE: u64 = 0x2;
    pub const MCL_ONFAULT: u64 = 0x4;
    pub const MNT_DETACH: u64 = 0x2;
    pub const MNT_EXPIRE: u64 = 0x4;
    pub const MNT_FORCE: u64 = 0x1;
    pub const MSG_CMSG_CLOEXEC: u64 = 0x40000000;
    pub const MSG_CONFIRM: u64 = 0x800;
    pub const MSG_CTRUNC: u64 = 0x8;
    pub const MSG_DONTROUTE: u64 = 0x4;
    pub const MSG_DONTWAIT: u64 = 0x40;
    pub const MSG_EOR: u64 = 0x80;
    pub const MSG_ERRQUEUE: u64 = 0x2000;
    pub const MSG_FASTOPEN: u64 = 0x20000000;
    pub const MSG_FIN: u64 = 0x200;
    pub const MSG_MORE: u64 = 0x8000;
    pub const MSG_NOSIGNAL: u64 = 0x4000;
    pub const MSG_OOB: u64 = 0x1;
    pub const MSG_PEEK: u64 = 0x2;
    pub const MSG_PROXY: u64 = 0x10;
    pub const MSG_RST: u64 = 0x1000;
    pub const MSG_SYN: u64 = 0x400;
    pub const MSG_TRUNC: u64 = 0x20;
    pub const MSG_TRYHARD: u64 = 0x4;
    pub const MSG_WAITALL: u64 = 0x100;
    pub const MSG_WAITFORONE: u64 = 0x10000;
    pub const MS_ACTIVE: u64 = 0x40000000;
    pub const MS_ASYNC: u64 = 0x1;
    pub const MS_BIND: u64 = 0x1000;
    pub const MS_DIRSYNC: u64 = 0x80;
    pub const MS_INVALIDATE: u64 = 0x2;
    pub const MS_I_VERSION: u64 = 0x800000;
    pub const MS_KERNMOUNT: u64 = 0x400000;
    pub const MS_MANDLOCK: u64 = 0x40;
    pub const MS_MGC_MSK: u64 = 0xffff0000;
    pub const MS_MGC_VAL: u64 = 0xc0ed0000;
    pub const MS_MOVE: u64 = 0x2000;
    pub const MS_NOATIME: u64 = 0x400;
    pub const MS_NODEV: u64 = 0x4;
    pub const MS_NODIRATIME: u64 = 0x800;
    pub const MS_NOEXEC: u64 = 0x8;
    pub const MS_NOSUID: u64 = 0x2;
    pub const MS_NOUSER: i64 = -0x80000000;
    pub const MS_POSIXACL: u64 = 0x10000;
    pub const MS_PRIVATE: u64 = 0x40000;
    pub const MS_RDONLY: u64 = 0x1;
    pub const MS_REC: u64 = 0x4000;
    pub const MS_RELATIME: u64 = 0x200000;
    pub const MS_REMOUNT: u64 = 0x20;
    pub const MS_RMT_MASK: u64 = 0x800051;
    pub const MS_SHARED: u64 = 0x100000;
    pub const MS_SILENT: u64 = 0x8000;
    pub const MS_SLAVE: u64 = 0x80000;
    pub const MS_STRICTATIME: u64 = 0x1000000;
    pub const MS_SYNC: u64 = 0x4;
    pub const MS_SYNCHRONOUS: u64 = 0x10;
    pub const MS_UNBINDABLE: u64 = 0x20000;
    pub const NAME_MAX: u64 = 0xff;
    pub const NETLINK_ADD_MEMBERSHIP: u64 = 0x1;
    pub const NETLINK_AUDIT: u64 = 0x9;
    pub const NETLINK_BROADCAST_ERROR: u64 = 0x4;
    pub const NETLINK_CONNECTOR: u64 = 0xb;
    pub const NETLINK_DNRTMSG: u64 = 0xe;
    pub const NETLINK_DROP_MEMBERSHIP: u64 = 0x2;
    pub const NETLINK_ECRYPTFS: u64 = 0x13;
    pub const NETLINK_FIB_LOOKUP: u64 = 0xa;
    pub const NETLINK_FIREWALL: u64 = 0x3;
    pub const NETLINK_GENERIC: u64 = 0x10;
    pub const NETLINK_INET_DIAG: u64 = 0x4;
    pub const NETLINK_IP6_FW: u64 = 0xd;
    pub const NETLINK_ISCSI: u64 = 0x8;
    pub const NETLINK_KOBJECT_UEVENT: u64 = 0xf;
    pub const NETLINK_NETFILTER: u64 = 0xc;
    pub const NETLINK_NFLOG: u64 = 0x5;
    pub const NETLINK_NO_ENOBUFS: u64 = 0x5;
    pub const NETLINK_PKTINFO: u64 = 0x3;
    pub const NETLINK_ROUTE: u64 = 0x0;
    pub const NETLINK_SCSITRANSPORT: u64 = 0x12;
    pub const NETLINK_SELINUX: u64 = 0x7;
    pub const NETLINK_UNUSED: u64 = 0x1;
    pub const NETLINK_USERSOCK: u64 = 0x2;
    pub const NETLINK_XFRM: u64 = 0x6;
    pub const NLA_ALIGNTO: u64 = 0x4;
    pub const NLA_F_NESTED: u64 = 0x8000;
    pub const NLA_F_NET_BYTEORDER: u64 = 0x4000;
    pub const NLA_HDRLEN: u64 = 0x4;
    pub const NLMSG_ALIGNTO: u64 = 0x4;
    pub const NLMSG_DONE: u64 = 0x3;
    pub const NLMSG_ERROR: u64 = 0x2;
    pub const NLMSG_HDRLEN: u64 = 0x10;
    pub const NLMSG_MIN_TYPE: u64 = 0x10;
    pub const NLMSG_NOOP: u64 = 0x1;
    pub const NLMSG_OVERRUN: u64 = 0x4;
    pub const NLM_F_ACK: u64 = 0x4;
    pub const NLM_F_APPEND: u64 = 0x800;
    pub const NLM_F_ATOMIC: u64 = 0x400;
    pub const NLM_F_CREATE: u64 = 0x400;
    pub const NLM_F_DUMP: u64 = 0x300;
    pub const NLM_F_ECHO: u64 = 0x8;
    pub const NLM_F_EXCL: u64 = 0x200;
    pub const NLM_F_MATCH: u64 = 0x200;
    pub const NLM_F_MULTI: u64 = 0x2;
    pub const NLM_F_REPLACE: u64 = 0x100;
    pub const NLM_F_REQUEST: u64 = 0x1;
    pub const NLM_F_ROOT: u64 = 0x100;
    pub const O_ACCMODE: u64 = 0x3;
    pub const O_APPEND: u64 = 0x400;
    pub const O_ASYNC: u64 = 0x2000;
    pub const O_CLOEXEC: u64 = 0x80000;
    pub const O_CREAT: u64 = 0x40;
    pub const O_DSYNC: u64 = 0x1000;
    pub const O_EXCL: u64 = 0x80;
    pub const O_FSYNC: u64 = 0x101000;
    pub const O_NDELAY: u64 = 0x800;
    pub const O_NOATIME: u64 = 0x40000;
    pub const O_NOCTTY: u64 = 0x100;
    pub const O_NONBLOCK: u64 = 0x800;
    pub const O_RDONLY: u64 = 0x0;
    pub const O_RDWR: u64 = 0x2;
    pub const O_RSYNC: u64 = 0x101000;
    pub const O_SYNC: u64 = 0x101000;
    pub const O_TRUNC: u64 = 0x200;
    pub const O_WRONLY: u64 = 0x1;
    pub const PACKET_ADD_MEMBERSHIP: u64 = 0x1;
    pub const PACKET_BROADCAST: u64 = 0x1;
    pub const PACKET_DROP_MEMBERSHIP: u64 = 0x2;
    pub const PACKET_FASTROUTE: u64 = 0x6;
    pub const PACKET_HOST: u64 = 0x0;
    pub const PACKET_LOOPBACK: u64 = 0x5;
    pub const PACKET_MR_ALLMULTI: u64 = 0x2;
    pub const PACKET_MR_MULTICAST: u64 = 0x0;
    pub const PACKET_MR_PROMISC: u64 = 0x1;
    pub const PACKET_MULTICAST: u64 = 0x2;
    pub const PACKET_OTHERHOST: u64 = 0x3;
    pub const PACKET_OUTGOING: u64 = 0x4;
    pub const PACKET_RECV_OUTPUT: u64 = 0x3;
    pub const PACKET_RX_RING: u64 = 0x5;
    pub const PACKET_STATISTICS: u64 = 0x6;
    pub const PRIO_PGRP: u64 = 0x1;
    pub const PRIO_PROCESS: u64 = 0x0;
    pub const PRIO_USER: u64 = 0x2;
    pub const PROT_EXEC: u64 = 0x4;
    pub const PROT_GROWSDOWN: u64 = 0x1000000;
    pub const PROT_GROWSUP: u64 = 0x2000000;
    pub const PROT_NONE: u64 = 0x0;
    pub const PROT_READ: u64 = 0x1;
    pub const PROT_WRITE: u64 = 0x2;
    pub const PR_CAPBSET_DROP: u64 = 0x18;
    pub const PR_CAPBSET_READ: u64 = 0x17;
    pub const PR_ENDIAN_BIG: u64 = 0x0;
    pub const PR_ENDIAN_LITTLE: u64 = 0x1;
    pub const PR_ENDIAN_PPC_LITTLE: u64 = 0x2;
    pub const PR_FPEMU_NOPRINT: u64 = 0x1;
    pub const PR_FPEMU_SIGFPE: u64 = 0x2;
    pub const PR_FP_EXC_ASYNC: u64 = 0x2;
    pub const PR_FP_EXC_DISABLED: u64 = 0x0;
    pub const PR_FP_EXC_DIV: u64 = 0x10000;
    pub const PR_FP_EXC_INV: u64 = 0x100000;
    pub const PR_FP_EXC_NONRECOV: u64 = 0x1;
    pub const PR_FP_EXC_OVF: u64 = 0x20000;
    pub const PR_FP_EXC_PRECISE: u64 = 0x3;
    pub const PR_FP_EXC_RES: u64 = 0x80000;
    pub const PR_FP_EXC_SW_ENABLE: u64 = 0x80;
    pub const PR_FP_EXC_UND: u64 = 0x40000;
    pub const PR_GET_DUMPABLE: u64 = 0x3;
    pub const PR_GET_ENDIAN: u64 = 0x13;
    pub const PR_GET_FPEMU: u64 = 0x9;
    pub const PR_GET_FPEXC: u64 = 0xb;
    pub const PR_GET_KEEPCAPS: u64 = 0x7;
    pub const PR_GET_NAME: u64 = 0x10;
    pub const PR_GET_PDEATHSIG: u64 = 0x2;
    pub const PR_GET_SECCOMP: u64 = 0x15;
    pub const PR_GET_SECUREBITS: u64 = 0x1b;
    pub const PR_GET_TIMERSLACK: u64 = 0x1e;
    pub const PR_GET_TIMING: u64 = 0xd;
    pub const PR_GET_TSC: u64 = 0x19;
    pub const PR_GET_UNALIGN: u64 = 0x5;
    pub const PR_MCE_KILL: u64 = 0x21;
    pub const PR_MCE_KILL_CLEAR: u64 = 0x0;
    pub const PR_MCE_KILL_DEFAULT: u64 = 0x2;
    pub const PR_MCE_KILL_EARLY: u64 = 0x1;
    pub const PR_MCE_KILL_GET: u64 = 0x22;
    pub const PR_MCE_KILL_LATE: u64 = 0x0;
    pub const PR_MCE_KILL_SET: u64 = 0x1;
    pub const PR_SET_DUMPABLE: u64 = 0x4;
    pub const PR_SET_ENDIAN: u64 = 0x14;
    pub const PR_SET_FPEMU: u64 = 0xa;
    pub const PR_SET_FPEXC: u64 = 0xc;
    pub const PR_SET_KEEPCAPS: u64 = 0x8;
    pub const PR_SET_NAME: u64 = 0xf;
    pub const PR_SET_PDEATHSIG: u64 = 0x1;
    pub const PR_SET_PTRACER: u64 = 0x59616d61;
    pub const PR_SET_SECCOMP: u64 = 0x16;
    pub const PR_SET_SECUREBITS: u64 = 0x1c;
    pub const PR_SET_TIMERSLACK: u64 = 0x1d;
    pub const PR_SET_TIMING: u64 = 0xe;
    pub const PR_SET_TSC: u64 = 0x1a;
    pub const PR_SET_UNALIGN: u64 = 0x6;
    pub const PR_TASK_PERF_EVENTS_DISABLE: u64 = 0x1f;
    pub const PR_TASK_PERF_EVENTS_ENABLE: u64 = 0x20;
    pub const PR_TIMING_STATISTICAL: u64 = 0x0;
    pub const PR_TIMING_TIMESTAMP: u64 = 0x1;
    pub const PR_TSC_ENABLE: u64 = 0x1;
    pub const PR_TSC_SIGSEGV: u64 = 0x2;
    pub const PR_UNALIGN_NOPRINT: u64 = 0x1;
    pub const PR_UNALIGN_SIGBUS: u64 = 0x2;
    pub const PTRACE_ARCH_PRCTL: u64 = 0x1e;
    pub const PTRACE_ATTACH: u64 = 0x10;
    pub const PTRACE_CONT: u64 = 0x7;
    pub const PTRACE_DETACH: u64 = 0x11;
    pub const PTRACE_EVENT_CLONE: u64 = 0x3;
    pub const PTRACE_EVENT_EXEC: u64 = 0x4;
    pub const PTRACE_EVENT_EXIT: u64 = 0x6;
    pub const PTRACE_EVENT_FORK: u64 = 0x1;
    pub const PTRACE_EVENT_VFORK: u64 = 0x2;
    pub const PTRACE_EVENT_VFORK_DONE: u64 = 0x5;
    pub const PTRACE_GETEVENTMSG: u64 = 0x4201;
    pub const PTRACE_GETFPREGS: u64 = 0xe;
    pub const PTRACE_GETFPXREGS: u64 = 0x12;
    pub const PTRACE_GETREGS: u64 = 0xc;
    pub const PTRACE_GETREGSET: u64 = 0x4204;
    pub const PTRACE_GETSIGINFO: u64 = 0x4202;
    pub const PTRACE_GET_THREAD_AREA: u64 = 0x19;
    pub const PTRACE_KILL: u64 = 0x8;
    pub const PTRACE_OLDSETOPTIONS: u64 = 0x15;
    pub const PTRACE_O_MASK: u64 = 0x7f;
    pub const PTRACE_O_TRACECLONE: u64 = 0x8;
    pub const PTRACE_O_TRACEEXEC: u64 = 0x10;
    pub const PTRACE_O_TRACEEXIT: u64 = 0x40;
    pub const PTRACE_O_TRACEFORK: u64 = 0x2;
    pub const PTRACE_O_TRACESYSGOOD: u64 = 0x1;
    pub const PTRACE_O_TRACEVFORK: u64 = 0x4;
    pub const PTRACE_O_TRACEVFORKDONE: u64 = 0x20;
    pub const PTRACE_PEEKDATA: u64 = 0x2;
    pub const PTRACE_PEEKTEXT: u64 = 0x1;
    pub const PTRACE_PEEKUSR: u64 = 0x3;
    pub const PTRACE_POKEDATA: u64 = 0x5;
    pub const PTRACE_POKETEXT: u64 = 0x4;
    pub const PTRACE_POKEUSR: u64 = 0x6;
    pub const PTRACE_SETFPREGS: u64 = 0xf;
    pub const PTRACE_SETFPXREGS: u64 = 0x13;
    pub const PTRACE_SETOPTIONS: u64 = 0x4200;
    pub const PTRACE_SETREGS: u64 = 0xd;
    pub const PTRACE_SETREGSET: u64 = 0x4205;
    pub const PTRACE_SETSIGINFO: u64 = 0x4203;
    pub const PTRACE_SET_THREAD_AREA: u64 = 0x1a;
    pub const PTRACE_SINGLEBLOCK: u64 = 0x21;
    pub const PTRACE_SINGLESTEP: u64 = 0x9;
    pub const PTRACE_SYSCALL: u64 = 0x18;
    pub const PTRACE_SYSEMU: u64 = 0x1f;
    pub const PTRACE_SYSEMU_SINGLESTEP: u64 = 0x20;
    pub const PTRACE_TRACEME: u64 = 0x0;
    pub const RLIMIT_AS: u64 = 0x9;
    pub const RLIMIT_CORE: u64 = 0x4;
    pub const RLIMIT_CPU: u64 = 0x0;
    pub const RLIMIT_DATA: u64 = 0x2;
    pub const RLIMIT_FSIZE: u64 = 0x1;
    pub const RLIMIT_NOFILE: u64 = 0x7;
    pub const RLIMIT_STACK: u64 = 0x3;
    pub const RLIM_INFINITY: i64 = -0x1;
    pub const RTAX_ADVMSS: u64 = 0x8;
    pub const RTAX_CWND: u64 = 0x7;
    pub const RTAX_FEATURES: u64 = 0xc;
    pub const RTAX_FEATURE_ALLFRAG: u64 = 0x8;
    pub const RTAX_FEATURE_ECN: u64 = 0x1;
    pub const RTAX_FEATURE_SACK: u64 = 0x2;
    pub const RTAX_FEATURE_TIMESTAMP: u64 = 0x4;
    pub const RTAX_HOPLIMIT: u64 = 0xa;
    pub const RTAX_INITCWND: u64 = 0xb;
    pub const RTAX_INITRWND: u64 = 0xe;
    pub const RTAX_LOCK: u64 = 0x1;
    pub const RTAX_MAX: u64 = 0xe;
    pub const RTAX_MTU: u64 = 0x2;
    pub const RTAX_REORDERING: u64 = 0x9;
    pub const RTAX_RTO_MIN: u64 = 0xd;
    pub const RTAX_RTT: u64 = 0x4;
    pub const RTAX_RTTVAR: u64 = 0x5;
    pub const RTAX_SSTHRESH: u64 = 0x6;
    pub const RTAX_UNSPEC: u64 = 0x0;
    pub const RTAX_WINDOW: u64 = 0x3;
    pub const RTA_ALIGNTO: u64 = 0x4;
    pub const RTA_MAX: u64 = 0x10;
    pub const RTCF_DIRECTSRC: u64 = 0x4000000;
    pub const RTCF_DOREDIRECT: u64 = 0x1000000;
    pub const RTCF_LOG: u64 = 0x2000000;
    pub const RTCF_MASQ: u64 = 0x400000;
    pub const RTCF_NAT: u64 = 0x800000;
    pub const RTCF_VALVE: u64 = 0x200000;
    pub const RTF_ADDRCLASSMASK: u64 = 0xf8000000;
    pub const RTF_ADDRCONF: u64 = 0x40000;
    pub const RTF_ALLONLINK: u64 = 0x20000;
    pub const RTF_BROADCAST: u64 = 0x10000000;
    pub const RTF_CACHE: u64 = 0x1000000;
    pub const RTF_DEFAULT: u64 = 0x10000;
    pub const RTF_DYNAMIC: u64 = 0x10;
    pub const RTF_FLOW: u64 = 0x2000000;
    pub const RTF_GATEWAY: u64 = 0x2;
    pub const RTF_HOST: u64 = 0x4;
    pub const RTF_INTERFACE: u64 = 0x40000000;
    pub const RTF_IRTT: u64 = 0x100;
    pub const RTF_LINKRT: u64 = 0x100000;
    pub const RTF_LOCAL: u64 = 0x80000000;
    pub const RTF_MODIFIED: u64 = 0x20;
    pub const RTF_MSS: u64 = 0x40;
    pub const RTF_MTU: u64 = 0x40;
    pub const RTF_MULTICAST: u64 = 0x20000000;
    pub const RTF_NAT: u64 = 0x8000000;
    pub const RTF_NOFORWARD: u64 = 0x1000;
    pub const RTF_NONEXTHOP: u64 = 0x200000;
    pub const RTF_NOPMTUDISC: u64 = 0x4000;
    pub const RTF_POLICY: u64 = 0x4000000;
    pub const RTF_REINSTATE: u64 = 0x8;
    pub const RTF_REJECT: u64 = 0x200;
    pub const RTF_STATIC: u64 = 0x400;
    pub const RTF_THROW: u64 = 0x2000;
    pub const RTF_UP: u64 = 0x1;
    pub const RTF_WINDOW: u64 = 0x80;
    pub const RTF_XRESOLVE: u64 = 0x800;
    pub const RTM_BASE: u64 = 0x10;
    pub const RTM_DELACTION: u64 = 0x31;
    pub const RTM_DELADDR: u64 = 0x15;
    pub const RTM_DELADDRLABEL: u64 = 0x49;
    pub const RTM_DELLINK: u64 = 0x11;
    pub const RTM_DELNEIGH: u64 = 0x1d;
    pub const RTM_DELQDISC: u64 = 0x25;
    pub const RTM_DELROUTE: u64 = 0x19;
    pub const RTM_DELRULE: u64 = 0x21;
    pub const RTM_DELTCLASS: u64 = 0x29;
    pub const RTM_DELTFILTER: u64 = 0x2d;
    pub const RTM_F_CLONED: u64 = 0x200;
    pub const RTM_F_EQUALIZE: u64 = 0x400;
    pub const RTM_F_NOTIFY: u64 = 0x100;
    pub const RTM_F_PREFIX: u64 = 0x800;
    pub const RTM_GETACTION: u64 = 0x32;
    pub const RTM_GETADDR: u64 = 0x16;
    pub const RTM_GETADDRLABEL: u64 = 0x4a;
    pub const RTM_GETANYCAST: u64 = 0x3e;
    pub const RTM_GETDCB: u64 = 0x4e;
    pub const RTM_GETLINK: u64 = 0x12;
    pub const RTM_GETMULTICAST: u64 = 0x3a;
    pub const RTM_GETNEIGH: u64 = 0x1e;
    pub const RTM_GETNEIGHTBL: u64 = 0x42;
    pub const RTM_GETQDISC: u64 = 0x26;
    pub const RTM_GETROUTE: u64 = 0x1a;
    pub const RTM_GETRULE: u64 = 0x22;
    pub const RTM_GETTCLASS: u64 = 0x2a;
    pub const RTM_GETTFILTER: u64 = 0x2e;
    pub const RTM_MAX: u64 = 0x4f;
    pub const RTM_NEWACTION: u64 = 0x30;
    pub const RTM_NEWADDR: u64 = 0x14;
    pub const RTM_NEWADDRLABEL: u64 = 0x48;
    pub const RTM_NEWLINK: u64 = 0x10;
    pub const RTM_NEWNDUSEROPT: u64 = 0x44;
    pub const RTM_NEWNEIGH: u64 = 0x1c;
    pub const RTM_NEWNEIGHTBL: u64 = 0x40;
    pub const RTM_NEWPREFIX: u64 = 0x34;
    pub const RTM_NEWQDISC: u64 = 0x24;
    pub const RTM_NEWROUTE: u64 = 0x18;
    pub const RTM_NEWRULE: u64 = 0x20;
    pub const RTM_NEWTCLASS: u64 = 0x28;
    pub const RTM_NEWTFILTER: u64 = 0x2c;
    pub const RTM_NR_FAMILIES: u64 = 0x10;
    pub const RTM_NR_MSGTYPES: u64 = 0x40;
    pub const RTM_SETDCB: u64 = 0x4f;
    pub const RTM_SETLINK: u64 = 0x13;
    pub const RTM_SETNEIGHTBL: u64 = 0x43;
    pub const RTNH_ALIGNTO: u64 = 0x4;
    pub const RTNH_F_DEAD: u64 = 0x1;
    pub const RTNH_F_ONLINK: u64 = 0x4;
    pub const RTNH_F_PERVASIVE: u64 = 0x2;
    pub const RTN_MAX: u64 = 0xb;
    pub const RTPROT_BIRD: u64 = 0xc;
    pub const RTPROT_BOOT: u64 = 0x3;
    pub const RTPROT_DHCP: u64 = 0x10;
    pub const RTPROT_DNROUTED: u64 = 0xd;
    pub const RTPROT_GATED: u64 = 0x8;
    pub const RTPROT_KERNEL: u64 = 0x2;
    pub const RTPROT_MRT: u64 = 0xa;
    pub const RTPROT_NTK: u64 = 0xf;
    pub const RTPROT_RA: u64 = 0x9;
    pub const RTPROT_REDIRECT: u64 = 0x1;
    pub const RTPROT_STATIC: u64 = 0x4;
    pub const RTPROT_UNSPEC: u64 = 0x0;
    pub const RTPROT_XORP: u64 = 0xe;
    pub const RTPROT_ZEBRA: u64 = 0xb;
    pub const RT_CLASS_DEFAULT: u64 = 0xfd;
    pub const RT_CLASS_LOCAL: u64 = 0xff;
    pub const RT_CLASS_MAIN: u64 = 0xfe;
    pub const RT_CLASS_MAX: u64 = 0xff;
    pub const RT_CLASS_UNSPEC: u64 = 0x0;
    pub const RUSAGE_CHILDREN: i64 = -0x1;
    pub const RUSAGE_SELF: u64 = 0x0;
    pub const RUSAGE_THREAD: u64 = 0x1;
    pub const SCM_CREDENTIALS: u64 = 0x2;
    pub const SCM_RIGHTS: u64 = 0x1;
    pub const SCM_TIMESTAMP: u64 = 0x1d;
    pub const SCM_TIMESTAMPING: u64 = 0x25;
    pub const SCM_TIMESTAMPNS: u64 = 0x23;
    pub const SHUT_RD: u64 = 0x0;
    pub const SHUT_RDWR: u64 = 0x2;
    pub const SHUT_WR: u64 = 0x1;
    pub const SIOCADDDLCI: u64 = 0x8980;
    pub const SIOCADDMULTI: u64 = 0x8931;
    pub const SIOCADDRT: u64 = 0x890b;
    pub const SIOCATMARK: u64 = 0x8905;
    pub const SIOCDARP: u64 = 0x8953;
    pub const SIOCDELDLCI: u64 = 0x8981;
    pub const SIOCDELMULTI: u64 = 0x8932;
    pub const SIOCDELRT: u64 = 0x890c;
    pub const SIOCDEVPRIVATE: u64 = 0x89f0;
    pub const SIOCDIFADDR: u64 = 0x8936;
    pub const SIOCDRARP: u64 = 0x8960;
    pub const SIOCGARP: u64 = 0x8954;
    pub const SIOCGIFADDR: u64 = 0x8915;
    pub const SIOCGIFBR: u64 = 0x8940;
    pub const SIOCGIFBRDADDR: u64 = 0x8919;
    pub const SIOCGIFCONF: u64 = 0x8912;
    pub const SIOCGIFCOUNT: u64 = 0x8938;
    pub const SIOCGIFDSTADDR: u64 = 0x8917;
    pub const SIOCGIFENCAP: u64 = 0x8925;
    pub const SIOCGIFFLAGS: u64 = 0x8913;
    pub const SIOCGIFHWADDR: u64 = 0x8927;
    pub const SIOCGIFINDEX: u64 = 0x8933;
    pub const SIOCGIFMAP: u64 = 0x8970;
    pub const SIOCGIFMEM: u64 = 0x891f;
    pub const SIOCGIFMETRIC: u64 = 0x891d;
    pub const SIOCGIFMTU: u64 = 0x8921;
    pub const SIOCGIFNAME: u64 = 0x8910;
    pub const SIOCGIFNETMASK: u64 = 0x891b;
    pub const SIOCGIFPFLAGS: u64 = 0x8935;
    pub const SIOCGIFSLAVE: u64 = 0x8929;
    pub const SIOCGIFTXQLEN: u64 = 0x8942;
    pub const SIOCGPGRP: u64 = 0x8904;
    pub const SIOCGRARP: u64 = 0x8961;
    pub const SIOCGSTAMP: u64 = 0x8906;
    pub const SIOCGSTAMPNS: u64 = 0x8907;
    pub const SIOCPROTOPRIVATE: u64 = 0x89e0;
    pub const SIOCRTMSG: u64 = 0x890d;
    pub const SIOCSARP: u64 = 0x8955;
    pub const SIOCSIFADDR: u64 = 0x8916;
    pub const SIOCSIFBR: u64 = 0x8941;
    pub const SIOCSIFBRDADDR: u64 = 0x891a;
    pub const SIOCSIFDSTADDR: u64 = 0x8918;
    pub const SIOCSIFENCAP: u64 = 0x8926;
    pub const SIOCSIFFLAGS: u64 = 0x8914;
    pub const SIOCSIFHWADDR: u64 = 0x8924;
    pub const SIOCSIFHWBROADCAST: u64 = 0x8937;
    pub const SIOCSIFLINK: u64 = 0x8911;
    pub const SIOCSIFMAP: u64 = 0x8971;
    pub const SIOCSIFMEM: u64 = 0x8920;
    pub const SIOCSIFMETRIC: u64 = 0x891e;
    pub const SIOCSIFMTU: u64 = 0x8922;
    pub const SIOCSIFNAME: u64 = 0x8923;
    pub const SIOCSIFNETMASK: u64 = 0x891c;
    pub const SIOCSIFPFLAGS: u64 = 0x8934;
    pub const SIOCSIFSLAVE: u64 = 0x8930;
    pub const SIOCSIFTXQLEN: u64 = 0x8943;
    pub const SIOCSPGRP: u64 = 0x8902;
    pub const SIOCSRARP: u64 = 0x8962;
    pub const SOCK_CLOEXEC: u64 = 0x80000;
    pub const SOCK_DCCP: u64 = 0x6;
    pub const SOCK_DGRAM: u64 = 0x2;
    pub const SOCK_NONBLOCK: u64 = 0x800;
    pub const SOCK_PACKET: u64 = 0xa;
    pub const SOCK_RAW: u64 = 0x3;
    pub const SOCK_RDM: u64 = 0x4;
    pub const SOCK_SEQPACKET: u64 = 0x5;
    pub const SOCK_STREAM: u64 = 0x1;

    // socket option level constants
    pub const SOL_AAL: u64 = 0x109;
    pub const SOL_ATM: u64 = 0x108;
    pub const SOL_DECNET: u64 = 0x105;
    pub const SOL_ICMPV6: u64 = 0x3a;
    pub const SOL_IP: u64 = 0x0;
    pub const SOL_IPV6: u64 = 0x29;
    pub const SOL_IRDA: u64 = 0x10a;
    pub const SOL_PACKET: u64 = 0x107;
    pub const SOL_RAW: u64 = 0xff;
    pub const SOL_SOCKET: u64 = 0x1;
    pub const SOL_TCP: u64 = 0x6;
    pub const SOL_X25: u64 = 0x106;

    pub const SOMAXCONN: u64 = 0x80;
    pub const SO_ACCEPTCONN: u64 = 0x1e;
    pub const SO_ATTACH_FILTER: u64 = 0x1a;
    pub const SO_BINDTODEVICE: u64 = 0x19;
    pub const SO_BROADCAST: u64 = 0x6;
    pub const SO_BSDCOMPAT: u64 = 0xe;
    pub const SO_REUSEPORT: u64 = 0xf;
    pub const SO_DEBUG: u64 = 0x1;
    pub const SO_DETACH_FILTER: u64 = 0x1b;
    pub const SO_DOMAIN: u64 = 0x27;
    pub const SO_DONTROUTE: u64 = 0x5;
    pub const SO_ERROR: u64 = 0x4;
    pub const SO_KEEPALIVE: u64 = 0x9;
    pub const SO_LINGER: u64 = 0xd;
    pub const SO_MARK: u64 = 0x24;
    pub const SO_NO_CHECK: u64 = 0xb;
    pub const SO_OOBINLINE: u64 = 0xa;
    pub const SO_PASSCRED: u64 = 0x10;
    pub const SO_PASSSEC: u64 = 0x22;
    pub const SO_PEERCRED: u64 = 0x11;
    pub const SO_PEERNAME: u64 = 0x1c;
    pub const SO_PEERSEC: u64 = 0x1f;
    pub const SO_PRIORITY: u64 = 0xc;
    pub const SO_PROTOCOL: u64 = 0x26;
    pub const SO_RCVBUF: u64 = 0x8;
    pub const SO_RCVBUFFORCE: u64 = 0x21;
    pub const SO_RCVLOWAT: u64 = 0x12;
    pub const SO_RCVTIMEO: u64 = 0x14;
    pub const SO_REUSEADDR: u64 = 0x2;
    pub const SO_RXQ_OVFL: u64 = 0x28;
    pub const SO_SECURITY_AUTHENTICATION: u64 = 0x16;
    pub const SO_SECURITY_ENCRYPTION_NETWORK: u64 = 0x18;
    pub const SO_SECURITY_ENCRYPTION_TRANSPORT: u64 = 0x17;
    pub const SO_SNDBUF: u64 = 0x7;
    pub const SO_SNDBUFFORCE: u64 = 0x20;
    pub const SO_SNDLOWAT: u64 = 0x13;
    pub const SO_SNDTIMEO: u64 = 0x15;
    pub const SO_TIMESTAMP: u64 = 0x1d;
    pub const SO_TIMESTAMPING: u64 = 0x25;
    pub const SO_TIMESTAMPNS: u64 = 0x23;
    pub const SO_TYPE: u64 = 0x3;
    pub const S_BLKSIZE: u64 = 0x200;
    pub const S_IEXEC: u64 = 0x40;
    pub const S_IFBLK: u64 = 0x6000;
    pub const S_IFCHR: u64 = 0x2000;
    pub const S_IFDIR: u64 = 0x4000;
    pub const S_IFIFO: u64 = 0x1000;
    pub const S_IFLNK: u64 = 0xa000;
    pub const S_IFMT: u64 = 0xf000;
    pub const S_IFREG: u64 = 0x8000;
    pub const S_IFSOCK: u64 = 0xc000;
    pub const S_IREAD: u64 = 0x100;
    pub const S_IRGRP: u64 = 0x20;
    pub const S_IROTH: u64 = 0x4;
    pub const S_IRUSR: u64 = 0x100;
    pub const S_IRWXG: u64 = 0x38;
    pub const S_IRWXO: u64 = 0x7;
    pub const S_IRWXU: u64 = 0x1c0;
    pub const S_ISGID: u64 = 0x400;
    pub const S_ISUID: u64 = 0x800;
    pub const S_ISVTX: u64 = 0x200;
    pub const S_IWGRP: u64 = 0x10;
    pub const S_IWOTH: u64 = 0x2;
    pub const S_IWRITE: u64 = 0x80;
    pub const S_IWUSR: u64 = 0x80;
    pub const S_IXGRP: u64 = 0x8;
    pub const S_IXOTH: u64 = 0x1;
    pub const S_IXUSR: u64 = 0x40;
    pub const TCIFLUSH: u64 = 0x0;
    pub const TCIOFLUSH: u64 = 0x2;
    pub const TCOFLUSH: u64 = 0x1;
    pub const TCP_CONGESTION: u64 = 0xd;
    pub const TCP_CORK: u64 = 0x3;
    pub const TCP_DEFER_ACCEPT: u64 = 0x9;
    pub const TCP_INFO: u64 = 0xb;
    pub const TCP_KEEPCNT: u64 = 0x6;
    pub const TCP_KEEPIDLE: u64 = 0x4;
    pub const TCP_KEEPINTVL: u64 = 0x5;
    pub const TCP_LINGER2: u64 = 0x8;
    pub const TCP_MAXSEG: u64 = 0x2;
    pub const TCP_MAXWIN: u64 = 0xffff;
    pub const TCP_MAX_WINSHIFT: u64 = 0xe;
    pub const TCP_MD5SIG: u64 = 0xe;
    pub const TCP_MD5SIG_MAXKEYLEN: u64 = 0x50;
    pub const TCP_MSS: u64 = 0x200;
    pub const TCP_NODELAY: u64 = 0x1;
    pub const TCP_QUICKACK: u64 = 0xc;
    pub const TCP_SYNCNT: u64 = 0x7;
    pub const TCP_WINDOW_CLAMP: u64 = 0xa;
    pub const TCP_INQ: u64 = 0x24;
    pub const TIOCCBRK: u64 = 0x5428;
    pub const TIOCCONS: u64 = 0x541d;
    pub const TIOCEXCL: u64 = 0x540c;
    pub const TIOCGDEV: u64 = 0x80045432;
    pub const TIOCGETD: u64 = 0x5424;
    pub const TIOCGICOUNT: u64 = 0x545d;
    pub const TIOCGLCKTRMIOS: u64 = 0x5456;
    pub const TIOCGPGRP: u64 = 0x540f;
    pub const TIOCGPTN: u64 = 0x80045430;
    pub const TIOCGRS485: u64 = 0x542e;
    pub const TIOCGSERIAL: u64 = 0x541e;
    pub const TIOCGSID: u64 = 0x5429;
    pub const TIOCGSOFTCAR: u64 = 0x5419;
    pub const TIOCGWINSZ: u64 = 0x5413;
    pub const TIOCINQ: u64 = 0x541b;
    pub const TIOCLINUX: u64 = 0x541c;
    pub const TIOCMBIC: u64 = 0x5417;
    pub const TIOCMBIS: u64 = 0x5416;
    pub const TIOCMGET: u64 = 0x5415;
    pub const TIOCMIWAIT: u64 = 0x545c;
    pub const TIOCMSET: u64 = 0x5418;
    pub const TIOCM_CAR: u64 = 0x40;
    pub const TIOCM_CD: u64 = 0x40;
    pub const TIOCM_CTS: u64 = 0x20;
    pub const TIOCM_DSR: u64 = 0x100;
    pub const TIOCM_DTR: u64 = 0x2;
    pub const TIOCM_LE: u64 = 0x1;
    pub const TIOCM_RI: u64 = 0x80;
    pub const TIOCM_RNG: u64 = 0x80;
    pub const TIOCM_RTS: u64 = 0x4;
    pub const TIOCM_SR: u64 = 0x10;
    pub const TIOCM_ST: u64 = 0x8;
    pub const TIOCNOTTY: u64 = 0x5422;
    pub const TIOCNXCL: u64 = 0x540d;
    pub const TIOCOUTQ: u64 = 0x5411;
    pub const TIOCPKT: u64 = 0x5420;
    pub const TIOCPKT_DATA: u64 = 0x0;
    pub const TIOCPKT_DOSTOP: u64 = 0x20;
    pub const TIOCPKT_FLUSHREAD: u64 = 0x1;
    pub const TIOCPKT_FLUSHWRITE: u64 = 0x2;
    pub const TIOCPKT_IOCTL: u64 = 0x40;
    pub const TIOCPKT_NOSTOP: u64 = 0x10;
    pub const TIOCPKT_START: u64 = 0x8;
    pub const TIOCPKT_STOP: u64 = 0x4;
    pub const TIOCSBRK: u64 = 0x5427;
    pub const TIOCSCTTY: u64 = 0x540e;
    pub const TIOCSERCONFIG: u64 = 0x5453;
    pub const TIOCSERGETLSR: u64 = 0x5459;
    pub const TIOCSERGETMULTI: u64 = 0x545a;
    pub const TIOCSERGSTRUCT: u64 = 0x5458;
    pub const TIOCSERGWILD: u64 = 0x5454;
    pub const TIOCSERSETMULTI: u64 = 0x545b;
    pub const TIOCSERSWILD: u64 = 0x5455;
    pub const TIOCSER_TEMT: u64 = 0x1;
    pub const TIOCSETD: u64 = 0x5423;
    pub const TIOCSIG: u64 = 0x40045436;
    pub const TIOCSLCKTRMIOS: u64 = 0x5457;
    pub const TIOCSPGRP: u64 = 0x5410;
    pub const TIOCSPTLCK: u64 = 0x40045431;
    pub const TIOCSRS485: u64 = 0x542f;
    pub const TIOCSSERIAL: u64 = 0x541f;
    pub const TIOCSSOFTCAR: u64 = 0x541a;
    pub const TIOCSTI: u64 = 0x5412;
    pub const TIOCSWINSZ: u64 = 0x5414;
    pub const TUNATTACHFILTER: u64 = 0x401054d5;
    pub const TUNDETACHFILTER: u64 = 0x401054d6;
    pub const TUNGETFEATURES: u64 = 0x800454cf;
    pub const TUNGETIFF: u64 = 0x800454d2;
    pub const TUNGETSNDBUF: u64 = 0x800454d3;
    pub const TUNGETVNETHDRSZ: u64 = 0x800454d7;
    pub const TUNSETDEBUG: u64 = 0x400454c9;
    pub const TUNSETGROUP: u64 = 0x400454ce;
    pub const TUNSETIFF: u64 = 0x400454ca;
    pub const TUNSETLINK: u64 = 0x400454cd;
    pub const TUNSETNOCSUM: u64 = 0x400454c8;
    pub const TUNSETOFFLOAD: u64 = 0x400454d0;
    pub const TUNSETOWNER: u64 = 0x400454cc;
    pub const TUNSETPERSIST: u64 = 0x400454cb;
    pub const TUNSETSNDBUF: u64 = 0x400454d4;
    pub const TUNSETTXFILTER: u64 = 0x400454d1;
    pub const TUNSETVNETHDRSZ: u64 = 0x400454d8;
    pub const WALL: u64 = 0x40000000;
    pub const WCLONE: u64 = 0x80000000;
    pub const WCONTINUED: u64 = 0x8;
    pub const WEXITED: u64 = 0x4;
    pub const WNOHANG: u64 = 0x1;
    pub const WNOTHREAD: u64 = 0x20000000;
    pub const WNOWAIT: u64 = 0x1000000;
    pub const WORDSIZE: u64 = 0x40;
    pub const WSTOPPED: u64 = 0x2;
    pub const WUNTRACED: u64 = 0x2;
}

pub struct Cmd {}

impl Cmd {
    pub const F_DUPFD: i32 = 0;
    pub const F_GETFD: i32 = 1;
    pub const F_SETFD: i32 = 2;
    pub const F_GETFL: i32 = 3;
    pub const F_SETFL: i32 = 4;
    pub const F_GETLK: i32 = 5;
    pub const F_SETLK: i32 = 6;
    pub const F_SETLKW: i32 = 7;
    pub const F_SETOWN: i32 = 8;
    pub const F_GETOWN: i32 = 9;
    pub const F_SETSIG: i32 = 10;
    pub const F_GETSIG: i32 = 11;
    pub const F_SETOWN_EX: i32 = 15;
    pub const F_GETOWN_EX: i32 = 16;
    pub const F_DUPFD_CLOEXEC: i32 = 1024 + 6;
    pub const F_SETPIPE_SZ: i32 = 1024 + 7;
    pub const F_GETPIPE_SZ: i32 = 1024 + 8;
    pub const F_ADD_SEALS: i32 = 1024 + 9;
    pub const F_GET_SEALS: i32 = 1024 + 10;

    pub const CLOSE_RANGE_UNSHARE: i32 = 1 << 1;
    pub const CLOSE_RANGE_CLOEXEC: i32 = 1 << 2;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PermMask {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl PermMask {
    pub fn NewReadWrite() -> Self {
        return Self {
            read: true,
            write: true,
            execute: false,
        };
    }

    pub fn FromFlags(mask: u32) -> Self {
        let mut res = PermMask::default();
        if mask & Flags::O_TRUNC as u32 != 0 {
            res.write = true;
        }

        let v = mask & Flags::O_ACCMODE as u32;
        if v == Flags::O_WRONLY as u32 {
            res.write = true;
        } else if v == Flags::O_RDWR as u32 {
            res.write = true;
            res.read = true;
        } else if v == Flags::O_RDONLY as u32 {
            res.read = true;
        }

        return res;
    }

    pub fn FromMode(mode: FileMode) -> Self {
        let mut res = Self::default();

        res.read = mode.OtherRead();
        res.write = mode.OtherWrite();
        res.execute = mode.OtherExec();

        return res;
    }

    pub fn OnlyRead(&self) -> bool {
        return self.read && !self.write && !self.execute;
    }

    pub fn Mode(&self) -> u32 {
        let mut ret = 0;
        if self.read {
            ret |= LibcConst::S_IROTH;
        }

        if self.write {
            ret |= LibcConst::S_IWOTH;
        }

        if self.execute {
            ret |= LibcConst::S_IXOTH;
        }

        return ret as u32;
    }

    pub fn SupersetOf(&self, other: &Self) -> bool {
        if !self.read && other.read {
            return false;
        }

        if !self.write && other.write {
            return false;
        }

        if !self.execute && other.execute {
            return false;
        }

        return true;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Flags(pub i32);

#[cfg(target_arch = "x86_64")]
impl Flags {
    pub const O_DIRECT: i32 = 0o00040000; //0x00004000;
    pub const O_LARGEFILE: i32 = 0o00100000; //0x00008000;
    pub const O_DIRECTORY: i32 = 0o00200000; //0x00010000;
    pub const O_NOFOLLOW: i32 = 0o00400000; //0x00020000;
}

#[cfg(target_arch = "aarch64")]
impl Flags {
    pub const O_DIRECT: i32 = 0o000200000;
    pub const O_LARGEFILE: i32 = 0o000400000;
    pub const O_DIRECTORY: i32 = 0o000040000;
    pub const O_NOFOLLOW: i32 = 0o000100000;
}

impl Flags {
    pub const O_ACCMODE: i32 = 0o00000003; //0x00000003;
    pub const O_RDONLY: i32 = 0o00000000; //0x00000000;
    pub const O_WRONLY: i32 = 0o00000001; //0x00000001;
    pub const O_RDWR: i32 = 0o00000002; //0x00000002;

    pub const O_CREAT: i32 = 0o00000100; //0x00000040;
    pub const O_EXCL: i32 = 0o00000200; //0x00000080;
    pub const O_NOCTTY: i32 = 0o00000400; //0x00000100;
    pub const O_TRUNC: i32 = 0o00001000; //0x00000200;
    pub const O_APPEND: i32 = 0o00002000; //0x00000400;
    pub const O_NONBLOCK: i32 = 0o00004000; //0x00000800;
    pub const O_DSYNC: i32 = 0o00010000; //0x00001000;
    pub const O_ASYNC: i32 = 0o00020000; //0x00002000;
    pub const O_NOATIME: i32 = 0o01000000; //0x00040000;
    pub const O_CLOEXEC: i32 = 0o02000000; //0x00080000;
    pub const O_SYNC: i32 = 0o04000000; //0x00100000;
    pub const O_PATH: i32 = 0o010000000; //0x00200000;
    pub const O_TMPFILE: i32 = 0o020000000;

    /* high priority request, poll if possible */
    pub const RWF_HIPRI: i32 = 0x00000001;

    /* per-IO O_DSYNC */
    pub const RWF_DSYNC: i32 = 0x00000002;

    /* per-IO O_SYNC */
    pub const RWF_SYNC: i32 = 0x00000004;

    /* per-IO, return -EAGAIN if operation would block */
    pub const RWF_NOWAIT: i32 = 0x00000008;

    /* per-IO O_APPEND */
    pub const RWF_APPEND: i32 = 0x00000010;

    pub const RWF_VALID: i32 = Self::RWF_HIPRI | Self::RWF_DSYNC | Self::RWF_SYNC;

    //pub fn Direct(&self) -> bool {
    //    return self.0 & Self::O_DIRECT != 0;
    //}

    pub fn Sync(&self) -> bool {
        return self.0 & Self::O_SYNC != 0;
    }

    pub fn NonBlocking(&self) -> bool {
        return self.0 & Self::O_NONBLOCK != 0;
    }

    pub fn Read(&self) -> bool {
        return self.0 & Self::O_ACCMODE != Self::O_WRONLY;
    }

    pub fn Write(&self) -> bool {
        return self.0 & Self::O_ACCMODE != Self::O_RDONLY;
    }

    pub fn Append(&self) -> bool {
        return self.0 & Self::O_APPEND != 0;
    }

    pub fn CloseOnExec(&self) -> bool {
        return self.0 & Self::O_CLOEXEC != 0;
    }

    //pub fn Directory(&self) -> bool {
    //    return self.0 & Self::O_DIRECTORY != 0;
    //}

    //pub fn Async(&self) -> bool {
    //    return self.0 & Self::O_ASYNC != 0;
    //}

    //pub fn LargeFile(&self) -> bool {
    //    return self.0 & Self::O_LARGEFILE != 0;
    //}

    pub fn ToPermission(&self) -> PermMask {
        let mut res = PermMask {
            read: false,
            write: false,
            execute: false,
        };

        if self.0 & Self::O_TRUNC != 0 {
            res.write = true;
        }

        let access = self.0 & Self::O_ACCMODE;

        if access == Self::O_WRONLY {
            res.write = true;
        } else if access == Self::O_RDWR {
            res.write = true;
            res.read = true;
        } else if access == Self::O_RDONLY {
            //
            res.read = true
        } else {
            panic!("...");
        }

        return res;
    }
}

pub struct MsgType {} //sendmsg/recvmsg flags

impl MsgType {
    pub const MSG_OOB: i32 = 0x1;
    pub const MSG_PEEK: i32 = 0x2;
    pub const MSG_DONTROUTE: i32 = 0x4;
    pub const MSG_TRYHARD: i32 = 0x4;
    pub const MSG_CTRUNC: i32 = 0x8;
    pub const MSG_PROBE: i32 = 0x10;
    pub const MSG_TRUNC: i32 = 0x20;
    pub const MSG_DONTWAIT: i32 = 0x40;
    pub const MSG_EOR: i32 = 0x80;
    pub const MSG_WAITALL: i32 = 0x100;
    pub const MSG_FIN: i32 = 0x200;
    pub const MSG_EOF: i32 = Self::MSG_FIN;
    pub const MSG_SYN: i32 = 0x400;
    pub const MSG_CONFIRM: i32 = 0x800;
    pub const MSG_RST: i32 = 0x1000;
    pub const MSG_ERRQUEUE: i32 = 0x2000;
    pub const MSG_NOSIGNAL: i32 = 0x4000;
    pub const MSG_MORE: i32 = 0x8000;
    pub const MSG_WAITFORONE: i32 = 0x10000;
    pub const MSG_SENDPAGE_NOTLAST: i32 = 0x20000;
    pub const MSG_REINJECT: i32 = 0x8000000;
    pub const MSG_ZEROCOPY: i32 = 0x4000000;
    pub const MSG_FASTOPEN: i32 = 0x20000000;
    pub const MSG_CMSG_CLOEXEC: i32 = 0x40000000;

    pub const BASE_RECV_FLAGS: i32 = Self::MSG_OOB
        | Self::MSG_DONTROUTE
        | Self::MSG_DONTWAIT
        | Self::MSG_NOSIGNAL
        | Self::MSG_WAITALL
        | Self::MSG_TRUNC
        | Self::MSG_CTRUNC;
}

pub struct AFType {} //Address Family

impl AFType {
    pub const AF_UNSPEC: i32 = 0;
    pub const AF_UNIX: i32 = 1;
    pub const AF_INET: i32 = 2;
    pub const AF_AX25: i32 = 3;
    pub const AF_IPX: i32 = 4;
    pub const AF_APPLETALK: i32 = 5;
    pub const AF_NETROM: i32 = 6;
    pub const AF_BRIDGE: i32 = 7;
    pub const AF_ATMPVC: i32 = 8;
    pub const AF_X25: i32 = 9;
    pub const AF_INET6: i32 = 10;
    pub const AF_ROSE: i32 = 11;
    pub const AF_DECNET: i32 = 12; //AF_DECnet
    pub const AF_NETBEUI: i32 = 13;
    pub const AF_SECURITY: i32 = 14;
    pub const AF_KEY: i32 = 15;
    pub const AF_NETLINK: i32 = 16;
    pub const AF_PACKET: i32 = 17;
    pub const AF_ASH: i32 = 18;
    pub const AF_ECONET: i32 = 19;
    pub const AF_ATMSVC: i32 = 20;
    pub const AF_RDS: i32 = 21;
    pub const AF_SNA: i32 = 22;
    pub const AF_IRDA: i32 = 23;
    pub const AF_PPPOX: i32 = 24;
    pub const AF_WANPIPE: i32 = 25;
    pub const AF_LLC: i32 = 26;
    pub const AF_IB: i32 = 27;
    pub const AF_MPLS: i32 = 28;
    pub const AF_CAN: i32 = 29;
    pub const AF_TIPC: i32 = 30;
    pub const AF_BLUETOOTH: i32 = 31;
    pub const AF_IUCV: i32 = 32;
    pub const AF_RXRPC: i32 = 33;
    pub const AF_ISDN: i32 = 34;
    pub const AF_PHONET: i32 = 35;
    pub const AF_IEEE802154: i32 = 36;
    pub const AF_CAIF: i32 = 37;
    pub const AF_ALG: i32 = 38;
    pub const AF_NFC: i32 = 39;
    pub const AF_VSOCK: i32 = 40;
}

pub struct SocketType {}

impl SocketType {
    pub const SOCK_STREAM: i32 = 1;
    pub const SOCK_DGRAM: i32 = 2;
    pub const SOCK_RAW: i32 = 3;
    pub const SOCK_RDM: i32 = 4;
    pub const SOCK_SEQPACKET: i32 = 5;
    pub const SOCK_DCCP: i32 = 6;
    pub const SOCK_PACKET: i32 = 10;

    pub const SOCK_TYPE_MASK: i32 = 0xf;
}

pub struct SocketFlags {}

impl SocketFlags {
    pub const SOCK_CLOEXEC: i32 = Flags::O_CLOEXEC;
    pub const SOCK_NONBLOCK: i32 = Flags::O_NONBLOCK;
}

pub const UIO_MAXIOV: usize = 1024;

#[repr(C)]
#[derive(Clone, Default, Debug, Copy, Eq, PartialEq)]
pub struct IoVec {
    pub start: u64,
    pub len: usize,
}

#[derive(Debug)]
pub struct DataBuff {
    pub buf: Vec<u8, GuestHostSharedAllocator>,
}

use super::kernel::tcpip::tcpip::SockAddrInet;
use super::mem::seq::BlockSeq;

impl DataBuff {
    pub fn New(size: usize) -> Self {
        // allocate memory even size is zero. So that Ptr() can get valid address
        let count = if size > 0 { size } else { 1 };
        let mut buf = Vec::with_capacity_in(count, GUEST_HOST_SHARED_ALLOCATOR);
        buf.resize(size, 0);
        return Self { buf: buf };
    }

    pub fn Zero(&mut self) {
        for i in 0..self.buf.len() {
            self.buf[i] = 0;
        }
    }

    pub fn Ptr(&self) -> u64 {
        return self.buf.as_ptr() as u64;
    }

    pub fn Len(&self) -> usize {
        return self.buf.len();
    }

    pub fn IoVec(&self, len: usize) -> IoVec {
        return IoVec {
            start: self.Ptr(),
            len: len,
        };
    }

    pub fn Iovs(&self, len: usize) -> Vec<IoVec, GuestHostSharedAllocator> {
        let mut iovs = Vec::with_capacity_in(1, GUEST_HOST_SHARED_ALLOCATOR);
        iovs.push(self.IoVec(len));
        return iovs;
    }

    pub fn BlockSeq(&self) -> BlockSeq {
        return BlockSeq::New(&self.buf);
    }

    pub fn BlockSeqWithLen(&self, len: usize) -> BlockSeq {
        return BlockSeq::New(&self.buf[0..len]);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct MsgHdr {
    pub msgName: u64,
    pub nameLen: u32,
    pub iov: u64,
    //*mut IoVec,
    pub iovLen: usize,
    pub msgControl: u64,
    pub msgControlLen: usize,
    pub msgFlags: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct MMsgHdr {
    pub msgHdr: MsgHdr,
    pub msgLen: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct PollFd {
    pub fd: i32,
    pub events: i16,
    pub revents: i16,
}

pub const AT_FDCWD: i32 = -100;

pub struct PollConst {}

impl PollConst {
    pub const POLLIN: i32 = 0x0001;
    pub const POLLPRI: i32 = 0x0002;
    pub const POLLOUT: i32 = 0x0004;
    pub const POLLERR: i32 = 0x0008;
    pub const POLLHUP: i32 = 0x0010;
    pub const POLLNVAL: i32 = 0x0020;
    pub const POLLRDNORM: i32 = 0x0040;
    pub const POLLRDBAND: i32 = 0x0080;
    pub const POLLWRNORM: i32 = 0x0100;
    pub const POLLWRBAND: i32 = 0x0200;
    pub const POLLMSG: i32 = 0x0400;
    pub const POLLREMOVE: i32 = 0x1000;
    pub const POLLRDHUP: i32 = 0x2000;
    pub const POLLFREE: i32 = 0x4000;
    pub const POLL_BUSY_LOOP: i32 = 0x8000;
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct iocb {
    pub aio_data: u64,
    pub aio_key: u32,
    pub aio_reserved1: i32,
    pub aio_lio_opcode: i16,
    pub aio_reqprio: i16,
    pub aio_fildes: u32,
    pub aio_buf: u64,
    pub aio_nbytes: u64,
    pub aio_offset: i64,
    pub aio_reserved2: u64,
    pub aio_flags: u32,
    pub aio_resfd: u32,
}

pub struct EflagsDef {}

impl EflagsDef {
    pub const EFLAGS_CF: u64 = 1 << 0;
    // pub const EFLAGS_PF is the mask for the parity flag.
    pub const EFLAGS_PF: u64 = 1 << 2;
    // pub const EFLAGS_AF is the mask for the auxiliary carry flag.
    pub const EFLAGS_AF: u64 = 1 << 4;
    // pub const EFLAGS_ZF is the mask for the zero flag.
    pub const EFLAGS_ZF: u64 = 1 << 6;
    // pub const EFLAGS_SF is the mask for the sign flag.
    pub const EFLAGS_SF: u64 = 1 << 7;
    // pub const EFLAGS_TF is the mask for the trap flag.
    pub const EFLAGS_TF: u64 = 1 << 8;
    // pub const EFLAGS_IF is the mask for the interrupt flag.
    pub const EFLAGS_IF: u64 = 1 << 9;
    // pub const EFLAGS_DF is the mask for the direction flag.
    pub const EFLAGS_DF: u64 = 1 << 10;
    // pub const EFLAGS_OF is the mask for the overflow flag.
    pub const EFLAGS_OF: u64 = 1 << 11;
    // pub const EFLAGS_IOPL is the mask for the I/O privilege level.
    pub const EFLAGS_IOPL: u64 = 3 << 12;
    // pub const EFLAGS_NT is the mask for the nested task bit.
    pub const EFLAGS_NT: u64 = 1 << 14;
    // pub const EFLAGS_RF is the mask for the resume flag.
    pub const EFLAGS_RF: u64 = 1 << 16;
    // pub const EFLAGS_VM is the mask for the virtual mode bit.
    pub const EFLAGS_VM: u64 = 1 << 17;
    // pub const EFLAGS_AC is the mask for the alignment check / access control bit.
    pub const EFLAGS_AC: u64 = 1 << 18;
    // pub const EFLAGS_VIF is the mask for the virtual interrupt flag.
    pub const EFLAGS_VIF: u64 = 1 << 19;
    // pub const EFLAGS_VIP is the mask for the virtual interrupt pending bit.
    pub const EFLAGS_VIP: u64 = 1 << 20;
    // pub const EFLAGS_ID is the mask for the CPUID detection bit.
    pub const EFLAGS_ID: u64 = 1 << 21;

    // eflagsPtraceMutable is the mask for the set of EFLAGS that may be
    // changed by ptrace(PTRACE_SETREGS). eflagsPtraceMutable is analogous to
    // Linux's FLAG_MASK.
    pub const EFLAGS_PTRACE_MUTABLE: u64 = Self::EFLAGS_CF
        | Self::EFLAGS_PF
        | Self::EFLAGS_AF
        | Self::EFLAGS_ZF
        | Self::EFLAGS_SF
        | Self::EFLAGS_TF
        | Self::EFLAGS_DF
        | Self::EFLAGS_OF
        | Self::EFLAGS_RF
        | Self::EFLAGS_AC
        | Self::EFLAGS_NT;

    // EFLAGS_Restorable is the mask for the set of EFLAGS_ that may be changed by
    // SignalReturn. EFLAGS_Restorable is analogous to Linux's FIX_EFLAGS_.
    pub const EFLAGS_RESTOREABLE: u64 = Self::EFLAGS_AC
        | Self::EFLAGS_OF
        | Self::EFLAGS_DF
        | Self::EFLAGS_TF
        | Self::EFLAGS_SF
        | Self::EFLAGS_ZF
        | Self::EFLAGS_AF
        | Self::EFLAGS_PF
        | Self::EFLAGS_CF
        | Self::EFLAGS_RF;
}

pub struct IoCtlCmd {}

impl IoCtlCmd {
    pub const TCGETS: u64 = 0x00005401;
    pub const TCSETS: u64 = 0x00005402;
    pub const TCSETSW: u64 = 0x00005403;
    pub const TCSETSF: u64 = 0x00005404;
    pub const TCSBRK: u64 = 0x00005409;
    pub const TIOCEXCL: u64 = 0x0000540c;
    pub const TIOCNXCL: u64 = 0x0000540d;
    pub const TIOCSCTTY: u64 = 0x0000540e;
    pub const TIOCGPGRP: u64 = 0x0000540f;
    pub const TIOCSPGRP: u64 = 0x00005410;
    pub const TIOCOUTQ: u64 = 0x00005411;
    pub const TIOCSTI: u64 = 0x00005412;
    pub const TIOCGWINSZ: u64 = 0x00005413;
    pub const TIOCSWINSZ: u64 = 0x00005414;
    pub const TIOCMGET: u64 = 0x00005415;
    pub const TIOCMBIS: u64 = 0x00005416;
    pub const TIOCMBIC: u64 = 0x00005417;
    pub const TIOCMSET: u64 = 0x00005418;
    pub const TIOCINQ: u64 = 0x0000541b;
    pub const FIONREAD: u64 = Self::TIOCINQ;
    pub const FIONBIO: u64 = 0x00005421;
    pub const TIOCSETD: u64 = 0x00005423;
    pub const TIOCNOTTY: u64 = 0x00005422;
    pub const TIOCGETD: u64 = 0x00005424;
    pub const TCSBRKP: u64 = 0x00005425;
    pub const TIOCSBRK: u64 = 0x00005427;
    pub const TIOCCBRK: u64 = 0x00005428;
    pub const TIOCGSID: u64 = 0x00005429;
    pub const TIOCGPTN: u64 = 0x80045430;
    pub const TIOCSPTLCK: u64 = 0x40045431;
    pub const TIOCGDEV: u64 = 0x80045432;
    pub const TIOCVHANGUP: u64 = 0x00005437;
    pub const TCFLSH: u64 = 0x0000540b;
    pub const TIOCCONS: u64 = 0x0000541d;
    pub const TIOCSSERIAL: u64 = 0x0000541f;
    pub const TIOCGEXCL: u64 = 0x80045440;
    pub const TIOCGPTPEER: u64 = 0x80045441;
    pub const TIOCGICOUNT: u64 = 0x0000545d;
    pub const FIONCLEX: u64 = 0x00005450;
    pub const FIOCLEX: u64 = 0x00005451;
    pub const FIOASYNC: u64 = 0x00005452;
    pub const FIOSETOWN: u64 = 0x00008901;
    pub const SIOCSPGRP: u64 = 0x00008902;
    pub const FIOGETOWN: u64 = 0x00008903;
    pub const SIOCGPGRP: u64 = 0x00008904;
}

#[derive(Clone, PartialEq, Copy, Debug)]
pub enum ProcessState {
    Running,
    Exiting,
    Zombie,
    Stop,
}

impl Default for ProcessState {
    fn default() -> Self {
        ProcessState::Running
    }
}

#[derive(Clone, PartialEq, Copy, Debug)]
pub enum ThreadState {
    Runable,
    Interruptable,
    UnInterruptable,
    Interrupted,
    Exiting,
    Zombie,
    Stoping,
    Stop,
}

impl Default for ThreadState {
    fn default() -> Self {
        ThreadState::Runable
    }
}

pub struct TaskEvent {}

impl TaskEvent {
    pub const EXIT: u32 = 1 << 0;
    pub const CHILD_GROUP_STOP: u32 = 1 << 1;
    pub const TRACE_STOP: u32 = 1 << 2;
    pub const GROUP_CONTINUE: u32 = 1 << 3;
}

pub struct WaitStatus(pub u32);

impl WaitStatus {
    pub const MASK: u32 = 0x7F;
    pub const CORE: u32 = 0x80;
    pub const EXITED: u32 = 0x00;
    pub const STOPPED: u32 = 0x7F;
    pub const SHIFT: u32 = 8;

    pub fn Exited(&self) -> bool {
        return self.0 & Self::MASK == Self::EXITED;
    }

    pub fn Signaled(&self) -> bool {
        return self.0 & Self::MASK != Self::STOPPED && self.0 & Self::MASK != Self::EXITED;
    }

    pub fn Stopped(&self) -> bool {
        return self.0 & 0xff == Self::STOPPED;
    }

    pub fn Continued(&self) -> bool {
        return self.0 == 0xFFFF;
    }

    pub fn CoreDump(&self) -> bool {
        return self.Signaled() && self.0 & Self::CORE != 0;
    }

    pub fn ExitStatus(&self) -> i32 {
        if !self.Exited() {
            return -1;
        }

        return (self.0 >> Self::SHIFT) as i32 & 0xff;
    }

    pub fn Signal(&self) -> i32 {
        if !self.Signaled() {
            return -1;
        }

        return (self.0 & Self::MASK) as i32;
    }

    pub fn StopSignal(&self) -> i32 {
        if !self.Stopped() {
            return -1;
        }

        return (self.0 >> Self::SHIFT) as i32 & 0xff;
    }

    pub fn TrapCause(&self) -> i32 {
        if self.StopSignal() != Signal::SIGTRAP {
            return -1;
        }

        return (self.0 >> Self::SHIFT) as i32 >> 8;
    }
}

#[derive(Clone, Copy)]
pub struct WaitOption {}

impl WaitOption {
    // Options for waitpid(2), wait4(2), and/or waitid(2), from
    // include/uapi/linux/wait.h.
    pub const WNOHANG: u32 = 0x00000001;
    pub const WUNTRACED: u32 = 0x00000002;
    pub const WSTOPPED: u32 = Self::WUNTRACED;
    pub const WEXITED: u32 = 0x00000004;
    pub const WCONTINUED: u32 = 0x00000008;
    pub const WNOWAIT: u32 = 0x01000000;
    pub const WNOTHREAD: u32 = 0x20000000;
    pub const WALL: u32 = 0x40000000;
    pub const WCLONE: u32 = 0x80000000;
}

pub struct IDType {}

impl IDType {
    pub const P_ALL: i32 = 0x0;
    pub const P_PID: i32 = 0x1;
    pub const P_PGID: i32 = 0x2;
}

pub struct MAdviseOp {}

impl MAdviseOp {
    pub const MADV_NORMAL: i32 = 0;
    pub const MADV_RANDOM: i32 = 1;
    pub const MADV_SEQUENTIAL: i32 = 2;
    pub const MADV_WILLNEED: i32 = 3;
    pub const MADV_DONTNEED: i32 = 4;
    pub const MADV_REMOVE: i32 = 9;
    pub const MADV_DONTFORK: i32 = 10;
    pub const MADV_DOFORK: i32 = 11;
    pub const MADV_MERGEABLE: i32 = 12;
    pub const MADV_UNMERGEABLE: i32 = 13;
    pub const MADV_HUGEPAGE: i32 = 14;
    pub const MADV_NOHUGEPAGE: i32 = 15;
    pub const MADV_DONTDUMP: i32 = 16;
    pub const MADV_DODUMP: i32 = 17;
    pub const MADV_HWPOISON: i32 = 100;
    pub const MADV_SOFT_OFFLINE: i32 = 101;
    pub const MADV_NOMAJFAULT: i32 = 200;
    pub const MADV_DONTCHGME: i32 = 201;
}

pub struct CloneOp {}

impl CloneOp {
    pub const CLONE_CHILD_CLEARTID: i32 = 0x200000;
    pub const CLONE_CHILD_SETTID: i32 = 0x1000000;
    pub const CLONE_DETACHED: i32 = 0x400000;
    pub const CLONE_FILES: i32 = 0x400;
    pub const CLONE_FS: i32 = 0x200;
    pub const CLONE_IO: u64 = 0x80000000;
    pub const CLONE_NEWIPC: i32 = 0x8000000;
    pub const CLONE_NEWNET: i32 = 0x40000000;
    pub const CLONE_NEWNS: i32 = 0x20000;
    pub const CLONE_NEWPID: i32 = 0x20000000;
    pub const CLONE_NEWUSER: i32 = 0x10000000;
    pub const CLONE_NEWUTS: i32 = 0x4000000;
    pub const CLONE_PARENT: i32 = 0x8000;
    pub const CLONE_PARENT_SETTID: i32 = 0x100000;
    pub const CLONE_PTRACE: i32 = 0x2000;
    pub const CLONE_SETTLS: i32 = 0x80000;
    pub const CLONE_SIGHAND: i32 = 0x800;
    pub const CLONE_SYSVSEM: i32 = 0x40000;
    pub const CLONE_THREAD: i32 = 0x10000;
    pub const CLONE_UNTRACED: i32 = 0x800000;
    pub const CLONE_VFORK: i32 = 0x4000;
    pub const CLONE_VM: i32 = 0x100;
}

pub struct FutexOp {}

impl FutexOp {
    pub const FUTEX_WAIT: u64 = 0;
    pub const FUTEX_WAKE: u64 = 1;
    pub const FUTEX_FD: u64 = 2;
    pub const FUTEX_REQUEUE: u64 = 3;
    pub const FUTEX_CMP_REQUEUE: u64 = 4;
    pub const FUTEX_WAKE_OP: u64 = 5;
    pub const FUTEX_LOCK_PI: u64 = 6;
    pub const FUTEX_UNLOCK_PI: u64 = 7;
    pub const FUTEX_TRYLOCK_PI: u64 = 8;
    pub const FUTEX_WAIT_BITSET: u64 = 9;
    pub const FUTEX_WAKE_BITSET: u64 = 10;
    pub const FUTEX_WAIT_REQUEUE_PI: u64 = 11;
    pub const FUTEX_CMP_REQUEUE_PI: u64 = 12;

    pub const FUTEX_PRIVATE_FLAG: u64 = 128;
    pub const FUTEX_CLOCK_REALTIME: u64 = 256;
}

pub struct FutexWakeOpOption {}

impl FutexWakeOpOption {
    pub const FUTEX_OP_SET: u64 = 0;
    pub const FUTEX_OP_ADD: u64 = 1;
    pub const FUTEX_OP_OR: u64 = 2;
    pub const FUTEX_OP_ANDN: u64 = 3;
    pub const FUTEX_OP_XOR: u64 = 4;
    pub const FUTEX_OP_OPARG_SHIFT: u64 = 8;
    pub const FUTEX_OP_CMP_EQ: u64 = 0;
    pub const FUTEX_OP_CMP_NE: u64 = 1;
    pub const FUTEX_OP_CMP_LT: u64 = 2;
    pub const FUTEX_OP_CMP_LE: u64 = 3;
    pub const FUTEX_OP_CMP_GT: u64 = 4;
    pub const FUTEX_OP_CMP_GE: u64 = 5;
}

pub struct SeekWhence {}

impl SeekWhence {
    pub const SEEK_SET: i32 = 0;
    pub const SEEK_CUR: i32 = 1;
    pub const SEEK_END: i32 = 2;
}

pub struct OpenFlags {}

impl OpenFlags {
    pub const O_RDONLY: i32 = 0;
    pub const O_WRONLY: i32 = 1;
    pub const O_RDWR: i32 = 2;
    pub const O_TRUNC: i32 = 512;
    pub const O_CLOEXEC: i32 = 0x80000;
}

pub struct MemoryDef {}

impl MemoryDef {
    pub const PTE_SHIFT: usize = 12;
    pub const PMD_SHIFT: usize = 21;
    pub const PUD_SHIFT: usize = 30;
    pub const PGD_SHIFT: usize = 39;

    // used for socket/tty buffer
    pub const DEFAULT_BUF_PAGE_COUNT: u64 = 16;

    pub const PTE_MASK: u64 = 0x1ff << Self::PTE_SHIFT;
    pub const PMD_MASK: u64 = 0x1ff << Self::PMD_SHIFT;
    pub const PUD_MASK: u64 = 0x1ff << Self::PUD_SHIFT;
    pub const PGD_MASK: u64 = 0x1ff << Self::PGD_SHIFT;

    pub const PTE_SIZE: u64 = 1 << Self::PTE_SHIFT;
    pub const PMD_SIZE: u64 = 1 << Self::PMD_SHIFT;
    pub const PUD_SIZE: u64 = 1 << Self::PUD_SHIFT;
    pub const PGD_SIZE: u64 = 1 << Self::PGD_SHIFT;

    //the blocks count put on stack to avoid heap allocation, total handle buffer size 32 * 4k = 128K
    pub const ON_STACK_BLOCKS: usize = 32;

    pub const PAGE_SHIFT: u64 = 12;
    pub const HUGE_PAGE_SHIFT: u64 = 21;
    pub const HUGE_1GPAGE_SHIFT: u64 = 30;

    pub const ONE_KB: u64 = 1 << 10; //0x400;
    pub const ONE_MB: u64 = 1 << 20; //0x100_000;
    pub const ONE_GB: u64 = 1 << 30; //0x40_000_000;
    pub const ONE_TB: u64 = 1 << 40; //0x10_000_000_000;
    pub const TWO_MB: u64 = 2 * Self::ONE_MB;

    //interrupt stack pages
    pub const INTERRUPT_STACK_PAGES: u64 = 1;

    pub const MSG_QLEN: usize = 1024;
    pub const QURING_SIZE: usize = 4096;
    pub const DEFAULT_STACK_PAGES: u64 = 32;

    pub const DEFAULT_STACK_SIZE: u64 = Self::DEFAULT_STACK_PAGES * Self::PAGE_SIZE; //64 KB
    pub const PAGE_SIZE: u64 = 1 << Self::PAGE_SHIFT; //0x1000;
    pub const HUGE_PAGE_SIZE: u64 = 1 << Self::HUGE_PAGE_SHIFT;
    pub const HUGE_PAGE_SIZE_1G: u64 = 1 << Self::HUGE_1GPAGE_SHIFT;

    pub const PAGE_MASK: u64 = Self::PAGE_SIZE - 1;

    pub const PAGE_SIZE_4K: u64 = 1 << Self::PAGE_SHIFT; //0x1000;
    pub const PAGE_SIZE_2M: u64 = (2 * Self::ONE_MB);
    pub const PAGE_SIZE_2M_MASK: u64 = !(Self::PAGE_SIZE_2M - 1);
    pub const BLOCK_SIZE: u64 = 64 * Self::ONE_GB;

    pub const PHY_LOWER_ADDR: u64 = 256 * Self::ONE_GB; // 256 ~ 512GB is Guest kernel space
    pub const PHY_UPPER_ADDR: u64 = Self::PHY_LOWER_ADDR + 256 * Self::ONE_GB; // 256 ~ 512GB is Guest kernel space

    // memory layout
    #[cfg(target_arch = "x86_64")]
    pub const USER_UPPER_ADDR: u64 = Self::PHY_LOWER_ADDR;

    // PHY_LOWER_ADDR: qkernel image 512MB
    pub const QKERNEL_IMAGE_SIZE: u64 = 512 * Self::ONE_MB;
    // RDMA Local share memory
    pub const RDMA_LOCAL_SHARE_OFFSET: u64 = Self::PHY_LOWER_ADDR + Self::QKERNEL_IMAGE_SIZE;
    pub const RDMA_LOCAL_SHARE_SIZE: u64 = 1024 * Self::ONE_MB; // 1GB
                                                                // RDMA global share memory
    pub const RDMA_GLOBAL_SHARE_OFFSET: u64 =
        Self::RDMA_LOCAL_SHARE_OFFSET + Self::RDMA_LOCAL_SHARE_SIZE;
    pub const RDMA_GLOBAL_SHARE_SIZE: u64 = 2 * Self::ONE_MB;

    // file map area
    pub const FILE_MAP_OFFSET: u64 = Self::RDMA_GLOBAL_SHARE_OFFSET + Self::RDMA_GLOBAL_SHARE_SIZE;
    pub const FILE_MAP_SIZE: u64 = Self::HEAP_OFFSET - Self::FILE_MAP_OFFSET;

    // heap
    pub const HEAP_OFFSET: u64 = MemoryDef::PHY_LOWER_ADDR
        + Self::KERNEL_MEM_INIT_REGION_SIZE * MemoryDef::ONE_GB
        - Self::HEAP_SIZE
        - Self::IO_HEAP_SIZE;
    pub const HEAP_SIZE: u64 = 10 * Self::ONE_GB;
    pub const HEAP_END: u64 = Self::HEAP_OFFSET + Self::HEAP_SIZE;
    pub const IO_HEAP_SIZE: u64 = 1 * Self::ONE_GB;
    pub const IO_HEAP_END: u64 = Self::HEAP_END + Self::IO_HEAP_SIZE;

    // Create 24GB Init memory region for KVM VM
    pub const KERNEL_MEM_INIT_REGION_SIZE: u64 = 24; // 24 GB

    // start address for memmap and dynamic load address space, there is heap address space between PHY_UPPER_ADDR + VIR_MMAP_START
    pub const VIR_MMAP_START: u64 = Self::PHY_UPPER_ADDR + 128 * Self::ONE_GB; // 512GB + 128 GB
    pub const SHARED_START: u64 = Self::VIR_MMAP_START + 1 * Self::ONE_TB; //512GB + 128 GB + 1TB
    pub const LOWER_TOP: u64 = 0x0000_8000_0000_0000;
    pub const UPPER_BOTTOM: u64 = 0xffff_8000_0000_0000;
    pub const ENTRY_COUNT: u16 = 512 as u16;

    pub const KERNEL_START_P2_ENTRY: usize = (Self::PHY_LOWER_ADDR / Self::ONE_GB) as usize; //256
    pub const KERNEL_END_P2_ENTRY: usize = (Self::PHY_UPPER_ADDR / Self::ONE_GB) as usize; //512
                                                                                           //
}

#[cfg(target_arch = "aarch64")]
impl MemoryDef {
    pub const USER_UPPER_ADDR: u64 = Self::HYPERCALL_MMIO_BASE;
    //
    // Page not backed up by guest physical frame, access causes KVM_EXIT_MMIO.
    //
    pub const HYPERCALL_MMIO_BASE: u64 = Self::PHY_LOWER_ADDR - Self::PAGE_SIZE;
    pub const HYPERCALL_MMIO_SIZE: u64 = Self::PAGE_SIZE;
}

impl MemoryDef {
    pub const HYPERCALL_PARA_PAGE_OFFSET: u64 =
        MemoryDef::GUEST_HOST_SHARED_HEAP_OFFSET + MemoryDef::PAGE_SIZE * 3;
    pub const HOST_INIT_HEAP_OFFSET: u64 = Self::IO_HEAP_END;
    pub const HOST_INIT_HEAP_SIZE: u64 = 1 * Self::ONE_GB;
    pub const HOST_INIT_HEAP_END: u64 = Self::HOST_INIT_HEAP_OFFSET + Self::HOST_INIT_HEAP_SIZE;
    pub const GUEST_PRIVATE_HEAP_OFFSET: u64 = Self::HEAP_OFFSET;
    pub const GUEST_PRIVATE_HEAP_SIZE: u64 = 5 * Self::ONE_GB;
    pub const GUEST_PRIVATE_HEAP_END: u64 =
        Self::GUEST_PRIVATE_HEAP_OFFSET + Self::GUEST_PRIVATE_HEAP_SIZE;
    pub const GUEST_HOST_SHARED_HEAP_OFFSET: u64 = Self::GUEST_PRIVATE_HEAP_END;
    pub const GUEST_HOST_SHARED_HEAP_SIZE: u64 = 5 * Self::ONE_GB;
    pub const GUEST_HOST_SHARED_HEAP_END: u64 =
        Self::GUEST_HOST_SHARED_HEAP_OFFSET + Self::GUEST_HOST_SHARED_HEAP_SIZE;
    pub const UNIDENTICAL_MAPPING_OFFSET: u64 = 30 * Self::ONE_GB;
    pub const GUEST_PRIVATE_INIT_HEAP_OFFSET: u64 = Self::HEAP_OFFSET;
    pub const GUEST_PRIVATE_INIT_HEAP_SIZE: u64 = 1 * Self::ONE_GB;
    pub const GUEST_PRIVATE_INIT_HEAP_END: u64 =
        Self::GUEST_PRIVATE_INIT_HEAP_OFFSET + Self::GUEST_PRIVATE_INIT_HEAP_SIZE;
    pub const GUEST_PRIVATE_RUNNING_HEAP_OFFSET: u64 = Self::GUEST_PRIVATE_INIT_HEAP_END;
    pub const GUEST_PRIVATE_RUNNING_HEAP_SIZE: u64 =
        Self::GUEST_PRIVATE_HEAP_SIZE - Self::GUEST_PRIVATE_INIT_HEAP_SIZE;
    pub const GUEST_PRIVATE_RUNNING_HEAP_END: u64 = Self::GUEST_PRIVATE_HEAP_END;
}

//mmap prot
pub struct MmapProt {}

impl MmapProt {
    pub const PROT_NONE: u64 = 0;
    pub const PROT_READ: u64 = 1 << 0;
    pub const PROT_WRITE: u64 = 1 << 1;
    pub const PROT_EXEC: u64 = 1 << 2;
    pub const PROT_SEM: u64 = 1 << 3;
    pub const PROT_GROWSDOWN: u64 = 1 << 24;
    pub const PROT_GROWSUP: u64 = 1 << 25;
}

//mmap flags
pub struct MmapFlags {}

impl MmapFlags {
    pub const MAP_SHARED: u64 = 1 << 0;
    pub const MAP_PRIVATE: u64 = 1 << 1;
    pub const MAP_FIXED: u64 = 1 << 4;
    pub const MAP_ANONYMOUS: u64 = 1 << 5;
    pub const MAP_32BIT: u64 = 1 << 6; // arch/x86/include/uapi/asm/mman.h
    pub const MAP_GROWSDOWN: u64 = 1 << 8;
    pub const MAP_DENYWRITE: u64 = 1 << 11;
    pub const MAP_EXECUTABLE: u64 = 1 << 12;
    pub const MAP_LOCKED: u64 = 1 << 13;
    pub const MAP_NORESERVE: u64 = 1 << 14;
    pub const MAP_POPULATE: u64 = 1 << 15;
    pub const MAP_NONBLOCK: u64 = 1 << 16;
    pub const MAP_STACK: u64 = 1 << 17;
    pub const MAP_HUGETLB: u64 = 1 << 18;

    pub const MAP_SOCKT_READ: u64 = 1 << 31;
}

//Linux: errors
pub struct SysErr {}

impl SysErr {
    pub const NONE: i32 = 0;
    pub const E2BIG: i32 = 0x7;
    pub const EACCES: i32 = 0xd;
    pub const EADDRINUSE: i32 = 0x62;
    pub const EADDRNOTAVAIL: i32 = 0x63;
    pub const EADV: i32 = 0x44;
    pub const EAFNOSUPPORT: i32 = 0x61;
    pub const EAGAIN: i32 = 0xb;
    pub const EALREADY: i32 = 0x72;
    pub const EBADE: i32 = 0x34;
    pub const EBADF: i32 = 0x9;
    pub const EBADFD: i32 = 0x4d;
    pub const EBADMSG: i32 = 0x4a;
    pub const EBADR: i32 = 0x35;
    pub const EBADRQC: i32 = 0x38;
    pub const EBADSLT: i32 = 0x39;
    pub const EBFONT: i32 = 0x3b;
    pub const EBUSY: i32 = 0x10;
    pub const ECANCELED: i32 = 0x7d;
    pub const ECHILD: i32 = 0xa;
    pub const ECHRNG: i32 = 0x2c;
    pub const ECOMM: i32 = 0x46;
    pub const ECONNABORTED: i32 = 0x67;
    pub const ECONNREFUSED: i32 = 0x6f;
    pub const ECONNRESET: i32 = 0x68;
    pub const EDEADLK: i32 = 0x23;
    pub const EDEADLOCK: i32 = 0x23;
    pub const EDESTADDRREQ: i32 = 0x59;
    pub const EDOM: i32 = 0x21;
    pub const EDOTDOT: i32 = 0x49;
    pub const EDQUOT: i32 = 0x7a;
    pub const EEXIST: i32 = 0x11;
    pub const EFAULT: i32 = 0xe;
    pub const EFBIG: i32 = 0x1b;
    pub const EHOSTDOWN: i32 = 0x70;
    pub const EHOSTUNREACH: i32 = 0x71;
    pub const EIDRM: i32 = 0x2b;
    pub const EILSEQ: i32 = 0x54;
    pub const EINPROGRESS: i32 = 0x73;
    pub const EINTR: i32 = 0x4;
    pub const EINVAL: i32 = 0x16;
    pub const EIO: i32 = 0x5;
    pub const EISCONN: i32 = 0x6a;
    pub const EISDIR: i32 = 0x15;
    pub const EISNAM: i32 = 0x78;
    pub const EKEYEXPIRED: i32 = 0x7f;
    pub const EKEYREJECTED: i32 = 0x81;
    pub const EKEYREVOKED: i32 = 0x80;
    pub const EL2HLT: i32 = 0x33;
    pub const EL2NSYNC: i32 = 0x2d;
    pub const EL3HLT: i32 = 0x2e;
    pub const EL3RST: i32 = 0x2f;
    pub const ELIBACC: i32 = 0x4f;
    pub const ELIBBAD: i32 = 0x50;
    pub const ELIBEXEC: i32 = 0x53;
    pub const ELIBMAX: i32 = 0x52;
    pub const ELIBSCN: i32 = 0x51;
    pub const ELNRNG: i32 = 0x30;
    pub const ELOOP: i32 = 0x28;
    pub const EMEDIUMTYPE: i32 = 0x7c;
    pub const EMFILE: i32 = 0x18;
    pub const EMLINK: i32 = 0x1f;
    pub const EMSGSIZE: i32 = 0x5a;
    pub const EMULTIHOP: i32 = 0x48;
    pub const ENAMETOOLONG: i32 = 0x24;
    pub const ENAVAIL: i32 = 0x77;
    pub const ENETDOWN: i32 = 0x64;
    pub const ENETRESET: i32 = 0x66;
    pub const ENETUNREACH: i32 = 0x65;
    pub const ENFILE: i32 = 0x17;
    pub const ENOANO: i32 = 0x37;
    pub const ENOBUFS: i32 = 0x69;
    pub const ENOCSI: i32 = 0x32;
    pub const ENODATA: i32 = 0x3d;
    pub const ENOATTR: i32 = Self::ENODATA;
    pub const ENODEV: i32 = 0x13;
    pub const ENOENT: i32 = 0x2;
    pub const ENOEXEC: i32 = 0x8;
    pub const ENOKEY: i32 = 0x7e;
    pub const ENOLCK: i32 = 0x25;
    pub const ENOLINK: i32 = 0x43;
    pub const ENOMEDIUM: i32 = 0x7b;
    pub const ENOMEM: i32 = 0xc;
    pub const ENOMSG: i32 = 0x2a;
    pub const ENONET: i32 = 0x40;
    pub const ENOPKG: i32 = 0x41;
    pub const ENOPROTOOPT: i32 = 0x5c;
    pub const ENOSPC: i32 = 0x1c;
    pub const ENOSR: i32 = 0x3f;
    pub const ENOSTR: i32 = 0x3c;
    pub const ENOSYS: i32 = 0x26;
    pub const ENOTBLK: i32 = 0xf;
    pub const ENOTCONN: i32 = 0x6b;
    pub const ENOTDIR: i32 = 0x14;
    pub const ENOTEMPTY: i32 = 0x27;
    pub const ENOTNAM: i32 = 0x76;
    pub const ENOTRECOVERABLE: i32 = 0x83;
    pub const ENOTSOCK: i32 = 0x58;
    pub const ENOTSUP: i32 = 0x5f;
    pub const ENOTTY: i32 = 0x19;
    pub const ENOTUNIQ: i32 = 0x4c;
    pub const ENXIO: i32 = 0x6;
    pub const EOPNOTSUPP: i32 = 0x5f;
    pub const EOVERFLOW: i32 = 0x4b;
    pub const EOWNERDEAD: i32 = 0x82;
    pub const EPERM: i32 = 0x1;
    pub const EPFNOSUPPORT: i32 = 0x60;
    pub const EPIPE: i32 = 0x20;
    pub const EPROTO: i32 = 0x47;
    pub const EPROTONOSUPPORT: i32 = 0x5d;
    pub const EPROTOTYPE: i32 = 0x5b;
    pub const ERANGE: i32 = 0x22;
    pub const EREMCHG: i32 = 0x4e;
    pub const EREMOTE: i32 = 0x42;
    pub const EREMOTEIO: i32 = 0x79;
    pub const ERESTART: i32 = 0x55;
    pub const ERFKILL: i32 = 0x84;
    pub const EROFS: i32 = 0x1e;
    pub const ESHUTDOWN: i32 = 0x6c;
    pub const ESOCKTNOSUPPORT: i32 = 0x5e;
    pub const ESPIPE: i32 = 0x1d;
    pub const ESRCH: i32 = 0x3;
    pub const ESRMNT: i32 = 0x45;
    pub const ESTALE: i32 = 0x74;
    pub const ESTRPIPE: i32 = 0x56;
    pub const ETIME: i32 = 0x3e;
    pub const ETIMEDOUT: i32 = 0x6e;
    pub const ETOOMANYREFS: i32 = 0x6d;
    pub const ETXTBSY: i32 = 0x1a;
    pub const EUCLEAN: i32 = 0x75;
    pub const EUNATCH: i32 = 0x31;
    pub const EUSERS: i32 = 0x57;
    pub const EWOULDBLOCK: i32 = 0xb;
    pub const EXDEV: i32 = 0x12;
    pub const EXFULL: i32 = 0x36;

    // ERESTARTSYS is returned by an interrupted syscall to indicate that it
    // should be converted to EINTR if interrupted by a signal delivered to a
    // user handler without SA_RESTART set, and restarted otherwise.
    pub const ERESTARTSYS: i32 = 512;

    // ERESTARTNOINTR is returned by an interrupted syscall to indicate that it
    // should always be restarted.
    pub const ERESTARTNOINTR: i32 = 513;

    // ERESTARTNOHAND is returned by an interrupted syscall to indicate that it
    // should be converted to EINTR if interrupted by a signal delivered to a
    // user handler, and restarted otherwise.
    pub const ERESTARTNOHAND: i32 = 514;

    // ERESTART_RESTARTBLOCK is returned by an interrupted syscall to indicate
    // that it should be restarted using a custom function. The interrupted
    // syscall must register a custom restart function by calling
    // Task.SetRestartSyscallFn.
    pub const ERESTART_RESTARTBLOCK: i32 = 515;
}

#[repr(C)]
pub struct RLimit {
    pub rlimCurr: u64,
    pub rlimMax: u64,
}

pub fn ComparePage(from: u64, to: u64) -> bool {
    unsafe {
        let cnt = 512;
        let fromArr = slice::from_raw_parts(from as *const u64, cnt);
        let toArr = slice::from_raw_parts_mut(to as *mut u64, cnt);
        for i in 0..cnt {
            if toArr[i] != fromArr[i] {
                return false;
            }
        }

        return true;
    }
}

/// UNSAFE CODE! the caller must make sure the to and from address are 4k aligned.
#[inline(always)]
pub fn CopyPage(to: u64, from: u64) {
    unsafe {
        CopyPageUnsafe(to, from);
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Debug, Default, Copy, Clone)]
#[repr(C)]
pub struct LibcStat {
    pub st_dev: u64,
    pub st_ino: u64,
    pub st_nlink: u64,
    pub st_mode: u32,
    pub st_uid: u32,
    pub st_gid: u32,
    pub pad0: i32,
    pub st_rdev: u64,
    pub st_size: i64,
    pub st_blksize: i64,
    pub st_blocks: i64,
    pub st_atime: i64,
    pub st_atime_nsec: i64,
    pub st_mtime: i64,
    pub st_mtime_nsec: i64,
    pub st_ctime: i64,
    pub st_ctime_nsec: i64,
    pub pad: [i64; 3],
}

#[cfg(target_arch = "aarch64")]
#[derive(Debug, Default, Copy, Clone)]
#[repr(C)]
pub struct LibcStat {
    pub st_dev: u64,
    pub st_ino: u64,
    pub st_mode: u32,
    pub st_nlink: u32,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_rdev: u64,
    pub pad0: u64,
    pub st_size: i64,
    pub st_blksize: i32,
    pub pad1: i32,
    pub st_blocks: i64,
    pub st_atime: i64,
    pub st_atime_nsec: i64,
    pub st_mtime: i64,
    pub st_mtime_nsec: i64,
    pub st_ctime: i64,
    pub st_ctime_nsec: i64,
    pub pad: [i32; 2],
}

impl LibcStat {
    pub fn IsRegularFile(&self) -> bool {
        let x = self.st_mode as u16 & ModeType::S_IFMT;
        return x == ModeType::S_IFREG;
    }
}
