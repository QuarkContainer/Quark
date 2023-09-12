// Copyright (c) 2021 Quark Container Authors
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
extern crate spin;

macro_rules! cfg_x86_64 {
    ($($item:item)*) => {
        $(
            #[cfg(target_arch="x86_64")]
            $item
        )*
    }
}

macro_rules! cfg_aarch64 {
    ($($item:item)*) => {
        $(
            #[cfg(target_arch="aarch64")]
            $item
        )*
    }
}

//#[macro_use]
//pub mod macros;
pub mod addr;
pub mod auxv;
pub mod buddyallocator;
pub mod common;
pub mod idallocator;
pub mod linux_def;
pub mod pagetable;
pub mod range;
//pub mod Process;
pub mod auth;
pub mod backtracer;
pub mod bytestream;
pub mod config;
pub mod control_msg;
pub mod cpuid;
pub mod cstring;
pub mod device;
pub mod eventchannel;
pub mod fileinfo;
pub mod limits;
pub mod linux;
pub mod loader;
pub mod lockfreebytestream;
pub mod lrc_cache;
pub mod mem;
pub mod metric;
pub mod mutex;
pub mod object_ref;
pub mod path;
pub mod perf_tunning;
pub mod platform;
pub mod qmsg;
pub mod singleton;
pub mod socket_buf;
pub mod sort_arr;
pub mod task_mgr;
pub mod uring;
pub mod usage;

pub mod kernel;
pub mod rdma_share;
pub mod ringbuf;
pub mod vcpu_mgr;

pub mod hiber_mgr;
pub mod proxy;
pub mod rdma_svc_cli;
pub mod rdmasocket;
pub mod unix_socket;

#[cfg(target_arch = "aarch64")]
mod pagetable_aarch64;

use self::mutex::*;
use alloc::vec::Vec;
use cache_padded::CachePadded;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use self::bytestream::*;
use self::config::*;
use self::control_msg::SignalArgs;
use self::fileinfo::*;
use self::hiber_mgr::*;
use self::kernel::kernel::futex::*;
use self::kernel::kernel::kernel::Kernel;
use self::kernel::kernel::timer::timekeeper::*;
use self::kernel::kernel::timer::timer_store::*;
use self::kernel::memmgr::pma::*;
use self::kernel::quring::uring_mgr::QUring;
use self::linux_def::*;
use self::object_ref::ObjectRef;
use self::qmsg::*;
use self::rdma_svc_cli::*;
use self::ringbuf::*;
use self::task_mgr::*;

pub fn InitSingleton() {
    unsafe {
        control_msg::InitSingleton();
        cpuid::InitSingleton();
        device::InitSingleton();
        eventchannel::InitSingleton();
        limits::InitSingleton();
        metric::InitSingleton();
        perf_tunning::InitSingleton();
        auth::id::InitSingleton();
        linux::limits::InitSingleton();
    }
}

pub const HYPERCALL_PANIC: u16 = 2;
pub const HYPERCALL_OOM: u16 = 4;
pub const HYPERCALL_MSG: u16 = 5;
pub const HYPERCALL_U64: u16 = 6;
pub const HYPERCALL_PRINT: u16 = 8;
pub const HYPERCALL_EXIT: u16 = 9;
pub const HYPERCALL_GETTIME: u16 = 11;
pub const HYPERCALL_QCALL: u16 = 12;
pub const HYPERCALL_HLT: u16 = 13;
pub const HYPERCALL_URING_WAKE: u16 = 14;
pub const HYPERCALL_HCALL: u16 = 15;
pub const HYPERCALL_IOWAIT: u16 = 16;
pub const HYPERCALL_WAKEUP_VCPU: u16 = 17;
pub const HYPERCALL_EXIT_VM: u16 = 18;
pub const HYPERCALL_VCPU_FREQ: u16 = 19;
pub const HYPERCALL_VCPU_YIELD: u16 = 20;
pub const HYPERCALL_VCPU_DEBUG: u16 = 21;
pub const HYPERCALL_VCPU_PRINT: u16 = 22;
pub const HYPERCALL_VCPU_WAIT: u16 = 23;
pub const HYPERCALL_RELEASE_VCPU: u16 = 24;

pub const DUMMY_TASKID: TaskId = TaskId::New(0xffff_ffff);

pub const MAX_VCPU_COUNT: usize = 64;

#[allow(non_camel_case_types)]
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SysCallID {
    sys_read = 0 as u64,
    sys_write,
    sys_open,
    sys_close,
    sys_stat,
    sys_fstat,
    sys_lstat,
    sys_poll,
    sys_lseek,
    sys_mmap,
    sys_mprotect,
    //10
    sys_munmap,
    sys_brk,
    sys_rt_sigaction,
    sys_rt_sigprocmask,
    sys_rt_sigreturn,
    sys_ioctl,
    sys_pread64,
    sys_pwrite64,
    sys_readv,
    sys_writev,
    //20
    sys_access,
    sys_pipe,
    sys_select,
    sys_sched_yield,
    sys_mremap,
    sys_msync,
    sys_mincore,
    sys_madvise,
    sys_shmget,
    sys_shmat,
    //30
    sys_shmctl,
    sys_dup,
    sys_dup2,
    sys_pause,
    sys_nanosleep,
    sys_getitimer,
    sys_alarm,
    sys_setitimer,
    sys_getpid,
    sys_sendfile,
    //40
    sys_socket,
    sys_connect,
    sys_accept,
    sys_sendto,
    sys_recvfrom,
    sys_sendmsg,
    sys_recvmsg,
    sys_shutdown,
    sys_bind,
    sys_listen,
    //50
    sys_getsockname,
    sys_getpeername,
    sys_socketpair,
    sys_setsockopt,
    sys_getsockopt,
    sys_clone,
    sys_fork,
    sys_vfork,
    sys_execve,
    sys_exit,
    //60
    sys_wait4,
    sys_kill,
    sys_uname,
    sys_semget,
    sys_semop,
    sys_semctl,
    sys_shmdt,
    sys_msgget,
    sys_msgsnd,
    sys_msgrcv,
    //70
    sys_msgctl,
    sys_fcntl,
    sys_flock,
    sys_fsync,
    sys_fdatasync,
    sys_truncate,
    sys_ftruncate,
    sys_getdents,
    sys_getcwd,
    sys_chdir,
    //80
    sys_fchdir,
    sys_rename,
    sys_mkdir,
    sys_rmdir,
    sys_creat,
    sys_link,
    sys_unlink,
    sys_symlink,
    sys_readlink,
    sys_chmod,
    //90
    sys_fchmod,
    sys_chown,
    sys_fchown,
    sys_lchown,
    sys_umask,
    sys_gettimeofday,
    sys_getrlimit,
    sys_getrusage,
    sys_sysinfo,
    sys_times,
    //100
    sys_ptrace,
    sys_getuid,
    sys_syslog,
    sys_getgid,
    sys_setuid,
    sys_setgid,
    sys_geteuid,
    sys_getegid,
    sys_setpgid,
    sys_getppid,
    //110
    sys_getpgrp,
    sys_setsid,
    sys_setreuid,
    sys_setregid,
    sys_getgroups,
    sys_setgroups,
    sys_setresuid,
    sys_getresuid,
    sys_setresgid,
    sys_getresgid,
    //120
    sys_getpgid,
    sys_setfsuid,
    sys_setfsgid,
    sys_getsid,
    sys_capget,
    sys_capset,
    sys_rt_sigpending,
    sys_rt_sigtimedwait,
    sys_rt_sigqueueinfo,
    sys_rt_sigsuspend,
    //130
    sys_sigaltstack,
    sys_utime,
    sys_mknod,
    sys_uselib,
    sys_personality,
    sys_ustat,
    sys_statfs,
    sys_fstatfs,
    sys_sysfs,
    sys_getpriority,
    //140
    sys_setpriority,
    sys_sched_setparam,
    sys_sched_getparam,
    sys_sched_setscheduler,
    sys_sched_getscheduler,
    sys_sched_get_priority_max,
    sys_sched_get_priority_min,
    sys_sched_rr_get_interval,
    sys_mlock,
    sys_munlock,
    //150
    sys_mlockall,
    sys_munlockall,
    sys_vhangup,
    sys_modify_ldt,
    sys_pivot_root,
    sys__sysctl,
    sys_prctl,
    sys_arch_prctl,
    sys_adjtimex,
    sys_setrlimit,
    sys_chroot,
    sys_sync,
    sys_acct,
    sys_settimeofday,
    sys_mount,
    sys_umount2,
    sys_swapon,
    sys_swapoff,
    sys_reboot,
    sys_sethostname,
    //160
    sys_setdomainname,
    sys_iopl,
    sys_ioperm,
    sys_create_module,
    sys_init_module,
    sys_delete_module,
    sys_get_kernel_syms,
    sys_query_module,
    sys_quotactl,
    sys_nfsservctl,
    //180
    sys_getpmsg,
    sys_putpmsg,
    sys_afs_syscall,
    sys_tuxcall,
    sys_security,
    sys_gettid,
    sys_readahead,
    sys_setxattr,
    sys_lsetxattr,
    sys_fsetxattr,
    //190
    sys_getxattr,
    sys_lgetxattr,
    sys_fgetxattr,
    sys_listxattr,
    sys_llistxattr,
    sys_flistxattr,
    sys_removexattr,
    sys_lremovexattr,
    sys_fremovexattr,
    sys_tkill,
    //200
    sys_time,
    sys_futex,
    sys_sched_setaffinity,
    sys_sched_getaffinity,
    sys_set_thread_area,
    sys_io_setup,
    sys_io_destroy,
    sys_io_getevents,
    sys_io_submit,
    sys_io_cancel,
    //210
    sys_get_thread_area,
    sys_lookup_dcookie,
    sys_epoll_create,
    sys_epoll_ctl_old,
    sys_epoll_wait_old,
    sys_remap_file_pages,
    sys_getdents64,
    sys_set_tid_address,
    sys_restart_syscall,
    sys_semtimedop,
    //220
    sys_fadvise64,
    sys_timer_create,
    sys_timer_settime,
    sys_timer_gettime,
    sys_timer_getoverrun,
    sys_timer_delete,
    sys_clock_settime,
    sys_clock_gettime,
    sys_clock_getres,
    sys_clock_nanosleep,
    //230
    sys_exit_group,
    sys_epoll_wait,
    sys_epoll_ctl,
    sys_tgkill,
    sys_utimes,
    sys_vserver,
    sys_mbind,
    sys_set_mempolicy,
    sys_get_mempolicy,
    sys_mq_open,
    //240
    sys_mq_unlink,
    sys_mq_timedsend,
    sys_mq_timedreceive,
    sys_mq_notify,
    sys_mq_getsetattr,
    sys_kexec_load,
    sys_waitid,
    sys_add_key,
    sys_request_key,
    sys_keyctl,
    //250
    sys_ioprio_set,
    sys_ioprio_get,
    sys_inotify_init,
    sys_inotify_add_watch,
    sys_inotify_rm_watch,
    sys_migrate_pages,
    sys_openat,
    sys_mkdirat,
    sys_mknodat,
    sys_fchownat,
    //260
    sys_futimesat,
    sys_newfstatat,
    sys_unlinkat,

    sys_renameat,
    sys_linkat,
    sys_symlinkat,
    sys_readlinkat,
    sys_fchmodat,
    sys_faccessat,
    sys_pselect6,
    //270
    sys_ppoll,
    sys_unshare,
    sys_set_robust_list,
    sys_get_robust_list,
    sys_splice,
    sys_tee,
    sys_sync_file_range,
    sys_vmsplice,
    sys_move_pages,
    sys_utimensat,
    //280
    sys_epoll_pwait,
    sys_signalfd,
    sys_timerfd_create,
    sys_eventfd,
    sys_fallocate,
    sys_timerfd_settime,
    sys_timerfd_gettime,
    sys_accept4,
    sys_signalfd4,
    sys_eventfd2,
    //290
    sys_epoll_create1,
    sys_dup3,
    sys_pipe2,
    sys_inotify_init1,
    sys_preadv,
    sys_pwritev,
    sys_rt_tgsigqueueinfo,
    sys_perf_event_open,
    sys_recvmmsg,
    sys_fanotify_init,
    //300
    sys_fanotify_mark,
    sys_prlimit64,
    sys_name_to_handle_at,
    sys_open_by_handle_at,
    sys_clock_adjtime,
    sys_syncfs,
    sys_sendmmsg,
    sys_setns,
    sys_getcpu,
    sys_process_vm_readv,
    //310
    sys_process_vm_writev,
    sys_kcmp,
    sys_finit_module,
    sys_sched_setattr,
    sys_sched_getattr,
    sys_renameat2,
    sys_seccomp,
    sys_getrandom,
    sys_memfd_create,
    sys_kexec_file_load,
    //320
    sys_bpf,
    sys_stub_execveat,
    sys_userfaultfd,
    sys_membarrier,
    sys_mlock2,
    sys_copy_file_range,
    sys_preadv2,
    sys_pwritev2,
    sys_pkey_mprotect,
    sys_pkey_alloc,
    // 330
    sys_pkey_free,
    sys_statx,

    syscall_333,
    syscall_334,
    syscall_335,
    syscall_336,
    syscall_337,
    syscall_338,
    syscall_339,
    syscall_340,
    syscall_341,
    syscall_342,
    syscall_343,
    syscall_344,
    syscall_345,
    syscall_346,
    syscall_347,
    syscall_348,
    syscall_349,
    syscall_350,
    syscall_351,
    syscall_352,
    syscall_353,
    syscall_354,
    syscall_355,
    syscall_356,
    syscall_357,
    syscall_358,
    syscall_359,
    syscall_360,
    syscall_361,
    syscall_362,
    syscall_363,
    syscall_364,
    syscall_365,
    syscall_366,
    syscall_367,
    syscall_368,
    syscall_369,
    syscall_370,
    syscall_371,
    syscall_372,
    syscall_373,
    syscall_374,
    syscall_375,
    syscall_376,
    syscall_377,
    syscall_378,
    syscall_379,
    syscall_380,
    syscall_381,
    syscall_382,
    syscall_383,
    syscall_384,
    syscall_385,
    syscall_386,
    syscall_387,
    syscall_388,
    syscall_389,
    syscall_390,
    syscall_391,
    syscall_392,
    syscall_393,
    syscall_394,
    syscall_395,
    syscall_396,
    syscall_397,
    syscall_398,
    syscall_399,
    syscall_400,
    syscall_401,
    syscall_402,
    syscall_403,
    syscall_404,
    syscall_405,
    syscall_406,
    syscall_407,
    syscall_408,
    syscall_409,
    syscall_410,
    syscall_411,
    syscall_412,
    syscall_413,
    syscall_414,
    syscall_415,
    syscall_416,
    syscall_417,
    syscall_418,
    syscall_419,
    syscall_420,
    syscall_421,
    syscall_422,
    syscall_423,
    sys_pidfd_send_signal,
    sys_io_uring_setup,
    sys_io_uring_enter,
    sys_io_uring_register,
    sys_open_tree,
    sys_move_mount,
    sys_fsopen,
    sys_fsconfig,
    sys_fsmount,
    sys_fspick,
    sys_pidfd_open,
    sys_clone3,
    sys_close_range,
    sys_openat2,
    sys_pidfd_getfd,
    sys_faccessat2,
    sys_process_madvise,
    sys_epoll_pwait2,
    nt_setattr,
    sys_quotactl_fd,
    sys_landlock_create_ruleset,
    sys_landlock_add_rule,
    sys_landlock_restrict_self,
    sys_memfd_secret,
    sys_process_mrelease,
    sys_futex_waitv,
    sys_set_mempolicy_home_node,
    UnknowSyscall = 451,
    sys_socket_produce = 10001,
    sys_socket_consume,
    sys_proxy,

    EXTENSION_MAX,
}

#[derive(Clone, Default, Debug, Copy)]
pub struct GetTimeCall {
    pub res: i64,
    pub clockId: i32,
}

#[derive(Clone, Default, Debug, Copy)]
pub struct VcpuFeq {
    pub res: i64,
}

pub enum IOState {
    Wait,
    Processing,
}

#[derive(Clone, Default, Debug, Copy)]
pub struct LoadAddr {
    pub phyAddr: u64,
    pub virtualAddr: u64,
    pub len: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Str {
    pub addr: u64,
    pub len: u32,
}

pub type ShareSpaceRef = ObjectRef<ShareSpace>;

#[repr(C)]
#[repr(align(128))]
#[derive(Default)]
pub struct ShareSpace {
    pub QOutput: QRingQueue<HostOutputMsg>, //QMutex<VecDeque<HostInputMsg>>,

    // add this pad can decrease the mariadb start time 25 sec to 12 sec
    //todo: root cause this. False share?
    //pub pad: [u64; 8],
    pub hostEpollProcessing: CachePadded<QMutex<()>>,

    pub scheduler: task_mgr::Scheduler,
    pub guestMsgCount: CachePadded<AtomicU64>,
    pub hostProcessor: CachePadded<AtomicU64>,
    pub VcpuSearchingCnt: CachePadded<AtomicU64>,

    pub shutdown: CachePadded<AtomicBool>,
    pub ioUring: CachePadded<QUring>,
    pub timerkeeper: CachePadded<TimeKeeper>,
    pub timerStore: CachePadded<TimerStore>,
    pub signalArgs: CachePadded<QMutex<Option<SignalArgs>>>,
    pub futexMgr: CachePadded<FutexMgr>,
    pub pageMgr: CachePadded<PageMgr>,
    pub ioMgr: CachePadded<IOMgr>,
    pub config: CachePadded<QRwLock<Config>>,
    pub rdmaSvcCli: CachePadded<RDMASvcClient>,

    pub logBuf: CachePadded<QMutex<Option<ByteStream>>>,
    pub logLock: CachePadded<QMutex<()>>,
    pub logfd: CachePadded<AtomicI32>,
    pub signalHandlerAddr: CachePadded<AtomicU64>,
    pub virtualizationHandlerAddr: CachePadded<AtomicU64>,
    pub kernel: CachePadded<QMutex<Option<Kernel>>>,
    pub tlbShootdownLock: CachePadded<QMutex<()>>,
    pub tlbShootdownMask: CachePadded<AtomicU64>,
    pub uid: CachePadded<AtomicU64>,
    pub inotifyCookie: CachePadded<AtomicU32>,
    pub waitMask: CachePadded<AtomicU64>,
    pub reapFileAvaiable: CachePadded<AtomicBool>,
    pub hibernatePause: CachePadded<AtomicBool>,
    pub hiberMgr: CachePadded<HiberMgr>,

    pub supportMemoryBarrier: bool,
    pub controlSock: i32,
    pub hostEpollfd: AtomicI32,

    pub values: Vec<[AtomicU64; 2]>,
}

impl ShareSpace {
    pub fn New() -> Self {
        return ShareSpace {
            ioUring: CachePadded::new(QUring::New(MemoryDef::QURING_SIZE)),
            ioMgr: CachePadded::new(IOMgr::Init().unwrap()),
            ..Default::default()
        };
    }

    pub fn NewUID(&self) -> u64 {
        return self.uid.fetch_add(1, Ordering::SeqCst) + 1;
    }

    pub fn NewInotifyCookie(&self) -> u32 {
        return self.inotifyCookie.fetch_add(1, Ordering::SeqCst) + 1;
    }

    pub fn MaskTlbShootdown(&self, vcpuId: u64) {
        self.tlbShootdownMask
            .fetch_or(1 << vcpuId, Ordering::SeqCst);
    }

    pub fn UnmaskTlbShootdown(&self, vcpuId: u64) {
        self.tlbShootdownMask
            .fetch_and(!(1 << vcpuId), Ordering::SeqCst);
    }

    pub fn TlbShootdownMask(&self) -> u64 {
        return self.tlbShootdownMask.load(Ordering::SeqCst);
    }

    pub fn ClearTlbShootdownMask(&self) {
        self.tlbShootdownMask.store(0, Ordering::SeqCst);
    }

    pub fn SetTlbShootdownMask(&self, mask: u64) {
        self.tlbShootdownMask.store(mask, Ordering::SeqCst);
    }

    pub fn SetIOUringsAddr(&self, addr: u64) {
        self.ioUring.SetIOUringsAddr(addr);
    }

    pub fn SetSignalHandlerAddr(&self, addr: u64) {
        self.signalHandlerAddr.store(addr, Ordering::SeqCst);
    }

    pub fn SignalHandlerAddr(&self) -> u64 {
        return self.signalHandlerAddr.load(Ordering::Relaxed);
    }

    pub fn SetvirtualizationHandlerAddr(&self, addr: u64) {
        self.virtualizationHandlerAddr.store(addr, Ordering::SeqCst);
    }

    pub fn VirtualizationHandlerAddr(&self) -> u64 {
        return self.virtualizationHandlerAddr.load(Ordering::Relaxed);
    }

    pub fn StoreShutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    pub fn Shutdown(&self) -> bool {
        return self.shutdown.load(Ordering::Relaxed);
    }

    pub fn GetPageMgrAddr(&self) -> u64 {
        return self.pageMgr.Addr();
    }

    pub fn GetFutexMgrAddr(&self) -> u64 {
        return self.futexMgr.Addr();
    }

    pub fn GetIOUringAddr(&self) -> u64 {
        return self.ioUring.Addr();
    }

    pub fn GetTimerKeeperAddr(&self) -> u64 {
        return self.timerkeeper.Addr();
    }

    pub fn GetTimerStoreAddr(&self) -> u64 {
        return self.timerStore.Addr();
    }

    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn TryLockEpollProcess(&self) -> Option<QMutexGuard<()>> {
        return self.hostEpollProcessing.try_lock();
    }

    pub fn HostHostEpollfd(&self) -> i32 {
        return self.hostEpollfd.load(Ordering::Relaxed);
    }

    pub fn SetValue(&self, cpuId: usize, idx: usize, val: u64) {
        self.values[cpuId][idx].store(val, Ordering::Relaxed);
    }

    pub fn GetValue(&self, cpuId: usize, idx: usize) -> u64 {
        return self.values[cpuId][idx].load(Ordering::Relaxed);
    }

    #[inline]
    pub fn AQHostOutputPop(&self) -> Option<HostOutputMsg> {
        return self.QOutput.Pop();
    }

    #[inline]
    pub fn IncrHostProcessor(&self) {
        self.hostProcessor.fetch_add(1, Ordering::SeqCst);
    }

    pub fn IncrVcpuSearching(&self) -> u64 {
        let ret = self.VcpuSearchingCnt.fetch_add(1, Ordering::SeqCst);
        return ret + 1;
    }

    pub fn DecrVcpuSearching(&self) -> u64 {
        let ret = self.VcpuSearchingCnt.fetch_sub(1, Ordering::SeqCst);
        return ret - 1;
    }

    #[inline]
    pub fn NeedHostProcess(&self) -> bool {
        match self
            .hostProcessor
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => return true,
            Err(_) => return false,
        }
    }

    // return Processor count after decrease
    #[inline]
    pub fn DecrHostProcessor(&self) -> u64 {
        let cnt = self.hostProcessor.fetch_sub(1, Ordering::SeqCst);
        return cnt - 1;
    }

    #[inline]
    pub fn HostProcessor(&self) -> u64 {
        return self.hostProcessor.load(Ordering::SeqCst);
    }

    pub fn SetLogfd(&self, fd: i32) {
        self.logfd.store(fd, Ordering::SeqCst);
    }

    pub fn Logfd(&self) -> i32 {
        return self.logfd.load(Ordering::SeqCst);
    }

    pub fn Log(&self, buf: &[u8]) -> bool {
        for i in 0..3 {
            let ret = self.logBuf.lock().as_mut().unwrap().lock().writeFull(buf);
            match ret {
                Err(_) => {
                    print!("log is full ... retry {}", i + 1);
                    Self::Yield();
                }
                Ok((trigger, _)) => return trigger,
            }
        }

        panic!("Log is full...")
    }

    pub fn ConsumeAndGetAvailableWriteBuf(&self, cnt: usize) -> (u64, usize) {
        let mut lock = self.logBuf.lock();
        lock.as_mut().unwrap().lock().Consume(cnt);
        let (addr, len) = lock.as_mut().unwrap().lock().GetDataBuf();
        return (addr, len);
    }

    pub fn GetDataBuf(&self) -> (u64, usize) {
        let mut lock = self.logBuf.lock();
        let (addr, len) = lock.as_mut().unwrap().lock().GetDataBuf();
        return (addr, len);
    }

    pub fn ReadLog(&self, buf: &mut [u8]) -> usize {
        let (_trigger, cnt) = self
            .logBuf
            .lock()
            .as_mut()
            .unwrap()
            .lock()
            .read(buf)
            .unwrap();
        return cnt;
    }

    #[inline]
    pub fn ReadyTaskCnt(&self, vcpuId: usize) -> u64 {
        return self.scheduler.ReadyTaskCnt(vcpuId) as u64;
    }
}
