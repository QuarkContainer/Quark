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

use super::super::syscalls::sys_aio::*;
use super::super::syscalls::sys_capability::*;
use super::super::syscalls::sys_chmod::*;
use super::super::syscalls::sys_epoll::*;
use super::super::syscalls::sys_eventfd::*;
use super::super::syscalls::sys_file::*;
use super::super::syscalls::sys_futex::*;
use super::super::syscalls::sys_getdents::*;
use super::super::syscalls::sys_identity::*;
use super::super::syscalls::sys_inotify::*;
use super::super::syscalls::sys_membarrier::*;
use super::super::syscalls::sys_memfd::*;
use super::super::syscalls::sys_mempolicy::*;
use super::super::syscalls::sys_mmap::*;
use super::super::syscalls::sys_mmap_socket::*;
use super::super::syscalls::sys_msgqueue::*;
use super::super::syscalls::sys_pipe::*;
use super::super::syscalls::sys_poll::*;
use super::super::syscalls::sys_prctl::*;
use super::super::syscalls::sys_proxy::*;
use super::super::syscalls::sys_random::*;
use super::super::syscalls::sys_read::*;
use super::super::syscalls::sys_rlimit::*;
use super::super::syscalls::sys_rusage::*;
use super::super::syscalls::sys_sched::*;
use super::super::syscalls::sys_sem::*;
use super::super::syscalls::sys_shm::*;
use super::super::syscalls::sys_signal::*;
use super::super::syscalls::sys_socket::*;
use super::super::syscalls::sys_splice::*;
use super::super::syscalls::sys_stat::*;
use super::super::syscalls::sys_sync::*;
use super::super::syscalls::sys_sysinfo::*;
use super::super::syscalls::sys_syslog::*;
use super::super::syscalls::sys_thread::*;
use super::super::syscalls::sys_time::*;
use super::super::syscalls::sys_timer::*;
use super::super::syscalls::sys_timerfd::*;
#[cfg(target_arch="x86_64")]
use super::super::syscalls::sys_tls::*;
use super::super::syscalls::sys_utsname::*;
use super::super::syscalls::sys_write::*;
use super::super::syscalls::sys_xattr::*;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::SysCallID;
use super::super::task::*;

//#[repr(align(128))]
#[derive(Debug)]
pub struct SyscallArguments {
    pub arg0: u64,
    pub arg1: u64,
    pub arg2: u64,
    pub arg3: u64,
    pub arg4: u64,
    pub arg5: u64,
}

#[inline]
pub fn SysCall(task: &mut Task, nr: u64, args: &SyscallArguments) -> TaskRunState {
    let idx = nr as usize;
    let func = match SYS_CALL_TABLE.get(idx) {
        Some(f) => f,
        None => EXTENSION_CALL_TABLE
            .get(idx - EXTENSION_CALL_OFFSET)
            .unwrap(),
    };

    match func(task, args) {
        Err(Error::SysCallRetCtrlWithRet(state, ret)) => {
            task.SetReturn(ret);
            return state;
        }
        Err(Error::ErrExceedsFileSizeLimit) => {
            match task.HandleExceedsFileSizeLimit() {
                Err(Error::SysError(e)) => {
                    task.haveSyscallReturn = true;
                    task.SetReturn(-e as u64);
                }
                Err(e) => {
                    panic!("ErrExceedsFileSizeLimit get unexpected error {:?}", e)
                }
                Ok(()) => {
                    panic!("ErrExceedsFileSizeLimit impossbile")
                }
            }

            return TaskRunState::RunApp;
        }
        Err(Error::SysCallRetCtrl(state)) => {
            return state;
        }
        Ok(res) => {
            task.SetReturn(res as u64);
            return TaskRunState::RunApp;
        }
        Err(Error::SysError(e)) => {
            task.haveSyscallReturn = true;
            task.SetReturn(-e as u64);
            return TaskRunState::RunApp;
        }
        Err(Error::SysCallNotImplement) => {
            let callId: SysCallID = unsafe { core::mem::transmute(nr as u64) };
            panic!("Sycall not implement syscall is {:?}", callId);
        }
        Err(e) => {
            panic!("Syscall[{}]: get unexpected error {:x?}", nr, e);
            //return TaskRunState::RunExit
        }
    }
}

pub type SyscallFn = fn(task: &mut Task, args: &SyscallArguments) -> Result<i64>;

pub const EXTENSION_CALL_OFFSET: usize = 10001;
pub const EXTENSION_CALL_TABLE: &'static [SyscallFn] = &[
    SysSocketProduce, // 10001 sys_socket_produce
    SysSocketConsume, // 10002 sys_socket_consume
    SysProxy,         // 10003 sys_proxy
];

#[cfg(target_arch = "x86_64")]
pub const SYS_CALL_TABLE: &'static [SyscallFn] = &[
    SysRead,                // 000 sys_read
    SysWrite,               // 001 sys_write,
    SysOpen,                // 002 sys_open,
    SysClose,               // 003 sys_close,
    SysStat,                // 004 sys_stat,
    SysFstat,               // 005 sys_fstat,
    SysLstat,               // 006 sys_lstat,
    SysPoll,                // 007 sys_poll,
    SysLseek,               // 008 sys_lseek,
    SysMmap,                // 009 sys_mmap,
    SysMprotect,            // 010 sys_mprotect,
    SysUnmap,               // 011 sys_munmap,
    SysBrk,                 // 012 sys_brk,
    SysRtSigaction,         // 013 sys_rt_sigaction,
    SysRtSigProcMask,       // 014 sys_rt_sigprocmask,
    SysRtSigreturn,         // 015 sys_rt_sigreturn,
    SysIoctl,               // 016 sys_ioctl,
    SysPread64,             // 017 sys_pread64,
    SysPwrite64,            // 018 sys_pwrite64,
    SysReadv,               // 019 sys_readv,
    SysWritev,              // 020 sys_writev,
    SysAccess,              // 021 sys_access,
    SysPipe,                // 022 sys_pipe,
    SysSelect,              // 023 sys_select,
    SysScheduleYield,       // 024 sys_sched_yield,
    SysMremap,              // 025 sys_mremap,
    SysMsync,               // 026 sys_msync,
    SysMincore,             // 027 sys_mincore,
    SysMadvise,             // 028 sys_madvise,
    SysShmget,              // 029 sys_shmget,
    SysShmat,               // 030 sys_shmat,
    SysShmctl,              // 031 sys_shmctl,
    SysDup,                 // 032 sys_dup,
    SysDup2,                // 033 sys_dup2,
    SysPause,               // 034 sys_pause,
    SysNanoSleep,           // 035 sys_nanosleep,
    SysGetitimer,           // 036 sys_getitimer,
    SysAlarm,               // 037 sys_alarm,
    SysSetitimer,           // 038 sys_setitimer,
    SysGetPid,              // 039 sys_getpid,
    SysSendfile,            // 040 sys_sendfile,
    SysSocket,              // 041 sys_socket,
    SysConnect,             // 042 sys_connect,
    SysAccept,              // 043 sys_accept,
    SysSendTo,              // 044 sys_sendto,
    SysRecvFrom,            // 045 sys_recvfrom,
    SysSendMsg,             // 046 sys_sendmsg,
    SysRecvMsg,             // 047 sys_recvmsg,
    SysShutdown,            // 048 sys_shutdown,
    SysBind,                // 049 sys_bind,
    SysListen,              // 050 sys_listen,
    SysGetSockName,         // 051 sys_getsockname,
    SysGetPeerName,         // 052 sys_getpeername,
    SysSocketPair,          // 053 sys_socketpair,
    SysSetSockOpt,          // 054 sys_setsockopt,
    SysGetSockOpt,          // 055 sys_getsockopt,
    SysClone,               // 056 sys_clone,
    SysFork,                // 057 sys_fork,
    SysVfork,               // 058 sys_vfork,
    SysExecve,              // 059 sys_execve,
    SysExit,                // 060 sys_exit,
    SysWait4,               // 061 sys_wait4,
    SysKill,                // 062 sys_kill,
    SysUname,               // 063 sys_uname,
    SysSemgetl,             // 064 sys_semget,
    SysSemop,               // 065 sys_semop,
    SysSemctl,              // 066 sys_semctl,
    SysShmdt,               // 067 sys_shmdt,
    SysMsgget,              // 068 sys_msgget,
    SysMsgsnd,              // 069 sys_msgsnd,
    SysMsgrcv,              // 070 sys_msgrcv,
    SysMsgctl,              // 071 sys_msgctl,
    SysFcntl,               // 072 sys_fcntl,
    SysFlock,               // 073 sys_flock,
    SysFsync,               // 074 sys_fsync,
    SysDatasync,            // 075 sys_fdatasync,
    SysTruncate,            // 076 sys_truncate,
    SysFtruncate,           // 077 sys_ftruncate,
    SysGetDents,            // 078 sys_getdents,
    SysGetcwd,              // 079 sys_getcwd,
    SysChdir,               // 080 sys_chdir,
    SysFchdir,              // 081 sys_fchdir,
    SysRename,              // 082 sys_rename,
    SysMkdir,               // 083 sys_mkdir,
    SysRmdir,               // 084 sys_rmdir,
    SysCreate,              // 085 sys_create,
    SysLink,                // 086 sys_link,
    SysUnlink,              // 087 sys_unlink,
    SysSymlink,             // 088 sys_symlink,
    SysReadLink,            // 089 sys_readlink,
    SysChmod,               // 090 sys_chmod,
    SysFchmod,              // 091 sys_fchmod,
    SysChown,               // 092 sys_chown,
    SysFchown,              // 093 sys_fchown,
    SysLchown,              // 094 sys_lchown,
    SysUmask,               // 095 sys_umask,
    SysGettimeofday,        // 096 sys_gettimeofday,
    SysGetrlimit,           // 097 sys_getrlimit,
    SysGetrusage,           // 098 sys_getrusage,
    SysInfo,                // 099 sys_sysinfo,
    SysTimes,               // 100 sys_times,
    SysNoSupport,           // 101 sys_ptrace,
    SysGetuid,              // 102 sys_getuid,
    SysSysLog,              // 103 sys_syslog,
    SysGetgid,              // 104 sys_getgid,
    SysSetuid,              // 105 sys_setuid,
    SysSetgid,              // 106 sys_setgid,
    SysGeteuid,             // 107 sys_geteuid,
    SysGetegid,             // 108 sys_getegid,
    SysSetpgid,             // 109 sys_setpgid,
    SysGetPpid,             // 110 sys_getppid,
    SysGetpgrp,             // 111 sys_getpgrp,
    SysSetsid,              // 112 sys_setsid,
    SysSetreuid,            // 113 sys_setreuid,
    SysSetregid,            // 114 sys_setregid,
    SysGetgroups,           // 115 sys_getgroups,
    SysSetgroups,           // 116 sys_setgroups,
    SysSetresuid,           // 117 sys_setresuid,
    SysGetresuid,           // 118 sys_getresuid,
    SysSetresgid,           // 119 sys_setresgid,
    SysGetresgid,           // 120 sys_getresgid,  //120
    SysGetpgid,             // 121 sys_getpgid,
    SysNoSys,               // 122 sys_setfsuid,
    SysNoSys,               // 123 sys_setfsgid,
    SysGetsid,              // 124 sys_getsid,
    SysCapget,              // 125 sys_capget,
    SysCapSet,              // 126 sys_capset,
    SysRtSigpending,        // 127 sys_rt_sigpending,
    SysRtSigtimedwait,      // 128 sys_rt_sigtimedwait,
    SysRtSigqueueinfo,      // 129 sys_rt_sigqueueinfo,
    SysRtSigsuspend,        // 130 sys_rt_sigsuspend,
    SysSigaltstack,         // 131 sys_sigaltstack,
    SysUtime,               // 132 sys_utime,
    SysMknode,              // 133 sys_mknod,
    SysObsolete,            // 134 sys_uselib,
    SysInvalid,             // 135 sys_personality,
    SysNoSys,               // 136 sys_ustat,      Needs filesystem support.
    SysStatfs,              // 137 sys_statfs,
    SysFstatfs,             // 138 sys_fstatfs,
    NotImplementSyscall,    // 139 sys_sysfs,
    SysGetpriority,         // 140 sys_getpriority,
    SysSetpriority,         // 141 sys_setpriority,
    SysCapErr,              // 142 sys_sched_setparam,
    SysSchedGetparam,       // 143 sys_sched_getparam	,
    SysSchedSetscheduler,   // 144 sys_sched_setscheduler,
    SysSchedGetscheduler,   // 145 sys_sched_getscheduler,
    SysSchedGetPriorityMax, // 146 sys_sched_get_priority_max,
    SysSchedGetPriorityMin, // 147 sys_sched_get_priority_min,
    SysNoPermission,        // 148 sys_sched_rr_get_interval,
    SysMlock,               // 149 sys_mlock,
    SysMunlock,             // 150 sys_munlock,
    SysMlockall,            // 151 sys_mlockall,
    SysMunlockall,          // 152 sys_munlockall,
    SysCapErr,              // 153 sys_vhangup,
    SysNoPermission,        // 154 sys_modify_ldt,
    SysNoPermission,        // 155 sys_pivot_root,
    SysNoPermission,        // 156 sys__sysctl,
    SysPrctl,               // 157 sys_prctl,
    SysArchPrctl,           // 158 sys_arch_prctl,
    SysCapErr,              // 159 sys_adjtimex,       CAP_SYS_TIME
    SysSetrlimit,           // 160 sys_setrlimit,
    SysChroot,              // 161 sys_chroot,
    SysSync,                // 162 sys_sync,
    SysCapErr,              // 163 sys_acct,
    SysCapErr,              // 164 sys_settimeofday,
    NotImplementSyscall,    // 165 sys_mount,
    NotImplementSyscall,    // 166 sys_umount2,
    SysCapErr,              // 167 sys_swapon,
    SysCapErr,              // 168 sys_swapoff,
    SysCapErr,              // 169 sys_reboot,
    SysSethostname,         // 170 sys_sethostname,
    SysSetdomainname,       // 171 sys_setdomainname,
    SysCapErr,              // 172 sys_iopl,
    SysCapErr,              // 173 sys_ioperm,
    SysCapErr,              // 174 sys_create_module,
    SysCapErr,              // 175 sys_init_module,
    SysCapErr,              // 176 sys_delete_module,
    SysNoSys,               // 177 sys_get_kernel_syms, Not supported in Linux > 2.6
    SysNoSys,               // 178 sys_query_module,    Not supported in Linux > 2.6
    SysCapErr,              // 179 sys_quotactl,
    SysNoSys,               // 180 sys_nfsservctl,      Removed after Linux 3.1
    SysNoSys,               // 181 sys_getpmsg,         Not implemented in Linux.
    SysNoSys,               // 182 sys_putpmsg,         Not implemented in Linux.
    SysNoSys,               // 183 sys_afs_syscall,     Not implemented in Linux.
    SysNoSys,               // 184 sys_tuxcall,         Not implemented in Linux.
    SysNoSys,               // 185 sys_security,        Not implemented in Linux.
    SysGetTid,              // 186 sys_gettid,
    SysReadahead,           // 187 sys_readahead,
    SysSetXattr,            // 188 sys_setxattr,
    SysLSetXattr,           // 189 sys_lsetxattr,
    SysFSetXattr,           // 190 sys_fsetxattr,
    SysGetXattr,            // 191 sys_getxattr,
    SysLGetXattr,           // 192 sys_lgetxattr,
    SysFGetXattr,           // 193 sys_fgetxattr,
    SysListXattr,           // 194 sys_listxattr,
    SysLListXattr,          // 195 sys_llistxattr,
    SysFListXattr,          // 196 sys_flistxattr,
    SysRemoveXattr,         // 197 sys_removexattr,
    SysLRemoveXattr,        // 198 sys_lremovexattr,
    SysFRemoveXattr,        // 199 sys_fremovexattr,
    SysTkill,               // 200 sys_tkill,
    SysTime,                // 201 sys_time,
    SysFutex,               // 202 sys_futex,
    SysSchedSetaffinity,    // 203 sys_sched_setaffinity,
    SysSchedGetaffinity,    // 204 sys_sched_getaffinity,
    SysNoSys,               // 205 sys_set_thread_area,     Expected to return ENOSYS on 64-bit
    SysIoSetup,             // 206 sys_io_setup,
    SysIoDestroy,           // 207 sys_io_destroy,
    SysIoGetevents,         // 208 sys_io_getevents,
    SysIOSubmit,            // 209 sys_io_submit,
    SysIOCancel,            // 210 sys_io_cancel,
    SysNoSys,               // 211 sys_get_thread_area,     Expected to return ENOSYS on 64-bit
    SysCapErr,              // 212 sys_lookup_dcookie,      CAP_SYS_ADMIN
    SysEpollCreate,         // 213 sys_epoll_create,
    SysNoSys,               // 214 sys_epoll_ctl_old,       Deprecated
    SysNoSys,               // 215 sys_epoll_wait_old,      Deprecated
    SysNoSys,               // 216 sys_remap_file_pages,    Deprecated since Linux 3.16.
    SysGetDents64,          // 217 sys_getdents64,
    SysSetTidAddr,          // 218 sys_set_tid_address,
    SysRestartSyscall,      // 219 sys_restart_syscall,
    SysSemtimedop,          // 220 sys_semtimedop,
    SysFadvise64,           // 221 sys_fadvise64,
    SysTimerCreate,         // 222 sys_timer_create,
    SysTimerSettime,        // 223 sys_timer_settime,
    SysTimerGettime,        // 224 sys_timer_gettime,
    SysTimerGetoverrun,     // 225 sys_timer_getoverrun,
    SysTimerDelete,         // 226 sys_timer_delete,
    SysClockSettime,        // 227 sys_clock_settime,
    SysClockGetTime,        // 228 sys_clock_gettime,
    SysClockGetRes,         // 229 sys_clock_getres,
    SysClockNanosleep,      // 230 sys_clock_nanosleep,
    SysExitThreadGroup,     // 231 sys_exit_group,
    SysEpollWait,           // 232 sys_epoll_wait,
    SysEpollCtl,            // 233 sys_epoll_ctl,
    SysTgkill,              // 234 sys_tgkill,
    SysUtimes,              // 235 sys_utimes,
    SysNoSys,               // 236 sys_vserver,             Not implemented by Linux
    SysMbind,               // 237 sys_mbind, just workaround
    SysSetMempolicy,        // 238 sys_set_mempolicy,
    SysGetMempolicy,        // 239 sys_get_mempolicy,
    SysNoSupport,           // 240 sys_mq_open,
    SysNoSupport,           // 241 sys_mq_unlink,
    SysNoSupport,           // 242 sys_mq_timedsend,
    SysNoSupport,           // 243 sys_mq_timedreceive,
    SysNoSupport,           // 244 sys_mq_notify,
    SysNoSupport,           // 245 sys_mq_getsetattr,
    SysCapErr,              // 246 sys_kexec_load,          CAP_SYS_BOOT
    SysWaitid,              // 247 sys_waitid,
    SysNoAccess,            // 248 sys_add_key,              Not available to user.
    SysNoAccess,            // 249 sys_request_key,          Not available to user.
    SysNoAccess,            // 250 sys_keyctl,    //250      Not available to user.
    SysCapErr,              // 251 sys_ioprio_set,           CAP_SYS_ADMIN
    SysCapErr,              // 252 sys_ioprio_get,           CAP_SYS_ADMIN
    SysInotifyInit,         // 253 sys_inotify_init,
    SysInotifyAddWatch,     // 254 sys_inotify_add_watch,
    SysInotifyRmWatch,      // 255 sys_inotify_rm_watch,
    SysCapErr,              // 256 sys_migrate_pages,        CAP_SYS_NICE
    SysOpenAt,              // 257 sys_openat,
    SysMkdirat,             // 258 sys_mkdirat,
    SysMknodeat,            // 259 sys_mknodat,
    SysFchownat,            // 260 sys_fchownat,
    SysFutimesat,           // 261 sys_futimesat,
    SysFstatat,             // 262 sys_newfstatat,
    SysUnlinkat,            // 263 sys_unlinkat,
    SysRenameat,            // 264 sys_renameat,
    SysLinkat,              // 265 sys_linkat,
    SysSymlinkat,           // 266 sys_symlinkat,
    SysReadLinkAt,          // 267 sys_readlinkat,
    SysFchmodat,            // 268 sys_fchmodat,
    SysFaccessat,           // 269 sys_faccessat,
    SysPSelect,             // 270 sys_pselect6,
    SysPpoll,               // 271 sys_ppoll,
    NotImplementSyscall,    // 272 sys_unshare,
    SysSetRobustList,       // 273 sys_set_robust_list,
    SysGetRobustList,       // 274 sys_get_robust_list,
    SysSplice,              // 275 sys_splice,
    SysTee,                 // 276 sys_tee,
    SysSyncFileRange,       // 277 sys_sync_file_range,
    NotImplementSyscall,    // 278 sys_vmsplice,
    SysCapErr,              // 279 sys_move_pages,          CAP_SYS_NICE
    SysUtimensat,           // 280 sys_utimensat,
    SysPwait,               // 281 sys_epoll_pwait,
    SysSignalfd,            // 282 sys_signalfd,
    SysTimerfdCreate,       // 283 sys_timerfd_create,
    SysEventfd,             // 284 sys_eventfd,
    SysFallocate,           // 285 sys_fallocate,
    SysTimerfdSettime,      // 286 sys_timerfd_settime,
    SysTimerfdGettime,      // 287 sys_timerfd_gettime,
    SysAccept4,             // 288 sys_accept4,
    SysSignalfd4,           // 289 sys_signalfd4,
    SysEventfd2,            // 290 sys_eventfd2,
    SysEpollCreate1,        // 291 sys_epoll_create1,
    SysDup3,                // 292 sys_dup3,
    SysPipe2,               // 293 sys_pipe2,
    SysInotifyInit1,        // 294 sys_inotify_init1,
    SysPreadv,              // 295 sys_preadv,
    SysPwritev,             // 296 sys_pwritev,
    SysRtTgsigqueueinfo,    // 297 sys_rt_tgsigqueueinfo,
    SysNoDev,               // 298 sys_perf_event_open,     No support for perf counters
    SysRecvMMsg,            // 299 sys_recvmmsg,
    SysNoSys,               //	300 sys_fanotify_init,       Needs CONFIG_FANOTIFY
    SysNoSys,               //	309 sys_fanotify_mark,       Needs CONFIG_FANOTIFY
    SysPrlimit64,           //	308 sys_prlimit64,
    SysOpNotSupport,        //	307 sys_name_to_handle_at,
    SysOpNotSupport,        //	306 sys_open_by_handle_at,
    SysCapErr,              //	305 sys_clock_adjtime,       CAP_SYS_TIME
    SysSyncFs,              //	304 sys_syncfs,
    SysSendMMsg,            //	303 sys_sendmmsg,
    SysOpNotSupport,        //	302 sys_setns,                   Needs filesystem support
    SysGetcpu,              //	301 sys_getcpu,
    SysNoSys,               //	310 sys_process_vm_readv    Need ptrace
    SysNoSys,               //	311 sys_process_vm_writev
    SysCapErr,              //	312 sys_kcmp,                CAP_SYS_PTRACE
    SysCapErr,              //	313 sys_finit_module,        CAP_SYS_MODULE
    SysNoSys,               //	314 sys_sched_setattr,       implement scheduler?
    SysNoSys,               //	315 sys_sched_getattr,       implement scheduler?
    SysNoSupport,           //	316 sys_renameat2,
    NotImplementSyscall,    //	317 sys_seccomp,
    SysGetRandom,           //	318 sys_getrandom,
    SysMemfdCreate,         //	319 sys_memfd_create,
    SysCapErr,              //	320 sys_kexec_file_load    CAP_SYS_BOOT
    SysCapErr,              //	321 sys_bpf,                 CAP_SYS_ADMIN
    SysExecveat,            //	322 sys_stub_execveat,
    NotImplementSyscall,    //	323 sys_userfaultfd,
    SysMembarrier,          //	324 sys_membarrier,
    SysMlock2,              //	325 mlock2,
    SysNoSys,               //	326 sys_copy_file_range,
    SysPreadv2,             //	327 sys_preadv2,
    SysPWritev2,            //	328 sys_pwritev2,
    NotImplementSyscall,    //	329 sys_pkey_mprotect,
    NotImplementSyscall,    //	330 sys_pkey_alloc,
    NotImplementSyscall,    //	331 sys_pkey_free,
    SysStatx,               //	332 sys_statx,
    NotImplementSyscall,    //	333 sys_io_pgetevents
    SysNoSys,               //	334 sys_rseq
    //don't use numbers 334 through 423
    ///////////////////////////////////////////////////////////////////////////////////////
    NotExisting, //	335
    NotExisting, //	336
    NotExisting, //	337
    NotExisting, //	338
    NotExisting, //	339
    NotExisting, //	340
    NotExisting, //	341
    NotExisting, //	342
    NotExisting, //	343
    NotExisting, //	344
    NotExisting, //	345
    NotExisting, //	346
    NotExisting, //	347
    NotExisting, //	348
    NotExisting, //	349
    NotExisting, //	350
    NotExisting, //	351
    NotExisting, //	352
    NotExisting, //	353
    NotExisting, //	354
    NotExisting, //	355
    NotExisting, //	356
    NotExisting, //	357
    NotExisting, //	358
    NotExisting, //	359
    NotExisting, //	360
    NotExisting, //	361
    NotExisting, //	362
    NotExisting, //	363
    NotExisting, //	364
    NotExisting, //	365
    NotExisting, //	366
    NotExisting, //	367
    NotExisting, //	368
    NotExisting, //	369
    NotExisting, //	370
    NotExisting, //	371
    NotExisting, //	372
    NotExisting, //	373
    NotExisting, //	374
    NotExisting, //	375
    NotExisting, //	376
    NotExisting, //	377
    NotExisting, //	378
    NotExisting, //	379
    NotExisting, //	380
    NotExisting, //	381
    NotExisting, //	382
    NotExisting, //	383
    NotExisting, //	384
    NotExisting, //	385
    NotExisting, //	386
    NotExisting, //	387
    NotExisting, //	388
    NotExisting, //	389
    NotExisting, //	390
    NotExisting, //	391
    NotExisting, //	392
    NotExisting, //	393
    NotExisting, //	394
    NotExisting, //	395
    NotExisting, //	396
    NotExisting, //	397
    NotExisting, //	398
    NotExisting, //	399
    NotExisting, //	400
    NotExisting, //	401
    NotExisting, //	402
    NotExisting, //	403
    NotExisting, //	404
    NotExisting, //	405
    NotExisting, //	406
    NotExisting, //	407
    NotExisting, //	408
    NotExisting, //	409
    NotExisting, //	410
    NotExisting, //	411
    NotExisting, //	412
    NotExisting, //	413
    NotExisting, //	414
    NotExisting, //	415
    NotExisting, //	416
    NotExisting, //	417
    NotExisting, //	418
    NotExisting, //	419
    NotExisting, //	420
    NotExisting, //	421
    NotExisting, //	422
    NotExisting, //	423
    ////////////////////////////////////////////////////////////////////////////
    //don't use numbers 334 through 423

    // Linux skips ahead to syscall 424 to sync numbers between arches.
    NotImplementSyscall, //	424 sys_pidfd_send_signal
    SysNoSys,            //	425 sys_io_uring_setup
    SysNoSys,            //	426 sys_io_uring_enter
    SysNoSys,            //	427 sys_io_uring_register
    NotImplementSyscall, //	428 sys_open_tree
    NotImplementSyscall, //	429 sys_move_mount
    NotImplementSyscall, //	430 sys_fsopen
    NotImplementSyscall, //	431 sys_fsconfig
    NotImplementSyscall, //	432 sys_fsmount
    NotImplementSyscall, //	433 sys_fspick
    NotImplementSyscall, //	434 sys_pidfd_open
    SysNoSys,            //	435 sys_clone3
    SysCloseRange,       //	436 sys_close_range
    NotImplementSyscall, //	437 sys_openat2
    NotImplementSyscall, //	438 sys_pidfd_getfd
    SysNoSys,            //	439 sys_faccessat2
    NotImplementSyscall, //	440 sys_process_madvise
    SysPwait2,           //	441 sys_epoll_pwait2
    NotImplementSyscall, //	442 sys_mouLoad(nt_setattr
    NotImplementSyscall, //	443 sys_quotactl_fd
    NotImplementSyscall, //	444 sys_landlock_create_ruleset
    NotImplementSyscall, //	445 sys_landlock_add_rule
    NotImplementSyscall, //	446 sys_landlock_restrict_self
    NotImplementSyscall, //	447 sys_memfd_secret
    NotImplementSyscall, //	448 sys_process_mrelease
    NotImplementSyscall, //	449 sys_futex_waitv
    NotImplementSyscall, //	450 sys_set_mempolicy_home_node
    NotExisting,         // 451 unknow syscall
];

#[cfg(target_arch = "aarch64")]
pub const SYS_CALL_TABLE: &'static [SyscallFn] = &[
    SysIoSetup,            // 0   sys_io_setup = 0 as u64,
    SysIoDestroy,          // 1   sys_io_destroy,
    SysIOSubmit,           // 2   sys_io_submit,
    SysIOCancel,           // 3   sys_io_cancel,
    SysIoGetevents,        // 4   sys_io_getevents,
    SysSetXattr,           // 5   sys_setxattr,
    SysLSetXattr,          // 6   sys_lsetxattr,
    SysFSetXattr,          // 7   sys_fsetxattr,
    SysGetXattr,           // 8   sys_getxattr,
    SysLGetXattr,          // 9   sys_lgetxattr,
    SysFGetXattr,          // 10  sys_fgetxattr, //10
    SysListXattr,          // 11  sys_listxattr,
    SysLListXattr,         // 12  sys_llistxattr,
    SysFListXattr,         // 13  sys_flistxattr,
    SysRemoveXattr,        // 14  sys_removexattr,
    SysLRemoveXattr,       // 15  sys_lremovexattr,
    SysFRemoveXattr,       // 16  sys_fremovexattr,
    SysGetcwd,             // 17  sys_getcwd,
    NotImplementSyscall,   // 18  sys_lookup_dcookie,
    SysEventfd2,           // 19  sys_eventfd2,
    SysEpollCreate1,       // 20  sys_epoll_create1, //20
    SysEpollCtl,           // 21  sys_epoll_ctl,
    SysPwait,              // 22  sys_epoll_pwait,
    SysDup,                // 23  sys_dup,
    SysDup3,               // 24  sys_dup3,
    SysFcntl,              // 25  sys_fcntl,
    SysInotifyInit,        // 26  sys_inotify_init1,
    SysInotifyAddWatch,    // 27  sys_inotify_add_watch,
    SysInotifyRmWatch,     // 28  sys_inotify_rm_watch,
    SysIoctl,              // 29  sys_ioctl,
    SysCapErr,             // 30  sys_ioprio_set, //30
    SysCapErr,             // 31  sys_ioprio_get,
    SysFlock,              // 32  sys_flock,
    SysMknodeat,           // 33  sys_mknodat,
    SysMkdirat,            // 34  sys_mkdirat,
    SysUnlinkat,           // 35  sys_unlinkat,
    SysSymlinkat,          // 36  sys_symlinkat,
    SysLinkat,             // 37  sys_linkat,
    SysRenameat,           // 38  sys_renameat,
    NotImplementSyscall,   // 39  sys_umount2,
    NotImplementSyscall,   // 40  sys_mount, //40
    SysNoPermission,       // 41  sys_pivot_root,
    SysNoSys,              // 42  sys_nfsservctl,
    SysStatfs,             // 43  sys_statfs,
    SysFstatfs,            // 44  sys_fstatfs,
    SysTruncate,           // 45  sys_truncate,
    SysFtruncate,          // 46  sys_ftruncate,
    SysFallocate,          // 47  sys_fallocate,
    SysFaccessat,          // 48  sys_faccessat,
    SysChdir,              // 49  sys_chdir,
    SysFchdir,             // 50  sys_fchdir, //50
    SysChroot,             // 51  sys_chroot,
    SysFchmod,             // 52  sys_fchmod,
    SysFchmodat,           // 53  sys_fchmodat,
    SysFchownat,           // 54  sys_fchownat,
    SysFchown,             // 55  sys_fchown,
    SysOpenAt,             // 56  sys_openat,
    SysClose,              // 57  sys_close,
    SysCapErr,             // 58  sys_vhangup,
    SysPipe2,              // 59  sys_pipe2,
    SysCapErr,             // 60  sys_quotactl, //60
    SysGetDents64,         // 61  sys_getdents64,
    SysLseek,              // 62  sys_lseek,
    SysRead,               // 63  sys_read,
    SysWrite,              // 64  sys_write,
    SysReadv,              // 65  sys_readv,
    SysWritev,             // 66  sys_writev,
    SysPread64,            // 67  sys_pread64,
    SysPwrite64,           // 68  sys_pwrite64,
    SysPreadv,             // 69  sys_preadv,
    SysPwritev,            // 70  sys_pwritev, //70
    SysSendfile,           // 71  sys_sendfile,
    SysPSelect,            // 72  sys_pselect6,
    SysPpoll,              // 73  sys_ppoll,
    SysSignalfd4,          // 74  sys_signalfd4,
    NotImplementSyscall,   // 75  sys_vmsplice,
    SysSplice,             // 76  sys_splice,
    SysTee,                // 77  sys_tee,
    SysReadLinkAt,         // 78  sys_readlinkat,
    SysFstatat,            // 79  sys_newfstatat,
    SysFstat,              // 80  sys_fstat, //80
    SysSync,               // 81  sys_sync,
    SysFsync,              // 82  sys_fsync,
    SysDatasync,           // 83  sys_fdatasync,
    SysSyncFileRange,      // 84  sys_sync_file_range,
    SysTimerfdCreate,      // 85  sys_timerfd_create,
    SysTimerfdSettime,     // 86  sys_timerfd_settime,
    SysTimerfdGettime,     // 87  sys_timerfd_gettime,
    SysUtimensat,          // 88  sys_utimensat,
    SysCapErr,             // 89  sys_acct,
    SysCapget,             // 90  sys_capget, //90
    SysCapSet,             // 91  sys_capset,
    SysInvalid,            // 92  sys_personality,
    SysExit,               // 93  sys_exit,
    SysExitThreadGroup,    // 94  sys_exit_group,
    SysWaitid,             // 95  sys_waitid,
    SysSetTidAddr,         // 96  sys_set_tid_address,
    SysUnshare,            // 97  sys_unshare,
    SysFutex,              // 98  sys_futex,
    SysSetRobustList,      // 99  sys_set_robust_list,
    SysGetRobustList,      // 100 sys_get_robust_list, //100
    SysNanoSleep,          // 101 sys_nanosleep,
    SysGetitimer,          // 102 sys_getitimer,
    SysSetitimer,          // 103 sys_setitimer,
    SysCapErr,             // 104 sys_kexec_load,
    SysCapErr,             // 105 sys_init_module,
    SysCapErr,             // 106 sys_delete_module,
    SysTimerCreate,        // 107 sys_timer_create,
    SysTimerGettime,       // 108 sys_timer_gettime,
    SysTimerGetoverrun,    // 109 sys_timer_getoverrun,
    SysTimerSettime,       // 110 sys_timer_settime, //110
    SysTimerDelete,        // 111 sys_timer_delete,
    SysClockSettime,       // 112 sys_clock_settime,
    SysClockGetTime,       // 113 sys_clock_gettime,
    SysClockGetRes,        // 114 sys_clock_getres,
    SysClockNanosleep,     // 115 sys_clock_nanosleep,
    SysSysLog,             // 116 sys_syslog,
    SysNoSupport,          // 117 sys_ptrace,
    SysCapErr,             // 118 sys_sched_setparam,
    SysSchedSetscheduler,  // 119 sys_sched_setscheduler,
    SysSchedGetscheduler,  // 120 sys_sched_getscheduler, //120
    SysSchedGetparam,      // 121 sys_sched_getparam,
    SysSchedSetaffinity,   // 122 sys_sched_setaffinity,
    SysSchedGetaffinity,   // 123 sys_sched_getaffinity,
    SysScheduleYield,      // 124 sys_sched_yield,
    SysSchedGetPriorityMax,// 125 sys_sched_get_priority_max,
    SysSchedGetPriorityMin,// 126 sys_sched_get_priority_min,
    SysNoPermission,       // 127 sys_sched_rr_get_interval,
    SysRestartSyscall,     // 128 sys_restart_syscall,
    SysKill,               // 129 sys_kill,
    SysTkill,              // 130 sys_tkill, //130
    SysTgkill,             // 131 sys_tgkill,
    SysSigaltstack,        // 132 sys_sigaltstack,
    SysRtSigsuspend,       // 133 sys_rt_sigsuspend,
    SysRtSigaction,        // 134 sys_rt_sigaction,
    SysRtSigProcMask,      // 135 sys_rt_sigprocmask,
    SysRtSigpending,       // 136 sys_rt_sigpending,
    SysRtSigtimedwait,     // 137 sys_rt_sigtimedwait,
    SysRtSigqueueinfo,     // 138 sys_rt_sigqueueinfo,
    SysRtSigreturn,        // 139 sys_rt_sigreturn,
    SysSetpriority,        // 140 sys_setpriority, //140
    SysGetpriority,        // 141 sys_getpriority,
    SysCapErr,             // 142 sys_reboot,
    SysSetregid,           // 143 sys_setregid,
    SysSetgid,             // 144 sys_setgid,
    SysSetreuid,           // 145 sys_setreuid,
    SysSetuid,             // 146 sys_setuid,
    SysSetresuid,          // 147 sys_setresuid,
    SysGetresuid,          // 148 sys_getresuid,
    SysSetresgid,          // 149 sys_setresgid,
    SysGetresgid,          // 150 sys_getresgid, //150
    SysNoSys,              // 151 sys_setfsuid,
    SysNoSys,              // 152 sys_setfsgid,
    SysTimes,              // 153 sys_times,
    SysSetpgid,            // 154 sys_setpgid,
    SysGetpgid,            // 155 sys_getpgid,
    SysGetsid,             // 156 sys_getsid,
    SysSetsid,             // 157 sys_setsid,
    SysGetgroups,          // 158 sys_getgroups,
    SysSetgroups,          // 159 sys_setgroups,
    SysUname,              // 160 sys_uname, // 160
    SysSethostname,        // 161 sys_sethostname,
    SysSetdomainname,      // 162 sys_setdomainname,
    SysGetrlimit,          // 163 sys_getrlimit,
    SysSetrlimit,          // 164 sys_setrlimit,
    SysGetrusage,          // 165 sys_getrusage,
    SysUmask,              // 166 sys_umask,
    SysPrctl,              // 167 sys_prctl,
    SysGetcpu,             // 168 sys_getcpu,
    SysGettimeofday,       // 169 sys_gettimeofday,
    SysCapErr,             // 170 sys_settimeofday, //170
    SysCapErr,             // 171 sys_adjtimex,
    SysGetPid,             // 172 sys_getpid,
    SysGetPpid,            // 173 sys_getppid,
    SysGetuid,             // 174 sys_getuid,
    SysGeteuid,            // 175 sys_geteuid,
    SysGetgid,             // 176 sys_getgid,
    SysGetegid,            // 177 sys_getegid,
    SysGetTid,             // 178 sys_gettid,
    SysInfo,               // 179 sys_sysinfo,
    SysNoSupport,          // 180 sys_mq_open, //180
    SysNoSupport,          // 181 sys_mq_unlink,
    SysNoSupport,          // 182 sys_mq_timedsend,
    SysNoSupport,          // 183 sys_mq_timedreceive,
    SysNoSupport,          // 184 sys_mq_notify,
    SysNoSupport,          // 185 sys_mq_getsetattr,
    SysMsgget,             // 186 sys_msgget,
    SysMsgctl,             // 187 sys_msgctl,
    SysMsgrcv,             // 188 sys_msgrcv,
    SysMsgsnd,             // 189 sys_msgsnd,
    SysSemgetl,            // 190 sys_semget, //190
    SysSemctl,             // 191 sys_semctl,
    SysSemtimedop,         // 192 sys_semtimedop,
    SysSemop,              // 193 sys_semop,
    SysShmget,             // 194 sys_shmget,
    SysShmctl,             // 195 sys_shmctl,
    SysShmat,              // 196 sys_shmat,
    SysShmdt,              // 197 sys_shmdt,
    SysSocket,             // 198 sys_socket,
    SysSocketPair,         // 199 sys_socketpair,
    SysBind,               // 200 sys_bind, //200
    SysListen,             // 201 sys_listen,
    SysAccept,             // 202 sys_accept,
    SysConnect,            // 203 sys_connect,
    SysGetSockName,        // 204 sys_getsockname,
    SysGetPeerName,        // 205 sys_getpeername,
    SysSendTo,             // 206 sys_sendto,
    SysRecvFrom,           // 207 sys_recvfrom,
    SysSetSockOpt,         // 208 sys_setsockopt,
    SysGetSockOpt,         // 209 sys_getsockopt,
    SysShutdown,           // 210 sys_shutdown, //210
    SysSendMsg,            // 211 sys_sendmsg,
    SysRecvMsg,            // 212 sys_recvmsg,
    SysReadahead,          // 213 sys_readahead,
    SysBrk,                // 214 sys_brk,
    SysUnmap,              // 215 sys_munmap,
    SysMremap,             // 216 sys_mremap,
    SysNoAccess,           // 217 sys_add_key,
    SysNoAccess,           // 218 sys_request_key,
    SysNoAccess,           // 219 sys_keyctl,
    SysClone,              // 220 sys_clone, //220
    SysExecve,             // 221 sys_execve,
    SysMmap,               // 222 sys_mmap,
    SysFadvise64,          // 223 sys_fadvise64,
    SysCapErr,             // 224 sys_swapon,
    SysCapErr,             // 225 sys_swapoff,
    SysMprotect,           // 226 sys_mprotect,
    SysMsync,              // 227 sys_msync,
    SysMlock,              // 228 sys_mlock,
    SysMunlock,            // 229 sys_munlock,
    SysMlockall,           // 230 sys_mlockall, //230
    SysMunlockall,         // 231 sys_munlockall,
    SysMincore,            // 232 sys_mincore,
    SysMadvise,            // 233 sys_madvise,
    SysNoSys,              // 234 sys_remap_file_pages,
    SysMbind,              // 235 sys_mbind,
    SysGetMempolicy,       // 236 sys_get_mempolicy,
    SysSetMempolicy,       // 237 sys_set_mempolicy,
    SysCapErr,             // 238 sys_migrate_pages,
    SysCapErr,             // 239 sys_move_pages,
    SysRtTgsigqueueinfo,   // 240 sys_rt_tgsigqueueinfo, //240
    SysNoDev,              // 241 sys_perf_event_open,
    SysAccept4,            // 242 sys_accept4,
    SysRecvMMsg,           // 243 sys_recvmmsg,
    NotImplementSyscall,   // 244 syscall_244,
    NotImplementSyscall,   // 245 syscall_245,
    NotImplementSyscall,   // 246 syscall_246,
    NotImplementSyscall,   // 247 syscall_247,
    NotImplementSyscall,   // 248 syscall_248,
    NotImplementSyscall,   // 249 syscall_249,
    NotImplementSyscall,   // 250 syscall_250, //250
    NotImplementSyscall,   // 251 syscall_251,
    NotImplementSyscall,   // 252 syscall_252,
    NotImplementSyscall,   // 253 syscall_253,
    NotImplementSyscall,   // 254 syscall_254,
    NotImplementSyscall,   // 255 syscall_255,
    NotImplementSyscall,   // 256 syscall_256,
    NotImplementSyscall,   // 257 syscall_257,
    NotImplementSyscall,   // 258 syscall_258,
    NotImplementSyscall,   // 259 syscall_259,
    SysWait4,              // 260 sys_wait4, //260
    SysPrlimit64,          // 261 sys_prlimit64,
    SysNoSys,              // 262 sys_fanotify_init,
    SysNoSys,              // 263 sys_fanotify_mark,
    SysOpNotSupport,       // 264 sys_name_to_handle_at,
    SysOpNotSupport,       // 265 sys_open_by_handle_at,
    SysCapErr,             // 266 sys_clock_adjtime,
    SysSyncFs,             // 267 sys_syncfs,
    SysOpNotSupport,       // 268 sys_setns,
    SysSendMMsg,           // 269 sys_sendmmsg,
    SysNoSys,              // 270 sys_process_vm_readv, //270
    SysNoSys,              // 271 sys_process_vm_writev,
    SysCapErr,             // 272 sys_kcmp,
    SysCapErr,             // 273 sys_finit_module,
    SysNoSys,              // 274 sys_sched_setattr,
    SysNoSys,              // 275 sys_sched_getattr,
    SysNoSupport,          // 276 sys_renameat2,
    NotImplementSyscall,   // 277 sys_seccomp,
    SysGetRandom,          // 278 sys_getrandom,
    SysMemfdCreate,        // 279 sys_memfd_create,
    SysCapErr,             // 280 sys_bpf, //280
    SysExecveat,           // 281 sys_execveat,
    NotImplementSyscall,   // 282 sys_userfaultfd,
    SysMembarrier,         // 283 sys_membarrier,
    SysMlock2,             // 284 sys_mlock2,
    SysNoSys,              // 285 sys_copy_file_range,
    SysPreadv2,            // 286 sys_preadv2,
    SysPWritev2,           // 287 sys_pwritev2,
    NotImplementSyscall,   // 288 sys_pkey_mprotect,
    NotImplementSyscall,   // 289 sys_pkey_alloc,
    NotImplementSyscall,   // 290 sys_pkey_free,
    SysStatx,              // 291 sys_statx,
    NotImplementSyscall,  // 292
    SysNoSys,              // 293 sys_rseq,
    NotImplementSyscall,  // 294
    NotImplementSyscall,  // 295
    NotImplementSyscall,  // 296
    NotImplementSyscall,  // 297
    NotImplementSyscall,  // 298
    NotImplementSyscall,  // 299
    NotImplementSyscall,  // 300
    NotImplementSyscall,  // 301
    NotImplementSyscall,  // 302
    NotImplementSyscall,  // 303
    NotImplementSyscall,  // 304
    NotImplementSyscall,  // 305
    NotImplementSyscall,  // 306
    NotImplementSyscall,  // 307
    NotImplementSyscall,  // 308
    NotImplementSyscall,  // 309
    NotImplementSyscall,  // 310
    NotImplementSyscall,  // 311
    NotImplementSyscall,  // 312
    NotImplementSyscall,  // 313
    NotImplementSyscall,  // 314
    NotImplementSyscall,  // 315
    NotImplementSyscall,  // 316
    NotImplementSyscall,  // 317
    NotImplementSyscall,  // 318
    NotImplementSyscall,  // 319
    NotImplementSyscall,  // 320
    NotImplementSyscall,  // 321
    NotImplementSyscall,  // 322
    NotImplementSyscall,  // 323
    NotImplementSyscall,  // 324
    NotImplementSyscall,  // 325
    NotImplementSyscall,  // 326
    NotImplementSyscall,  // 327
    NotImplementSyscall,  // 328
    NotImplementSyscall,  // 329
    NotImplementSyscall,  // 330
    NotImplementSyscall,  // 331
    NotImplementSyscall,  // 332
    NotImplementSyscall,  // 333
    NotImplementSyscall,  // 334
    NotImplementSyscall,  //	335
    NotImplementSyscall,  //	336
    NotImplementSyscall,  //	337
    NotImplementSyscall,  //	338
    NotImplementSyscall,  //	339
    NotImplementSyscall,  //	340
    NotImplementSyscall,  //	341
    NotImplementSyscall,  //	342
    NotImplementSyscall,  //	343
    NotImplementSyscall,  //	344
    NotImplementSyscall,  //	345
    NotImplementSyscall,  //	346
    NotImplementSyscall,  //	347
    NotImplementSyscall,  //	348
    NotImplementSyscall,  //	349
    NotImplementSyscall,  //	350
    NotImplementSyscall,  //	351
    NotImplementSyscall,  //	352
    NotImplementSyscall,  //	353
    NotImplementSyscall,  //	354
    NotImplementSyscall,  //	355
    NotImplementSyscall,  //	356
    NotImplementSyscall,  //	357
    NotImplementSyscall,  //	358
    NotImplementSyscall,  //	359
    NotImplementSyscall,  //	360
    NotImplementSyscall,  //	361
    NotImplementSyscall,  //	362
    NotImplementSyscall,  //	363
    NotImplementSyscall,  //	364
    NotImplementSyscall,  //	365
    NotImplementSyscall,  //	366
    NotImplementSyscall,  //	367
    NotImplementSyscall,  //	368
    NotImplementSyscall,  //	369
    NotImplementSyscall,  //	370
    NotImplementSyscall,  //	371
    NotImplementSyscall,  //	372
    NotImplementSyscall,  //	373
    NotImplementSyscall,  //	374
    NotImplementSyscall,  //	375
    NotImplementSyscall,  //	376
    NotImplementSyscall,  //	377
    NotImplementSyscall,  //	378
    NotImplementSyscall,  //	379
    NotImplementSyscall,  //	380
    NotImplementSyscall,  //	381
    NotImplementSyscall,  //	382
    NotImplementSyscall,  //	383
    NotImplementSyscall,  //	384
    NotImplementSyscall,  //	385
    NotImplementSyscall,  //	386
    NotImplementSyscall,  //	387
    NotImplementSyscall,  //	388
    NotImplementSyscall,  //	389
    NotImplementSyscall,  //	390
    NotImplementSyscall,  //	391
    NotImplementSyscall,  //	392
    NotImplementSyscall,  //	393
    NotImplementSyscall,  //	394
    NotImplementSyscall,  //	395
    NotImplementSyscall,  //	396
    NotImplementSyscall,  //	397
    NotImplementSyscall,  //	398
    NotImplementSyscall,  //	399
    NotImplementSyscall,  //	400
    NotImplementSyscall,  //	401
    NotImplementSyscall,  //	402
    NotImplementSyscall,  //	403
    NotImplementSyscall,  //	404
    NotImplementSyscall,  //	405
    NotImplementSyscall,  //	406
    NotImplementSyscall,  //	407
    NotImplementSyscall,  //	408
    NotImplementSyscall,  //	409
    NotImplementSyscall,  //	410
    NotImplementSyscall,  //	411
    NotImplementSyscall,  //	412
    NotImplementSyscall,  //	413
    NotImplementSyscall,  //	414
    NotImplementSyscall,  //	415
    NotImplementSyscall,  //	416
    NotImplementSyscall,  //	417
    NotImplementSyscall,  //	418
    NotImplementSyscall,  //	419
    NotImplementSyscall,  //	420
    NotImplementSyscall,  //	421
    NotImplementSyscall,  //	422
    NotImplementSyscall,  //	423
    NotImplementSyscall,  //	424
    NotImplementSyscall,  //	425
    NotImplementSyscall,  //	426
    NotImplementSyscall,  //	427
    NotImplementSyscall,  //	428
    NotImplementSyscall,  //	429
    NotImplementSyscall,  //	430
    NotImplementSyscall,  //	431
    NotImplementSyscall,  //	432
    NotImplementSyscall,  //	433
    NotImplementSyscall,  //	434
    NotImplementSyscall,  //	435
    NotImplementSyscall,  //	436
    NotImplementSyscall,  //	437
    NotImplementSyscall,  //	438
    SysNoSys,             //	439
    NotImplementSyscall,  //	440
    NotImplementSyscall,  //	441
    NotImplementSyscall,  //	442
    NotImplementSyscall,  //	443
    NotImplementSyscall,  //	444
    NotImplementSyscall,  //	445
    NotImplementSyscall,  //	446
    NotImplementSyscall,  //	447
    NotImplementSyscall,  //	448
    NotImplementSyscall,  //	449
    NotImplementSyscall,  //	450
    NotExisting,          // 451 unknow syscall
];

pub fn NotImplementSyscall(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("NotImplementSyscall syscall {:x?}", args);
    return Err(Error::SysCallNotImplement);
}

pub fn NotExisting(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("NotExisting syscall {:x?}", args);
    return Err(Error::SysError(SysErr::ENODATA));
}

pub fn SysNoSupport(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("SysNoSupport syscall {:x?}", args);
    return Err(Error::SysError(SysErr::ENODATA));
    //return Err(Error::SysError(SysErr::ENOTSUP));
}

pub fn SysNoAccess(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::EACCES));
}

pub fn SysInvalid(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::EINVAL));
    //return Err(Error::SysError(SysErr::ENOTSUP));
}

pub fn SysNoPermission(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::EPERM));
}

pub fn SysObsolete(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("SysObsolete syscall {:x?}", args);
    return Err(Error::SysError(SysErr::ENOSYS));
    //return Err(Error::SysError(SysErr::ENOTSUP));
}

pub fn SysNoSys(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS));
}

pub fn SysNoDev(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENODEV));
}

pub fn SysOpNotSupport(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::EOPNOTSUPP));
}

pub fn SysCapErr(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::EPERM));
    //return Err(Error::SysError(SysErr::ENOSYS));
}
