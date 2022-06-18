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
use super::super::syscalls::sys_membarrier::*;
use super::super::syscalls::sys_mempolicy::*;
use super::super::syscalls::sys_mmap::*;
use super::super::syscalls::sys_pipe::*;
use super::super::syscalls::sys_poll::*;
use super::super::syscalls::sys_prctl::*;
use super::super::syscalls::sys_random::*;
use super::super::syscalls::sys_read::*;
use super::super::syscalls::sys_rlimit::*;
use super::super::syscalls::sys_rusage::*;
use super::super::syscalls::sys_signal::*;
use super::super::syscalls::sys_socket::*;
use super::super::syscalls::sys_splice::*;
use super::super::syscalls::sys_stat::*;
use super::super::syscalls::sys_sync::*;
use super::super::syscalls::sys_sysinfo::*;
use super::super::syscalls::sys_thread::*;
use super::super::syscalls::sys_time::*;
use super::super::syscalls::sys_timer::*;
use super::super::syscalls::sys_timerfd::*;
use super::super::syscalls::sys_tls::*;
use super::super::syscalls::sys_utsname::*;
use super::super::syscalls::sys_write::*;
use super::super::syscalls::sys_memfd::*;
use super::super::syscalls::sys_sched::*;
use super::super::syscalls::sys_inotify::*;
use super::super::syscalls::sys_xattr::*;
use super::super::syscalls::sys_sem::*;
use super::super::syscalls::sys_shm::*;
use super::super::syscalls::sys_msgqueue::*;

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
    let func = SYS_CALL_TABLE.get(idx).unwrap();
    match func(task, args) {
        Err(Error::SysCallRetCtrlWithRet(state, ret)) => {
            task.SetReturn(ret);
            return state;
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

pub const SYS_CALL_TABLE: &'static [SyscallFn] = &[
    SysRead,             // 000 sys_read
    SysWrite,            // 001 sys_write,
    SysOpen,             // 002 sys_open,
    SysClose,            // 003 sys_close,
    SysStat,             // 004 sys_stat,
    SysFstat,            // 005 sys_fstat,
    SysLstat,            // 006 sys_lstat,
    SysPoll,             // 007 sys_poll,
    SysLseek,            // 008 sys_lseek,
    SysMmap,             // 009 sys_mmap,
    SysMprotect,         // 010 sys_mprotect,
    SysUnmap,            // 011 sys_munmap,
    SysBrk,              // 012 sys_brk,
    SysRtSigaction,      // 013 sys_rt_sigaction,
    SysRtSigProcMask,    // 014 sys_rt_sigprocmask,
    SysRtSigreturn,      // 015 sys_rt_sigreturn,
    SysIoctl,            // 016 sys_ioctl,
    SysPread64,          // 017 sys_pread64,
    SysPwrite64,         // 018 sys_pwrite64,
    SysReadv,            // 019 sys_readv,
    SysWritev,           // 020 sys_writev,
    SysAccess,           // 021 sys_access,
    SysPipe,             // 022 sys_pipe,
    SysSelect,           // 023 sys_select,
    SysScheduleYield,    // 024 sys_sched_yield,
    SysMremap,           // 025 sys_mremap,
    SysMsync,            // 026 sys_msync,
    SysMincore,          // 027 sys_mincore,
    SysMadvise,          // 028 sys_madvise,
    SysShmget,           // 029 sys_shmget,
    SysShmat,            // 030 sys_shmat,
    SysShmctl,           // 031 sys_shmctl,
    SysDup,              // 032 sys_dup,
    SysDup2,             // 033 sys_dup2,
    SysPause,            // 034 sys_pause,
    SysNanoSleep,        // 035 sys_nanosleep,
    SysGetitimer,        // 036 sys_getitimer,
    SysAlarm,            // 037 sys_alarm,
    SysSetitimer,        // 038 sys_setitimer,
    SysGetPid,           // 039 sys_getpid,
    SysSendfile,         // 040 sys_sendfile,
    SysSocket,           // 041 sys_socket,
    SysConnect,          // 042 sys_connect,
    SysAccept,           // 043 sys_accept,
    SysSendTo,           // 044 sys_sendto,
    SysRecvFrom,         // 045 sys_recvfrom,
    SysSendMsg,          // 046 sys_sendmsg,
    SysRecvMsg,          // 047 sys_recvmsg,
    SysShutdown,         // 048 sys_shutdown,
    SysBind,             // 049 sys_bind,
    SysListen,           // 050 sys_listen,
    SysGetSockName,      // 051 sys_getsockname,
    SysGetPeerName,      // 052 sys_getpeername,
    SysSocketPair,       // 053 sys_socketpair,
    SysSetSockOpt,       // 054 sys_setsockopt,
    SysGetSockOpt,       // 055 sys_getsockopt,
    SysClone,            // 056 sys_clone,
    SysFork,             // 057 sys_fork,
    SysVfork,            // 058 sys_vfork,
    SysExecve,           // 059 sys_execve,
    SysExit,             // 060 sys_exit,
    SysWait4,            // 061 sys_wait4,
    SysKill,             // 062 sys_kill,
    SysUname,            // 063 sys_uname,
    SysSemgetl,          // 064 sys_semget,
    SysSemop,            // 065 sys_semop,
    SysSemctl,           // 066 sys_semctl,
    SysShmdt,            // 067 sys_shmdt,
    SysMsgget,           // 068 sys_msgget,
    SysMsgsnd,           // 069 sys_msgsnd,
    SysMsgrcv,           // 070 sys_msgrcv,
    SysMsgctl,           // 071 sys_msgctl,
    SysFcntl,            // 072 sys_fcntl,
    SysFlock,            // 073 sys_flock,
    SysFsync,            // 074 sys_fsync,
    SysDatasync,         // 075 sys_fdatasync,
    SysTruncate,         // 076 sys_truncate,
    SysFtruncate,        // 077 sys_ftruncate,
    SysGetDents,         // 078 sys_getdents,
    SysGetcwd,           // 079 sys_getcwd,
    SysChdir,            // 080 sys_chdir,
    SysFchdir,           // 081 sys_fchdir,
    SysRename,           // 082 sys_rename,
    SysMkdir,            // 083 sys_mkdir,
    SysRmdir,            // 084 sys_rmdir,
    SysCreate,           // 085 sys_create,
    SysLink,             // 086 sys_link,
    SysUnlink,           // 087 sys_unlink,
    SysSymlink,          // 088 sys_symlink,
    SysReadLink,         // 089 sys_readlink,
    SysChmod,            // 090 sys_chmod,
    SysFchmod,           // 091 sys_fchmod,
    SysChown,            // 092 sys_chown,
    SysFchown,           // 093 sys_fchown,
    SysLchown,           // 094 sys_lchown,
    SysUmask,            // 095 sys_umask,
    SysGettimeofday,     // 096 sys_gettimeofday,
    SysGetrlimit,        // 097 sys_getrlimit,
    SysGetrusage,        // 098 sys_getrusage,
    SysInfo,             // 099 sys_sysinfo,
    SysTimes,            // 100 sys_times,
    SysNoSupport,        // 101 sys_ptrace,
    SysGetuid,           // 102 sys_getuid,
    NotImplementSyscall, // 103 sys_syslog,
    SysGetgid,           // 104 sys_getgid,
    SysSetuid,           // 105 sys_setuid,
    SysSetgid,           // 106 sys_setgid,
    SysGeteuid,          // 107 sys_geteuid,
    SysGetegid,          // 108 sys_getegid,
    SysSetpgid,          // 109 sys_setpgid,
    SysGetPpid,          // 110 sys_getppid,
    SysGetpgrp,          // 111 sys_getpgrp,
    SysSetsid,           // 112 sys_setsid,
    SysSetreuid,         // 113 sys_setreuid,
    SysSetregid,         // 114 sys_setregid,
    SysGetgroups,        // 115 sys_getgroups,
    SysSetgroups,        // 116 sys_setgroups,
    SysSetresuid,        // 117 sys_setresuid,
    SysGetresuid,        // 118 sys_getresuid,
    SysSetresgid,        // 119 sys_setresgid,
    SysGetresgid,        // 120 sys_getresgid,  //120
    SysGetpgid,          // 121 sys_getpgid,
    NotImplementSyscall, // 122 sys_setfsuid,
    NotImplementSyscall, // 123 sys_setfsgid,
    SysGetsid,           // 124 sys_getsid,
    SysCapget,           // 125 sys_capget,
    SysCapSet,           // 126 sys_capset,
    SysRtSigpending,     // 127 sys_rt_sigpending,
    SysRtSigtimedwait,   // 128 sys_rt_sigtimedwait,
    SysRtSigqueueinfo,   // 129 sys_rt_sigqueueinfo,
    SysRtSigsuspend,     // 130 sys_rt_sigsuspend,
    SysSigaltstack,      // 131 sys_sigaltstack,
    SysUtime,            // 132 sys_utime,
    SysMknode,           // 133 sys_mknod,
    SysObsolete,         // 134 sys_uselib,
    SysInvalid,          // 135 sys_personality,
    SysNoSys,            // 136 sys_ustat,      Needs filesystem support.
    SysStatfs,           // 137 sys_statfs,
    SysFstatfs,          // 138 sys_fstatfs,
    NotImplementSyscall, // 139 sys_sysfs,
    SysGetpriority,      // 140 sys_getpriority,
    SysSetpriority,      // 141 sys_setpriority,
    SysCapErr,           // 142 sys_sched_setparam,
    SysSchedGetparam,    // 143 sys_sched_getparam	,
    SysSchedSetscheduler,// 144 sys_sched_setscheduler,
    SysSchedGetscheduler, // 145 sys_sched_getscheduler,
    SysSchedGetPriorityMax,// 146 sys_sched_get_priority_max,
    SysSchedGetPriorityMin,// 147 sys_sched_get_priority_min,
    SysNoPermission,     // 148 sys_sched_rr_get_interval,
    SysMlock,            // 149 sys_mlock,
    SysMunlock,          // 150 sys_munlock,
    SysMlockall,         // 151 sys_mlockall,
    SysMunlockall,       // 152 sys_munlockall,
    SysCapErr,           // 153 sys_vhangup,
    SysNoPermission,     // 154 sys_modify_ldt,
    SysNoPermission,     // 155 sys_pivot_root,
    SysNoPermission,     // 156 sys__sysctl,
    SysPrctl,            // 157 sys_prctl,
    SysArchPrctl,        // 158 sys_arch_prctl,
    SysCapErr,           // 159 sys_adjtimex,       CAP_SYS_TIME
    SysSetrlimit,        // 160 sys_setrlimit,
    SysChroot,           // 161 sys_chroot,
    SysSync,             // 162 sys_sync,
    SysCapErr,           // 163 sys_acct,
    SysCapErr,           // 164 sys_settimeofday,
    NotImplementSyscall, // 165 sys_mount,
    NotImplementSyscall, // 166 sys_umount2,
    SysCapErr,           // 167 sys_swapon,
    SysCapErr,           // 168 sys_swapoff,
    SysCapErr,           // 169 sys_reboot,
    SysSethostname,      // 170 sys_sethostname,
    SysSetdomainname,    // 171 sys_setdomainname,
    SysCapErr,           // 172 sys_iopl,
    SysCapErr,           // 173 sys_ioperm,
    SysCapErr,           // 174 sys_create_module,
    SysCapErr,           // 175 sys_init_module,
    SysCapErr,           // 176 sys_delete_module,
    SysNoSys,            // 177 sys_get_kernel_syms, Not supported in Linux > 2.6
    SysNoSys,            // 178 sys_query_module,    Not supported in Linux > 2.6
    SysCapErr,           // 179 sys_quotactl,
    SysNoSys,            // 180 sys_nfsservctl,      Removed after Linux 3.1
    SysNoSys,            // 181 sys_getpmsg,         Not implemented in Linux.
    SysNoSys,            // 182 sys_putpmsg,         Not implemented in Linux.
    SysNoSys,            // 183 sys_afs_syscall,     Not implemented in Linux.
    SysNoSys,            // 184 sys_tuxcall,         Not implemented in Linux.
    SysNoSys,            // 185 sys_security,        Not implemented in Linux.
    SysGetTid,           // 186 sys_gettid,
    SysReadahead,        // 187 sys_readahead,
    SysSetXattr,         // 188 sys_setxattr,
    SysLSetXattr,        // 189 sys_lsetxattr,
    SysFSetXattr,        // 190 sys_fsetxattr,
    SysGetXattr,         // 191 sys_getxattr,
    SysLGetXattr,        // 192 sys_lgetxattr,
    SysFGetXattr,        // 193 sys_fgetxattr,
    SysListXattr,        // 194 sys_listxattr,
    SysLListXattr,       // 195 sys_llistxattr,
    SysFListXattr,       // 196 sys_flistxattr,
    SysRemoveXattr,      // 197 sys_removexattr,
    SysLRemoveXattr,     // 198 sys_lremovexattr,
    SysFRemoveXattr,     // 199 sys_fremovexattr,
    SysTkill,            // 200 sys_tkill,
    SysTime,             // 201 sys_time,
    SysFutex,            // 202 sys_futex,
    SysSchedSetaffinity, // 203 sys_sched_setaffinity,
    SysSchedGetaffinity, // 204 sys_sched_getaffinity,
    SysNoSys,            // 205 sys_set_thread_area,     Expected to return ENOSYS on 64-bit
    SysIoSetup,          // 206 sys_io_setup,
    SysIoDestroy,        // 207 sys_io_destroy,
    SysIoGetevents,      // 208 sys_io_getevents,
    SysIOSubmit,         // 209 sys_io_submit,
    SysIOCancel,         // 210 sys_io_cancel,
    SysNoSys,            // 211 sys_get_thread_area,     Expected to return ENOSYS on 64-bit
    SysCapErr,           // 212 sys_lookup_dcookie,      CAP_SYS_ADMIN
    SysEpollCreate,      // 213 sys_epoll_create,
    SysNoSys,            // 214 sys_epoll_ctl_old,       Deprecated
    SysNoSys,            // 215 sys_epoll_wait_old,      Deprecated
    SysNoSys,            // 216 sys_remap_file_pages,    Deprecated since Linux 3.16.
    SysGetDents64,       // 217 sys_getdents64,
    SysSetTidAddr,       // 218 sys_set_tid_address,
    SysRestartSyscall,   // 219 sys_restart_syscall,
    SysSemtimedop,       // 220 sys_semtimedop,
    SysFadvise64,        // 221 sys_fadvise64,
    SysTimerCreate,      // 222 sys_timer_create,
    SysTimerSettime,     // 223 sys_timer_settime,
    SysTimerGettime,     // 224 sys_timer_gettime,
    SysTimerGetoverrun,  // 225 sys_timer_getoverrun,
    SysTimerDelete,      // 226 sys_timer_delete,
    SysClockSettime,     // 227 sys_clock_settime,
    SysClockGetTime,     // 228 sys_clock_gettime,
    SysClockGetRes,      // 229 sys_clock_getres,
    SysClockNanosleep,   // 230 sys_clock_nanosleep,
    SysExitThreadGroup,  // 231 sys_exit_group,
    SysEpollWait,        // 232 sys_epoll_wait,
    SysEpollCtl,         // 233 sys_epoll_ctl,
    SysTgkill,           // 234 sys_tgkill,
    SysUtimes,           // 235 sys_utimes,
    SysNoSys,            // 236 sys_vserver,             Not implemented by Linux
    SysMbind,            // 237 sys_mbind, just workaround
    SysSetMempolicy,     // 238 sys_set_mempolicy,
    SysGetMempolicy,     // 239 sys_get_mempolicy,
    SysNoSupport,        // 240 sys_mq_open,
    SysNoSupport,        // 241 sys_mq_unlink,
    SysNoSupport,        // 242 sys_mq_timedsend,
    SysNoSupport,        // 243 sys_mq_timedreceive,
    SysNoSupport,        // 244 sys_mq_notify,
    SysNoSupport,        // 245 sys_mq_getsetattr,
    SysCapErr,           // 246 sys_kexec_load,          CAP_SYS_BOOT
    SysWaitid,           // 247 sys_waitid,
    SysNoAccess,         // 248 sys_add_key,              Not available to user.
    SysNoAccess,         // 249 sys_request_key,          Not available to user.
    SysNoAccess,         // 250 sys_keyctl,    //250      Not available to user.
    SysCapErr,           // 251 sys_ioprio_set,           CAP_SYS_ADMIN
    SysCapErr,           // 252 sys_ioprio_get,           CAP_SYS_ADMIN
    SysInotifyInit,      // 253 sys_inotify_init,
    SysInotifyAddWatch,  // 254 sys_inotify_add_watch,
    SysInotifyRmWatch,   // 255 sys_inotify_rm_watch,
    SysCapErr,           // 256 sys_migrate_pages,        CAP_SYS_NICE
    SysOpenAt,           // 257 sys_openat,
    SysMkdirat,          // 258 sys_mkdirat,
    SysMknodeat,         // 259 sys_mknodat,
    SysFchownat,         // 260 sys_fchownat,
    SysFutimesat,        // 261 sys_futimesat,
    SysFstatat,          // 262 sys_newfstatat,
    SysUnlinkat,         // 263 sys_unlinkat,
    SysRenameat,         // 264 sys_renameat,
    SysLinkat,           // 265 sys_linkat,
    SysSymlinkat,        // 266 sys_symlinkat,
    SysReadLinkAt,       // 267 sys_readlinkat,
    SysFchmodat,         // 268 sys_fchmodat,
    SysFaccessat,        // 269 sys_faccessat,
    SysPSelect,          // 270 sys_pselect6,
    SysPpoll,            // 271 sys_ppoll,
    NotImplementSyscall, // 272 sys_unshare,
    SysSetRobustList,    // 273 sys_set_robust_list,
    SysGetRobustList,    // 274 sys_get_robust_list,
    SysSplice,           // 275 sys_splice,
    SysTee,              // 276 sys_tee,
    SysSyncFileRange,    // 277 sys_sync_file_range,
    NotImplementSyscall, // 278 sys_vmsplice,
    SysCapErr,           // 279 sys_move_pages,          CAP_SYS_NICE
    SysUtimensat,        // 280 sys_utimensat,
    SysPwait,            // 281 sys_epoll_pwait,
    SysSignalfd,         // 282 sys_signalfd,
    SysTimerfdCreate,    // 283 sys_timerfd_create,
    SysEventfd,          // 284 sys_eventfd,
    SysFallocate,        // 285 sys_fallocate,
    SysTimerfdSettime,   // 286 sys_timerfd_settime,
    SysTimerfdGettime,   // 287 sys_timerfd_gettime,
    SysAccept4,          // 288 sys_accept4,
    SysSignalfd4,        // 289 sys_signalfd4,
    SysEventfd2,         // 290 sys_eventfd2,
    SysEpollCreate1,     // 291 sys_epoll_create1,
    SysDup3,             // 292 sys_dup3,
    SysPipe2,            // 293 sys_pipe2,
    SysInotifyInit1,     // 294 sys_inotify_init1,
    SysPreadv,           // 295 sys_preadv,
    SysPwritev,          // 296 sys_pwritev,
    SysRtTgsigqueueinfo, // 297 sys_rt_tgsigqueueinfo,
    SysNoDev,            // 298 sys_perf_event_open,     No support for perf counters
    SysRecvMMsg,         // 299 sys_recvmmsg,
    SysNoSys,            //	300 sys_fanotify_init,  //300 Needs CONFIG_FANOTIFY
    SysNoSys,            //	309 sys_fanotify_mark,       Needs CONFIG_FANOTIFY
    SysPrlimit64,        //	308 sys_prlimit64,
    SysOpNotSupport,     //	307 sys_name_to_handle_at,
    SysOpNotSupport,     //	306 sys_open_by_handle_at,
    SysCapErr,           //	305 sys_clock_adjtime,       CAP_SYS_TIME
    SysSyncFs,           //	304 sys_syncfs,
    SysSendMMsg,         //	303 sys_sendmmsg,
    SysOpNotSupport,     //	302 sys_setns,                   Needs filesystem support
    SysGetcpu,           //	301 sys_getcpu,
    SysNoSys,            //	310 sys_process_vm_readv    Need ptrace
    SysNoSys,            //	311 sys_process_vm_writev
    SysCapErr,           //	312 sys_kcmp,                CAP_SYS_PTRACE
    SysCapErr,           //	313 sys_finit_module,        CAP_SYS_MODULE
    SysNoSys,            //	314 sys_sched_setattr,       implement scheduler?
    SysNoSys,            //	315 sys_sched_getattr,       implement scheduler?
    SysNoSupport,        //	316 sys_renameat2,
    NotImplementSyscall, //	317 sys_seccomp,
    SysGetRandom,        //	318 sys_getrandom,
    SysMemfdCreate,      //	319 sys_memfd_create,
    SysCapErr,           //	320 sys_kexec_file_load    CAP_SYS_BOOT
    SysCapErr,           //	321 sys_bpf,                 CAP_SYS_ADMIN
    SysExecveat,         //	322 sys_stub_execveat,
    NotImplementSyscall, //	323 sys_userfaultfd,
    SysMembarrier,       //	324 sys_membarrier,
    SysMlock2,           //	325 mlock2,

    SysNoSys,            //	326 sys_copy_file_range,
    SysPreadv2,          //	327 sys_preadv2,
    SysPWritev2,         //	328 sys_pwritev2,
    NotImplementSyscall, //	329 sys_pkey_mprotect,
    NotImplementSyscall, //	330 sys_pkey_alloc,
    NotImplementSyscall, //	331 sys_pkey_free,
    SysStatx,            //	332 sys_statx,
    NotImplementSyscall, //	333 sys_io_pgetevents
    SysNoSys,            //	334 sys_rseq

    //don't use numbers 334 through 423
    ///////////////////////////////////////////////////////////////////////////////////////
    NotExisting,         //	335
    NotExisting,         //	336
    NotExisting,         //	337
    NotExisting,         //	338
    NotExisting,         //	339
    NotExisting,         //	340
    NotExisting,         //	341
    NotExisting,         //	342
    NotExisting,         //	343
    NotExisting,         //	344
    NotExisting,         //	345
    NotExisting,         //	346
    NotExisting,         //	347
    NotExisting,         //	348
    NotExisting,         //	349
    NotExisting,         //	350
    NotExisting,         //	351
    NotExisting,         //	352
    NotExisting,         //	353
    NotExisting,         //	354
    NotExisting,         //	355
    NotExisting,         //	356
    NotExisting,         //	357
    NotExisting,         //	358
    NotExisting,         //	359
    NotExisting,         //	360
    NotExisting,         //	361
    NotExisting,         //	362
    NotExisting,         //	363
    NotExisting,         //	364
    NotExisting,         //	365
    NotExisting,         //	366
    NotExisting,         //	367
    NotExisting,         //	368
    NotExisting,         //	369
    NotExisting,         //	370
    NotExisting,         //	371
    NotExisting,         //	372
    NotExisting,         //	373
    NotExisting,         //	374
    NotExisting,         //	375
    NotExisting,         //	376
    NotExisting,         //	377
    NotExisting,         //	378
    NotExisting,         //	379
    NotExisting,         //	380
    NotExisting,         //	381
    NotExisting,         //	382
    NotExisting,         //	383
    NotExisting,         //	384
    NotExisting,         //	385
    NotExisting,         //	386
    NotExisting,         //	387
    NotExisting,         //	388
    NotExisting,         //	389
    NotExisting,         //	390
    NotExisting,         //	391
    NotExisting,         //	392
    NotExisting,         //	393
    NotExisting,         //	394
    NotExisting,         //	395
    NotExisting,         //	396
    NotExisting,         //	397
    NotExisting,         //	398
    NotExisting,         //	399
    NotExisting,         //	400
    NotExisting,         //	401
    NotExisting,         //	402
    NotExisting,         //	403
    NotExisting,         //	404
    NotExisting,         //	405
    NotExisting,         //	406
    NotExisting,         //	407
    NotExisting,         //	408
    NotExisting,         //	409
    NotExisting,         //	410
    NotExisting,         //	411
    NotExisting,         //	412
    NotExisting,         //	413
    NotExisting,         //	414
    NotExisting,         //	415
    NotExisting,         //	416
    NotExisting,         //	417
    NotExisting,         //	418
    NotExisting,         //	419
    NotExisting,         //	420
    NotExisting,         //	421
    NotExisting,         //	422
    NotExisting,         //	423
    ////////////////////////////////////////////////////////////////////////////
    //don't use numbers 334 through 423

    // Linux skips ahead to syscall 424 to sync numbers between arches.
    NotImplementSyscall, //	424 sys_pidfd_send_signal
    NotImplementSyscall, //	425 sys_io_uring_setup
    NotImplementSyscall, //	426 sys_io_uring_enter
    NotImplementSyscall, //	427 sys_io_uring_register
    NotImplementSyscall, //	428 sys_open_tree
    NotImplementSyscall, //	429 sys_move_mount
    NotImplementSyscall, //	430 sys_fsopen
    NotImplementSyscall, //	431 sys_fsconfig
    NotImplementSyscall, //	432 sys_fsmount
    NotImplementSyscall, //	433 sys_fspick
    NotImplementSyscall, //	434 sys_pidfd_open
    NotImplementSyscall, //	435 sys_clone3
    NotImplementSyscall, //	436 sys_close_range
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
];

pub fn NotImplementSyscall(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("NotImplementSyscall syscall {:x?}", args);
    return Err(Error::SysCallNotImplement);
}

pub fn NotExisting(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    panic!("NotExisting syscall {:x?}", args);
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