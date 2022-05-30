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
    SysRead,             //sys_read = 0 as u64,
    SysWrite,            //sys_write,
    SysOpen,             //sys_open,
    SysClose,            //sys_close,
    SysStat,             //sys_stat,
    SysFstat,            //sys_fstat,
    SysLstat,            //sys_lstat,
    SysPoll,             //sys_poll,
    SysLseek,            //sys_lseek,
    SysMmap,             //sys_mmap,
    SysMprotect,         //sys_mprotect,   //10
    SysUnmap,            //sys_munmap,
    SysBrk,              //sys_brk,
    SysRtSigaction,      //sys_rt_sigaction,
    SysRtSigProcMask,    //sys_rt_sigprocmask,
    SysRtSigreturn,      //sys_rt_sigreturn,
    SysIoctl,            //sys_ioctl,
    SysPread64,          //sys_pread64,
    SysPwrite64,         //sys_pwrite64,
    SysReadv,            //sys_readv,
    SysWritev,           //sys_writev,    //20
    SysAccess,           //sys_access,
    SysPipe,             //sys_pipe,
    SysSelect,           //sys_select,
    SysScheduleYield,    //sys_sched_yield,
    SysMremap,           //sys_mremap,
    SysMsync,            //sys_msync,
    SysMincore,          //sys_mincore,
    SysMadvise,          //sys_madvise,
    NotImplementSyscall, //sys_shmget,
    NotImplementSyscall, //sys_shmat,   //30
    NotImplementSyscall, //sys_shmctl,
    SysDup,              //sys_dup,
    SysDup2,             //sys_dup2,
    SysPause,            //sys_pause,
    SysNanoSleep,        //sys_nanosleep,
    SysGetitimer,        //sys_getitimer,
    SysAlarm,            //sys_alarm,
    SysSetitimer,        //sys_setitimer,
    SysGetPid,           //sys_getpid,
    SysSendfile,         //sys_sendfile,   //40
    SysSocket,           //sys_socket,
    SysConnect,          //sys_connect,
    SysAccept,           //sys_accept,
    SysSendTo,           //sys_sendto,
    SysRecvFrom,         //sys_recvfrom,
    SysSendMsg,          //sys_sendmsg,
    SysRecvMsg,          //sys_recvmsg,
    SysShutdown,         //sys_shutdown,
    SysBind,             //sys_bind,
    SysListen,           //sys_listen,    //50
    SysGetSockName,      //sys_getsockname,
    SysGetPeerName,      //sys_getpeername,
    SysSocketPair,       //sys_socketpair,
    SysSetSockOpt,       //sys_setsockopt,
    SysGetSockOpt,       //sys_getsockopt,
    SysClone,            //sys_clone,
    SysFork,             //sys_fork,
    SysVfork,            //sys_vfork,
    SysExecve,           //sys_execve,
    SysExit,             //sys_exit,    //60
    SysWait4,            //sys_wait4,
    SysKill,             //sys_kill,
    SysUname,            //sys_uname,
    NotImplementSyscall, //sys_semget,
    NotImplementSyscall, //sys_semop,
    NotImplementSyscall, //sys_semctl,
    NotImplementSyscall, //sys_shmdt,
    NotImplementSyscall, //sys_msgget,
    NotImplementSyscall, //sys_msgsnd,
    NotImplementSyscall, //sys_msgrcv,    //70
    NotImplementSyscall, //sys_msgctl,
    SysFcntl,            //sys_fcntl,
    SysFlock,            //sys_flock,
    SysFsync,            //sys_fsync,
    SysDatasync,         //sys_fdatasync,
    SysTruncate,         //sys_truncate,
    SysFtruncate,        //sys_ftruncate,
    SysGetDents,         //sys_getdents,
    SysGetcwd,           //sys_getcwd,
    SysChdir,            //sys_chdir,    //80
    SysFchdir,           //sys_fchdir,
    SysRename,           //sys_rename,
    SysMkdir,            //sys_mkdir,
    SysRmdir,            //sys_rmdir,
    SysCreate,           //sys_creat,
    SysLink,             //sys_link,
    SysUnlink,           //sys_unlink,
    SysSymlink,          //sys_symlink,
    SysReadLink,         //sys_readlink,
    SysChmod,            //sys_chmod,    //90
    SysFchmod,           //sys_fchmod,
    SysChown,            //sys_chown,
    SysFchown,           //sys_fchown,
    SysLchown,           //sys_lchown,
    SysUmask,            //sys_umask,
    SysGettimeofday,     //sys_gettimeofday,
    SysGetrlimit,        //sys_getrlimit,
    SysGetrusage,        //sys_getrusage,
    SysInfo,             //sys_sysinfo,
    SysTimes,            //sys_times,    //100
    SysNoSupport,        //sys_ptrace,
    SysGetuid,           //sys_getuid,
    NotImplementSyscall, //sys_syslog,
    SysGetgid,           //sys_getgid,
    SysSetuid,           //sys_setuid,
    SysSetgid,           //sys_setgid,
    SysGeteuid,          //sys_geteuid,
    SysGetegid,          //sys_getegid,
    SysSetpgid,          //sys_setpgid,
    SysGetPpid,          //sys_getppid,    //110
    SysGetpgrp,          //sys_getpgrp,
    SysSetsid,           //sys_setsid,
    SysSetreuid,         //sys_setreuid,
    SysSetregid,         //sys_setregid,
    SysGetgroups,        //sys_getgroups,
    SysSetgroups,        //sys_setgroups,
    SysSetresuid,        //sys_setresuid,
    SysGetresuid,        //sys_getresuid,
    SysSetresgid,        //sys_setresgid,
    SysGetresgid,        //sys_getresgid,  //120
    SysGetpgid,          //sys_getpgid,
    NotImplementSyscall, //sys_setfsuid,
    NotImplementSyscall, //sys_setfsgid,
    SysGetsid,           //sys_getsid,
    SysCapget,           //sys_capget,
    SysCapet,            //sys_capset,
    NotImplementSyscall, //sys_rt_sigpending,
    SysRtSigtimedwait,   //sys_rt_sigtimedwait,
    SysRtSigqueueinfo,   //sys_rt_sigqueueinfo,
    SysRtSigsuspend,     //sys_rt_sigsuspend,  //130
    SysSigaltstack,      //sys_sigaltstack,
    SysUtime,            //sys_utime,
    SysMknode,           //sys_mknod,
    NotImplementSyscall, //sys_uselib,
    NotImplementSyscall, //sys_personality,
    NotImplementSyscall, //sys_ustat,
    SysStatfs,           //sys_statfs,
    SysFstatfs,          //sys_fstatfs,
    NotImplementSyscall, //sys_sysfs,
    SysGetpriority,      //sys_getpriority,    //140
    SysSetpriority,      //sys_setpriority,
    SysCapErr,           //sys_sched_setparam,
    SysSchedGetparam,    //sys_sched_getparam	,
    SysSchedSetscheduler,//sys_sched_setscheduler,
    SysSchedGetscheduler, //sys_sched_getscheduler,
    SysSchedGetPriorityMax,//sys_sched_get_priority_max,
    SysSchedGetPriorityMin,//sys_sched_get_priority_min,
    NotImplementSyscall, //sys_sched_rr_get_interval,
    SysMlock,            //sys_mlock,
    SysMunlock,          //sys_munlock,    //150
    SysMlockall,         //sys_mlockall,
    SysMunlockall,       //sys_munlockall,
    NotImplementSyscall, //sys_vhangup,
    NotImplementSyscall, //sys_modify_ldt,
    NotImplementSyscall, //sys_pivot_root,
    NotImplementSyscall, //sys__sysctl,
    SysPrctl,            //sys_prctl,
    SysArchPrctl,        //sys_arch_prctl,
    NotImplementSyscall, //sys_adjtimex,
    SysSetrlimit,        //sys_setrlimit, // 160
    SysChroot,           //sys_chroot,
    SysSync,             //sys_sync,
    NotImplementSyscall, //sys_acct,
    NotImplementSyscall, //sys_settimeofday,
    NotImplementSyscall, //sys_mount,
    NotImplementSyscall, //sys_umount2,
    NotImplementSyscall, //sys_swapon,
    NotImplementSyscall, //sys_swapoff,
    NotImplementSyscall, //sys_reboot,
    SysSethostname,      //sys_sethostname,    //170
    SysSetdomainname,    //sys_setdomainname,
    NotImplementSyscall, //sys_iopl,
    NotImplementSyscall, //sys_ioperm,
    NotImplementSyscall, //sys_create_module,
    NotImplementSyscall, //sys_init_module,
    NotImplementSyscall, //sys_delete_module,
    NotImplementSyscall, //sys_get_kernel_syms,
    NotImplementSyscall, //sys_query_module,
    NotImplementSyscall, //sys_quotactl,
    NotImplementSyscall, //sys_nfsservctl,    //180
    NotImplementSyscall, //sys_getpmsg,
    NotImplementSyscall, //sys_putpmsg,
    NotImplementSyscall, //sys_afs_syscall,
    NotImplementSyscall, //sys_tuxcall,
    NotImplementSyscall, //sys_security,
    SysGetTid,           //sys_gettid,
    NotImplementSyscall, //sys_readahead,
    SysNoSupport,        //sys_setxattr,
    SysNoSupport,        //sys_lsetxattr,
    SysNoSupport,        //sys_fsetxattr,    //190
    SysNoSupport,        //sys_getxattr,
    SysNoSupport,        //sys_lgetxattr,
    SysNoSupport,        //sys_fgetxattr,
    SysNoSupport,        //sys_listxattr,
    SysNoSupport,        //sys_llistxattr,
    SysNoSupport,        //sys_flistxattr,
    SysNoSupport,        //sys_removexattr,
    SysNoSupport,        //sys_lremovexattr,
    SysNoSupport,        //sys_fremovexattr,
    SysTkill,            //sys_tkill,    //200
    SysTime,             //sys_time,
    SysFutex,            //sys_futex,
    SysSchedSetaffinity, //sys_sched_setaffinity,
    SysSchedGetaffinity, //sys_sched_getaffinity,
    NotImplementSyscall, //sys_set_thread_area,
    SysIoSetup,          //sys_io_setup,
    SysIoDestroy,        //sys_io_destroy,
    SysIoGetevents,      //sys_io_getevents,
    SysIOSubmit,         //sys_io_submit,
    SysIOCancel,         //sys_io_cancel,    //210
    NotImplementSyscall, //sys_get_thread_area,
    NotImplementSyscall, //sys_lookup_dcookie,
    SysEpollCreate,      //sys_epoll_create,
    NotImplementSyscall, //sys_epoll_ctl_old,
    NotImplementSyscall, //sys_epoll_wait_old,
    NotImplementSyscall, //sys_remap_file_pages,
    SysGetDents64,       //sys_getdents64,
    SysSetTidAddr,       //sys_set_tid_address,
    SysRestartSyscall,   //sys_restart_syscall,
    NotImplementSyscall, //sys_semtimedop,    //220
    SysFadvise64,        //sys_fadvise64,
    SysTimerCreate,      //sys_timer_create,
    SysTimerSettime,     //sys_timer_settime,
    SysTimerGettime,     //sys_timer_gettime,
    SysTimerGetoverrun,  //sys_timer_getoverrun,
    SysTimerDelete,      //sys_timer_delete,
    SysClockSettime,     //sys_clock_settime,
    SysClockGetTime,     //sys_clock_gettime,
    SysClockGetRes,      //sys_clock_getres,
    SysClockNanosleep,   //sys_clock_nanosleep,//230
    SysExitThreadGroup,  //sys_exit_group,
    SysEpollWait,        //sys_epoll_wait,
    SysEpollCtl,         //sys_epoll_ctl,
    SysTgkill,           //sys_tgkill,
    SysUtimes,           //sys_utimes,
    NotImplementSyscall, //sys_vserver,
    SysMbind,            //sys_mbind, just workaround
    SysSetMempolicy,     //sys_set_mempolicy,
    SysGetMempolicy,     //sys_get_mempolicy,
    SysNoSupport,       //sys_mq_open,    //240
    SysNoSupport,       //sys_mq_unlink,
    SysNoSupport,       //sys_mq_timedsend,
    SysNoSupport,       //sys_mq_timedreceive,
    SysNoSupport,       //sys_mq_notify,
    SysNoSupport,       //sys_mq_getsetattr,
    NotImplementSyscall, //sys_kexec_load,
    SysWaitid,           //sys_waitid,
    NotImplementSyscall, //sys_add_key,
    NotImplementSyscall, //sys_request_key,
    NotImplementSyscall, //sys_keyctl,    //250
    NotImplementSyscall, //sys_ioprio_set,
    NotImplementSyscall, //sys_ioprio_get,
    SysInotifyInit, //sys_inotify_init,
    SysInotifyAddWatch, //sys_inotify_add_watch,
    SysInotifyRmWatch, //sys_inotify_rm_watch,
    NotImplementSyscall, //sys_migrate_pages,
    SysOpenAt,           //sys_openat,
    SysMkdirat,          //sys_mkdirat,
    SysMknodeat,         //sys_mknodat,
    SysFchownat,         //sys_fchownat,    //260
    SysFutimesat,        //sys_futimesat,
    SysFstatat,          //sys_newfstatat,
    SysUnlinkat,         //sys_unlinkat,
    SysRenameat,         //sys_renameat,
    SysLinkat,           //sys_linkat,
    SysSymlinkat,        //sys_symlinkat,
    SysReadLinkAt,       //sys_readlinkat,
    SysFchmodat,         //sys_fchmodat,
    SysFaccessat,        //sys_faccessat,
    SysPSelect,          //sys_pselect6,    //270
    SysPpoll,            //sys_ppoll,
    NotImplementSyscall, //sys_unshare,
    SysSetRobustList,    //sys_set_robust_list,
    SysGetRobustList,    //sys_get_robust_list,
    SysSplice,           //sys_splice,
    SysTee,              //sys_tee,
    SysSyncFileRange,    //sys_sync_file_range,
    NotImplementSyscall, //sys_vmsplice,
    NotImplementSyscall, //sys_move_pages,
    SysUtimensat,        //sys_utimensat,    //280
    SysPwait,            //sys_epoll_pwait,
    SysSignalfd,         //sys_signalfd,
    SysTimerfdCreate,    //sys_timerfd_create,
    SysEventfd,          //sys_eventfd,
    SysFallocate,        //sys_fallocate,
    SysTimerfdSettime,   //sys_timerfd_settime,
    SysTimerfdGettime,   //sys_timerfd_gettime,
    SysAccept4,          //sys_accept4,
    SysSignalfd4,        //sys_signalfd4,
    SysEventfd2,         //sys_eventfd2,    //290
    SysEpollCreate1,     //sys_epoll_create1,
    SysDup3,             //sys_dup3,
    SysPipe2,            //sys_pipe2,
    SysInotifyInit1,     //sys_inotify_init1,
    SysPreadv,           //sys_preadv,
    SysPwritev,          //sys_pwritev,
    SysRtTgsigqueueinfo, //sys_rt_tgsigqueueinfo,
    NotImplementSyscall, //sys_perf_event_open,
    SysRecvMMsg,         //sys_recvmmsg,
    NotImplementSyscall, //sys_fanotify_init,  //300
    NotImplementSyscall, //sys_fanotify_mark,
    SysPrlimit64,        //sys_prlimit64,
    NotImplementSyscall, //sys_name_to_handle_at,
    NotImplementSyscall, //sys_open_by_handle_at,
    NotImplementSyscall, //sys_clock_adjtime,
    SysSyncFs,           //sys_syncfs,
    SysSendMMsg,         //sys_sendmmsg,
    NotImplementSyscall, //sys_setns,
    SysGetcpu,           //sys_getcpu,
    NotImplementSyscall, //sys_process_vm_readv,//310
    NotImplementSyscall, //sys_process_vm_writev,
    NotImplementSyscall, //sys_kcmp,
    NotImplementSyscall, //sys_finit_module,
    NotImplementSyscall, //sys_sched_setattr,
    NotImplementSyscall, //sys_sched_getattr,
    SysNoSupport, //sys_renameat2,
    NotImplementSyscall, //sys_seccomp,
    SysGetRandom,        //sys_getrandom,
    SysMemfdCreate,     //sys_memfd_create,
    NotImplementSyscall, //sys_kexec_file_load,//320
    NotImplementSyscall, //sys_bpf,
    NotImplementSyscall, //sys_stub_execveat,
    NotImplementSyscall, //sys_userfaultfd,
    SysMembarrier,       //sys_membarrier,
    SysMlock2,           //mlock2,
    SysNoSys,            //sys_copy_file_range,
    SysPreadv2,          //sys_preadv2,
    SysPWritev2,         //sys_pwritev2,
    NotImplementSyscall, //sys_pkey_mprotect,
    NotImplementSyscall, //sys_pkey_alloc,//330
    NotImplementSyscall, //sys_pkey_free, 331
    SysStatx,            //sys_statx, 332
    NotImplementSyscall, //	333
    NotImplementSyscall, //	334
    NotImplementSyscall, //	335
    NotImplementSyscall, //	336
    NotImplementSyscall, //	337
    NotImplementSyscall, //	338
    NotImplementSyscall, //	339
    NotImplementSyscall, //	340
    NotImplementSyscall, //	341
    NotImplementSyscall, //	342
    NotImplementSyscall, //	343
    NotImplementSyscall, //	344
    NotImplementSyscall, //	345
    NotImplementSyscall, //	346
    NotImplementSyscall, //	347
    NotImplementSyscall, //	348
    NotImplementSyscall, //	349
    NotImplementSyscall, //	350
    NotImplementSyscall, //	351
    NotImplementSyscall, //	352
    NotImplementSyscall, //	353
    NotImplementSyscall, //	354
    NotImplementSyscall, //	355
    NotImplementSyscall, //	356
    NotImplementSyscall, //	357
    NotImplementSyscall, //	358
    NotImplementSyscall, //	359
    NotImplementSyscall, //	360
    NotImplementSyscall, //	361
    NotImplementSyscall, //	362
    NotImplementSyscall, //	363
    NotImplementSyscall, //	364
    NotImplementSyscall, //	365
    NotImplementSyscall, //	366
    NotImplementSyscall, //	367
    NotImplementSyscall, //	368
    NotImplementSyscall, //	369
    NotImplementSyscall, //	370
    NotImplementSyscall, //	371
    NotImplementSyscall, //	372
    NotImplementSyscall, //	373
    NotImplementSyscall, //	374
    NotImplementSyscall, //	375
    NotImplementSyscall, //	376
    NotImplementSyscall, //	377
    NotImplementSyscall, //	378
    NotImplementSyscall, //	379
    NotImplementSyscall, //	380
    NotImplementSyscall, //	381
    NotImplementSyscall, //	382
    NotImplementSyscall, //	383
    NotImplementSyscall, //	384
    NotImplementSyscall, //	385
    NotImplementSyscall, //	386
    NotImplementSyscall, //	387
    NotImplementSyscall, //	388
    NotImplementSyscall, //	389
    NotImplementSyscall, //	390
    NotImplementSyscall, //	391
    NotImplementSyscall, //	392
    NotImplementSyscall, //	393
    NotImplementSyscall, //	394
    NotImplementSyscall, //	395
    NotImplementSyscall, //	396
    NotImplementSyscall, //	397
    NotImplementSyscall, //	398
    NotImplementSyscall, //	399
    NotImplementSyscall, //	400
    NotImplementSyscall, //	401
    NotImplementSyscall, //	402
    NotImplementSyscall, //	403
    NotImplementSyscall, //	404
    NotImplementSyscall, //	405
    NotImplementSyscall, //	406
    NotImplementSyscall, //	407
    NotImplementSyscall, //	408
    NotImplementSyscall, //	409
    NotImplementSyscall, //	410
    NotImplementSyscall, //	411
    NotImplementSyscall, //	412
    NotImplementSyscall, //	413
    NotImplementSyscall, //	414
    NotImplementSyscall, //	415
    NotImplementSyscall, //	416
    NotImplementSyscall, //	417
    NotImplementSyscall, //	418
    NotImplementSyscall, //	419
    NotImplementSyscall, //	420
    NotImplementSyscall, //	421
    NotImplementSyscall, //	422
    NotImplementSyscall, //	423
    NotImplementSyscall, //	424
    NotImplementSyscall, //	425
    NotImplementSyscall, //	426
    NotImplementSyscall, //	427
    NotImplementSyscall, //	428
    NotImplementSyscall, //	429
    NotImplementSyscall, //	430
    NotImplementSyscall, //	431
    NotImplementSyscall, //	432
    NotImplementSyscall, //	433
    NotImplementSyscall, //	434
    NotImplementSyscall, //	435
    NotImplementSyscall, //	436
    NotImplementSyscall, //	437
    NotImplementSyscall, //	438
    SysNoSys,            //	439
    NotImplementSyscall, //	440
    SysNoSys, //	441
    NotImplementSyscall, //	442
];

pub fn NotImplementSyscall(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("NotImplementSyscall syscall {:x?}", args);
    return Err(Error::SysCallNotImplement);
}

pub fn SysNoSupport(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("SysNoSupport syscall {:x?}", args);
    return Err(Error::SysError(SysErr::ENODATA));
    //return Err(Error::SysError(SysErr::ENOTSUP));
}

pub fn SysObsolete(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("SysObsolete syscall {:x?}", args);
    return Err(Error::SysError(SysErr::ENOSYS));
    //return Err(Error::SysError(SysErr::ENOTSUP));
}

pub fn SysNoSys(_task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    error!("No support syscall {:x?}", args);
    return Err(Error::SysError(SysErr::ENOSYS));
}

pub fn SysCapErr(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::EPERM));
    //return Err(Error::SysError(SysErr::ENOSYS));
}