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

pub mod HostFileMap;
//pub mod TimerMgr;
pub mod hibernate;
pub mod host_pma_keeper;
pub mod host_uring;
pub mod hostfdnotifier;
pub mod kernel_io_thread;
pub mod limits;
pub mod random;
pub mod syscall;
pub mod time;
pub mod uringMgr;
pub mod nvidia;
pub mod tsot_agent;
pub mod xpu;

use core::arch::asm;
use core::sync::atomic;
use core::sync::atomic::AtomicU64;
use lazy_static::lazy_static;
use libc::*;
use serde_json;
use std::env::temp_dir;
use std::fs;
use std::marker::Send;
use std::os::unix::io::IntoRawFd;
use std::slice;
use std::str;
use uuid::Uuid;

use crate::qlib::cstring::CString;
use crate::qlib::fileinfo::*;
use crate::qlib::kernel::socket::control::*;
use crate::qlib::kernel::socket::control::Parse;
use crate::qlib::nvproxy::frontend::FrontendIoctlCmd;
use crate::qlib::nvproxy::frontend_type::NV_ESC_CHECK_VERSION_STR;
use crate::qlib::nvproxy::frontend_type::RMAPIVersion;
use crate::qlib::range::Range;
use crate::qlib::proxy::*;
use crate::vmspace::kernel::GlobalIOMgr;
use crate::vmspace::kernel::GlobalRDMASvcCli;

use self::limits::*;
use self::nvidia::NvidiaProxy;
use self::random::*;
use self::syscall::*;
use self::tsot_agent::TSOT_AGENT;
use self::tsot_msg::TsotMessage;
use super::kvm_vcpu::HostPageAllocator;
use super::kvm_vcpu::KVMVcpu;
use super::namespace::MountNs;
use super::qlib::addr::Addr;
use super::qlib::common::{Error, Result};
use super::qlib::control_msg::*;
use super::qlib::linux::membarrier::*;
use super::qlib::linux_def::*;
use super::qlib::pagetable::PageTables;
use super::qlib::pagetable::PageTableFlags;
use super::qlib::perf_tunning::*;
use super::qlib::qmsg::*;
use super::qlib::socket_buf::*;
use super::qlib::task_mgr::*;
use super::qlib::*;
use super::runc::runtime::loader::*;
use super::runc::runtime::signal_handle::*;
use super::runc::specutils::specutils::*;
use super::ucall::usocket::*;
use super::*;




const ARCH_SET_GS: u64 = 0x1001;
const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_GET_GS: u64 = 0x1004;

lazy_static! {
    static ref UID: AtomicU64 = AtomicU64::new(1);
}

macro_rules! scan {
    ( $string:expr, $sep:expr, $( $x:ty ),+ ) => {{
        let mut iter = $string.split($sep);
        ($(iter.next().and_then(|word| word.parse::<$x>().ok()),)*)
    }}
}

pub fn NewUID() -> u64 {
    return UID.fetch_add(1, atomic::Ordering::SeqCst);
}

pub fn Init() {
    //self::fs::Init();
}

#[derive(Clone, Copy, Debug)]
pub struct WaitingMsgCall {
    pub taskId: TaskId,
    pub addr: u64,
    pub len: usize,
    pub retAddr: u64,
}

pub struct VMSpace {
    pub podUid: String,
    pub pageTables: PageTables,
    pub allocator: HostPageAllocator,
    pub vdsoAddrGpa: u64,
    pub vcpuCount: usize,
    pub cpuAffinit: bool,
    pub vcpuMappingDelta: usize,

    pub rng: RandGen,
    pub args: Option<Args>,
    pub pivot: bool,
    pub waitingMsgCall: Option<WaitingMsgCall>,
    pub controlSock: i32,
    pub vcpus: Vec<Arc<KVMVcpu>>,
    pub haveMembarrierGlobal: bool,
    pub haveMembarrierPrivateExpedited: bool,

    pub rdmaSvcCliSock: i32,
    pub podId: [u8;64],
    pub kvmfd: i32,
    pub vmfd: i32,
}


unsafe impl Sync for VMSpace {}
unsafe impl Send for VMSpace {}

impl VMSpace {
    pub fn Init() -> Self {
        let (haveMembarrierGlobal, haveMembarrierPrivateExpedited) = Self::MembarrierInit();
        
        return VMSpace {
            podUid: "".to_owned(),
            allocator: HostPageAllocator::New(),
            pageTables: PageTables::default(),
            vdsoAddrGpa: 0,
            cpuAffinit: false,
            vcpuCount: 0,
            vcpuMappingDelta: 0,
            rng: RandGen::Init(),
            args: None,
            pivot: false,
            waitingMsgCall: None,
            controlSock: -1,
            vcpus: Vec::new(),
            haveMembarrierGlobal: haveMembarrierGlobal,
            haveMembarrierPrivateExpedited: haveMembarrierPrivateExpedited,
            rdmaSvcCliSock: 0,
            podId: [0u8;64],
            kvmfd: 0,
            vmfd: 0,
        };
    }

    ///////////start of file operation//////////////////////////////////////////////
    pub fn GetOsfd(hostfd: i32) -> Option<i32> {
        return GlobalIOMgr().GetFdByHost(hostfd);
    }

    pub fn GetFdInfo(hostfd: i32) -> Option<FdInfo> {
        return GlobalIOMgr().GetByHost(hostfd);
    }

    pub fn ReadDir(dirfd: i32, addr: u64, len: usize, reset: bool) -> i64 {
        let fdInfo = match Self::GetFdInfo(dirfd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOReadDir(addr, len, reset);
    }


    pub fn PivotRoot(&self, rootfs: &str) {
                let mns = MountNs::New(rootfs.to_string());
        mns.PivotRoot();
    }

    pub fn WriteControlMsgResp(fd: i32, addr: u64, len: usize, close: bool) -> i64 {
        let buf = {
            let ptr = addr as *const u8;
            unsafe { slice::from_raw_parts(ptr, len) }
        };

        let resp: UCallResp = serde_json::from_slice(&buf[0..len]).expect("ControlMsgRet des fail");

        let usock = USocket { socket: fd };

        match usock.SendResp(&resp) {
            Err(e) => error!("ControlMsgRet send resp fail with error {:?}", e),
            Ok(()) => (),
        }

        if close {
            usock.Drop();
        }

        return 0;
    }

    pub fn VCPUCount() -> usize {
        let mut cpuCount = num_cpus::get();

        if cpuCount < 2 {
            cpuCount = 2; // at least 2 vcpu (one for host io and the other for process vcpu)
        }

        if cpuCount > MAX_VCPU_COUNT {
            cpuCount = MAX_VCPU_COUNT;
        }

        return cpuCount;
    }

    pub fn LoadProcessKernel(&mut self, processAddr: u64) -> i64 {
        info!("LoadProcessKernel processAddr:0x{:x}",processAddr);
        unsafe{core::ptr::write(processAddr as *mut loader::Process, loader::Process::default());};
        let process = unsafe { &mut *(processAddr as *mut loader::Process) };
        process.ID = self.args.as_ref().unwrap().ID.to_string();
        let spec = &mut self.args.as_mut().unwrap().Spec;

        let mut cwd = spec.process.cwd.to_string();
        if cwd.len() == 0 {
            cwd = "/".to_string();
        }
        process.Cwd = cwd;
        SetConole(spec.process.terminal);
        process.Terminal = spec.process.terminal;
        process.Args.append(&mut spec.process.args);
        process.Envs.append(&mut spec.process.env);

        //todo: credential fix.
        //error!("LoadProcessKernel: need to study the user mapping handling...");
        process.UID = spec.process.user.uid;
        process.GID = spec.process.user.gid;
        process
            .AdditionalGids
            .append(&mut spec.process.user.additional_gids);
        process.limitSet = CreateLimitSet(&spec)
            .expect("load limitSet fail")
            .GetInternalCopy();
        process.Caps = Capabilities(false, &spec.process.capabilities);

        process.HostName = spec.hostname.to_string();

        process.NumCpu = self.vcpuCount as u32;
        process.ExecId = Some("".to_string());
        for i in 0..process.Stdiofds.len() {
            let osfd = unsafe { dup(i as i32) as i32 };

            URING_MGR.lock().Addfd(osfd).unwrap();

            if osfd < 0 {
                return osfd as i64;
            }

            let hostfd = GlobalIOMgr().AddFile(osfd);

            process.Stdiofds[i] = hostfd;
        }
        process.Root = format!("/{}", &process.ID);
        //process.Root = "/".to_string();

        let rootfs = self.args.as_ref().unwrap().Rootfs.to_string();

        if self.pivot {
            self.PivotRoot(&rootfs);
        }
        StartSignalHandle();

        return 0;
    }

    pub fn TgKill(tgid: i32, tid: i32, signal: i32) -> i64 {
        let nr = SysCallID::sys_tgkill as usize;
        let ret = unsafe { syscall3(nr, tgid as usize, tid as usize, signal as usize) as i32 };
        return ret as _;
    }

    pub fn CreateMemfd(len: i64, flags: u32) -> i64 {
        let uid = NewUID();
        let path = format!("/tmp/memfd_{}", uid);
        let cstr = CString::New(&path);

        let nr = SysCallID::sys_memfd_create as usize;
        let fd =
            unsafe { syscall2(nr, cstr.Ptr() as *const c_char as usize, flags as usize) as i32 };

        if fd < 0 {
            return Self::GetRet(fd as i64);
        }

        let ret = unsafe { ftruncate(fd, len) };

        if ret < 0 {
            unsafe {
                libc::close(fd);
            }
            return Self::GetRet(ret as i64);
        }

        let hostfd = GlobalIOMgr().AddFile(fd);
        return hostfd as i64;
    }

    pub fn Fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { fallocate(fd, mode, offset, len) };

        return Self::GetRet(ret as i64);
    }

    pub fn RenameAt(olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        let olddirfd = {
            if olddirfd > 0 {
                match Self::GetOsfd(olddirfd) {
                    Some(olddirfd) => olddirfd,
                    None => return -SysErr::EBADF as i64,
                }
            } else {
                olddirfd
            }
        };

        let newdirfd = {
            if newdirfd > 0 {
                match Self::GetOsfd(newdirfd) {
                    Some(newdirfd) => newdirfd,
                    None => return -SysErr::EBADF as i64,
                }
            } else {
                newdirfd
            }
        };

        let ret = unsafe {
            renameat(
                olddirfd,
                oldpath as *const c_char,
                newdirfd,
                newpath as *const c_char,
            )
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Ftruncate(fd: i32, len: i64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { ftruncate64(fd, len) };

        return Self::GetRet(ret as i64);
    }

    pub fn GetStr(string: u64) -> &'static str {
        let ptr = string as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, 1024) };

        let len = {
            let mut res: usize = 0;
            for i in 0..1024 {
                if slice[i] == 0 {
                    res = i;
                    break;
                }
            }

            res
        };

        return str::from_utf8(&slice[0..len]).unwrap();
    }

    pub fn GetStrWithLen(string: u64, len: u64) -> &'static str {
        let ptr = string as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, len as usize) };

        return str::from_utf8(&slice[0..len as usize]).unwrap();
    }

    pub fn GetStrLen(string: u64) -> i64 {
        let ptr = string as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, 1024) };

        let len = {
            let mut res: usize = 0;
            for i in 0..1024 {
                if slice[i] == 0 {
                    res = i;
                    break;
                }
            }

            res
        };

        return (len + 1) as i64;
    }

    pub fn TryOpenWrite(dirfd: i32, oldfd: i32, name: u64) -> i64 {
        let flags = Flags::O_NOFOLLOW;

        let fd = unsafe {
            libc::openat(
                dirfd,
                name as *const c_char,
                (flags | Flags::O_RDWR) as i32,
                0,
            )
        };

        if fd < 0 {
            return fd as i64;
        }

        let ret = unsafe {
            libc::dup2(fd, oldfd)
        };

        if ret < 0 {
            error!("TryOpenWrite can't dup new fd to old fd with error {}", errno::errno().0);
            unsafe {
                libc::close(fd);
            }
            return ret as i64;
        }

        unsafe {
            libc::close(fd);
        }

        return 0;
    }

    pub unsafe fn TryOpenHelper(dirfd: i32, name: u64, skiprw: bool) -> (i32, bool) {
        let flags = Flags::O_NOFOLLOW;

        if !skiprw {
            let ret = libc::openat(
                dirfd,
                name as *const c_char,
                (flags | Flags::O_RDWR) as i32,
                0,
            );
            if ret > 0 {
                return (ret, true);
            }

            let err = Self::GetRet(ret as i64) as i32;
            if err == -SysErr::ENOENT {
                return (-SysErr::ENOENT, false);
            }
        }

        let ret = libc::openat(
            dirfd,
            name as *const c_char,
            (flags | Flags::O_RDONLY) as i32,
            0,
        );
        if ret > 0 {
            return (ret, false);
        }

        if skiprw {
            let err = Self::GetRet(ret as i64) as i32;
            if err == -SysErr::ENOENT {
                return (-SysErr::ENOENT, false);
            }
        }

        let ret = libc::openat(
            dirfd,
            name as *const c_char,
            (flags | Flags::O_WRONLY) as i32,
            0,
        );
        if ret > 0 {
            return (ret, true);
        }

        let ret = libc::openat(
            dirfd,
            name as *const c_char,
            flags as i32 | Flags::O_PATH,
            0,
        );
        if ret > 0 {
            return (ret, false);
        }

        return (Self::GetRet(ret as i64) as i32, false);
    }

    pub fn TryOpenAt(dirfd: i32, name: u64, addr: u64, skiprw: bool) -> i64 {
        //info!("TryOpenAt1: the filename is {}", Self::GetStr(name));
        let dirfd = if dirfd < 0 {
            dirfd
        } else {
            match Self::GetOsfd(dirfd) {
                Some(fd) => fd,
                None => return -SysErr::EBADF as i64,
            }
        };

        let tryOpenAt = unsafe { &mut *(addr as *mut TryOpenStruct) };
        let ret =
            unsafe { 
                libc::fstatat(
                    dirfd, 
                    name as *const c_char,
                    tryOpenAt.fstat as *const _ as u64 as *mut stat,
                    libc::AT_SYMLINK_NOFOLLOW

                ) as i64 
            };

        if ret < 0 {
            return Self::GetRet(ret as i64);
        }
        
        let (fd, writeable) = unsafe { 
            Self::TryOpenHelper(
                dirfd, 
                name, 
                skiprw && tryOpenAt.fstat.IsRegularFile()
            ) 
        };
        
        //error!("TryOpenAt dirfd {}, name {} ret {}", dirfd, Self::GetStr(name), fd);

        if fd < 0 {
            return fd as i64;
        }

        tryOpenAt.writeable = writeable;
        let hostfd = GlobalIOMgr().AddFile(fd);

        if tryOpenAt.fstat.IsRegularFile() {
            URING_MGR.lock().Addfd(hostfd).unwrap();
        }

        return hostfd as i64;
    }

    pub fn OpenAt(dirfd: i32, name: u64, flags: i32, addr: u64) -> i64 {
        let tryOpenAt = unsafe { &mut *(addr as *mut TryOpenStruct) };

        let ret = unsafe { libc::openat(dirfd, name as *const c_char, flags, 0) };

        let fd = Self::GetRet(ret as i64) as i32;
        if fd < 0 {
            return fd as i64;
        }

        let ret =
            unsafe { libc::fstat(fd, tryOpenAt.fstat as *const _ as u64 as *mut stat) as i64 };

        if ret < 0 {
            unsafe {
                libc::close(fd);
            }
        }

        let hostfd = GlobalIOMgr().AddFile(fd);

        if tryOpenAt.fstat.IsRegularFile() {
            URING_MGR.lock().Addfd(hostfd).unwrap();
        }

        return Self::GetRet(fd as i64);
    }

    pub fn OpenDevFile(dirfd: i32, name: u64, flags: i32) -> i64 {
        let ret = unsafe { libc::openat(dirfd, name as *const c_char, flags, 0) };
        let fd = Self::GetRet(ret as i64) as i32;
        if fd < 0 {
            return fd as i64;
        }

        let hostfd = GlobalIOMgr().AddFile(fd);
        return Self::GetRet(hostfd as i64);
    }

    pub fn NividiaDriverVersion(ioctlParamsAddr: u64) -> i64 {
        let ioctlParams = unsafe {
            &mut *(ioctlParamsAddr as * mut RMAPIVersion) 
        };

        let drvName = CString::New("/dev/nvidiactl");

        let ret = unsafe { 
            libc::openat(
                -1, 
                drvName.Ptr() as *const c_char, 
                O_RDONLY | O_NOFOLLOW, 
                0
            ) 
        };
        let fd = Self::GetRet(ret as i64) as i32;
        if fd < 0 {
            return fd as i64;
        }

        // From src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h:
	    const NV_RM_API_VERSION_REPLY_RECOGNIZED : u32 = 1;
        ioctlParams.cmd = '2' as _;

        let req = FrontendIoctlCmd(NV_ESC_CHECK_VERSION_STR, core::mem::size_of::<RMAPIVersion>() as _);

        let ret = unsafe {
            ioctl(fd, req, ioctlParamsAddr)
        };

        unsafe {
            close(fd);
        }

        return Self::GetRet(ret as i64);
    }

    pub fn NvidiaMMap(addr: u64, len: u64, prot: i32, flags: i32, fd: i32, offset: u64) -> i64 {
        let ret = unsafe {
            libc::mmap(addr as _, len as usize, prot, flags, fd, offset as i64) as i64
        };

        let ret: i64 = Self::GetRet(ret);
        return ret;
    }

    pub fn RemapGuestMemRanges(len: u64, addr: u64, count: usize) -> i64 {
        let ptr = addr as *const Range;
        let ranges = unsafe { slice::from_raw_parts(ptr, count) };
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
        let ret = unsafe {
            libc::mmap(
                0 as _,
                len as usize,
                libc::PROT_NONE,
                flags,
                -1,
                0,
            ) as i64
        };

        if ret < 0 {
            return -errno::errno().0 as i64;
        }

        let mut addr = ret as u64;
        let flags = libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED;// | libc::MREMAP_DONTUNMAP;
        for r in ranges {
            let ret = unsafe {
                libc::mremap(
                    r.Start() as _,
                    0, 
                    r.len as usize, 
                    flags,
                    addr
                ) as i32
            } as i64;

            if ret == -1 {
                return -errno::errno().0 as i64;
            }

            addr += r.len;
        }

        return ret;
    }

    pub fn UnmapGuestMemRange(start: u64, len: u64) -> i64 {
        let ret = unsafe {
            libc::munmap(start as _, len as usize) as i64
        };

        return Self::GetRet(ret as i64);
    }

    pub fn CreateAt(
        dirfd: i32,
        fileName: u64,
        flags: i32,
        mode: i32,
        uid: u32,
        gid: u32,
        fstatAddr: u64,
    ) -> i32 {
        info!("CreateAt: the filename is {}, flag is {:x}, the mode is {:b}, owenr is {}:{}, dirfd is {}",
            Self::GetStr(fileName), flags, mode, uid, gid, dirfd);

        let dirfd = if dirfd < 0 {
            dirfd
        } else {
            match Self::GetOsfd(dirfd) {
                Some(fd) => fd,
                None => return -SysErr::EBADF as i32,
            }
        };

        unsafe {
            let osfd = libc::openat(
                dirfd,
                fileName as *const c_char,
                flags as c_int,
                mode as c_int,
            );
            if osfd <= 0 {
                return Self::GetRet(osfd as i64) as i32;
            }

            let ret = libc::fchown(osfd, uid, gid);
            if ret < 0 {
                libc::close(osfd);
                return Self::GetRet(ret as i64) as i32;
            }

            let ret = libc::fstat(osfd, fstatAddr as *mut stat) as i64;

            if ret < 0 {
                libc::close(osfd);
                return Self::GetRet(ret as i64) as i32;
            }

            let hostfd = GlobalIOMgr().AddFile(osfd);

            URING_MGR.lock().Addfd(osfd).unwrap();

            return hostfd;
        }
    }

    pub fn Close(fd: i32) -> i64 {
        let info = GlobalIOMgr().RemoveFd(fd);

        URING_MGR.lock().Removefd(fd).unwrap();
        let res = if info.is_some() {
            let fdInfo = info.unwrap();
            let fdInfoLock = fdInfo.lock();
            let sockInfo = fdInfoLock.sockInfo.lock().clone();
            match sockInfo {
                SockInfo::RDMADataSocket(dataSock) => {
                    // error!("VMSpace::Close, dataSock fd: {}", fd);
                    GlobalRDMASvcCli()
                        .channelToSocketMappings
                        .lock()
                        .remove(&dataSock.channelId);
                    GlobalRDMASvcCli()
                        .rdmaIdToSocketMappings
                        .lock()
                        .remove(&dataSock.rdmaId);
                    GlobalRDMASvcCli()
                        .tcpPortAllocator
                        .lock()
                        .Free(dataSock.localPort as u64);
                    let _res = GlobalRDMASvcCli().close(dataSock.channelId);
                }
                SockInfo::RDMAServerSocket(serverSock) => {
                    GlobalRDMASvcCli()
                        .rdmaIdToSocketMappings
                        .lock()
                        .remove(&serverSock.rdmaId);
                    //TODO: handle server close
                }
                SockInfo::RDMAUDPSocket(sock) => {
                    GlobalRDMASvcCli()
                        .portToFdInfoMappings
                        .lock()
                        .remove(&sock.port);
                    GlobalRDMASvcCli()
                        .udpPortAllocator
                        .lock()
                        .Free(sock.port as u64);
                }
                _ => {}
            }
            0
        } else {
            -SysErr::EINVAL as i64
        };

        return res;
    }

    pub fn IORead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { readv(fd as c_int, iovs as *const iovec, iovcnt) as i64 };

        return Self::GetRet(ret as i64);
    }

    pub fn IOTTYRead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            let opt: i32 = 1;
            // in some cases, tty read will blocked even after set unblock with fcntl
            // todo: this workaround, fix this
            ioctl(fd, FIONBIO, &opt);

            readv(fd as c_int, iovs as *const iovec, iovcnt) as i64
        };

        unsafe {
            let opt: i32 = 0;
            ioctl(fd, FIONBIO, &opt);
        }

        return Self::GetRet(ret as i64);
    }

    pub fn IOBufWrite(fd: i32, addr: u64, len: usize, offset: isize) -> i64 {
        PerfGoto(PerfType::BufWrite);
        defer!(PerfGofrom(PerfType::BufWrite));

        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOBufWrite(addr, len, offset);
    }

    pub fn IOWrite(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOWrite(iovs, iovcnt);
    }

    pub fn UpdateWaitInfo(fd: i32, waitInfo: FdWaitInfo) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        fdInfo.UpdateWaitInfo(waitInfo);
        return 0;
    }

    pub fn IOAppend(fd: i32, iovs: u64, iovcnt: i32, fileLenAddr: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOAppend(iovs, iovcnt, fileLenAddr);
    }

    pub fn IOReadAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOReadAt(iovs, iovcnt, offset);
    }

    pub fn IOWriteAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOWriteAt(iovs, iovcnt, offset);
    }

    pub fn IOAccept(fd: i32, addr: u64, addrlen: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOAccept(addr, addrlen);
    }

    pub fn NewSocket(fd: i32) -> i64 {
        GlobalIOMgr().AddSocket(fd);
        URING_MGR.lock().Addfd(fd).unwrap();
        return 0;
    }

    pub fn IOConnect(fd: i32, addr: u64, addrlen: u32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOConnect(addr, addrlen);
    }

    pub fn IORecvMsg(fd: i32, msghdr: u64, flags: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IORecvMsg(msghdr, flags);
    }

    pub fn IORecvfrom(fd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IORecvfrom(buf, size, flags, addr, len);
    }

    pub fn IOSendMsg(fd: i32, msghdr: u64, flags: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOSendMsg(msghdr, flags);
    }

    pub fn IOSendto(fd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOSendto(buf, size, flags, addr, len);
    }

    pub fn Fcntl(fd: i32, cmd: i32, arg: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOFcntl(cmd, arg);
    }

    pub fn IoCtl(fd: i32, cmd: u64, argp: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOIoCtl(cmd, argp);
    }

    pub fn SysSync() -> i64 {
        // as quark running inside container, assume sys_sync only works for the current fs namespace
        // todo: confirm this
        unsafe { libc::sync() };

        return 0;
    }

    pub fn SyncFs(fd: i32) -> i64 {
        let osfd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { libc::syncfs(osfd) as i64 };

        return Self::GetRet(ret);
    }

    pub fn SyncFileRange(fd: i32, offset: i64, nbytes: i64, flags: u32) -> i64 {
        let osfd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { libc::sync_file_range(osfd, offset, nbytes, flags) as i64 };

        return Self::GetRet(ret);
    }

    pub fn FSync(fd: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOFSync(false);
    }

    pub fn FDataSync(fd: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOFSync(true);
    }

    pub fn Seek(fd: i32, offset: i64, whence: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOSeek(offset, whence);
    }

    pub fn FSetXattr(fd: i32, name: u64, value: u64, size: usize, flags: u32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOFSetXattr(name, value, size, flags);
    }

    pub fn FGetXattr(fd: i32, name: u64, value: u64, size: usize) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOFGetXattr(name, value, size);
    }

    pub fn FRemoveXattr(fd: i32, name: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOFRemoveXattr(name);
    }

    pub fn FListXattr(fd: i32, list: u64, size: usize) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOFListXattr(list, size);
    }

    pub fn ReadLinkAt(dirfd: i32, path: u64, buf: u64, bufsize: u64) -> i64 {
        //info!("ReadLinkAt: the path is {}", Self::GetStr(path));

        let dirfd = {
            if dirfd == -100 {
                dirfd
            } else {
                match Self::GetOsfd(dirfd) {
                    Some(dirfd) => dirfd,
                    None => return -SysErr::EBADF as i64,
                }
            }
        };

        let res = unsafe {
            readlinkat(
                dirfd,
                path as *const c_char,
                buf as *mut c_char,
                bufsize as usize,
            )
        };
        return Self::GetRet(res as i64);
    }

    pub fn Fstat(fd: i32, buf: u64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { libc::fstat(fd, buf as *mut stat) as i64 };

        return Self::GetRet(ret);
    }

    pub fn Getxattr(path: u64, name: u64, value: u64, size: u64) -> i64 {
        info!(
            "Getxattr: the path is {}, name is {}",
            Self::GetStr(path),
            Self::GetStr(name)
        );
        let ret = unsafe {
            getxattr(
                path as *const c_char,
                name as *const c_char,
                value as *mut c_void,
                size as usize,
            ) as i64
        };

        return Self::GetRet(ret);
    }

    pub fn Lgetxattr(path: u64, name: u64, value: u64, size: u64) -> i64 {
        info!(
            "Lgetxattr: the path is {}, name is {}",
            Self::GetStr(path),
            Self::GetStr(name)
        );
        let ret = unsafe {
            lgetxattr(
                path as *const c_char,
                name as *const c_char,
                value as *mut c_void,
                size as usize,
            ) as i64
        };

        return Self::GetRet(ret);
    }

    pub fn Fgetxattr(fd: i32, name: u64, value: u64, size: u64) -> i64 {
        let fd = Self::GetOsfd(fd).expect("fgetxattr");
        let ret = unsafe {
            fgetxattr(
                fd,
                name as *const c_char,
                value as *mut c_void,
                size as usize,
            ) as i64
        };

        return Self::GetRet(ret);
    }

    pub fn GetRet(ret: i64) -> i64 {
        if ret == -1 {
            //info!("get error, errno is {}", errno::errno().0);
            return -errno::errno().0 as i64;
        }

        return ret;
    }

    pub fn Fstatat(dirfd: i32, pathname: u64, buf: u64, flags: i32) -> i64 {
        let dirfd = {
            if dirfd > 0 {
                Self::GetOsfd(dirfd).expect("Fstatat")
            } else {
                dirfd
            }
        };

        return unsafe {
            Self::GetRet(
                libc::fstatat(dirfd, pathname as *const c_char, buf as *mut stat, flags) as i64,
            )
        };
    }

    pub fn Fstatfs(fd: i32, buf: u64) -> i64 {
        let fd = Self::GetOsfd(fd).expect("Fstatfs");

        let ret = unsafe { fstatfs(fd, buf as *mut statfs) };

        return Self::GetRet(ret as i64);
    }

    pub fn Unlinkat(dirfd: i32, pathname: u64, flags: i32) -> i64 {
        info!("Unlinkat: the pathname is {}", Self::GetStr(pathname));
        let dirfd = {
            if dirfd > 0 {
                match Self::GetOsfd(dirfd) {
                    Some(dirfd) => dirfd,
                    None => return -SysErr::EBADF as i64,
                }
            } else {
                dirfd
            }
        };

        let ret = unsafe { unlinkat(dirfd, pathname as *const c_char, flags) };

        return Self::GetRet(ret as i64);
    }

    pub fn Mkfifoat(dirfd: i32, name: u64, mode: u32, uid: u32, gid: u32) -> i64 {
        info!("Mkfifoat: the pathname is {}", Self::GetStr(name));
        let dirfd = {
            if dirfd > 0 {
                match Self::GetOsfd(dirfd) {
                    Some(dirfd) => dirfd,
                    None => return -SysErr::EBADF as i64,
                }
            } else {
                dirfd
            }
        };

        let ret = unsafe { mkfifoat(dirfd, name as *const c_char, mode as mode_t) };

        Self::ChDirOwnerat(dirfd, name, uid, gid);

        return Self::GetRet(ret as i64);
    }

    pub fn Mkdirat(dirfd: i32, pathname: u64, mode_: u32, uid: u32, gid: u32) -> i64 {
        info!("Mkdirat: the pathname is {}", Self::GetStr(pathname));

        let dirfd = {
            if dirfd > 0 {
                match Self::GetOsfd(dirfd) {
                    Some(dirfd) => dirfd,
                    None => return -SysErr::EBADF as i64,
                }
            } else {
                dirfd
            }
        };

        let ret = unsafe { mkdirat(dirfd, pathname as *const c_char, mode_ as mode_t) };

        Self::ChDirOwnerat(dirfd, pathname, uid, gid);

        return Self::GetRet(ret as i64);
    }

    pub fn ChDirOwnerat(dirfd: i32, pathname: u64, uid: u32, gid: u32) {
        unsafe {
            let ret = libc::fchownat(dirfd, pathname as *const c_char, uid, gid, 0);
            if ret < 0 {
                panic!("fchownat fail with error {}", Self::GetRet(ret as i64))
            }
        }
    }

    pub fn MSync(addr: u64, len: usize, flags: i32) -> i64 {
        let ret = unsafe { msync(addr as *mut c_void, len, flags) };

        return Self::GetRet(ret as i64);
    }

    pub fn MAdvise(addr: u64, len: usize, advise: i32) -> i64 {
        let ret = unsafe { madvise(addr as *mut c_void, len, advise) };

        return Self::GetRet(ret as i64);
    }

    pub fn FAccessAt(dirfd: i32, pathname: u64, mode: i32, flags: i32) -> i64 {
        info!("FAccessAt: the pathName is {}", Self::GetStr(pathname));
        let dirfd = {
            if dirfd == -100 {
                dirfd
            } else {
                match Self::GetOsfd(dirfd) {
                    Some(dirfd) => dirfd,
                    None => return -SysErr::EBADF as i64,
                }
            }
        };

        let ret = unsafe { faccessat(dirfd, pathname as *const c_char, mode, flags) };

        return Self::GetRet(ret as i64);
    }

    ///////////end of file operation//////////////////////////////////////////////

    ///////////start of network operation//////////////////////////////////////////////////////////////////

    pub fn HostUnixRecvMsg(fd: i32, msghdr: u64, flags: i32) -> i64 {
        match Self::HostUnixRecvMsgHelper(fd, msghdr, flags) {
            Err(Error::SysError(errno)) => {
                return -errno as i64
            }
            Ok(()) => return 0,
            _ => panic!("HostUnixRecvMsg impossible"),
        }
    }

    pub fn HostUnixRecvMsgHelper(fd: i32, msghdr: u64, flags: i32) -> Result<()> {
        let ret = unsafe {
            libc::recvmsg(fd, msghdr as * mut _, flags)
        };

        if ret < 0 {
            return Err(Error::SysError(Self::GetRet(ret as i64) as i32));
        }

        let hdr = unsafe {
            &mut *(msghdr as * mut MsgHdr)
        };

        if hdr.msgControlLen > 0 {
            let controlVec = unsafe {
                slice::from_raw_parts_mut(hdr.msgControl as *mut u8, hdr.msgControlLen)
            };

            let ctrlMsg = Parse(controlVec)?;
            let mut fds = Vec::new();
            match &ctrlMsg.Rights {
                Some(right) => {
                    for fd in &right.0 {
                        let fd = *fd;
                        let mut stat = LibcStat::default();
                        unsafe { 
                            libc::fstat(
                                fd, 
                                &mut stat as * mut _ as u64 as *mut _
                            ) as i64 
                        };
    
                        if true || stat.IsRegularFile() {
                            let hostfd = GlobalIOMgr().AddFile(fd);
                            URING_MGR.lock().Addfd(hostfd).unwrap();
                            fds.push(hostfd);
                        } else {
                            error!("HostUnixRecvMsg get unsupport fd with state {:x?}", &stat);
                        }
                    }
                }
                None => ()
            }

            let totalLen = controlVec.len();
            let controlData = &mut controlVec[..];
            let (controlData, _) = ControlMessageRights(fds).EncodeInto(controlData, hdr.msgFlags);

            let new_size = totalLen - controlData.len();
            hdr.msgControlLen = new_size;
        }

        return Ok(())
    }

    pub fn HostUnixConnect(type_: i32, addr: u64, len: usize) -> i64 {
        let blockedType = type_ & (!SocketFlags::SOCK_NONBLOCK);

        let fd = unsafe {
            libc::socket(
                AFType::AF_UNIX,
                blockedType | SocketFlags::SOCK_CLOEXEC,
                0
            )
        };

        if fd < 0 {
            return Self::GetRet(fd as i64);
        }

        let mut socketAddr = libc::sockaddr_un {
            sun_family: libc::AF_UNIX as u16,
            sun_path: [0; 108],
        };

        let slice = unsafe {
            slice::from_raw_parts_mut(addr as *mut i8, len)
        };

        for i in 0..slice.len() {
            socketAddr.sun_path[i] = slice[i]
        };

        let ret = unsafe {
            libc::connect(fd, &socketAddr as * const _ as u64 as * const _, 108 + 2)
        };

        if ret < 0 {
            unsafe {
                libc::close(fd);
            }
            
            return Self::GetRet(ret as i64);
        }

        unsafe {
            let flags = fcntl(fd, Cmd::F_GETFL, 0);
            let ret = fcntl(fd, Cmd::F_SETFL, flags | Flags::O_NONBLOCK);
            assert!(ret == 0, "UnblockFd fail");
        }

        let hostfd = GlobalIOMgr().AddSocket(fd);
        URING_MGR.lock().Addfd(fd).unwrap();
        return Self::GetRet(hostfd as i64);
    }

    pub fn TsotRecvMsg(msgAddr: u64) -> i64 {
        match TSOT_AGENT.RecvMsg() {
            Ok(msg) => {
                let m = unsafe {
                    &mut *(msgAddr as * mut TsotMessage)
                };
                *m = msg;
                return 0;
            }
            Err(_) => return -1,
        }
    }

    pub fn TsotSendMsg(msgAddr: u64) -> i64 {
        let msg = unsafe {
            &*(msgAddr as * const TsotMessage)
        };
        match TSOT_AGENT.SendMsg(msg) {
            Ok(()) => {
                return 0;
            }
            Err(e) => {
                error!("TsotSendMsg fail with error {:?}", e);
                return -1;
            }
        }
    }

    pub fn Socket(domain: i32, type_: i32, protocol: i32) -> i64 {
        let fd = unsafe {
            socket(
                domain,
                type_ | SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC,
                protocol,
            )
        };

        if fd < 0 {
            return Self::GetRet(fd as i64);
        }

        let hostfd = GlobalIOMgr().AddSocket(fd);
        URING_MGR.lock().Addfd(fd).unwrap();
        return Self::GetRet(hostfd as i64);
    }

    pub fn GetSockName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOGetSockName(addr, addrlen);
    }

    pub fn GetPeerName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOGetPeerName(addr, addrlen);
    }

    pub fn GetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOGetSockOpt(level, optname, optval, optlen);
    }

    pub fn SetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOSetSockOpt(level, optname, optval, optlen);
    }

    pub fn Bind(sockfd: i32, sockaddr: u64, addrlen: u32, umask: u32) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOBind(sockaddr, addrlen, umask);
    }

    pub fn Listen(sockfd: i32, backlog: i32, block: bool) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOListen(backlog, block);
    }

    pub fn RDMAListen(sockfd: i32, backlog: i32, block: bool, acceptQueue: AcceptQueue) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.RDMAListen(backlog, block, acceptQueue);
    }

    pub fn RDMANotify(sockfd: i32, typ: RDMANotifyType) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.RDMANotify(typ);
    }

    pub fn PostRDMAConnect(msg: &'static mut PostRDMAConnect) {
        let fdInfo = match Self::GetFdInfo(msg.fd) {
            Some(fdInfo) => fdInfo,
            None => {
                msg.Finish(-SysErr::EBADF as i64);
                return;
            }
        };

        fdInfo.PostRDMAConnect(msg);
    }

    pub fn Shutdown(sockfd: i32, how: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(sockfd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOShutdown(how);
    }

    ///////////end of network operation//////////////////////////////////////////////////////////////////
    pub fn ReadControlMsg(fd: i32, addr: u64) -> i64 {
        match super::ucall::ucall_server::ReadControlMsg(fd) {
            Err(_e) => return -1,
            Ok(msg) => {
                let controlMsg = unsafe { &mut *(addr as *mut ControlMsg) };
                *controlMsg = msg;
                return 0;
            }
        }
    }

    pub fn SchedGetAffinity(pid: i32, cpuSetSize: u64, mask: u64) -> i64 {
        //todo: fix this
        //let pid = 0;

        let ret = unsafe {
            sched_getaffinity(pid as pid_t, cpuSetSize as size_t, mask as *mut cpu_set_t)
        };

        //todo: fix this.
        if ret == 0 {
            return 8;
        } else {
            Self::GetRet(ret as i64)
        }
    }

    pub fn GetTimeOfDay(tv: u64, tz: u64) -> i64 {
        //let res = unsafe{ gettimeofday(tv as *mut timeval, tz as *mut timezone) };
        //return Self::GetRet(res as i64)

        let nr = SysCallID::sys_gettimeofday as usize;
        unsafe {
            let res = syscall2(nr, tv as usize, tz as usize) as i64;
            //error!("finish GetTimeOfDay");
            return res;
        }
    }

    pub fn GetRandom(&mut self, buf: u64, len: u64, _flags: u32) -> i64 {
        unsafe {
            let slice = slice::from_raw_parts_mut(buf as *mut u8, len as usize);
            self.rng.Fill(slice);
        }

        return len as i64;
    }

    pub fn GetRandomU8(&mut self) -> u8 {
        let mut data: [u8; 1] = [0; 1];
        self.rng.Fill(&mut data);
        return data[0];
    }

    pub fn RandomVcpuMapping(&mut self) {
        let delta = self.GetRandomU8() as usize;
        self.vcpuMappingDelta = delta % Self::VCPUCount();
    }

    pub fn ComputeVcpuCoreId(&self, threadId: usize) -> usize {
        let id = (threadId + self.vcpuMappingDelta) % Self::VCPUCount();

        return id;
    }

    pub fn Fchdir(fd: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { fchdir(fd) };

        return Self::GetRet(ret as i64);
    }

    pub fn Sysinfo(info: u64) -> i64 {
        unsafe {
            return Self::GetRet(sysinfo(info as *mut sysinfo) as i64);
        }
    }

    pub fn Fadvise(fd: i32, offset: u64, len: u64, advice: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { posix_fadvise(fd, offset as i64, len as i64, advice) };

        return Self::GetRet(ret as i64);
    }

    pub fn Mlock2(addr: u64, len: u64, flags: u32) -> i64 {
        let nr = SysCallID::sys_mlock2 as usize;
        let ret = unsafe { syscall3(nr, addr as usize, len as usize, flags as usize) as i64 };

        return Self::GetRet(ret as i64);
    }

    pub fn MUnlock(addr: u64, len: u64) -> i64 {
        let ret = unsafe { munlock(addr as *const c_void, len as size_t) };

        return Self::GetRet(ret as i64);
    }

    pub fn Chown(pathname: u64, owner: u32, group: u32) -> i64 {
        info!("Chown: the pathname is {}", Self::GetStr(pathname));

        let ret = unsafe { chown(pathname as *const c_char, owner, group) };

        return Self::GetRet(ret as i64);
    }

    pub fn FChown(fd: i32, owner: u32, group: u32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { fchown(fd, owner, group) };

        return Self::GetRet(ret as i64);
    }

    pub fn Chmod(pathname: u64, mode: u32) -> i64 {
        let ret = unsafe { chmod(pathname as *const c_char, mode as mode_t) };

        return Self::GetRet(ret as i64);
    }

    pub fn Fchmod(fd: i32, mode: u32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { fchmod(fd, mode as mode_t) };

        return Self::GetRet(ret as i64);
    }

    pub fn EventfdWrite(fd: i32) -> i64 {
        let val: u64 = 8;

        let ret = unsafe { write(fd, &val as *const _ as _, 8) };

        return Self::GetRet(ret as i64);
    }

    pub fn NonBlockingPoll(fd: i32, mask: EventMask) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let mut e = pollfd {
            fd: fd,
            events: mask as i16,
            revents: 0,
        };

        loop {
            let ret = unsafe { poll(&mut e, 1, 0) };

            let ret = Self::GetRet(ret as i64) as i32;
            // Interrupted by signal, try again.
            if ret == -SysErr::EINTR {
                continue;
            }

            // If an error occur we'll conservatively say the FD is ready for
            // whatever is being checked.
            if ret < 0 {
                return mask as i64;
            }

            // If no FDs were returned, it wasn't ready for anything.
            if ret == 0 {
                return 0;
            }

            return e.revents as i64;
        }
    }

    pub fn NewTmpfile(dir: bool, addr: u64) -> i64 {
        let mut td = temp_dir();

        let file_name = format!("{}", Uuid::new_v4());
        td.push(file_name);

        let fd = if dir {
            let folder = td.into_os_string().into_string().unwrap();
            let cstr = CString::New(&folder);
            let ret = unsafe { libc::mkdir(cstr.Ptr() as *const c_char, 0o777) };

            if ret != 0 {
                return Self::GetRet(ret as i64);
            }

            let fd = unsafe {
                libc::openat(
                    -100,
                    cstr.Ptr() as *const c_char,
                    libc::O_DIRECTORY | libc::O_RDONLY,
                    0o777,
                )
            };

            Self::GetRet(fd as i64) as i32
        } else {
            let file = fs::File::create(td).expect("tmp file create fail");
            file.into_raw_fd()
        };

        let ret = unsafe { fstat(fd, addr as *mut stat) };

        if ret < 0 {
            unsafe {
                close(fd);
            }

            return Self::GetRet(ret as i64);
        }

        let guestfd = GlobalIOMgr().AddFile(fd);

        return guestfd as i64;
    }

    pub fn NewFifo() -> i64 {
        let uid = NewUID();
        let path = format!("/tmp/fifo_{}", uid);
        let cstr = CString::New(&path);
        let ret = unsafe { mkfifo(cstr.Ptr() as *const c_char, 0o666) };

        error!("NewFifo apth is {}, id is {}", path, ret);

        if ret < 0 {
            return Self::GetRet(ret as i64);
        }

        return uid as i64;
    }

    pub fn NewTmpfsFile(typ: TmpfsFileType, addr: u64) -> i64 {
        match typ {
            TmpfsFileType::Dir => Self::NewTmpfile(true, addr),
            TmpfsFileType::File => Self::NewTmpfile(false, addr),
        }
    }

    pub fn Statm(buf: u64) -> i64 {
        const STATM: &str = "/proc/self/statm";
        let contents = fs::read_to_string(STATM).expect("Something went wrong reading the file");

        let output = scan!(&contents, char::is_whitespace, u64, u64);
        let statm = unsafe { &mut *(buf as *mut StatmInfo) };

        statm.vss = output.0.unwrap();
        statm.rss = output.1.unwrap();
        return 0;
    }

    pub fn HostEpollWaitProcess() -> i64 {
        let ret = FD_NOTIFIER.HostEpollWait();
        return ret;
    }

    #[cfg(target_arch = "x86_64")]
    pub fn HostID(axArg: u32, cxArg: u32) -> (u32, u32, u32, u32) {
        let mut ax: u32 = axArg;
        let bx: u32;
        let mut cx: u32 = cxArg;
        let dx: u32;
        unsafe {
            asm!("
              CPUID
              mov edi, ebx
            ",
            inout("eax") ax,
            out("edi") bx,
            inout("ecx") cx,
            out("edx") dx,
            );
        }

        return (ax, bx, cx, dx);
    }

    #[cfg(target_arch = "aarch64")]
    pub fn HostID(axArg: u32, cxArg: u32) -> (u32, u32, u32, u32) {
        return (0, 0, 0, 0);
    }

    pub fn SymLinkAt(oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        let newdirfd = match Self::GetOsfd(newdirfd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret =
            unsafe { symlinkat(oldpath as *const c_char, newdirfd, newpath as *const c_char) };

        return Self::GetRet(ret as i64);
    }

    pub fn LinkAt(olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64, flags: i32) -> i64 {
        let newdirfd = match Self::GetOsfd(newdirfd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let olddirfd = match Self::GetOsfd(olddirfd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            linkat(
                olddirfd,
                oldpath as *const c_char,
                newdirfd,
                newpath as *const c_char,
                flags,
            )
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Futimens(fd: i32, times: u64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe { futimens(fd, times as *const timespec) };

        return Self::GetRet(ret as i64);
    }

    //map kernel table
    pub fn KernelMap(
        &mut self,
        start: Addr,
        end: Addr,
        physical: Addr,
        flags: PageTableFlags,
    ) -> Result<bool> {
        info!("KernelMap start is {:x}, end is {:x}", start.0, end.0);
        return self
            .pageTables
            .Map(start, end, physical, flags, &mut self.allocator, true);
    }

    pub fn KernelMapHugeTable(
        &mut self,
        start: Addr,
        end: Addr,
        physical: Addr,
        flags: PageTableFlags,
    ) -> Result<bool> {
        info!("KernelMap1G start is {:x}, end is {:x}", start.0, end.0);
        return self
            .pageTables
            .MapWith1G(start, end, physical, flags, &mut self.allocator, true);
    }

    #[cfg (feature = "cc")]
    pub fn KernelMapHugeTableSevSnp(
        &mut self,
        start: Addr,
        end: Addr,
        physical: Addr,
        flags: PageTableFlags,
        c_bit: u64,
    ) -> Result<bool> {
        info!("KernelMap1G start is {:x}, end is {:x}", start.0, end.0);
        return self
            .pageTables
            .MapWith1GSevSnp(start, end, physical, flags, &mut self.allocator, c_bit, true);
    }

    pub fn PrintStr(phAddr: u64) {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            let ptr = phAddr as *const u8;
            #[cfg(target_arch = "x86_64")]
            let ptr = phAddr as *const i8;
            info!(
                "the Str: {} ",
                str::from_utf8_unchecked(slice::from_raw_parts(
                    phAddr as *const u8,
                    strlen(ptr) + 1
                ))
            );
        }
    }

    pub fn Proxy(cmd: ProxyCommand, parameters: &ProxyParameters) -> i64 {
        match NvidiaProxy(cmd, parameters) {
            Ok(v) => return v,
            Err(e) => {
                error!("nvidia proxy get error {:?}", e);
                return 0;
            }
        }
    }

    pub fn SwapInPage(addr: u64) -> i64 {
        match SHARE_SPACE.hiberMgr.SwapIn(addr) {
            Ok(_) => return 0,
            Err(Error::SysError(e)) => return e as i64,
            _ => panic!("imposible"),
        }
    }

    pub fn UnblockFd(fd: i32) {
        unsafe {
            let flags = fcntl(fd, Cmd::F_GETFL, 0);
            let ret = fcntl(fd, Cmd::F_SETFL, flags | Flags::O_NONBLOCK);
            assert!(ret == 0, "UnblockFd fail");
        }
    }

    pub fn BlockFd(fd: i32) {
        unsafe {
            let flags = fcntl(fd, Cmd::F_GETFL, 0);
            let ret = fcntl(fd, Cmd::F_SETFL, flags & !Flags::O_NONBLOCK);
            assert!(ret == 0, "UnblockFd fail");
        }
    }

    pub fn GetStdfds(addr: u64) -> i64 {
        let ptr = addr as *mut i32;
        let stdfds = unsafe { slice::from_raw_parts_mut(ptr, 3) };

        for i in 0..stdfds.len() {
            let osfd = unsafe { dup(i as i32) as i32 };

            if osfd < 0 {
                return osfd as i64;
            }

            Self::UnblockFd(osfd);

            let hostfd = GlobalIOMgr().AddFile(osfd);
            stdfds[i] = hostfd;
        }

        return 0;
    }

    pub fn LibcFstat(osfd: i32) -> Result<LibcStat> {
        let mut stat = LibcStat::default();
        let ret = unsafe { fstat(osfd, &mut stat as *mut _ as u64 as *mut stat) };

        if ret < 0 {
            info!("can't fstat osfd {}", osfd);
            return Err(Error::SysError(errno::errno().0));
        }

        //Self::LibcStatx(osfd);

        return Ok(stat);
    }

    pub fn LibcStatx(osfd: i32) {
        let statx = Statx::default();
        #[cfg(target_arch = "x86_64")]
        let addr: i8 = 0;
        #[cfg(target_arch = "aarch64")]
        let addr: u8 = 0;
        let ret = unsafe {
            libc::statx(
                osfd,
                &addr as *const c_char,
                libc::AT_EMPTY_PATH,
                libc::STATX_BASIC_STATS,
                &statx as *const _ as u64 as *mut statx,
            )
        };

        error!(
            "LibcStatx osfd is {} ret is {} error is {}",
            osfd,
            ret,
            errno::errno().0
        );
    }

    #[cfg(target_arch = "x86_64")]
    pub fn GetVcpuFreq(&self) -> i64 {
        let freq = self.vcpus[0].vcpu.get_tsc_khz().unwrap() * 1000;
        return freq as i64;
    }

    #[cfg(target_arch = "aarch64")]
    pub fn GetVcpuFreq(&self) -> i64 {
        return 0;
    }

    pub fn Membarrier(cmd: i32) -> i32 {
        let nr = SysCallID::sys_membarrier as usize;
        let ret = unsafe {
            syscall3(
                nr,
                cmd as usize,
                0 as usize, /*flag*/
                0 as usize, /*unused*/
            ) as i32
        };
        return ret as _;
    }

    pub fn HostMemoryBarrier() -> i64 {
        let haveMembarrierPrivateExpedited = VMS.read().haveMembarrierPrivateExpedited;
        let cmd = if haveMembarrierPrivateExpedited {
            MEMBARRIER_CMD_PRIVATE_EXPEDITED
        } else {
            MEMBARRIER_CMD_GLOBAL
        };

        return Self::Membarrier(cmd) as _;
    }

    //return (haveMembarrierGlobal, haveMembarrierPrivateExpedited)
    pub fn MembarrierInit() -> (bool, bool) {
        let supported = Self::Membarrier(MEMBARRIER_CMD_QUERY);
        if supported < 0 {
            return (false, false);
        }

        let mut haveMembarrierGlobal = false;
        let mut haveMembarrierPrivateExpedited = false;
        // We don't use MEMBARRIER_CMD_GLOBAL_EXPEDITED because this sends IPIs to
        // all CPUs running tasks that have previously invoked
        // MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED, which presents a DOS risk.
        // (MEMBARRIER_CMD_GLOBAL is synchronize_rcu(), i.e. it waits for an RCU
        // grace period to elapse without bothering other CPUs.
        // MEMBARRIER_CMD_PRIVATE_EXPEDITED sends IPIs only to CPUs running tasks
        // sharing the caller's MM.)
        if supported & MEMBARRIER_CMD_GLOBAL != 0 {
            haveMembarrierGlobal = true;
        }

        let req = MEMBARRIER_CMD_PRIVATE_EXPEDITED | MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED;
        if supported & req == req {
            let ret = Self::Membarrier(MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED);
            if ret >= 0 {
                haveMembarrierPrivateExpedited = true;
            }
        }

        return (haveMembarrierGlobal, haveMembarrierPrivateExpedited);
    }

}

impl PostRDMAConnect {
    pub fn Finish(&mut self, ret: i64) {
        self.ret = ret;
        SHARE_SPACE
            .scheduler
            .ScheduleQ(self.taskId, self.taskId.Queue(), true)
    }

    pub fn ToRef(addr: u64) -> &'static mut Self {
        let msgRef = unsafe { &mut *(addr as *mut Self) };

        return msgRef;
    }
}
