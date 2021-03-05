// Copyright (c) 2021 QuarkSoft LLC
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
pub mod syscall;
pub mod timer_keeper;
pub mod hostfdnotifier;
pub mod time;
pub mod host_pma_keeper;
pub mod random;
pub mod limits;
pub mod uringMgr;
pub mod host_uring;
pub mod kernel_io_thread;

use std::str;
use std::slice;
use libc::*;
use std::time::Duration;
use std::marker::Send;
use serde_json;
use alloc::collections::btree_map::BTreeMap;
use x86_64::structures::paging::PageTableFlags;
use std::collections::VecDeque;
use alloc::boxed::Box;
use tempfile::tempfile;
use std::os::unix::io::IntoRawFd;
use lazy_static::lazy_static;
use core::sync::atomic::AtomicU64;
use core::sync::atomic;

use super::runc::runtime::loader::*;
use super::runc::specutils::specutils::*;
use super::runc::container::mounts::*;
use super::qlib::*;
use super::qlib::task_mgr::*;
use super::qlib::common::{Error, Result};
use super::qlib::linux_def::*;
use super::qlib::pagetable::{PageTables};
use super::qlib::addr::{Addr};
use super::qlib::control_msg::*;
use super::qlib::buddyallocator::MemAllocator;
//use super::qlib::boot::loader;
use super::qlib::qmsg::*;
use super::qlib::cstring::*;
use super::qlib::perf_tunning::*;
use super::qcall::*;
use super::namespace::MountNs;
use super::runc::runtime::vm::*;
use super::ucall::usocket::*;
use super::*;
use self::HostFileMap::fdinfo::*;
use self::syscall::*;
use self::time::*;
use self::random::*;
use self::limits::*;
use super::runc::runtime::signal_handle::*;

const ARCH_SET_GS:u64 = 0x1001;
const ARCH_SET_FS:u64 = 0x1002;
const ARCH_GET_FS:u64 = 0x1003;
const ARCH_GET_GS:u64 = 0x1004;

lazy_static! {
    static ref UID: AtomicU64 = AtomicU64::new(1);
}

pub fn NewUID() -> u64 {
    return UID.fetch_add(1, atomic::Ordering::SeqCst);
}

pub fn Init() {
    //self::fs::Init();
}

#[derive(Clone, Copy)]
pub struct WaitingMsgCall {
    pub taskId: TaskIdQ,
    pub addr: u64,
    pub len: usize,
    pub retAddr: u64,
}

pub struct VMSpace {
    pub pageTables : PageTables,
    pub allocator: Option<MemAllocator>,
    pub hostAddrTop: u64,
    pub sharedLoasdOffset: u64,
    pub vdsoAddr: u64,

    pub shareSpace: &'static ShareSpace,
    pub rng: RandGen,
    pub args: Option<Args>,
    pub pivot: bool,
    pub waitingMsgCall: Option<WaitingMsgCall>,
    pub controlMsgCallBack: BTreeMap<u64, USocket>,
    pub controlMsgQueue: VecDeque<Box<(USocket, ControlMsg)>>,
}

unsafe impl Sync for VMSpace {}
unsafe impl Send for VMSpace {}

impl VMSpace {
    pub fn CloseVMSpace(&mut self) {
        for (_, sock) in self.controlMsgCallBack.iter() {
            sock.SendResp(&UCallResp::UCallRespErr("container shutdown...".to_string())).ok();
        }
        self.controlMsgCallBack.clear();
    }

    ///////////start of file operation//////////////////////////////////////////////
    pub fn GetOsfd(hostfd: i32) -> Option<i32> {
        return IO_MGR.lock().GetFdByHost(hostfd);
    }

    pub fn GetFdInfo(hostfd: i32) -> Option<FdInfo> {
        return IO_MGR.lock().GetByHost(hostfd);
    }

    pub fn GetDents(_taskId: u64, fd: i32, dirp: u64, count: u32) -> i64 {
        let fd = Self::GetOsfd(fd).expect("GetDents64");

        let nr = SysCallID::sys_getdents as usize;
        unsafe {
            return syscall3(nr, fd as usize, dirp as usize, count as usize) as i64;
        }
    }

    pub fn GetDents64(_taskId: u64, fd: i32, dirp: u64, count: u32) -> i64 {
        let fd = Self::GetOsfd(fd).expect("GetDents64");

        let nr = SysCallID::sys_getdents64 as usize;


        //info!("sys_getdents64 is {}", nr);
        unsafe {
            return syscall3(nr, fd as usize, dirp as usize, count as usize) as i64;
        }
    }

    pub fn GetUid(_taskId: u64) -> i64 {
        unsafe {
            return getuid() as i64; //always success
        }
    }

    pub fn GetEUid(_taskId: u64) -> i64 {
        unsafe {
            return geteuid() as i64;//always success
        }
    }

    pub fn GetGid(_taskId: u64) -> i64 {
        unsafe {
            return getgid() as i64;//always success
        }
    }

    pub fn SetGid(_taskId: u64, _gid: u32) -> i64 {
        //unsafe {
            //return setgid(gid) as i64;
        //}

        return 0;
    }

    pub fn GetEGid(_taskId: u64) -> i64 {
        unsafe {
            return getegid() as i64; //always success
        }
    }

    pub fn GetGroups(_taskId: u64, size: i32, list: u64) -> i64 {
        //info!("GetGroups the list is {:x}", list);

        unsafe {
            return getgroups(size, list as *mut gid_t) as i64;
        }
    }

    pub fn SetGroups(_taskId: u64, size: usize, list: u64) -> i64 {
        unsafe {
            return setgroups(size, list as *mut gid_t) as i64;
        }
    }

    pub fn Sysinfo(_taskId: u64, info: u64) -> i64 {
        unsafe {
            return Self::GetRet(sysinfo(info as *mut sysinfo) as i64);
        }
    }

    pub fn GetCwd(_taskId: u64, buf: u64, size: u64) -> i64 {
        let nr = SysCallID::sys_getcwd as usize;

        unsafe {
            let res = syscall2(nr, buf as usize, size as usize) as i64;
            return res
        }

        /*let ret = unsafe {
            getcwd(buf as *mut c_char, size as size_t) as i64
        };
        info!("GetCwd: the local path is {}", Self::GetStr(ret as u64));

        return Self::GetRet(ret as i64)*/
        //return ret;
        //return Self::GetStrLen(ret as u64);
    }

    pub fn Mount(&self, id: &str, rootfs: &str) -> Result<()> {
        let spec = &self.args.as_ref().unwrap().Spec;
        //let rootfs : &str = &spec.root.path;
        let cpath = format!("/{}", id);

        init_rootfs(spec, rootfs, &cpath, false)?;
        pivot_rootfs(&*rootfs)?;
        return Ok(())
    }

    pub fn PivotRoot(&self, rootfs: &str) {
        let mns = MountNs::New(rootfs.to_string());
        mns.PivotRoot();
    }

    pub fn ControlMsgCall(&mut self, taskId: TaskIdQ, addr: u64, len: usize, retAddr: u64) -> QcallRet {
        match self.controlMsgQueue.pop_back() {
            Some(data) => {
                self.CopyControlMsg(&WaitingMsgCall{
                    taskId: taskId,
                    addr: addr,
                    len: len,
                    retAddr,
                }, data.0, &data.1).expect("ControlMsgCall CopyControlMsg fail");

                return QcallRet::Normal
            }
            None => ()
        };

        self.waitingMsgCall = Some(WaitingMsgCall{
            taskId: taskId,
            addr: addr,
            len: len,
            retAddr,
        });

        return QcallRet::Block
    }

    pub fn ControlMsgRet(&mut self, _taskId: u64, msgId: u64, addr: u64, len: usize) -> i64 {
        let buf = {
            let ptr = addr as * const u8;
            unsafe { slice::from_raw_parts(ptr, len) }
        };

        let resp : UCallResp = serde_json::from_slice(&buf[0..len]).expect("ControlMsgRet des fail");

        let usock = match self.controlMsgCallBack.remove(&msgId) {
            None => panic!("ControlMsgRet get non-exist msgid {}", msgId),
            Some(s) => s,
        };

        match usock.SendResp(&resp) {
            Err(e) => error!("ControlMsgRet send resp fail with error {:?}", e),
            Ok(()) => (),
        }

        return 0;
    }

    pub fn CopyControlMsg(&mut self, waitMsg: &WaitingMsgCall, usock: USocket, msg: &ControlMsg) -> Result<()> {
        let msgId = msg.msgId;

        let vec : Vec<u8> = serde_json::to_vec(msg).expect("SendControlMsg ser fail...");
        let buff = {
            let ptr = waitMsg.addr as *mut u8;
            unsafe { slice::from_raw_parts_mut(ptr, waitMsg.len) }
        };

        if vec.len() > buff.len() {
            return Err(Error::Common(format!("ExecProcess not enough space..., required len is {}, buff len is {}", vec.len(), buff.len())));
        }

        for i in 0..vec.len() {
            buff[i] = vec[i];
        }

        let addr = waitMsg.retAddr;
        unsafe {
            *(addr as * mut u64) = vec.len() as u64;
        }

        self.controlMsgCallBack.insert(msgId, usock);
        return Ok(())
    }

    pub fn SendControlMsg(&mut self, usock: USocket, msg: ControlMsg) -> Result<()> {
        let waitMsg = match self.waitingMsgCall {
            None => {
                self.controlMsgQueue.push_front(Box::new((usock, msg)));
                return Ok(());
            },
            Some(m) => m,
        };

        self.CopyControlMsg(&waitMsg, usock, &msg)?;

        VirtualMachine::Schedule(self.GetShareSpace(), waitMsg.taskId);
        return Ok(())
    }

    pub fn VCPUCount() -> usize {
        let mut cpuCount = num_cpus::get();

        if cpuCount < 3 {
            cpuCount = 3; // at least 3 vcpu (one for kernel io, one for host io and other for process vcpu)
        }

        if cpuCount > MAX_VCPU_COUNT {
            cpuCount = MAX_VCPU_COUNT;
        }

        return cpuCount
    }

    pub fn LoadProcessKernel(&mut self, _taskId: u64, processAddr: u64, buffLen: usize) -> i64 {
        let mut process = loader::Process::default();
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
        error!("LoadProcessKernel: need to study the user mapping handling...");
        process.UID = spec.process.user.uid;
        process.GID = spec.process.user.gid;
        process.AdditionalGids.append(&mut spec.process.user.additional_gids);
        process.limitSet = CreateLimitSet(&spec).expect("load limitSet fail").GetInternalCopy();
        process.Caps = Capabilities(false, &spec.process.capabilities);

        process.HostName = spec.hostname.to_string();

        process.NumCpu = Self::VCPUCount() as u32;

        for i in 0..process.Stdiofds.len() {
            let osfd = unsafe {
                dup(i as i32) as i32
            };

            URING_MGR.lock().Addfd(osfd).unwrap();

            if  osfd < 0 {
                return osfd as i64
            }

            let stat = Self::LibcFstat(osfd).expect("LoadProcessKernel: can't fstat for the stdio fds");

            //Self::UnblockFd(osfd);

            let st_mode = stat.st_mode & ModeType::S_IFMT as u32;
            let epollable = st_mode == S_IFIFO || st_mode == S_IFSOCK || st_mode == S_IFCHR;

            let hostfd = IO_MGR.lock().AddFd(osfd, epollable);

            // can block wait
            if epollable {
                FD_NOTIFIER.AddFd(osfd, Box::new(GuestFd{hostfd: hostfd}));
            }

            process.Stdiofds[i] = hostfd;
        }

        process.Root = "/".to_string();

        let rootfs = self.args.as_ref().unwrap().Rootfs.to_string();

        if self.pivot {
            self.PivotRoot(&rootfs);
        }

        //error!("LoadProcessKernel proces is {:?}", &process);

        let vec : Vec<u8> = serde_json::to_vec(&process).expect("LoadProcessKernel ser fail...");
        let buff = {
            let ptr = processAddr as *mut u8;
            unsafe { slice::from_raw_parts_mut(ptr, buffLen) }
        };

        assert!(vec.len() <= buff.len(), "LoadProcessKernel not enough space...");
        for i in 0..vec.len() {
            buff[i] = vec[i];
         }

        StartSignalHandle();

        //self.shareSpace.lock().AQHostInputCall(HostMsg::ExecProcess);

        return vec.len() as i64
    }

    pub fn Pipe2(_taskId: u64, fds: u64, flags: i32) -> i64 {
        unsafe {
            let ret = pipe2(fds as *mut c_int, flags | O_NONBLOCK);

            if ret < 0 {
                return Self::GetRet(ret as i64)
            }

            let ptr = fds as * mut i32;
            let fds = slice::from_raw_parts_mut(ptr, 2);

            let hostfd0 = IO_MGR.lock().AddFd(fds[0], true);
            let hostfd1 = IO_MGR.lock().AddFd(fds[1], true);

            FD_NOTIFIER.AddFd(fds[0], Box::new(GuestFd{hostfd: hostfd0}));
            FD_NOTIFIER.AddFd(fds[1], Box::new(GuestFd{hostfd: hostfd1}));

            fds[0] = hostfd0;
            fds[1] = hostfd1;

            return Self::GetRet(ret as i64)
        }
    }

    pub fn Fallocate(_taskId: u64, fd: i32, mode: i32, offset: i64, len: i64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            fallocate(fd, mode, offset, len)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn RenameAt(_taskId: u64, olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
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
            renameat(olddirfd, oldpath as *const c_char, newdirfd, newpath as *const c_char)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Truncate(_taskId: u64, path: u64, len: i64) -> i64 {
        let ret = unsafe {
            truncate64(path as *const c_char, len)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Ftruncate(_taskId: u64, fd: i32, len: i64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            ftruncate64(fd, len)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Open(_taskId: u64, fileName: u64, flags: i32, mode: i32) -> i64 {
        info!("Open: the filename is {}, flag is {:b}, the mode is {:b}", Self::GetStr(fileName), flags, mode);

        unsafe {
            let fd = open(fileName as *const c_char, flags as c_int, mode as c_int);
            if fd < 0 {
                return Self::GetRet(fd as i64)
            }

            let guestfd = IO_MGR.lock().AddFd(fd, false);

            return guestfd as i64
        }
    }

    pub fn Eventfd(_taskId: u64, initval: u32, flags: i32) -> i64 {
        unsafe {
            let fd = eventfd(initval, flags);
            if fd < 0 {
                return Self::GetRet(fd as i64)
            }

            let hostfd = IO_MGR.lock().AddFd(fd, true);
            FD_NOTIFIER.AddFd(fd, Box::new(GuestFd{hostfd: hostfd}));
            return hostfd as i64
        }
    }

    pub fn GetStr(string: u64) -> &'static str {
        let ptr = string as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, 1024) };

        let len = {
            let mut res : usize = 0;
            for i in 0..1024 {
                if slice[i] == 0 {
                    res = i;
                    break
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
            let mut res : usize = 0;
            for i in 0..1024 {
                if slice[i] == 0 {
                    res = i;
                    break
                }
            }

            res
        };

        return (len+1) as i64
    }

    pub fn OpenAt(_taskId: u64, dirfd: i32, fileName: u64, flags: i32, mode: i32) -> i32 {
        info!("Openat: the filename is {}, flag is {:x}, the mode is {:b}, dirfd is {}", Self::GetStr(fileName), flags, mode, dirfd);

        let dirfd = if dirfd < 0 {
            dirfd
        } else {
            match Self::GetOsfd(dirfd) {
                Some(fd) => fd,
                None => return -SysErr::EBADF as i32,
            }
        };

        unsafe {
            let osfd = libc::openat(dirfd, fileName as *const c_char, flags as c_int, mode as c_int);
            if osfd <= 0 {
                return Self::GetRet(osfd as i64) as i32
            }

            let hostfd = IO_MGR.lock().AddFd(osfd, false);
            return hostfd
        }
    }

    pub unsafe fn TryOpenHelper(dirfd: i32, name: u64) -> (i32, bool) {
        let flags = Flags::O_NOFOLLOW;
        let ret = libc::openat(dirfd, name as *const c_char, (flags | Flags::O_RDWR) as i32, 0);
        if ret > 0 {
            return (ret, true);
        }

        let err = Self::GetRet(ret as i64) as i32;
        if err == ENOENT {
            return (-ENOENT, false)
        }

        let ret = libc::openat(dirfd, name as *const c_char, (flags | Flags::O_RDONLY) as i32, 0);
        if ret > 0 {
            return (ret, false);
        }

        let ret = libc::openat(dirfd, name as *const c_char, (flags | Flags::O_WRONLY) as i32, 0);
        if ret > 0 {
            return (ret, true);
        }

        // when the file is unix socket file, needs to return SysErr::ENXIO
        let ret = libc::openat(dirfd, name as *const c_char, flags as i32 | Flags::O_PATH, 0);
        if ret > 0 {
            return (ret, false);
        }

        return (Self::GetRet(ret as i64) as i32, false)
    }

    pub fn TryOpenAt(_taskId: u64, dirfd: i32, name: u64, addr: u64) -> i64 {
        //info!("TryOpenAt: the filename is {}", Self::GetStr(name));
        let dirfd = if dirfd < 0 {
            dirfd
        } else {
            match Self::GetOsfd(dirfd) {
                Some(fd) => fd,
                None => return -SysErr::EBADF as i64,
            }
        };

        let tryOpenAt = unsafe {
            &mut *(addr as * mut TryOpenStruct)
        };

        let (fd, writeable) = unsafe {
            Self::TryOpenHelper(dirfd, name)
        };

        if fd < 0 {
            return fd as i64
        }

        let ret = unsafe {
            libc::fstat(fd, tryOpenAt.fstat as * const _ as u64 as *mut stat) as i64
        };

        if ret < 0 {
            unsafe {
                libc::close(fd);
            }
            return Self::GetRet(ret as i64)
        }

        tryOpenAt.writeable = writeable;
        let hostfd = IO_MGR.lock().AddFd(fd, false);

        if tryOpenAt.fstat.IsRegularFile() {
            URING_MGR.lock().Addfd(hostfd).unwrap();
        }

        return hostfd as i64
    }

    pub fn CreateAt(_taskId: u64, dirfd: i32, fileName: u64, flags: i32, mode: i32, uid: u32, gid: u32, fstatAddr: u64) -> i32 {
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
            let osfd = libc::openat(dirfd, fileName as *const c_char, flags as c_int, mode as c_int);
            if osfd <= 0 {
                return Self::GetRet(osfd as i64) as i32
            }

            let ret = libc::fchown(osfd, uid, gid);
            if ret < 0 {
                libc::close(osfd);
                return Self::GetRet(ret as i64) as i32
            }

            let ret = libc::fstat(osfd, fstatAddr as *mut stat) as i64;

            if ret < 0 {
                libc::close(osfd);
                return Self::GetRet(ret as i64) as i32
            }

            let hostfd = IO_MGR.lock().AddFd(osfd, false);

            URING_MGR.lock().Addfd(osfd).unwrap();

            return hostfd
        }
    }

    pub fn Close(_taskId: u64, fd: i32) -> i64 {
        let info = IO_MGR.lock().RemoveFd(fd);

        URING_MGR.lock().Removefd(fd).unwrap();
        let res = if let Some(info) = info {
            let waitable = info.lock().epollable;
            if waitable {
                FD_NOTIFIER.RemoveFd(info.lock().osfd).expect("close FD_NOTIFIER.RemoveFd fail");
            }
            0
        } else {
            -SysErr::EINVAL as i64
        };

        return res;
    }

    pub fn IORead(_taskId: u64, fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            readv(fd as c_int, iovs as *const iovec, iovcnt) as i64
        };

        return Self::GetRet(ret as i64)
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

    pub fn IOWrite(taskId: u64, fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOWrite(taskId, iovs, iovcnt)
    }

    pub fn IOAppend(taskId: u64, fd: i32, iovs: u64, iovcnt: i32, fileLenAddr: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOAppend(taskId, iovs, iovcnt, fileLenAddr)
    }

    pub fn IOReadAt(taskId: u64, fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOReadAt(taskId, iovs, iovcnt, offset)
    }

    pub fn IOWriteAt(taskId: u64, fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOWriteAt(taskId, iovs, iovcnt, offset)
    }

    pub fn IOAccept(taskId: u64, fd: i32, addr: u64, addrlen: u64, flags: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOAccept(taskId, addr, addrlen, flags)
    }

    pub fn IOConnect(taskId: u64, fd: i32, addr: u64, addrlen: u32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOConnect(taskId, addr, addrlen)
    }

    pub fn IORecvMsg(taskId: u64, fd: i32, msghdr: u64, flags: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IORecvMsg(taskId, msghdr, flags)
    }

    pub fn IOSendMsg(taskId: u64, fd: i32, msghdr: u64, flags: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IOSendMsg(taskId, msghdr, flags)
    }

    pub fn Fcntl(taskId: u64, fd: i32, cmd: i32, arg: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(info) => info,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.Fcntl(taskId, cmd, arg)
    }

    pub fn IoCtl(taskId: u64, fd: i32, cmd: u64, argp: u64) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.IoCtl(taskId, cmd, argp)
    }

    pub fn FSync(taskId: u64, fd: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.FSync(taskId, false)
    }

    pub fn FDataSync(taskId: u64, fd: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.FSync(taskId, true)
    }

    pub fn Seek(taskId: u64, fd: i32, offset: i64, whence: i32) -> i64 {
        let fdInfo = match Self::GetFdInfo(fd) {
            Some(fdInfo) => fdInfo,
            None => return -SysErr::EBADF as i64,
        };

        return fdInfo.Seek(taskId, offset, whence)
    }

    pub fn ReadLink(_taskId: u64, path: u64, buf: u64, bufsize: u64) -> i64 {
        //info!("ReadLink: the path is {}", Self::GetStr(path));

        let res = unsafe{ readlink(path as *const c_char, buf as *mut c_char, bufsize as usize) };
        return Self::GetRet(res as i64)
    }

    pub fn ReadLinkAt(_taskId: u64, dirfd: i32, path: u64, buf: u64, bufsize: u64) -> i64 {
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

        let res = unsafe{ readlinkat(dirfd, path as *const c_char, buf as *mut c_char, bufsize as usize) };
        return Self::GetRet(res as i64)
    }

    pub fn IOSetup(_taskId: u64, nr_events: u64, ctx_idp: u64) -> i64 {
        let nr = SysCallID::sys_io_setup as usize;
        unsafe {
            let res = syscall2(nr, nr_events as usize, ctx_idp as usize) as i64;

            let context = *(ctx_idp as * const u64);
            error!("IOSetup context id is {:x}", context);
            return res
        }
    }

    pub fn IOSubmit(_taskId: u64, _ctx_id: u64, _nr: u64, _iocbpp: u64) -> i64 {
        /*let callId = SysCallID::sys_io_submit as usize;

        unsafe {
            for i in 0..nr {
                let addr = (iocbpp + i*8) as *const u64;
                let iocbp = &mut *((*addr) as *mut iocb);
                iocbp.aio_fildes = match Self::GetOsfd(taskId, iocbp.aio_fildes as i32) {
                    Some(fd) => fd as u32,
                    None => return -SysErr::EBADF as i64,
                };

                if iocbp.aio_resfd != 0 {
                    iocbp.aio_resfd = match Self::GetOsfd(taskId, iocbp.aio_resfd as i32) {
                        Some(fd) => fd as u32,
                        None => return -SysErr::EBADF as i64,
                    };
                }
            }
        }

        unsafe {
            let res = syscall3(callId, ctx_id as usize, nr as usize, iocbpp as usize) as i64;

            if res < 0 {
                error!("IOSubmit get error {}", -res);
            }
            return res
        }*/

        //todo: disable aio now. enable it later
        return -EINVAL as i64;
    }

    pub fn Stat(pathName: u64, statBuff: u64) -> i64 {
        info!("Stat: the filename is {}", Self::GetStr(pathName));
        let ret = unsafe{
            stat(pathName as *const c_char, statBuff as *mut stat)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Lstat(pathName: u64, statBuff: u64) -> i64 {
        //info!("Lstat: the filename is {}", Self::GetStr(pathName));

        let ret = unsafe{
            lstat(pathName as *const c_char, statBuff as *mut stat)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Fstat(_taskId: u64, fd: i32, buf: u64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            libc::fstat(fd, buf as *mut stat) as i64
        };

        //Self::LibcStatx(fd);
        return Self::GetRet(ret);
    }

    pub fn Getxattr(_taskId: u64, path: u64, name: u64, value: u64, size: u64) -> i64 {
        info!("Getxattr: the path is {}, name is {}", Self::GetStr(path), Self::GetStr(name));
        let ret = unsafe {
            getxattr(path as *const c_char, name as *const c_char, value as *mut c_void, size as usize) as i64
        };

        return Self::GetRet(ret);
    }

    pub fn Lgetxattr(_taskId: u64, path: u64, name: u64, value: u64, size: u64) -> i64 {
        info!("Lgetxattr: the path is {}, name is {}", Self::GetStr(path), Self::GetStr(name));
        let ret = unsafe {
            lgetxattr(path as *const c_char, name as *const c_char, value as *mut c_void, size as usize) as i64
        };

        return Self::GetRet(ret);
    }

    pub fn Fgetxattr(_taskId: u64, fd: i32, name: u64, value: u64, size: u64) -> i64 {
        let fd = Self::GetOsfd(fd).expect("fgetxattr");
        let ret = unsafe {
            fgetxattr(fd, name as *const c_char, value as *mut c_void, size as usize) as i64
        };

        return Self::GetRet(ret);
    }

    pub fn GetRet(ret: i64) -> i64 {
        if ret == -1 {
            //info!("get error, errno is {}", errno::errno().0);
            return -errno::errno().0 as i64
        }

        return ret
    }

    pub fn Fstatat(_taskId: u64, dirfd: i32, pathname: u64, buf: u64, flags: i32) -> i64 {
        let dirfd = {
            if dirfd > 0 {
                Self::GetOsfd(dirfd).expect("Fstatat")
            } else {
                dirfd
            }
        };

        return unsafe {
            Self::GetRet(libc::fstatat(dirfd, pathname as *const c_char, buf as *mut stat, flags) as i64)
        };
    }

    pub fn Statfs(path: u64, buf: u64) -> i64 {
        let ret = unsafe{
            statfs(path as *const c_char, buf as *mut statfs)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Fstatfs(_taskId: u64, fd: i32, buf: u64) -> i64 {
        let fd = Self::GetOsfd(fd).expect("Fstatfs");

        let ret = unsafe{
            fstatfs(fd, buf as *mut statfs)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn PRLimit(_pid: i32, resource: i32, newLimit: u64, oldLimit: u64) -> i64 {
        let ret = unsafe{
            prlimit(0 as pid_t, resource as u32, newLimit as *const rlimit, oldLimit as *mut rlimit)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn GetRLimit(resource: u32, rlimit: u64) -> i64 {
        let ret = unsafe {
            getrlimit(resource, rlimit as *mut rlimit )
        };

        return Self::GetRet(ret as i64);
    }

    pub fn SetRLimit(resource: u32, rlimit: u64) -> i64 {
        let ret = unsafe {
            setrlimit(resource, rlimit as *const rlimit)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Unlink(_taskId: u64, pathname: u64) -> i64 {
        info!("Unlink: the pathname is {}", Self::GetStr(pathname));
        let ret = unsafe {
            unlink(pathname as *const c_char)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Unlinkat(_taskId: u64, dirfd: i32, pathname: u64, flags: i32) -> i64 {
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

        let ret = unsafe {
            unlinkat(dirfd, pathname as *const c_char, flags)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Mkdir(_taskId: u64, pathname: u64, mode_ : u32) -> i64 {
        info!("Mkdir: the pathname is {}", Self::GetStr(pathname));

        let ret = unsafe {
            mkdir(pathname as *const c_char, mode_ as mode_t)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Mkdirat(_taskId: u64, dirfd: i32, pathname: u64, mode_ : u32, uid: u32, gid: u32) -> i64 {
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

        let ret = unsafe {
            mkdirat(dirfd, pathname as *const c_char, mode_ as mode_t)
        };

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

    pub fn MSync(_taskId: u64, addr: u64, len: usize, flags: i32) -> i64 {
        let ret = unsafe{
            msync(addr as *mut c_void, len, flags)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Uname(_taskId: u64, buff: u64) -> i64 {
        let ret = unsafe{
            uname(buff as *mut utsname)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Umask(_taskId: u64, mask: u32) -> i64 {
        let ret = unsafe{
            umask(mask)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Access(_taskId: u64, pathName: u64, mode: i32) -> i64 {
        info!("Access: the pathName is {}", Self::GetStr(pathName));
        let ret = unsafe{
            access(pathName as *const c_char, mode)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn FAccessAt(_taskId: u64, dirfd: i32, pathname: u64, mode: i32, flags: i32) -> i64 {
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

        let ret = unsafe{
            faccessat(dirfd, pathname as *const c_char, mode, flags)
        };

        return Self::GetRet(ret as i64);
    }

    ///////////end of file operation//////////////////////////////////////////////


    ///////////start of network operation//////////////////////////////////////////////////////////////////

    pub fn Socket(_taskId: u64, domain: i32, type_: i32, protocol: i32) -> i64 {
        let fd = unsafe{
            socket(domain, type_ | SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC, protocol)
        };

        if fd < 0 {
            return Self::GetRet(fd as i64);
        }

        let hostfd = IO_MGR.lock().AddFd(fd, true);
        FD_NOTIFIER.AddFd(fd, Box::new(GuestFd{hostfd: hostfd}));
        URING_MGR.lock().Addfd(fd).unwrap();
        return Self::GetRet(hostfd as i64);
    }

    pub fn SocketPair(_taskId: u64, domain: i32, type_: i32, protocol: i32, socketVect: u64) -> i64 {
        let res = unsafe{
            socketpair(domain, type_ | SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC, protocol, socketVect as *mut i32)
        };

        if res < 0 {
            return Self::GetRet(res as i64);
        }

        let ptr = socketVect as * mut i32;
        let fds = unsafe { slice::from_raw_parts_mut(ptr, 2) };

        let hostfd0 = IO_MGR.lock().AddFd(fds[0], true);
        let hostfd1 = IO_MGR.lock().AddFd(fds[1], true);

        FD_NOTIFIER.AddFd(fds[0], Box::new(GuestFd{hostfd: hostfd0}));
        FD_NOTIFIER.AddFd(fds[1], Box::new(GuestFd{hostfd: hostfd1}));

        fds[0] = hostfd0;
        fds[1] = hostfd1;

        return Self::GetRet(res as i64);
    }

    pub fn GetSockName(_taskId: u64, sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let sockfd = match Self::GetOsfd(sockfd) {
            Some(sockfd) => sockfd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            getsockname(sockfd, addr as *mut sockaddr, addrlen as *mut socklen_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn GetPeerName(_taskId: u64, sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let sockfd = match Self::GetOsfd(sockfd) {
            Some(sockfd) => sockfd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            getpeername(sockfd, addr as *mut sockaddr, addrlen as *mut socklen_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn GetSockOpt(_taskId: u64, sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let sockfd = match Self::GetOsfd(sockfd) {
            Some(sockfd) => sockfd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            getsockopt(sockfd, level, optname, optval as *mut c_void, optlen as *mut socklen_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn SetSockOpt(_taskId: u64, sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let sockfd = match Self::GetOsfd(sockfd) {
            Some(sockfd) => sockfd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            setsockopt(sockfd, level, optname, optval as *const c_void, optlen as socklen_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Bind(_taskId: u64, sockfd: i32, sockaddr: u64, addrlen: u32, umask: u32) -> i64 {
        let sockfd = match Self::GetOsfd(sockfd) {
            Some(sockfd) => sockfd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            // todo: this is not thread safe, need to add lock when implement multiple io threads
            let oldUmask = libc::umask(umask);
            let ret = bind(sockfd, sockaddr as *const sockaddr, addrlen as socklen_t);
            libc::umask(oldUmask);
            ret
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Listen(_taskId: u64, sockfd: i32, backlog: i32) -> i64 {
        let sockfd = match Self::GetOsfd(sockfd) {
            Some(sockfd) => sockfd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            listen(sockfd, backlog)
        };

        return Self::GetRet(ret as i64);
    }

    pub fn Shutdown(_taskId: u64, sockfd: i32, how: i32) -> i64 {
        let sockfd = match Self::GetOsfd(sockfd) {
            Some(sockfd) => sockfd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            shutdown(sockfd, how)
        };

        return Self::GetRet(ret as i64)
    }

    ///////////end of network operation//////////////////////////////////////////////////////////////////
    pub fn SchedGetAffinity(_taskId: u64,  pid: i32, cpuSetSize: u64, mask: u64) -> i64 {
        //todo: fix this
        //let pid = 0;

        let ret = unsafe{
            sched_getaffinity(pid as pid_t, cpuSetSize as size_t, mask as *mut cpu_set_t)
        };

        //todo: fix this.
        if ret == 0 {
            return 8;
        } else {
            Self::GetRet(ret as i64)
        }
    }

    pub fn MinCore(_taskId: u64,  addr: u64, len: u64, vec: u64) -> i64 {
        let ret = unsafe{
            mincore(addr as *mut c_void, len as size_t, vec as *mut c_uchar)
        };

        return Self::GetRet(ret as i64) as i64;
    }

    pub fn Pause() -> QcallRet {
        return QcallRet::Block
    }

    pub fn GetDuration(timeout: u64) -> Duration {
        let timespec = unsafe {
            &*(timeout as *const libc::timespec)
        };

        let dur = Duration::new(timespec.tv_sec as u64, timespec.tv_nsec as u32);
        return dur;
    }

    pub fn Time(tloc: u64) -> i64 {
        //info!("---------------call in SysTime tloc = {:x}", tloc);
        let ret = unsafe{
            time(tloc as *mut time_t)
        };

        //info!("The sysTime ret is {:x}", ret);

        return Self::GetRet(ret as i64);
    }

    pub fn GetTimeOfDay(_taskId: u64, tv: u64, tz: u64) -> i64 {
        //let res = unsafe{ gettimeofday(tv as *mut timeval, tz as *mut timezone) };
        //return Self::GetRet(res as i64)

        let nr = SysCallID::sys_gettimeofday as usize;
        unsafe {
            let res = syscall2(nr, tv as usize, tz as usize) as i64;
            //error!("finish GetTimeOfDay");
            return res
        }
    }

    pub fn ClockGetRes(_taskId: u64, clkId: i32, ts: u64) -> i64 {
        let res = unsafe{ clock_getres(clkId as clockid_t, ts as *mut timespec) };
        return Self::GetRet(res as i64)
    }

    pub fn ClockGetTime(_taskId: u64, clkId: i32, ts: u64) -> i64 {
        let res = unsafe{ clock_gettime(clkId as clockid_t, ts as *mut timespec) };
        return Self::GetRet(res as i64)
    }

    pub fn ClockSetTime(_taskId: u64, clkId: i32, ts: u64) -> i64 {
        let res = unsafe{ clock_getres(clkId as clockid_t, ts as *mut timespec) };
        return Self::GetRet(res as i64)
    }

    pub fn Times(_taskId: u64, tms: u64) -> i64 {
        let res = unsafe {
            times(tms as *mut tms)
        };

        return Self::GetRet(res as i64)
    }

    pub fn GetRandom(&mut self, _taskId: u64, buf: u64, len: u64, _flags: u32) -> i64 {
        unsafe {
            let slice = slice::from_raw_parts_mut(buf as *mut u8, len as usize);
            self.rng.Fill(slice);
        }

        return len as i64;
    }

    pub fn Chdir(_taskId: u64, path: u64) -> i64 {
        let ret = unsafe {
            chdir(path as *const c_char)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Fchdir(_taskId: u64, fd: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            fchdir(fd)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Fadvise(_taskId: u64, fd: i32, offset: u64, len: u64, advice: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            posix_fadvise(fd, offset as i64, len as i64, advice)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Prctl(_taskId: u64, option: i32, arg2:u64, arg3 :u64, arg4 :u64, arg5: u64) -> i64 {
        let ret = unsafe {
            prctl(option, arg2, arg3, arg4, arg5)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Mlock(_taskId: u64, addr: u64, len: u64) -> i64 {
        let ret = unsafe {
            mlock(addr as *const c_void, len as size_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn MUnlock(_taskId: u64, addr: u64, len: u64) -> i64 {
        let ret = unsafe {
            munlock(addr as *const c_void, len as size_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Rename(_taskId: u64, oldpath: u64, newpath: u64) -> i64 {
        let ret = unsafe {
            rename(oldpath as *const c_char, newpath as *const c_char)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Rmdir(_taskId: u64, pathname: u64) -> i64 {
        let ret = unsafe {
            rmdir(pathname as *const c_char)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Chown(_taskId: u64, pathname: u64, owner: u32, group: u32) -> i64 {
        info!("Chown: the pathname is {}", Self::GetStr(pathname));

        let ret = unsafe {
            chown(pathname as *const c_char, owner, group)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn FChown(_taskId: u64, fd: i32, owner: u32, group: u32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            fchown(fd, owner, group)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn TimerFdCreate(_taskId: u64, clockId: i32, flags: i32) -> i64 {
        let fd = unsafe {
            timerfd_create(clockId, flags)
        };

        if fd < 0 {
            return Self::GetRet(fd as i64)
        }

        let guestfd = IO_MGR.lock().AddFd(fd, true);

        return guestfd as i64
    }

    pub fn TimerFdSetTime(_taskId: u64, fd: i32, flags: i32, newValue: u64, oldValue: u64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            timerfd_settime(fd, flags, newValue as *const itimerspec, oldValue as *mut itimerspec)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn TimerFdGetTime(_taskId: u64, fd: i32, currVal: u64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            timerfd_gettime(fd, currVal as *mut itimerspec)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Chmod(_taskId: u64, pathname: u64, mode: u32) -> i64 {
        let ret = unsafe {
            chmod(pathname as *const c_char, mode as mode_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Fchmod(_taskId: u64, fd: i32, mode: u32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            fchmod(fd, mode as mode_t)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn WaitFD(fd: i32, mask: u32) -> i64 {
        let osfd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        match FD_NOTIFIER.WaitFd(osfd, mask) {
            Ok(()) => return 0,
            Err(Error::SysError(syserror)) => return -syserror as i64,
            Err(e) => {
                panic!("WaitFD get error {:?}", e);
            }
        }
    }

    pub fn NonBlockingPoll(_taskId: u64, fd: i32, mask: u32) -> i64 {
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
            let ret = unsafe {
                poll(&mut e, 1, 0)
            };

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

    pub fn NewTmpfile(addr: u64) -> i64 {
        let file = match tempfile() {
            Err(e) => {
                error!("create tempfs file fail with error {:?}", e);
                return -SysErr::ENOENT as i64;
            }
            Ok(f) => f,
        };

        //take the ownership of the fd
        let fd = file.into_raw_fd();

        let ret = unsafe {
            fstat(fd, addr as * mut stat)
        };

        if ret < 0 {
            unsafe {
                close(fd);
            }

            return Self::GetRet(ret as i64);
        }

        let guestfd = IO_MGR.lock().AddFd(fd, false);

        return guestfd as i64
    }

    pub fn NewFifo() -> i64 {
        let uid = NewUID();
        let path = format!("/tmp/fifo_{}", uid);
        let cstr = CString::New(&path);
        let ret = unsafe {
            mkfifo(cstr.Ptr() as *const c_char, 0o666)
        };

        error!("NewFifo apth is {}, id is {}", path, ret);

        if ret < 0 {
            return Self::GetRet(ret as i64);
        }

        return uid as i64;
    }

    pub fn NewTmpfsFile(_taskId: u64, typ: TmpfsFileType, addr: u64) -> i64 {
        match typ {
            TmpfsFileType::File => Self::NewTmpfile(addr),
            TmpfsFileType::Fifo => {
                // Self::NewFifo()
                panic!("NewTmpfsFile doesn't support fifo");
            },
        }
    }

    pub fn OpenFifo(_taskId: u64, uid: u64, flags: i32) -> i64 {
        let path = format!("/tmp/fifo_{}", uid);
        error!("OpenFifo path is {}, flag is {:x}", &path, flags);
        let cstr = CString::New(&path);
        let osfd = unsafe {
            open(cstr.Ptr() as *const c_char, flags)
        };

        {
            let flags = unsafe {
                fcntl(osfd, F_GETFL)
            };

            let flags = flags | Flags::O_NONBLOCK as i32;

            let ret = unsafe {
                fcntl(osfd, F_SETFL, flags)
            };

            if ret == -1 {
                panic!("SetUnblock: can't F_SETFL for fd");
            }
        }

        error!("OpenFifo path is {}, flag is {:x}, osfd is {}", &path, flags, osfd);

        if osfd < 0 {
            return Self::GetRet(osfd as i64);
        }

        let hostfd = IO_MGR.lock().AddFd(osfd, false);
        FD_NOTIFIER.AddFd(osfd, Box::new(GuestFd{hostfd: hostfd}));

        return hostfd as i64;
    }

    pub fn HostID(axArg: u32, cxArg: u32) -> (u32, u32, u32, u32) {
        let ax: u32;
        let bx: u32;
        let cx: u32;
        let dx: u32;
        unsafe {
            llvm_asm!("
              CPUID
            "
            : "={eax}"(ax), "={ebx}"(bx), "={ecx}"(cx), "={edx}"(dx)
            : "{eax}"(axArg), "{ecx}"(cxArg)
            :
            : );
        }

        return (ax, bx, cx, dx)
    }

    pub fn HostCPUInfo(_taskId: u64, axArg: u32, cxArg: u32, addr: u64) -> i64 {
        let (ax, bx, cx, dx) = Self::HostID(axArg, cxArg);

        let CPUIDInfo = unsafe {
            &mut *(addr as * mut CPUIDInfo)
        };

        CPUIDInfo.ax = ax;
        CPUIDInfo.bx = bx;
        CPUIDInfo.cx = cx;
        CPUIDInfo.dx = dx;

        return 0;
    }

    pub fn SetHostName(_taskId: u64, name: u64, len: usize) -> i64 {
        let ret = unsafe {
            sethostname(name as *const c_char, len)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn SymLinkAt(_taskId: u64, oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        let newdirfd = match Self::GetOsfd(newdirfd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            symlinkat(oldpath as *const c_char, newdirfd, newpath as *const c_char)
        };

        return Self::GetRet(ret as i64)
    }

    pub fn Futimens(_taskId: u64, fd: i32, times: u64) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            futimens(fd, times as *const timespec)
        };

        return Self::GetRet(ret as i64)
    }

    //map kernel table
    pub fn KernelMap(&mut self, start: Addr, end: Addr, physical: Addr, flags: PageTableFlags) -> Result<bool> {
        error!("KernelMap start is {:x}, end is {:x}", start.0, end.0);
        return self.pageTables.write().Map(start, end, physical, flags, self.allocator.as_mut().unwrap(), true);
    }

    pub fn PrintStr(phAddr: u64) {
        unsafe {
            info!("the Str: {} ", str::from_utf8_unchecked(slice::from_raw_parts(phAddr as *const u8, strlen(phAddr as *const i8)+1)));
        }
    }

    pub fn UnblockFd(fd: i32) {
        unsafe {
            let flags = fcntl(fd, Cmd::F_GETFL, 0);
            let ret = fcntl(fd, Cmd::F_SETFL, flags | Flags::O_NONBLOCK);
            assert!(ret==0, "UnblockFd fail");
        }
    }

    pub fn GetStdfds(_taskId: u64, addr: u64) -> i64 {
        let ptr = addr as * mut i32;
        let stdfds = unsafe { slice::from_raw_parts_mut(ptr, 3) };

        for i in 0..stdfds.len() {
            let osfd = unsafe {
                dup(i as i32) as i32
            };

            if  osfd < 0 {
                return  osfd as i64
            }

            Self::UnblockFd(osfd);

            let hostfd = IO_MGR.lock().AddFd(osfd, true);
            FD_NOTIFIER.AddFd(osfd, Box::new(GuestFd{hostfd: hostfd}));
            stdfds[i] = hostfd;
        }

        return 0;
    }

    pub fn GetShareSpace(&self) -> &'static ShareSpace {
        return &self.shareSpace
    }

    pub fn FdNotify(&self, fd: i32, mask: u32) {
        self.shareSpace.AQHostInputCall(HostInputMsg::FdNotify(FdNotify{
            fd: fd,
            mask: mask,
        }));
    }

    pub fn LibcFstat(osfd: i32) -> Result<LibcStat> {
        let mut stat = LibcStat::default();
        let ret = unsafe {
            fstat(osfd, &mut stat as * mut _ as u64 as * mut stat)
        };

        if ret < 0 {
            info!("can't fstat osfd {}", osfd);
            return Err(Error::SysError(errno::errno().0))
        }

        //Self::LibcStatx(osfd);

        return Ok(stat)
    }

    pub fn LibcStatx(osfd: i32) {
        let statx = Statx::default();
        let addr : i8 = 0;
        let ret = unsafe {
            libc::statx(osfd, &addr as *const c_char, libc::AT_EMPTY_PATH, libc::STATX_BASIC_STATS, &statx as * const _ as u64 as * mut statx)
        };

        error!("LibcStatx osfd is {} ret is {} error is {}", osfd, ret, errno::errno().0);
    }

    pub fn Init(start: u64, len: u64) -> Self {
        error!("vmspace init start is {:x}, len is {:x}", start, len);
        PMA_KEEPER.lock().Reset(start, len);

        return VMSpace {
            allocator: None,
            pageTables: PageTables::default(),
            hostAddrTop: 0,
            sharedLoasdOffset: 0x0000_5555_0000_0000,
            vdsoAddr: 0,
            shareSpace: unsafe {
                &mut *(0 as * mut ShareSpace)
            },
            rng: RandGen::Init(),
            args: None,
            pivot: false,
            waitingMsgCall: None,
            controlMsgCallBack: BTreeMap::new(),
            controlMsgQueue: VecDeque::with_capacity(5),
        }
    }
}

pub fn SendControlMsg(usock: USocket, msg: ControlMsg) -> Result<()> {
    VMS.lock().SendControlMsg(usock, msg)?;

    return Ok(())
}