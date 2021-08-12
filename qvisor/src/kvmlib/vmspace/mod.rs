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
use std::fs;
use libc::*;
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
use super::kvm_vcpu::SimplePageAllocator;

const ARCH_SET_GS:u64 = 0x1001;
const ARCH_SET_FS:u64 = 0x1002;
const ARCH_GET_FS:u64 = 0x1003;
const ARCH_GET_GS:u64 = 0x1004;

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
    pub taskId: TaskIdQ,
    pub addr: u64,
    pub len: usize,
    pub retAddr: u64,
}

pub struct VMSpace {
    pub pageTables : PageTables,
    pub allocator: Option<SimplePageAllocator>,
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

    pub fn GetDents64(_taskId: u64, fd: i32, dirp: u64, count: u32) -> i64 {
        let fd = Self::GetOsfd(fd).expect("GetDents64");

        let nr = SysCallID::sys_getdents64 as usize;


        //info!("sys_getdents64 is {}", nr);
        unsafe {
            return syscall3(nr, fd as usize, dirp as usize, count as usize) as i64;
        }
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
        let waitMsg = match self.waitingMsgCall.take() {
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

    pub fn CreateMemfd(_taskId: u64, len: i64) -> i64 {
        let uid = NewUID();
        let path = format!("/tmp/memfd_{}", uid);
        let cstr = CString::New(&path);

        let nr = SysCallID::sys_memfd_create as usize;
        let fd = unsafe {
            syscall2(nr, cstr.Ptr() as *const c_char as usize, 0) as i32
        };

        if fd < 0 {
            return Self::GetRet(fd as i64)
        }

        let ret = unsafe {
            ftruncate(fd, len)
        };

        if ret < 0 {
            unsafe {
                libc::close(fd);
            }
            return Self::GetRet(ret as i64)
        }

        let hostfd = IO_MGR.lock().AddFd(fd, true);
        FD_NOTIFIER.AddFd(fd, Box::new(GuestFd{hostfd: hostfd}));
        return hostfd as i64
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

    pub unsafe fn TryOpenHelper(dirfd: i32, name: u64) -> (i32, bool) {
        let flags = Flags::O_NOFOLLOW;
        let ret = libc::openat(dirfd, name as *const c_char, (flags | Flags::O_RDWR) as i32, 0);
        if ret > 0 {
            return (ret, true);
        }

        let err = Self::GetRet(ret as i64) as i32;
        if err == -SysErr::ENOENT {
            return (-SysErr::ENOENT, false)
        }

        let ret = libc::openat(dirfd, name as *const c_char, (flags | Flags::O_RDONLY) as i32, 0);
        if ret > 0 {
            return (ret, false);
        }

        let ret = libc::openat(dirfd, name as *const c_char, (flags | Flags::O_WRONLY) as i32, 0);
        if ret > 0 {
            return (ret, true);
        }

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

    pub fn IOTTYRead(_taskId: u64, fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let fd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe{
            let opt : i32 = 1;
            // in some cases, tty read will blocked even after set unblock with fcntl
            // todo: this workaround, fix this
            ioctl(fd, FIONBIO, &opt);

            readv(fd as c_int, iovs as *const iovec, iovcnt) as i64
        };

        unsafe {
            let opt : i32 = 0;
            ioctl(fd, FIONBIO, &opt);
        }

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

    pub fn SysSync(_taskId: u64) -> i64 {
        // as quark running inside container, assume sys_sync only works for the current fs namespace
        // todo: confirm this
        unsafe {
            libc::sync()
        };

        return 0;
    }

    pub fn SyncFs(_taskId: u64, fd: i32) -> i64 {
        let osfd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            libc::syncfs(osfd) as i64
        };

        return Self::GetRet(ret);
    }

    pub fn SyncFileRange(_taskId: u64, fd: i32, offset: i64, nbytes: i64, flags: u32) -> i64 {
        let osfd = match Self::GetOsfd(fd) {
            Some(fd) => fd,
            None => return -SysErr::EBADF as i64,
        };

        let ret = unsafe {
            libc::sync_file_range(osfd, offset, nbytes, flags) as i64
        };

        return Self::GetRet(ret);
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


    pub fn BatchFstatat(_taskId: u64, addr: u64, count: usize) -> i64 {
        let mut stat: LibcStat = Default::default();

        let ptr = addr as * mut FileType;
        let filetypes = unsafe { slice::from_raw_parts_mut(ptr, count) };

        for ft in filetypes {
            let dirfd = {
                if ft.dirfd > 0 {
                    Self::GetOsfd(ft.dirfd).expect("Fstatat")
                } else {
                    ft.dirfd
                }
            };

            let ret = unsafe {
                Self::GetRet(libc::fstatat(dirfd, ft.pathname as *const c_char, &mut stat as *mut _ as u64 as *mut stat, AT_SYMLINK_NOFOLLOW) as i64)
            };

            ft.mode = stat.st_mode;
            ft.device = stat.st_dev;
            ft.inode = stat.st_ino;
            ft.ret = ret as i32;
        }

        return 0;
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

    pub fn Fstatfs(_taskId: u64, fd: i32, buf: u64) -> i64 {
        let fd = Self::GetOsfd(fd).expect("Fstatfs");

        let ret = unsafe{
            fstatfs(fd, buf as *mut statfs)
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

    pub fn MAdvise(_taskId: u64, addr: u64, len: usize, advise: i32) -> i64 {
        let ret = unsafe{
            madvise(addr as *mut c_void, len, advise)
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

    pub fn GetRandom(&mut self, _taskId: u64, buf: u64, len: u64, _flags: u32) -> i64 {
        unsafe {
            let slice = slice::from_raw_parts_mut(buf as *mut u8, len as usize);
            self.rng.Fill(slice);
        }

        return len as i64;
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

    pub fn Mlock2(_taskId: u64, addr: u64, len: u64, flags: u32) -> i64 {
        let nr = SysCallID::sys_mlock2 as usize;
        let ret = unsafe {
            syscall3(nr, addr as usize, len as usize, flags as usize) as i64
        };

        return Self::GetRet(ret as i64)
    }

    pub fn MUnlock(_taskId: u64, addr: u64, len: u64) -> i64 {
        let ret = unsafe {
            munlock(addr as *const c_void, len as size_t)
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

    pub fn WaitFD(fd: i32, mask: EventMask) -> i64 {
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

    pub fn NonBlockingPoll(_taskId: u64, fd: i32, mask: EventMask) -> i64 {
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

    pub fn Statm(_taskId: u64, buf: u64) -> i64 {
        const STATM : &str = "/proc/self/statm";
        let contents = fs::read_to_string(STATM)
            .expect("Something went wrong reading the file");

        let output = scan!(&contents, char::is_whitespace, u64, u64);
        let mut statm = unsafe {
            &mut *(buf as * mut StatmInfo)
        };

        statm.vss = output.0.unwrap();
        statm.rss = output.1.unwrap();
        return 0;
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
        return self.pageTables.Map(start, end, physical, flags, self.allocator.as_mut().unwrap(), true);
    }

    pub fn KernelMapHugeTable(&mut self, start: Addr, end: Addr, physical: Addr, flags: PageTableFlags) -> Result<bool> {
        error!("KernelMap1G start is {:x}, end is {:x}", start.0, end.0);
        return self.pageTables.MapWith1G(start, end, physical, flags, self.allocator.as_mut().unwrap(), true);
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

    pub fn FdNotify(&self, fd: i32, mask: EventMask) {
        self.shareSpace.AQHostInputCall(&HostInputMsg::FdNotify(FdNotify{
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