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

use alloc::sync::Arc;
use libc::*;
use spin::Mutex;

use crate::qlib::fileinfo::*;
use crate::qlib::kernel::GlobalIOMgr;
use crate::qlib::socket_buf::*;

use super::super::super::util::*;
use super::super::qlib::common::*;
use super::super::qlib::rdmasocket::*;
use super::super::FD_NOTIFIER;
use super::super::*;
// use super::socket_info::*;

impl FdInfo {
    pub fn SockInfo(&self) -> SockInfo {
        return self.lock().sockInfo.lock().clone();
    }

    pub fn BufWrite(fd: i32, addr: u64, len: usize, offset: isize) -> i64 {
        let ret = unsafe {
            if offset < 0 {
                write(fd as c_int, addr as *const c_void, len as size_t)
            } else {
                pwrite(
                    fd as c_int,
                    addr as *const c_void,
                    len as size_t,
                    offset as off_t,
                )
            }
        };

        return SysRet(ret as i64);
    }

    pub fn ReadDir(dirfd: i32, addr: u64, len: usize, reset: bool) -> i64 {
        if reset {
            let res = unsafe { libc::lseek(dirfd, 0, SeekWhence::SEEK_SET) as i32 };

            let res = SysRet(res as i64);
            if res < 0 {
                if -res == SysErr::ESPIPE as i64 {
                    return -SysErr::ENOTDIR as i64;
                }

                return res;
            }
        }

        let ret = Self::GetDents64(dirfd, addr, len as u32);
        return ret;
    }

    pub fn GetDents64(fd: i32, dirp: u64, count: u32) -> i64 {
        let nr = SysCallID::sys_getdents64 as usize;

        unsafe {
            return syscall3(nr, fd as usize, dirp as usize, count as usize) as i64;
        }
    }

    pub fn Write(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let ret = unsafe { writev(fd as c_int, iovs as *const iovec, iovcnt) as i64 };

        return SysRet(ret as i64);
    }

    pub fn Append(fd: i32, iovs: u64, iovcnt: i32, fileLenAddr: u64) -> i64 {
        let end = unsafe { lseek(fd as c_int, 0, libc::SEEK_END) };

        if end < 0 {
            panic!("IOAppend lseek fail")
        }

        let size = unsafe {
            //todo: don't know why RWF_APPEND doesn't work. need to fix.
            //syscall5(nr, fd as usize, iovs as usize, iovcnt as usize, -1 as i32 as usize, Flags::RWF_APPEND as usize) as i64
            pwritev(fd as c_int, iovs as *const iovec, iovcnt, end as i64) as i64
        };

        //error!("IOAppend: end is {:x}, size is {:x}, new end is {:x}", end, size, end + size);
        if size < 0 {
            return SysRet(size as i64);
        }

        unsafe { *(fileLenAddr as *mut i64) = (end + size) as i64 }

        return size;

        // the pwritev2 doesn't work. It will bread the bazel build.
        // Todo: root cause this.
        /*let fd = self.lock().fd;

        let size = unsafe{
            pwritev2(fd as c_int, iovs as *const iovec, iovcnt, -1, Flags::RWF_APPEND) as i64
        };

        if size < 0 {
            return SysRet(size as i64)
        }

        let end = unsafe {
            lseek(fd as c_int, 0, libc::SEEK_END)
        };

        unsafe {
            *(fileLenAddr as * mut i64) = end as i64
        }

        return size as i64*/
    }

    pub fn ReadAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let ret = unsafe {
            if offset as i64 == -1 {
                readv(fd as c_int, iovs as *const iovec, iovcnt) as i64
            } else {
                preadv(fd as c_int, iovs as *const iovec, iovcnt, offset as i64) as i64
            }
        };

        return SysRet(ret as i64);
    }

    pub fn WriteAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let ret = unsafe {
            if offset as i64 == -1 {
                writev(fd as c_int, iovs as *const iovec, iovcnt) as i64
            } else {
                pwritev(fd as c_int, iovs as *const iovec, iovcnt, offset as i64) as i64
            }
        };

        return SysRet(ret as i64);
    }

    pub fn IoCtl(fd: i32, cmd: u64, argp: u64) -> i64 {
        //todo: fix this.
        /* when run /bin/bash, the second command as below return ENOTTY. Doesn't know why
        ioctl(0, TCGETS, {B38400 opost isig icanon echo ...}) = 0
        ioctl(2, TCGETS, 0x7ffdf82a09a0)        = -1 ENOTTY (Inappropriate ioctl for device)
        ioctl(-1, TIOCGPGRP, 0x7ffdf82a0a14)    = -1 EBADF (Bad file descriptor)
        */
        if fd == 2 {
            return -SysErr::ENOTTY as i64;
        }

        //error!("IoCtl fd is {}, cmd is {:x}, argp is {:x}", fd, cmd, argp);

        let ret = unsafe { ioctl(fd as c_int, cmd, argp) };

        return SysRet(ret as i64);
    }

    pub fn FSync(fd: i32, dataSync: bool) -> i64 {
        let ret = if dataSync {
            unsafe { fsync(fd) }
        } else {
            unsafe { fdatasync(fd) }
        };

        return SysRet(ret as i64);
    }

    pub fn Seek(fd: i32, offset: i64, whence: i32) -> i64 {
        let ret = unsafe { libc::lseek(fd, offset, whence) };

        return SysRet(ret as i64);
    }

    pub fn FSetXattr(fd: i32, name: u64, value: u64, size: usize, flags: u32) -> i64 {
        let ret = unsafe { libc::fsetxattr(fd, name as _, value as _, size, flags as _) };
        return SysRet(ret as i64);
    }

    pub fn FGetXattr(fd: i32, name: u64, value: u64, size: usize) -> i64 {
        let ret = unsafe { libc::fgetxattr(fd, name as _, value as _, size) };
        return SysRet(ret as i64);
    }

    pub fn FRemoveXattr(fd: i32, name: u64) -> i64 {
        let ret = unsafe { libc::fremovexattr(fd, name as _) };
        return SysRet(ret as i64);
    }

    pub fn FListXattr(fd: i32, list: u64, size: usize) -> i64 {
        let ret = unsafe { libc::flistxattr(fd, list as _, size) };
        return SysRet(ret as i64);
    }

    ///////////////////////////socket operation//////////////////////////////
    pub fn Accept(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let newfd = unsafe {
            accept4(
                sockfd,
                addr as *mut sockaddr,
                addrlen as *mut socklen_t,
                SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC,
            )
        };

        if newfd < 0 {
            return SysRet(newfd as i64);
        }

        let hostfd = GlobalIOMgr().AddSocket(newfd);
        return SysRet(hostfd as i64);
    }

    pub fn Connect(sockfd: i32, addr: u64, addrlen: u32) -> i64 {
        let ret = unsafe { connect(sockfd, addr as *const sockaddr, addrlen as socklen_t) };

        return SysRet(ret as i64);
    }

    pub fn RecvMsg(sockfd: i32, msghdr: u64, flags: i32) -> i64 {
        let ret = unsafe { recvmsg(sockfd, msghdr as *mut msghdr, flags as c_int) };

        return SysRet(ret as i64);
    }

    pub fn Recvfrom(sockfd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u64) -> i64 {
        let ret = unsafe { recvfrom(sockfd, buf as _, size, flags, addr as _, len as _) };
        return SysRet(ret as i64);
    }

    pub fn SendMsg(sockfd: i32, msghdr: u64, flags: i32) -> i64 {
        let ret = unsafe { sendmsg(sockfd, msghdr as *mut msghdr, flags as c_int) };
        return SysRet(ret as i64);
    }

    pub fn Sendto(sockfd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u32) -> i64 {
        let ret = unsafe { sendto(sockfd, buf as _, size, flags, addr as _, len) };
        return SysRet(ret as i64);
    }

    pub fn GetSockName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let ret = unsafe { getsockname(sockfd, addr as *mut sockaddr, addrlen as *mut socklen_t) };

        return SysRet(ret as i64);
    }

    pub fn GetPeerName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let ret = unsafe { getpeername(sockfd, addr as *mut sockaddr, addrlen as *mut socklen_t) };

        return SysRet(ret as i64);
    }

    pub fn GetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let ret = unsafe {
            getsockopt(
                sockfd,
                level,
                optname,
                optval as *mut c_void,
                optlen as *mut socklen_t,
            )
        };

        return SysRet(ret as i64);
    }

    pub fn SetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let ret = unsafe {
            setsockopt(
                sockfd,
                level,
                optname,
                optval as *const c_void,
                optlen as socklen_t,
            )
        };

        return SysRet(ret as i64);
    }

    pub fn Bind(sockfd: i32, sockaddr: u64, addrlen: u32, umask: u32) -> i64 {
        // use global lock to avoid race condition
        //let _ = GLOCK.lock();
        let ret = unsafe {
            let oldUmask = libc::umask(umask);
            let ret = bind(sockfd, sockaddr as *const sockaddr, addrlen as socklen_t);
            libc::umask(oldUmask);
            ret
        };

        return SysRet(ret as i64);
    }

    pub fn Listen(sockfd: i32, backlog: i32, block: bool) -> i64 {
        let ret = unsafe { listen(sockfd, backlog) };

        if block {
            VMSpace::BlockFd(sockfd);
        }

        return SysRet(ret as i64);
    }

    pub fn Shutdown(sockfd: i32, how: i32) -> i64 {
        let ret = unsafe { shutdown(sockfd, how) };

        return SysRet(ret as i64);
    }

    ///////////////////////////socket operation//////////////////////////////
}

impl FdInfo {
    pub fn NewFile(fd: i32) -> Self {
        return Self(Arc::new(Mutex::new(FdInfoIntern::NewFile(fd))));
    }

    pub fn NewSocket(fd: i32) -> Self {
        return Self(Arc::new(Mutex::new(FdInfoIntern::NewSocket(fd))));
    }

    pub fn Notify(&self, mask: EventMask) {
        let sockInfo = self.SockInfo();
        sockInfo.Notify(mask, self.WaitInfo());
    }

    pub fn WaitInfo(&self) -> FdWaitInfo {
        return self.lock().waitInfo.clone();
    }

    pub fn Fd(&self) -> i32 {
        return self.lock().fd;
    }

    pub fn IOReadDir(&self, addr: u64, len: usize, reset: bool) -> i64 {
        let fd = self.lock().fd;
        return Self::ReadDir(fd, addr, len, reset);
    }

    pub fn IOBufWrite(&self, addr: u64, len: usize, offset: isize) -> i64 {
        let fd = self.lock().fd;
        return Self::BufWrite(fd, addr, len, offset);
    }

    pub fn IOWrite(&self, iovs: u64, iovcnt: i32) -> i64 {
        let fd = self.lock().fd;
        return Self::Write(fd, iovs, iovcnt);
    }

    pub fn IOAppend(&self, iovs: u64, iovcnt: i32, fileLenAddr: u64) -> i64 {
        let fd = self.lock().fd;
        return Self::Append(fd, iovs, iovcnt, fileLenAddr);
    }

    pub fn IOReadAt(&self, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let fd = self.lock().fd;
        return Self::ReadAt(fd, iovs, iovcnt, offset);
    }

    pub fn IOWriteAt(&self, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let fd = self.lock().fd;
        return Self::WriteAt(fd, iovs, iovcnt, offset);
    }

    pub fn IOFcntl(&self, cmd: i32, arg: u64) -> i64 {
        assert!(
            cmd == Cmd::F_GETFL || cmd == Cmd::F_GET_SEALS || cmd == Cmd::F_ADD_SEALS,
            "we only support Cmd::F_GETFL in Fcntl"
        );
        if cmd == Cmd::F_GETFL {
            return self.lock().GetFlags() as i64;
        } else {
            let fd = self.lock().fd;
            let ret = unsafe { fcntl(fd, cmd, arg) };
            return SysRet(ret as i64);
        }
    }

    pub fn IOIoCtl(&self, cmd: u64, argp: u64) -> i64 {
        let fd = self.lock().fd;
        return Self::IoCtl(fd, cmd, argp);
    }

    pub fn IOFSync(&self, dataSync: bool) -> i64 {
        let fd = self.lock().fd;
        return Self::FSync(fd, dataSync);
    }

    pub fn IOSeek(&self, offset: i64, whence: i32) -> i64 {
        let fd = self.lock().fd;
        return Self::Seek(fd, offset, whence);
    }

    pub fn IOFSetXattr(&self, name: u64, value: u64, size: usize, flags: u32) -> i64 {
        let fd = self.lock().fd;
        return Self::FSetXattr(fd, name, value, size, flags);
    }

    pub fn IOFGetXattr(&self, name: u64, value: u64, size: usize) -> i64 {
        let fd = self.lock().fd;
        return Self::FGetXattr(fd, name, value, size);
    }

    pub fn IOFRemoveXattr(&self, name: u64) -> i64 {
        let fd = self.lock().fd;
        return Self::FRemoveXattr(fd, name);
    }

    pub fn IOFListXattr(&self, list: u64, size: usize) -> i64 {
        let fd = self.lock().fd;
        return Self::FListXattr(fd, list, size);
    }

    ///////////////////////////socket operation//////////////////////////////
    pub fn IOAccept(&self, addr: u64, addrlen: u64) -> i64 {
        let fd = self.lock().fd;
        return Self::Accept(fd, addr, addrlen);
    }

    pub fn IOConnect(&self, addr: u64, addrlen: u32) -> i64 {
        let fd = self.lock().fd;
        return Self::Connect(fd, addr, addrlen);
    }

    pub fn IORecvMsg(&self, msghdr: u64, flags: i32) -> i64 {
        let fd = self.lock().fd;
        return Self::RecvMsg(fd, msghdr, flags);
    }

    pub fn IORecvfrom(&self, buf: u64, size: usize, flags: i32, addr: u64, len: u64) -> i64 {
        let fd = self.lock().fd;
        return Self::Recvfrom(fd, buf, size, flags, addr, len);
    }

    pub fn IOSendMsg(&self, msghdr: u64, flags: i32) -> i64 {
        let fd = self.lock().fd;
        return Self::SendMsg(fd, msghdr, flags);
    }

    pub fn IOSendto(&self, buf: u64, size: usize, flags: i32, addr: u64, len: u32) -> i64 {
        let fd = self.lock().fd;
        return Self::Sendto(fd, buf, size, flags, addr, len);
    }

    pub fn IOGetSockName(&self, addr: u64, addrlen: u64) -> i64 {
        let sockfd = self.lock().fd;
        return Self::GetSockName(sockfd, addr, addrlen);
    }

    pub fn IOGetPeerName(&self, addr: u64, addrlen: u64) -> i64 {
        let sockfd = self.lock().fd;
        return Self::GetPeerName(sockfd, addr, addrlen);
    }

    pub fn IOGetSockOpt(&self, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let sockfd = self.lock().fd;
        return Self::GetSockOpt(sockfd, level, optname, optval, optlen);
    }

    pub fn IOSetSockOpt(&self, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let sockfd = self.lock().fd;
        return Self::SetSockOpt(sockfd, level, optname, optval, optlen);
    }

    pub fn IOBind(&self, sockaddr: u64, addrlen: u32, umask: u32) -> i64 {
        let sockfd = self.lock().fd;
        return Self::Bind(sockfd, sockaddr, addrlen, umask);
    }

    pub fn IOListen(&self, backlog: i32, block: bool) -> i64 {
        let sockfd = self.lock().fd;
        return Self::Listen(sockfd, backlog, block);
    }

    pub fn RDMAListen(&self, backlog: i32, block: bool, acceptQueue: AcceptQueue) -> i64 {
        let sockfd = self.lock().fd;
        let ret = Self::Listen(sockfd, backlog, block);
        if ret < 0 {
            return errno::errno().0 as _;
        }

        match self.SockInfo() {
            SockInfo::Socket(_socketInfo) => {
                //TODO: should double check this is needed or not
                let rdmaSocket = RDMAServerSock::New(sockfd as u32, acceptQueue, 0, 0);
                *self.lock().sockInfo.lock() = SockInfo::RDMAServerSocket(rdmaSocket);
                self.lock()
                    .AddWait(EVENT_READ | EVENT_WRITE)
                    .expect("RDMAListen EpollCtlAdd fail");
                // the accept4 with SOCK_NONBLOCK doesn't work, have to fcntl it to unblock
                super::super::VMSpace::UnblockFd(sockfd);
            }
            _ => {
                error!(
                    "RDMAListen listen fail with wrong state {:?}",
                    self.SockInfo()
                );
                return -SysErr::EINVAL as i64;
            }
        }
        return 0;
    }

    pub fn ProcessRDMAWriteImmFinish(&self) {
        match self.SockInfo() {
            SockInfo::RDMADataSocket(sock) => sock.ProcessRDMAWriteImmFinish(self.WaitInfo()),
            _ => {
                panic!(
                    "ProcessRDMAWriteImmFinish get unexpected socket {:?}",
                    self.SockInfo()
                )
            }
        }
    }

    pub fn ProcessRDMARecvWriteImm(&self, recvCount: u64, writeCount: u64) {
        match self.SockInfo() {
            SockInfo::RDMADataSocket(sock) => {
                sock.ProcessRDMARecvWriteImm(recvCount, writeCount, self.WaitInfo())
            }
            _ => {
                panic!(
                    "ProcessRDMARecvWriteImm get unexpected socket {:?}",
                    self.SockInfo()
                )
            }
        }
    }

    pub fn RDMANotify(&self, typ: RDMANotifyType) -> i64 {
        match self.SockInfo() {
            SockInfo::RDMAServerSocket(RDMAServerSock) => {
                RDMAServerSock.Accept(self.WaitInfo());
            }
            SockInfo::RDMADataSocket(sock) => {
                match typ {
                    RDMANotifyType::Read => {
                        sock.Notify(EVENT_IN, self.WaitInfo());
                        //self.lock().AddWait(EVENT_READ).unwrap();
                    }
                    RDMANotifyType::Write => {
                        sock.Notify(EVENT_OUT, self.WaitInfo());
                        //self.lock().AddWait(EVENT_WRITE).unwrap();
                    }
                    RDMANotifyType::RDMARead => {
                        sock.RDMARead();
                        //self.lock().AddWait(EVENT_WRITE).unwrap();
                    }
                    RDMANotifyType::RDMAWrite => {
                        sock.RDMAWrite();
                        //self.lock().AddWait(EVENT_WRITE).unwrap();
                    }
                    _ => {
                        panic!("RDMANotify wrong state {:?}", typ);
                    }
                }
            }
            _ => {
                error!(
                    "RDMAListen RDMANotify fail with wrong state {:?}",
                    self.SockInfo()
                );
            }
        }

        return 0;
    }

    pub fn PostRDMAConnect(&self, msg: &mut PostRDMAConnect) {
        debug!("FdInfo::PostRDMAConnect");
        let sockfd = self.Fd();
        match self.SockInfo() {
            SockInfo::Socket(_) => {
                let sockBuf = msg.socketBuf.clone();
                let _rdmaType = if RDMA_ENABLE {
                    let addr = msg as *const _ as u64;
                    RDMAType::Client(addr)
                } else {
                    RDMAType::None
                };

                let rdmaSocket = RDMADataSock::New(sockfd as u32, sockBuf, 1, 0, 0, 0, 0);
                *self.lock().sockInfo.lock() = SockInfo::RDMADataSocket(rdmaSocket);
                self.lock()
                    .AddWait(EVENT_READ | EVENT_WRITE)
                    .expect("RDMAListen EpollCtlAdd fail");

                // the accept4 with SOCK_NONBLOCK doesn't work, have to fcntl it to unblock
                super::super::VMSpace::UnblockFd(sockfd);
            }
            _ => {
                error!(
                    "PostRDMAConnect fail with wrong state {:?}",
                    self.SockInfo()
                );
            }
        }

        if !RDMA_ENABLE {
            msg.Finish(0)
        }
    }

    pub fn IOShutdown(&self, how: i32) -> i64 {
        let sockfd = self.lock().fd;
        return Self::Shutdown(sockfd, how);
    }

    ///////////////////////////socket operation//////////////////////////////
}

impl Drop for FdInfoIntern {
    fn drop(&mut self) {
        //error!("in fdInfo drop: guest fd is {}, fd is {}", self.hostfd, self.fd);
        self.Close();
    }
}

impl FdInfoIntern {
    pub fn NewFile(fd: i32) -> Self {
        let flags = unsafe { fcntl(fd, F_GETFL) };

        let res = Self {
            fd: fd,
            waitInfo: FdWaitInfo::default(),
            flags: Flags(flags),
            sockInfo: Mutex::new(SockInfo::File),
        };

        return res;
    }

    pub fn NewSocket(fd: i32) -> Self {
        //info!("New fd {}, hostfd{}: epollable is {}", fd, hostfd, epollable);
        let flags = unsafe { fcntl(fd, F_GETFL) };

        let res = Self {
            fd: fd,
            waitInfo: FdWaitInfo::default(),
            flags: Flags(flags),
            sockInfo: Mutex::new(SockInfo::Socket(SocketInfo::default())),
        };

        return res;
    }

    pub fn Flags(&self) -> Flags {
        return self.flags;
    }

    pub fn GetSockErr(&self) -> Result<u64> {
        let mut err = 0;
        let mut len: u32 = 8;

        let ret = unsafe {
            getsockopt(
                self.fd,
                SOL_SOCKET,
                SO_ERROR,
                &mut err as *mut _ as *mut c_void,
                &mut len as *mut socklen_t,
            )
        };

        if ret == -1 {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(err);
    }

    pub fn Close(&self) -> i32 {
        let _ioMgr = GlobalIOMgr().fdTbl.lock(); //global lock
        if self.fd >= 0 {
            unsafe {
                // shutdown for socket, without shutdown, it the uring read won't be wake up
                // todo: handle this elegant
                shutdown(self.fd, 2);
                // debug!("fdinfo close, fd: {}", self.fd);
                return close(self.fd);
            }
        }

        return 0;
    }

    pub fn GetFlags(&mut self) -> i32 {
        return self.Flags().0;
    }

    pub fn RemoveWait(&mut self, mask: EventMask) -> Result<()> {
        let mask = self.waitInfo.lock().mask & !mask;
        return self.WaitFd(mask);
    }

    pub fn AddWait(&mut self, mask: EventMask) -> Result<()> {
        let mask = self.waitInfo.lock().mask | mask;
        return self.WaitFd(mask);
    }

    pub fn WaitFd(&mut self, mask: EventMask) -> Result<()> {
        let op;
        {
            let mut wi = self.waitInfo.lock();
            if mask == wi.mask {
                return Ok(());
            }

            if wi.mask == 0 {
                op = LibcConst::EPOLL_CTL_ADD;
            } else if mask == 0 {
                op = LibcConst::EPOLL_CTL_DEL;
            } else {
                op = LibcConst::EPOLL_CTL_MOD;
            }

            wi.mask = mask;
        }

        return FD_NOTIFIER.WaitFd(self.fd, op as u32, mask);
    }
}
