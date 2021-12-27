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
use spin::Mutex;
use core::ops::Deref;
use libc::*;

use super::socket_info::*;
use super::rdma_socket::*;
use super::super::*;
use super::super::qlib::common::*;
use super::super::super::util::*;

#[derive(Clone, Debug)]
pub struct FdInfo (pub Arc<Mutex<FdInfoIntern>>);

impl Deref for FdInfo {
    type Target = Arc<Mutex<FdInfoIntern>>;

    fn deref(&self) -> &Arc<Mutex<FdInfoIntern>> {
        &self.0
    }
}

impl FdInfo {
    pub fn SockInfo(&self) -> SockInfo {
        return self.lock().sockInfo.lock().clone();
    }

    pub fn BufWrite(osfd: i32, addr: u64, len: usize, offset: isize) -> i64 {
        let ret = unsafe{
            if offset < 0 {
                write(osfd as c_int, addr as *const c_void, len as size_t)
            } else {
                pwrite(osfd as c_int, addr as *const c_void, len as size_t, offset as off_t)
            }
        };

        return SysRet(ret as i64)
    }

    pub fn Write(osfd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let ret = unsafe {
            writev(osfd as c_int, iovs as *const iovec, iovcnt) as i64
        };

        return SysRet(ret as i64)
    }

    pub fn Append(osfd: i32, iovs: u64, iovcnt: i32, fileLenAddr: u64) -> i64 {
        let end = unsafe {
            lseek(osfd as c_int, 0, libc::SEEK_END)
        };

        if end < 0 {
            panic!("IOAppend lseek fail")
        }

        let size = unsafe{
            //todo: don't know why RWF_APPEND doesn't work. need to fix.
            //syscall5(nr, osfd as usize, iovs as usize, iovcnt as usize, -1 as i32 as usize, Flags::RWF_APPEND as usize) as i64
            pwritev(osfd as c_int, iovs as *const iovec, iovcnt, end as i64) as i64
        };

        //error!("IOAppend: end is {:x}, size is {:x}, new end is {:x}", end, size, end + size);
        if size < 0 {
            return SysRet(size as i64)
        }

        unsafe {
            *(fileLenAddr as * mut i64) = (end + size) as i64
        }

        return size;

        // the pwritev2 doesn't work. It will bread the bazel build.
        // Todo: root cause this.
        /*let osfd = self.lock().osfd;

        let size = unsafe{
            pwritev2(osfd as c_int, iovs as *const iovec, iovcnt, -1, Flags::RWF_APPEND) as i64
        };

        if size < 0 {
            return SysRet(size as i64)
        }

        let end = unsafe {
            lseek(osfd as c_int, 0, libc::SEEK_END)
        };

        unsafe {
            *(fileLenAddr as * mut i64) = end as i64
        }

        return size as i64*/
    }

    pub fn ReadAt(osfd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let ret = unsafe {
            if offset as i64 == -1 {
                readv(osfd as c_int, iovs as *const iovec, iovcnt) as i64
            } else {
                preadv(osfd as c_int, iovs as *const iovec, iovcnt, offset as i64) as i64
            }
        };

        return SysRet(ret as i64)
    }

    pub fn WriteAt(osfd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let ret = unsafe{
            if offset as i64 == -1 {
                writev(osfd as c_int, iovs as *const iovec, iovcnt) as i64
            } else {
                pwritev(osfd as c_int, iovs as *const iovec, iovcnt, offset as i64) as i64
            }
        };

        return SysRet(ret as i64)
    }

    pub fn IoCtl(osfd: i32, cmd: u64, argp: u64) -> i64 {
        //todo: fix this.
        /* when run /bin/bash, the second command as below return ENOTTY. Doesn't know why
        ioctl(0, TCGETS, {B38400 opost isig icanon echo ...}) = 0
        ioctl(2, TCGETS, 0x7ffdf82a09a0)        = -1 ENOTTY (Inappropriate ioctl for device)
        ioctl(-1, TIOCGPGRP, 0x7ffdf82a0a14)    = -1 EBADF (Bad file descriptor)
        */
        if osfd == 2 {
            return -SysErr::ENOTTY as i64
        }

        //error!("IoCtl osfd is {}, cmd is {:x}, argp is {:x}", osfd, cmd, argp);

        let ret = unsafe{
            ioctl(osfd as c_int, cmd, argp)
        };

        return SysRet(ret as i64);
    }

    pub fn FSync(osfd: i32, dataSync: bool) -> i64 {
        let ret = if dataSync {
            unsafe{
                fsync(osfd)
            }
        } else {
            unsafe{
                fdatasync(osfd)
            }
        };

        return SysRet(ret as i64);
    }

    pub fn Seek(osfd: i32, offset: i64, whence: i32) -> i64 {
        let ret = unsafe {
            libc::lseek(osfd, offset, whence)
        };

        return SysRet(ret as i64)
    }

    ///////////////////////////socket operation//////////////////////////////
    pub fn Accept(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let newOsfd = unsafe{
            accept4(sockfd, addr as  *mut sockaddr, addrlen as  *mut socklen_t, SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC)
        };

        if newOsfd < 0 {
            return SysRet(newOsfd as i64);
        }

        let hostfd = IO_MGR.AddSocket(newOsfd);
        URING_MGR.lock().Addfd(newOsfd).unwrap();
        return SysRet(hostfd as i64);
    }

    pub fn Connect(sockfd: i32, addr: u64, addrlen: u32) -> i64 {
        let ret = unsafe{
            connect(sockfd, addr as *const sockaddr, addrlen as socklen_t)
        };

        return SysRet(ret as i64)
    }

    pub fn RecvMsg(sockfd: i32, msghdr: u64, flags: i32) -> i64 {
        let ret = unsafe{
            recvmsg(sockfd, msghdr as *mut msghdr, flags as c_int)
        };

        return SysRet(ret as i64);
    }

    pub fn SendMsg(sockfd: i32, msghdr: u64, flags: i32) -> i64 {
        let ret = unsafe{
            sendmsg(sockfd, msghdr as *mut msghdr, flags as c_int)
        };

        return SysRet(ret as i64);
    }

    pub fn GetSockName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let ret = unsafe{
            getsockname(sockfd, addr as *mut sockaddr, addrlen as *mut socklen_t)
        };

        return SysRet(ret as i64)
    }

    pub fn GetPeerName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let ret = unsafe{
            getpeername(sockfd, addr as *mut sockaddr, addrlen as *mut socklen_t)
        };

        return SysRet(ret as i64)
    }

    pub fn GetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let ret = unsafe{
            getsockopt(sockfd, level, optname, optval as *mut c_void, optlen as *mut socklen_t)
        };

        return SysRet(ret as i64)
    }

    pub fn SetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let ret = unsafe{
            setsockopt(sockfd, level, optname, optval as *const c_void, optlen as socklen_t)
        };

        return SysRet(ret as i64)
    }

    pub fn Bind(sockfd: i32, sockaddr: u64, addrlen: u32, umask: u32) -> i64 {
        // use global lock to avoid race condition
        //let _ = GLOCK.lock();
        let ret = unsafe{
            let oldUmask = libc::umask(umask);
            let ret = bind(sockfd, sockaddr as *const sockaddr, addrlen as socklen_t);
            libc::umask(oldUmask);
            ret
        };

        return SysRet(ret as i64);
    }

    pub fn Listen(sockfd: i32, backlog: i32, block: bool) -> i64 {
        let ret = unsafe{
            listen(sockfd, backlog)
        };

        if block {
            VMSpace::BlockFd(sockfd);
        }

        return SysRet(ret as i64);
    }

    pub fn Shutdown(sockfd: i32, how: i32) -> i64 {
        let ret = unsafe{
            shutdown(sockfd, how)
        };

        return SysRet(ret as i64)
    }

    ///////////////////////////socket operation//////////////////////////////
}

impl FdInfo {
    pub fn NewFile(osfd: i32) -> Self {
        return Self(Arc::new(Mutex::new(FdInfoIntern::NewFile(osfd))))
    }

    pub fn Notify(&self, mask: EventMask) {
        let fd = self.Fd();
        let sockInfo = self.SockInfo();
        sockInfo.Trigger(fd, mask);
    }

    pub fn Fd(&self) -> i32 {
        return self.lock().osfd;
    }

    pub fn NewSocket(osfd: i32) -> Self {
        return Self(Arc::new(Mutex::new(FdInfoIntern::NewSocket(osfd))))
    }

    pub fn IOBufWrite(&self, addr: u64, len: usize, offset: isize) -> i64 {
        let osfd = self.lock().osfd;
        return Self::BufWrite(osfd, addr, len, offset)
    }

    pub fn IOWrite(&self, iovs: u64, iovcnt: i32) -> i64 {
        let osfd = self.lock().osfd;
        return Self::Write(osfd, iovs, iovcnt)
    }

    pub fn IOAppend(&self, iovs: u64, iovcnt: i32, fileLenAddr: u64) -> i64 {
        let osfd = self.lock().osfd;
        return Self::Append(osfd, iovs, iovcnt, fileLenAddr)
    }

    pub fn IOReadAt(&self, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let osfd = self.lock().osfd;
        return Self::ReadAt(osfd, iovs, iovcnt, offset)
    }

    pub fn IOWriteAt(&self, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let osfd = self.lock().osfd;
        return Self::WriteAt(osfd, iovs, iovcnt, offset)
    }

    pub fn IOFcntl(&self, cmd: i32, _arg: u64) -> i64 {
        assert!(cmd == Cmd::F_GETFL, "we only support Cmd::F_GETFL in Fcntl");
        return self.lock().GetFlags() as i64;
    }

    pub fn IOIoCtl(&self, cmd: u64, argp: u64) -> i64 {
        let osfd = self.lock().osfd;
        return Self::IoCtl(osfd, cmd, argp);
    }

    pub fn IOFSync(&self, dataSync: bool) -> i64 {
        let osfd = self.lock().osfd;
        return Self::FSync(osfd, dataSync);
    }

    pub fn IOSeek(&self, offset: i64, whence: i32) -> i64 {
        let osfd = self.lock().osfd;
        return Self::Seek(osfd, offset, whence);
    }

    ///////////////////////////socket operation//////////////////////////////
    pub fn IOAccept(&self, addr: u64, addrlen: u64) -> i64 {
        let osfd = self.lock().osfd;
        return Self::Accept(osfd, addr, addrlen);
    }

    pub fn IOConnect(&self, addr: u64, addrlen: u32) -> i64 {
        let osfd = self.lock().osfd;
        return Self::Connect(osfd, addr, addrlen);
    }

    pub fn IORecvMsg(&self, msghdr: u64, flags: i32) -> i64 {
        let osfd = self.lock().osfd;
        return Self::RecvMsg(osfd, msghdr, flags);
    }

    pub fn IOSendMsg(&self, msghdr: u64, flags: i32) -> i64 {
        let osfd = self.lock().osfd;
        return Self::SendMsg(osfd, msghdr, flags);
    }

    pub fn IOGetSockName(&self, addr: u64, addrlen: u64) -> i64 {
        let sockfd = self.lock().osfd;
        return Self::GetSockName(sockfd, addr, addrlen);
    }

    pub fn IOGetPeerName(&self, addr: u64, addrlen: u64) -> i64 {
        let sockfd = self.lock().osfd;
        return Self::GetPeerName(sockfd, addr, addrlen);
    }

    pub fn IOGetSockOpt(&self, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let sockfd = self.lock().osfd;
        return Self::GetSockOpt(sockfd, level, optname, optval, optlen);
    }

    pub fn IOSetSockOpt(&self, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let sockfd = self.lock().osfd;
        return Self::SetSockOpt(sockfd, level, optname, optval, optlen);
    }

    pub fn IOBind(&self, sockaddr: u64, addrlen: u32, umask: u32) -> i64 {
        let sockfd = self.lock().osfd;
        return Self::Bind(sockfd, sockaddr, addrlen, umask);
    }

    pub fn IOListen(&self, backlog: i32, block: bool) -> i64 {
        let sockfd = self.lock().osfd;
        return Self::Listen(sockfd, backlog, block);
    }

    pub fn RDMAListen(&self, backlog: i32, block: bool, acceptQueue: AcceptQueue) -> i64 {
        let sockfd = self.lock().osfd;
        let ret = Self::Listen(sockfd, backlog, block);
        if ret < 0 {
            return ret;
        }

        match self.SockInfo() {
            SockInfo::Socket => {
                let rdmaSocket = RDMAServerSock::New(sockfd, acceptQueue);
                *self.lock().sockInfo.lock() = SockInfo::RDMAServerSocket(rdmaSocket);
            }
            _ => {
                error!("RDMAListen listen fail with wrong state {:?}", self.SockInfo());
            }
        }

        return 0;
    }

    pub fn IOShutdown(&self, how: i32) -> i64 {
        let sockfd = self.lock().osfd;
        return Self::Shutdown(sockfd, how);
    }

    ///////////////////////////socket operation//////////////////////////////
}

#[derive(Debug)]
pub struct FdInfoIntern {
    pub osfd: i32,

    pub flags: Flags,
    pub sockInfo: Mutex<SockInfo>,
}

impl Drop for FdInfoIntern {
    fn drop(&mut self) {
        //error!("in fdInfo drop: guest fd is {}, osfd is {}", self.hostfd, self.osfd);
        self.Close();
    }
}

impl FdInfoIntern {
    pub fn NewFile(osfd: i32) -> Self {
        //info!("New osfd {}, hostfd{}: epollable is {}", osfd, hostfd, epollable);
        let flags = unsafe {
            fcntl(osfd, F_GETFL)
        };

        let res = Self {
            osfd: osfd,
            flags: Flags(flags),
            sockInfo: Mutex::new(SockInfo::File)
        };

        return res;
    }

    pub fn NewSocket(osfd: i32) -> Self {
        //info!("New osfd {}, hostfd{}: epollable is {}", osfd, hostfd, epollable);
        let flags = unsafe {
            fcntl(osfd, F_GETFL)
        };

        let res = Self {
            osfd: osfd,
            flags: Flags(flags),
            sockInfo: Mutex::new(SockInfo::Socket)
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
            getsockopt(self.osfd, SOL_SOCKET, SO_ERROR, &mut err as *mut _ as *mut c_void, &mut len as *mut socklen_t)
        };

        if ret == -1 {
            return Err(Error::SysError(errno::errno().0))
        }

        return Ok(err);
    }

    pub fn Close(&self) -> i32 {
        let _ioMgr = IO_MGR.fdTbl.lock(); //global lock
        if self.osfd >= 0 {
            unsafe {
                // shutdown for socket, without shutdown, it the uring read won't be wake up
                // todo: handle this elegant
                shutdown(self.osfd, 2);
                return close(self.osfd)
            }
        }

        return 0;
    }

    pub fn GetFlags(&mut self) -> i32 {
        return self.Flags().0
    }
}

pub fn FdNotify(fd: i32, mask: EventMask) {
    SHARE_SPACE.AQHostInputCall(&HostInputMsg::FdNotify(FdNotify{
        fd: fd,
        mask: mask,
    }));
}