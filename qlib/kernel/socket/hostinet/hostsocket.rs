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

use crate::qlib::mutex::*;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;
use core::ptr;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;

use super::super::super::super::common::*;
use super::super::super::super::linux::time::Timeval;
use super::super::super::super::linux_def::*;
use super::super::super::super::socket_buf::*;
use super::super::super::fd::*;
use super::super::super::fs::attr::*;
use super::super::super::fs::dentry::*;
use super::super::super::fs::dirent::*;
use super::super::super::fs::file::*;
use super::super::super::fs::flags::*;
use super::super::super::fs::host::hostinodeop::*;
use super::super::super::guestfdnotifier::*;
use super::super::super::kernel::async_wait::*;
use super::super::super::kernel::fd_table::*;
use super::super::super::kernel::kernel::GetKernel;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::*;
use super::super::super::task::*;
use super::super::super::tcpip::tcpip::*;
use super::super::super::Kernel;
use super::super::super::Kernel::HostSpace;
use super::super::super::IOURING;
use super::super::control::*;
use super::super::socket::*;
use super::socket::*;

pub fn newHostSocketFile(
    task: &Task,
    family: i32,
    fd: i32,
    stype: i32,
    nonblock: bool,
    addr: Option<Vec<u8>>,
) -> Result<File> {
    let dirent = NewSocketDirent(task, SOCKET_DEVICE.clone(), fd)?;
    let inode = dirent.Inode();
    let iops = inode.lock().InodeOp.clone();
    let hostiops = iops.HostInodeOp().unwrap();
    let s = HostSocketOperations::New(family, fd, stype, hostiops.Queue(), hostiops.clone(), addr)?;

    let file = File::New(
        &dirent,
        &FileFlags {
            NonBlocking: nonblock,
            Read: true,
            Write: true,
            ..Default::default()
        },
        s.into(),
    );

    GetKernel().sockets.AddSocket(&file);
    return Ok(file);
}

pub struct HostSocketOperationsIntern {
    pub send: AtomicI64,
    pub recv: AtomicI64,
    pub family: i32,
    pub stype: i32,
    pub fd: i32,
    pub queue: Queue,
    pub remoteAddr: QMutex<Option<SockAddr>>,
    pub hostops: HostInodeOp,
    passInq: AtomicBool,
}

#[derive(Clone)]
pub struct HostSocketOperations(Arc<HostSocketOperationsIntern>);

impl HostSocketOperations {
    pub fn New(
        family: i32,
        fd: i32,
        stype: i32,
        queue: Queue,
        hostops: HostInodeOp,
        addr: Option<Vec<u8>>,
    ) -> Result<Self> {
        let addr = match addr {
            None => None,
            Some(v) => {
                if v.len() == 0 {
                    None
                } else {
                    Some(GetAddr(v[0] as i16, &v[0..v.len()]).unwrap())
                }
            }
        };

        let ret = HostSocketOperationsIntern {
            send: AtomicI64::new(0),
            recv: AtomicI64::new(0),
            family,
            stype,
            fd,
            queue,
            remoteAddr: QMutex::new(addr),
            hostops: hostops,
            passInq: AtomicBool::new(false),
        };

        let ret = Self(Arc::new(ret));
        return Ok(ret);
    }

    pub fn IOAccept(&self) -> Result<AcceptItem> {
        let mut ai = AcceptItem::default();
        ai.len = ai.addr.data.len() as _;
        let res = Kernel::HostSpace::IOAccept(
            self.fd,
            &ai.addr as *const _ as u64,
            &ai.len as *const _ as u64,
        ) as i32;
        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        ai.fd = res;
        return Ok(ai);
    }

    pub fn Notify(&self, mask: EventMask) {
        self.queue.Notify(EventMaskFromLinux(mask as u32));
    }
}

impl Deref for HostSocketOperations {
    type Target = Arc<HostSocketOperationsIntern>;

    fn deref(&self) -> &Arc<HostSocketOperationsIntern> {
        &self.0
    }
}

impl HostSocketOperations {
    pub fn SetRemoteAddr(&self, addr: Vec<u8>) -> Result<()> {
        let addr = GetAddr(addr[0] as i16, &addr[0..addr.len()])?;

        *self.remoteAddr.lock() = Some(addr);
        return Ok(());
    }

    pub fn GetRemoteAddr(&self) -> Option<Vec<u8>> {
        return match *self.remoteAddr.lock() {
            None => None,
            Some(ref v) => Some(v.ToVec().unwrap()),
        };
    }
}

pub const SIZEOF_SOCKADDR: usize = SocketSize::SIZEOF_SOCKADDR_INET6;

impl Waitable for HostSocketOperations {
    fn AsyncReadiness(&self, _task: &Task, mask: EventMask, wait: &MultiWait) -> Future<EventMask> {
        let fd = self.fd;
        let future = IOURING.UnblockPollAdd(fd, mask as u32, wait);
        return future;
    }

    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let fd = self.fd;
        return NonBlockingPoll(fd, mask);
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.queue.clone();
        queue.EventRegister(task, e, mask);
        let fd = self.fd;
        UpdateFD(fd).unwrap();
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.queue.clone();
        queue.EventUnregister(task, e);
        let fd = self.fd;
        UpdateFD(fd).unwrap();
    }
}

impl SpliceOperations for HostSocketOperations {}

impl FileOperations for HostSocketOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::SocketOperations;
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(
        &self,
        _task: &Task,
        _f: &File,
        _whence: i32,
        _current: i64,
        _offset: i64,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    fn ReadDir(
        &self,
        _task: &Task,
        _f: &File,
        _offset: i64,
        _serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let size = IoVec::NumBytes(dsts);
        let buf = DataBuff::New(size);
        let iovs = buf.Iovs(size);
        let ret = IORead(self.fd, &iovs)?;

        // handle partial memcopy
        task.CopyDataOutToIovs(&buf.buf[0..ret as usize], dsts, false)?;
        return Ok(ret);
    }

    fn WriteAt(
        &self,
        task: &Task,
        _f: &File,
        srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let size = IoVec::NumBytes(srcs);
        if size == 0 {
            return Ok(0);
        }

        let size = IoVec::NumBytes(srcs);
        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
        let iovs = buf.Iovs(len);
        return IOWrite(self.fd, &iovs);
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0));
    }

    fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        _syncType: SyncType,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn Ioctl(&self, task: &Task, _f: &File, _fd: i32, request: u64, val: u64) -> Result<u64> {
        let flags = request as i32;

        let hostfd = self.fd;
        match flags as u64 {
            LibcConst::SIOCGIFFLAGS
            | LibcConst::SIOCGIFADDR
            | LibcConst::SIOCGIFBRDADDR
            | LibcConst::SIOCGIFDSTADDR
            | LibcConst::SIOCGIFHWADDR
            | LibcConst::SIOCGIFINDEX
            | LibcConst::SIOCGIFMAP
            | LibcConst::SIOCGIFMETRIC
            | LibcConst::SIOCGIFMTU
            | LibcConst::SIOCGIFNAME
            | LibcConst::SIOCGIFNETMASK
            | LibcConst::SIOCGIFTXQLEN => {
                let addr = val;
                HostIoctlIFReq(task, hostfd, request, addr)?;

                return Ok(0);
            }
            LibcConst::SIOCGIFCONF => {
                let addr = val;
                HostIoctlIFConf(task, hostfd, request, addr)?;

                return Ok(0);
            }
            LibcConst::TIOCINQ => {
                let tmp: i32 = 0;
                let res = Kernel::HostSpace::IoCtl(self.fd, request, &tmp as *const _ as u64,core::mem::size_of::<i32>());
                if res < 0 {
                    return Err(Error::SysError(-res as i32));
                }
                task.CopyOutObj(&tmp, val)?;
                return Ok(0);
            }
            _ => {
                let tmp: i32 = 0;
                let res = Kernel::HostSpace::IoCtl(self.fd, request, &tmp as *const _ as u64,core::mem::size_of::<i32>());
                if res < 0 {
                    return Err(Error::SysError(-res as i32));
                }
                task.CopyOutObj(&tmp, val)?;
                return Ok(0);
            }
        }
    }

    fn IterateDir(
        &self,
        _task: &Task,
        _d: &Dirent,
        _dirCtx: &mut DirCtx,
        _offset: i32,
    ) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for HostSocketOperations {
    fn Connect(&self, task: &Task, sockaddr: &[u8], blocking: bool) -> Result<i64> {
        let mut socketaddr = sockaddr;

        if (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            && socketaddr.len() > SIZEOF_SOCKADDR
        {
            socketaddr = &socketaddr[..SIZEOF_SOCKADDR]
        }

        let res = Kernel::HostSpace::IOConnect(
            self.fd,
            &socketaddr[0] as *const _ as u64,
            socketaddr.len() as u32,
        ) as i32;

        if res == 0 {
            let family = unsafe { *(&sockaddr[0] as *const u8 as *const i16) } as i32;
            if family == AFType::AF_INET || family == AFType::AF_INET6 {
                self.SetRemoteAddr(socketaddr.to_vec())?;
            }

            return Ok(0);
        }

        if res != 0 {
            if -res != SysErr::EINPROGRESS || !blocking {
                return Err(Error::SysError(-res));
            }

            //todo: which one is more efficent?
            let general = task.blocker.generalEntry.clone();
            self.EventRegister(task, &general, EVENT_OUT);
            defer!(self.EventUnregister(task, &general));

            if self.Readiness(task, WRITEABLE_EVENT) == 0 {
                match task.blocker.BlockWithMonoTimer(true, None) {
                    Err(Error::ErrInterrupted) => {
                        return Err(Error::SysError(SysErr::ERESTARTSYS));
                    }
                    Err(e) => {
                        return Err(e);
                    }
                    _ => (),
                }
            }
        }

        let mut val: i32 = 0;
        let len: i32 = 4;
        let res = HostSpace::GetSockOpt(
            self.fd,
            LibcConst::SOL_SOCKET as i32,
            LibcConst::SO_ERROR as i32,
            &mut val as *mut i32 as u64,
            &len as *const i32 as u64,
        ) as i32;

        if res < 0 {
            return Err(Error::SysError(-res));
        }

        if val != 0 {
            return Err(Error::SysError(val as i32));
        }

        self.SetRemoteAddr(socketaddr.to_vec())?;
        return Ok(0);
    }

    fn Accept(
        &self,
        task: &Task,
        addr: &mut [u8],
        addrlen: &mut u32,
        flags: i32,
        blocking: bool,
    ) -> Result<i64> {
        let mut acceptItem = AcceptItem::default();
        if !blocking {
            let ai = self.IOAccept();

            match ai {
                Err(Error::SysError(SysErr::EAGAIN)) => {
                    if !blocking {
                        return Err(Error::SysError(SysErr::EAGAIN));
                    }
                }
                Err(e) => return Err(e),
                Ok(item) => {
                    acceptItem = item;
                }
            }
        } else {
            let general = task.blocker.generalEntry.clone();
            self.EventRegister(task, &general, EVENT_IN);
            defer!(self.EventUnregister(task, &general));

            loop {
                let ai = self.IOAccept();

                match ai {
                    Err(Error::SysError(SysErr::EAGAIN)) => (),
                    Err(e) => return Err(e),
                    Ok(item) => {
                        acceptItem = item;
                        break;
                    }
                }
                match task.blocker.BlockWithMonoTimer(true, None) {
                    Err(e) => {
                        return Err(e);
                    }
                    _ => (),
                }
            }
        }

        let mut len: usize = acceptItem.addr.data.len();
        if addr.len() > 0 {
            len = core::cmp::min(
                core::cmp::min(acceptItem.len as usize, addr.len()),
                acceptItem.addr.data.len(),
            );
            for i in 0..len {
                addr[i] = acceptItem.addr.data[i];
            }

            *addrlen = len as u32;
        }

        let fd = acceptItem.fd;

        let remoteAddr = &acceptItem.addr.data[0..len];

        let file = newHostSocketFile(
            task,
            self.family,
            fd as i32,
            self.stype,
            flags & SocketFlags::SOCK_NONBLOCK != 0,
            Some(remoteAddr.to_vec()),
        )?;

        let fdFlags = FDFlags {
            CloseOnExec: flags & SocketFlags::SOCK_CLOEXEC != 0,
        };

        let fd = task.NewFDFrom(0, &Arc::new(file), &fdFlags)?;
        return Ok(fd as i64);
    }

    fn Bind(&self, task: &Task, sockaddr: &[u8]) -> Result<i64> {
        let mut socketaddr = sockaddr;

        info!(
            "hostinet socket bind {:?}, addr is {:?}",
            self.family, socketaddr
        );
        if (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            && socketaddr.len() > SIZEOF_SOCKADDR
        {
            socketaddr = &socketaddr[..SIZEOF_SOCKADDR]
        }

        let res = Kernel::HostSpace::Bind(
            self.fd,
            &socketaddr[0] as *const _ as u64,
            socketaddr.len() as u32,
            task.Umask(),
        );
        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(res);
    }

    fn Listen(&self, _task: &Task, backlog: i32) -> Result<i64> {
        let len = if backlog <= 0 { 5 } else { backlog };

        let res = Kernel::HostSpace::Listen(self.fd, len, false);

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(res);
    }

    fn Shutdown(&self, _task: &Task, how: i32) -> Result<i64> {
        let how = how as u64;

        if how == LibcConst::SHUT_RD || how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR {
            let res = Kernel::HostSpace::Shutdown(self.fd, how as i32);
            if res < 0 {
                return Err(Error::SysError(-res as i32));
            }

            return Ok(res);
        }

        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn GetSockOpt(&self, _task: &Task, level: i32, name: i32, opt: &mut [u8]) -> Result<i64> {
        /*
        let optlen = match level as u64 {
            LibcConst::SOL_IPV6 => {
                match name as u64 {
                    LibcConst::IPV6_V6ONLY => SocketSize::SIZEOF_INT32,
                    LibcConst::IPV6_TCLASS => SocketSize::SIZEOF_INfAT32,
                    _ => 0,
                }
            }
            LibcConst::SOL_SOCKET => {
                match name as u64 {
                    LibcConst::SO_ERROR
                    | LibcConst::SO_KEEPALIVE
                    | LibcConst::SO_SNDBUF
                    | LibcConst::SO_RCVBUF
                    | LibcConst::SO_REUSEADDR
                    | LibcConst::SO_TYPE => SocketSize::SIZEOF_INT32,
                    LibcConst::SO_LINGER => SocketSize::SIZEOF_LINGER,
                    _ => 0,
                }
            }
            LibcConst::SOL_TCP => {
                match name as u64 {
                    LibcConst::TCP_NODELAY => SocketSize::SIZEOF_INT32,
                    LibcConst::TCP_INFO => SocketSize::SIZEOF_TCPINFO,
                    _ => 0,
                }
            }
            LibcConst::SOL_IP => {
                match name as u64 {
                    LibcConst::IP_TTL => SocketSize::SIZEOF_INT32,
                    LibcConst::IP_TOS => SocketSize::SIZEOF_INT32,
                    _ => 0,
                }
            }
            _ => 0,
        };

        if optlen == 0 {
            return Err(Error::SysError(SysErr::ENOPROTOOPT))
        }

        let bufferSize = opt.len();

        if bufferSize < optlen {
            // provide special handling for options like IP_TOS, which allow inadequate buffer for optval
            match name as u64 {
                LibcConst::IP_TOS => {
                    let res = if bufferSize == 0 {
                        // dirty, any better way?
                        Kernel::HostSpace::GetSockOpt(self.fd, level, name, &bufferSize as *const _ as u64, &bufferSize as *const _ as u64)
                    } else {
                        Kernel::HostSpace::GetSockOpt(self.fd, level, name, &opt[0] as *const _ as u64, &bufferSize as *const _ as u64)
                    };
                    if res < 0 {
                        return Err(Error::SysError(-res as i32))
                    }
                    // if optlen < sizeof(i32), the return of getsockopt will be of sizeof(i8)
                    return Ok(bufferSize as i64)
                },
                _ => return Err(Error::SysError(SysErr::EINVAL))
            };
        };

        let opt = &opt[..optlen];
        let res = Kernel::HostSpace::GetSockOpt(self.fd, level, name, &opt[0] as *const _ as u64, &optlen as *const _ as u64);
        if res < 0 {
            return Err(Error::SysError(-res as i32))
        }

        return Ok(optlen as i64)
        */

        let mut optLen = opt.len();
        let res = if optLen == 0 {
            Kernel::HostSpace::GetSockOpt(
                self.fd,
                level,
                name,
                ptr::null::<u8>() as u64,
                &mut optLen as *mut _ as u64,
            )
        } else {
            Kernel::HostSpace::GetSockOpt(
                self.fd,
                level,
                name,
                &mut opt[0] as *mut _ as u64,
                &mut optLen as *mut _ as u64,
            )
        };

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(optLen as i64);
    }

    fn SetSockOpt(&self, task: &Task, level: i32, name: i32, opt: &[u8]) -> Result<i64> {
        if (level as u64) == LibcConst::SOL_SOCKET && (name as u64) == LibcConst::SO_SNDTIMEO {
            if opt.len() >= SocketSize::SIZEOF_TIMEVAL {
                let timeVal = task.CopyInObj::<Timeval>(&opt[0] as *const _ as u64)?;
                self.SetSendTimeout(timeVal.ToDuration() as i64);
            } else {
                //TODO: to be aligned with Linux, Linux allows shorter length for this flag.
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        if (level as u64) == LibcConst::SOL_SOCKET && (name as u64) == LibcConst::SO_RCVTIMEO {
            if opt.len() >= SocketSize::SIZEOF_TIMEVAL {
                let timeVal = task.CopyInObj::<Timeval>(&opt[0] as *const _ as u64)?;
                self.SetRecvTimeout(timeVal.ToDuration() as i64);
            } else {
                //TODO: to be aligned with Linux, Linux allows shorter length for this flag.
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        // TCP_INQ is bound to buffer implementation
        if (level as u64) == LibcConst::SOL_TCP && (name as u64) == LibcConst::TCP_INQ {
            let val = unsafe { *(&opt[0] as *const _ as u64 as *const i32) };
            if val == 1 {
                self.passInq.store(true, Ordering::Relaxed);
            } else {
                self.passInq.store(false, Ordering::Relaxed);
            }
        }

        let optLen = opt.len();
        let res = if optLen == 0 {
            Kernel::HostSpace::SetSockOpt(
                self.fd,
                level,
                name,
                ptr::null::<u8>() as u64,
                optLen as u32,
            )
        } else {
            Kernel::HostSpace::SetSockOpt(
                self.fd,
                level,
                name,
                &opt[0] as *const _ as u64,
                optLen as u32,
            )
        };

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(res);
    }

    fn GetSockName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let len = socketaddr.len() as i32;

        let res = Kernel::HostSpace::GetSockName(
            self.fd,
            &socketaddr[0] as *const _ as u64,
            &len as *const _ as u64,
        );
        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(len as i64);
    }

    fn GetPeerName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let len = socketaddr.len() as i32;
        let res = Kernel::HostSpace::GetPeerName(
            self.fd,
            &socketaddr[0] as *const _ as u64,
            &len as *const _ as u64,
        );
        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(len as i64);
    }

    fn RecvMsg(
        &self,
        task: &Task,
        dsts: &mut [IoVec],
        flags: i32,
        deadline: Option<Time>,
        senderRequested: bool,
        controlDataLen: usize,
    ) -> Result<(i64, i32, Option<(SockAddr, usize)>, Vec<u8>)> {
        if flags
            & !(MsgType::MSG_DONTWAIT
                | MsgType::MSG_PEEK
                | MsgType::MSG_TRUNC
                | MsgType::MSG_ERRQUEUE
                | MsgType::MSG_CTRUNC
                | MsgType::MSG_WAITALL)
            != 0
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let size = IoVec::NumBytes(dsts);
        let buf = DataBuff::New(size);
        let iovs = buf.Iovs(size);

        let mut msgHdr = MsgHdr::default();
        msgHdr.iov = &iovs[0] as *const _ as u64;
        msgHdr.iovLen = iovs.len();

        let mut addr: [u8; SIZEOF_SOCKADDR] = [0; SIZEOF_SOCKADDR];
        if senderRequested {
            msgHdr.msgName = &mut addr[0] as *mut _ as u64;
            msgHdr.nameLen = SIZEOF_SOCKADDR as u32;
        }

        let mut controlVec: Vec<u8> = vec![0; controlDataLen];
        msgHdr.msgControlLen = controlDataLen;
        if msgHdr.msgControlLen != 0 {
            msgHdr.msgControl = &mut controlVec[0] as *mut _ as u64;
        } else {
            msgHdr.msgControl = ptr::null::<u8>() as u64;
        }

        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_READ);
        defer!(self.EventUnregister(task, &general));

        let mut res = Kernel::HostSpace::IORecvMsg(
            self.fd,
            &mut msgHdr as *mut _ as u64,
            flags | MsgType::MSG_DONTWAIT,
            false,
        ) as i32;

        while res == -SysErr::EWOULDBLOCK
            && flags & (MsgType::MSG_DONTWAIT | MsgType::MSG_ERRQUEUE) == 0
        {
            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::ErrInterrupted) => {
                    return Err(Error::SysError(SysErr::ERESTARTSYS));
                }
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    return Err(Error::SysError(SysErr::EAGAIN));
                }
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }

            res = Kernel::HostSpace::IORecvMsg(
                self.fd,
                &mut msgHdr as *mut _ as u64,
                flags | MsgType::MSG_DONTWAIT,
                false,
            ) as i32;
        }

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        let msgFlags = msgHdr.msgFlags & !MsgType::MSG_CTRUNC;
        let senderAddr = if senderRequested
            // for tcp connect, recvmsg get nameLen=0 msg
            && msgHdr.nameLen >= 4
        {
            let addr = GetAddr(addr[0] as i16, &addr[0..msgHdr.nameLen as usize])?;
            let l = addr.Len();
            Some((addr, l))
        } else {
            None
        };

        controlVec.resize(msgHdr.msgControlLen, 0);

        // todo: need to handle partial copy
        let count = if res < buf.buf.len() as i32 {
            res
        } else {
            buf.buf.len() as i32
        };
        let _len = task.CopyDataOutToIovs(&buf.buf[0..count as usize], dsts, false)?;
        return Ok((res as i64, msgFlags, senderAddr, controlVec));
    }

    fn SendMsg(
        &self,
        task: &Task,
        srcs: &[IoVec],
        flags: i32,
        msgHdr: &mut MsgHdr,
        deadline: Option<Time>,
    ) -> Result<i64> {
        if flags
            & !(MsgType::MSG_DONTWAIT
                | MsgType::MSG_EOR
                | MsgType::MSG_FASTOPEN
                | MsgType::MSG_MORE
                | MsgType::MSG_NOSIGNAL)
            != 0
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let size = IoVec::NumBytes(srcs);
        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
        let iovs = buf.Iovs(len);

        msgHdr.iov = &iovs[0] as *const _ as u64;
        msgHdr.iovLen = iovs.len();
        msgHdr.msgFlags = 0;

        let mut res = if msgHdr.msgControlLen > 0 {
            Kernel::HostSpace::IOSendMsg(
                self.fd,
                msgHdr as *const _ as u64,
                flags | MsgType::MSG_DONTWAIT,
                false,
            ) as i32
        } else {
            Kernel::HostSpace::IOSendto(
                self.fd,
                buf.Ptr(),
                len,
                flags | MsgType::MSG_DONTWAIT,
                msgHdr.msgName,
                msgHdr.nameLen,
            ) as i32
        };

        while res == -SysErr::EWOULDBLOCK && flags & MsgType::MSG_DONTWAIT == 0 {
            let general = task.blocker.generalEntry.clone();

            self.EventRegister(task, &general, EVENT_WRITE);
            defer!(self.EventUnregister(task, &general));
            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }

            res = if msgHdr.msgControlLen > 0 {
                Kernel::HostSpace::IOSendMsg(
                    self.fd,
                    msgHdr as *const _ as u64,
                    flags | MsgType::MSG_DONTWAIT,
                    false,
                ) as i32
            } else {
                Kernel::HostSpace::IOSendto(
                    self.fd,
                    buf.Ptr(),
                    len,
                    flags | MsgType::MSG_DONTWAIT,
                    msgHdr.msgName,
                    msgHdr.nameLen,
                ) as i32
            };
        }

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(res as i64);
    }

    fn SetRecvTimeout(&self, ns: i64) {
        self.recv.store(ns, Ordering::Relaxed)
    }

    fn SetSendTimeout(&self, ns: i64) {
        self.send.store(ns, Ordering::Relaxed)
    }

    fn RecvTimeout(&self) -> i64 {
        return self.recv.load(Ordering::Relaxed);
    }

    fn SendTimeout(&self) -> i64 {
        return self.send.load(Ordering::Relaxed);
    }

    fn State(&self) -> u32 {
        let mut info = TCPInfo::default();
        let mut len = SocketSize::SIZEOF_TCPINFO;

        let ret = HostSpace::GetSockOpt(
            self.fd,
            LibcConst::SOL_TCP as _,
            LibcConst::TCP_INFO as _,
            &mut info as *mut _ as u64,
            &mut len as *mut _ as u64,
        ) as i32;

        if ret < 0 {
            if ret != -SysErr::ENOPROTOOPT {
                error!(
                    "fail to Failed to get TCP socket info from {} with error {}",
                    self.fd, ret
                );

                // For non-TCP sockets, silently ignore the failure.
                return 0;
            }
        }

        if len != SocketSize::SIZEOF_TCPINFO {
            return 0;
        }

        return info.State as u32;
    }

    fn Type(&self) -> (i32, i32, i32) {
        return (self.family, self.stype, -1);
    }
}
