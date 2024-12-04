// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor& Authors.
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
use alloc::sync::Weak;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt;
use core::ops::Deref;
use core::ptr;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::AtomicU16;
use core::sync::atomic::Ordering;

use super::super::super::super::common::*;
use super::super::super::super::linux::time::Timeval;
use super::super::super::super::linux_def::*;
use super::super::super::super::mem::block::*;
use super::super::super::super::socket_buf::*;
use super::super::super::fs::attr::*;
use super::super::super::fs::dentry::*;
use super::super::super::fs::dirent::*;
use super::super::super::fs::file::*;
use super::super::super::fs::flags::*;
use super::super::super::fs::host::hostinodeop::*;
use super::super::super::guestfdnotifier::*;
use super::super::super::kernel::fd_table::*;
use super::super::super::kernel::kernel::GetKernel;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::*;
use super::super::super::quring::QUring;
use super::super::super::task::*;
use super::super::super::tcpip::tcpip::*;
use super::super::super::Kernel;
use super::super::super::Kernel::HostSpace;
use super::super::super::IOURING;
use super::super::super::SHARESPACE;
use super::super::control::*;
use super::super::socket::*;
use super::super::unix::transport::unix::*;
use crate::qlib::bytestream::*;
use crate::qlib::kernel::kernel::abstract_socket_namespace::*;
use crate::qlib::kernel::kernel::waiter::Queue;
use crate::qlib::kernel::socket::hostinet::loopbacksocket::*;
use crate::qlib::kernel::socket::hostinet::socket::HostIoctlIFConf;
use crate::qlib::kernel::socket::hostinet::socket::HostIoctlIFReq;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::GuestHostSharedAllocator;

pub fn newUringSocketFile(
    task: &Task,
    family: i32,
    fd: i32,
    stype: i32,
    nonblock: bool,
    queue: Queue,
    socketType: UringSocketType,
    addr: Option<Vec<u8>>,
) -> Result<File> {
    let dirent = NewSocketDirent(task, SOCKET_DEVICE.clone(), fd)?;
    let inode = dirent.Inode();
    let iops = inode.lock().InodeOp.clone();
    let hostiops = iops.HostInodeOp().unwrap();
    let s = UringSocketOperations::New(
        family,
        fd,
        stype,
        queue, //hostiops.Queue(),
        hostiops.clone(),
        socketType,
        addr,
    )?;

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

#[repr(u64)]
#[derive(Clone)]
pub enum UringSocketType {
    TCPInit,                      // Init TCP Socket, no listen and no connect
    TCPConnecting,                // Start connecting, content is connect
    TCPUringlServer(AcceptQueue), // Uring TCP Server socket, when socket start to listen
    Uring(SocketBuff),
    Loopback(LoopbackSocket),
}

impl fmt::Debug for UringSocketType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::TCPInit => write!(f, "UringSocketType::TCPInit"),
            Self::TCPConnecting => write!(f, "UringSocketType::TCPConnecting"),
            Self::TCPUringlServer(_) => write!(f, "UringSocketType::TCPUringlServer"),
            Self::Uring(_) => write!(f, "UringSocketType::Uring"),
            Self::Loopback(_) => write!(f, "UringSocketType::Loopback"),
        }
    }
}

impl UringSocketType {
    pub fn Accept(&self, accept: AcceptSocket) -> Self {
        match self {
            UringSocketType::TCPUringlServer(_) => match accept {
                AcceptSocket::SocketBuff(socketBuf) => return UringSocketType::Uring(socketBuf),
                AcceptSocket::LoopbackSocket(loopback) => {
                    return UringSocketType::Loopback(loopback)
                }
                AcceptSocket::None => {
                    panic!("UringSocketType::Accept unexpect AcceptSocket::None")
                }
            },
            _ => {
                panic!("UringSocketType::Accept unexpect type {:?}", self)
            }
        }
    }

    pub fn RClosed(&self) -> bool {
        let ret = match self {
            UringSocketType::Uring(buf) => buf.RClosed(),
            UringSocketType::Loopback(loopback) => loopback.sockBuff.RClosed(),
            _ => {
                panic!("Uring socket Rclose unexpect type {:?}", self);
            }
        };

        return ret;
    }

    pub fn WClosed(&self) -> bool {
        let ret = match self {
            UringSocketType::Uring(buf) => buf.WClosed(),
            UringSocketType::Loopback(loopback) => loopback.sockBuff.WClosed(),
            _ => {
                panic!("Uring socket WClosed unexpect type {:?}", self);
            }
        };

        return ret;
    }
}

pub struct UringSocketOperationsIntern {
    pub send: AtomicI64,
    pub recv: AtomicI64,
    pub family: i32,
    pub stype: i32,
    pub fd: i32,
    pub connErrNo: AtomicI32,
    pub queue: Queue,
    pub remoteAddr: QMutex<Option<SockAddr>>,
    pub socketType: QMutex<UringSocketType>,
    pub enableAsyncAccept: AtomicBool,
    pub loopbackPort: AtomicU16,
    pub hostops: HostInodeOp,
    passInq: AtomicBool,
}

#[derive(Clone)]
pub struct UringSocketOperationsWeak(pub Weak<UringSocketOperationsIntern, GuestHostSharedAllocator>);

impl UringSocketOperationsWeak {
    pub fn Upgrade(&self) -> Option<UringSocketOperations> {
        let f = match self.0.upgrade() {
            None => return None,
            Some(f) => f,
        };

        return Some(UringSocketOperations(f));
    }
}

#[derive(Clone)]
pub struct UringSocketOperations(Arc<UringSocketOperationsIntern, GuestHostSharedAllocator>);

impl Drop for UringSocketOperations {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            let loopbackPort = self.loopbackPort.load(Ordering::Acquire);
            if loopbackPort > 0 {
                TCP_SOCKET.Remove(loopbackPort).unwrap();
            }
        }
    }
}

impl UringSocketOperations {
    pub fn Downgrade(&self) -> UringSocketOperationsWeak {
        return UringSocketOperationsWeak(Arc::downgrade(&self.0));
    }

    pub fn InnerGetSockName(&self) -> Result<SockAddrInet> {
        let socketAddr = SockAddrInet::default();
        let len = core::mem::size_of_val(&socketAddr);
        let res = Kernel::HostSpace::GetSockName(
            self.fd,
            &socketAddr as *const _ as u64,
            &len as *const _ as u64,
        );
        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(socketAddr);
    }

    pub fn SetConnErrno(&self, errno: i32) {
        self.connErrNo.store(errno, Ordering::Release);
    }

    pub fn ConnErrno(&self) -> i32 {
        return self.connErrNo.load(Ordering::Acquire);
    }

    pub fn New(
        family: i32,
        fd: i32,
        stype: i32,
        queue: Queue,
        hostops: HostInodeOp,
        socketBuf: UringSocketType,
        addr: Option<Vec<u8>>,
    ) -> Result<Self> {
        let addr = match addr {
            None => None,
            Some(v) => {
                if v.len() == 0 {
                    None
                } else {
                    match GetAddr(v[0] as i16, &v[0..v.len()]) {
                        Ok(addr) => Some(addr),
                        Err(_) => {
                            error!("can't get address through accept...");
                            None
                        }
                    }
                }
            }
        };

        match &socketBuf {
            UringSocketType::Uring(ref buf) => {
                QUring::BufSockInit(fd, queue.clone(), buf.clone(), true).unwrap();
            }
            _ => (),
        }

        let ret = UringSocketOperationsIntern {
            send: AtomicI64::new(0),
            recv: AtomicI64::new(0),
            family,
            stype,
            fd,
            connErrNo: AtomicI32::new(0),
            queue,
            remoteAddr: QMutex::new(addr),
            socketType: QMutex::new(socketBuf.clone()),
            enableAsyncAccept: AtomicBool::new(false),
            loopbackPort: AtomicU16::new(0),
            hostops: hostops,
            passInq: AtomicBool::new(false),
        };

        let ret = Self(Arc::new_in(ret, GUEST_HOST_SHARED_ALLOCATOR));

        return Ok(ret);
    }

    pub fn Produce(&self, task: &Task, count: usize, iovs: &mut SocketBufIovs) -> Result<()> {
        let sockBufType = self.socketType.lock().clone();
        match sockBufType {
            UringSocketType::Uring(buf) => {
                if buf.WClosed() {
                    return Err(Error::SysError(SysErr::EPIPE));
                }

                return QUring::SocketProduce(
                    task,
                    self.fd,
                    self.queue.clone(),
                    buf,
                    count as usize,
                    self,
                    iovs,
                );
            }
            _ => {
                return Err(Error::SysError(SysErr::EPIPE));
            }
        }
    }

    pub fn Consume(&self, task: &Task, count: usize, iovs: &mut SocketBufIovs) -> Result<()> {
        let sockBufType = self.socketType.lock().clone();
        match sockBufType {
            UringSocketType::Uring(buf) => {
                if buf.WClosed() {
                    return Err(Error::SysError(SysErr::EPIPE));
                }

                return QUring::SocketConsume(
                    task,
                    self.fd,
                    self.queue.clone(),
                    buf,
                    count as usize,
                    iovs,
                );
            }
            _ => {
                return Err(Error::SysError(SysErr::EPIPE));
            }
        }
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

    fn prepareControlMessage(&self, controlDataLen: usize) -> (i32, Vec<u8>) {
        // shortcut for no controldata wanted
        if controlDataLen == 0 {
            return (0, Vec::new());
        }

        let mut controlData: Vec<u8> = vec![0; controlDataLen];
        if self.passInq.load(Ordering::Relaxed) {
            let inqMessage = ControlMessageTCPInq {
                Size: self.SocketBuf().readBuf.lock().AvailableDataSize() as u32,
            };

            let (remaining, updated_flags) = inqMessage.EncodeInto(&mut controlData[..], 0);
            let remainSize = remaining.len();
            controlData.resize(controlDataLen - remainSize, 0);
            return (updated_flags, controlData);
        } else {
            return (0, Vec::new());
        }
    }

    pub fn AsyncAcceptEnabled(&self) -> bool {
        return self.enableAsyncAccept.load(Ordering::Relaxed);
    }

    pub fn SocketType(&self) -> UringSocketType {
        return self.socketType.lock().clone();
    }

    pub fn SocketBuf(&self) -> SocketBuff {
        match self.SocketType() {
            UringSocketType::Uring(b) => return b,
            _ => panic!(
                "UringSocketType::None has no SockBuff {:?}",
                self.SocketType()
            ),
        }
    }

    pub fn SocketBufEnabled(&self) -> bool {
        match self.SocketType() {
            UringSocketType::Uring(_) => return true,
            _ => false,
        }
    }

    pub fn PostConnect(&self) {
        let socketBuf = SocketBuff(Arc::new_in(
            SocketBuffIntern::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT),
            crate::GUEST_HOST_SHARED_ALLOCATOR,
        ));
        *self.socketType.lock() = UringSocketType::Uring(socketBuf.clone());
        QUring::BufSockInit(self.fd, self.queue.clone(), socketBuf, true).unwrap();
    }

    pub fn Notify(&self, mask: EventMask) {
        self.queue.Notify(EventMaskFromLinux(mask as u32));
    }

    pub fn AcceptData(&self) -> Result<AcceptItem> {
        let sockBufType = self.socketType.lock().clone();
        match sockBufType {
            UringSocketType::TCPUringlServer(ref queue) => {
                return IOURING.Accept(self.fd, &self.queue, queue)
            }
            _ => {
                error!("UringSocketType invalid accept {:?}", sockBufType);
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }
    }

    pub fn ReadFromBuf(
        &self,
        task: &Task,
        buf: &UringSocketType,
        dsts: &mut [IoVec],
        peek: bool,
    ) -> Result<i64> {
        let ret = match buf {
            UringSocketType::Uring(buf) => QUring::RingFileRead(
                task,
                self.fd,
                self.queue.clone(),
                buf.clone(),
                dsts,
                true,
                peek,
            )?,
            UringSocketType::Loopback(loopback) => loopback.Readv(task, dsts, peek)?,
            _ => {
                //return Err(Error::SysError(SysErr::ECONNREFUSED));
                return Err(Error::SysError(SysErr::ENOTCONN));
            }
        };

        return Ok(ret);
    }

    pub fn WriteToBuf(&self, task: &Task, buf: &UringSocketType, srcs: &[IoVec]) -> Result<i64> {
        let ret = match buf {
            UringSocketType::Uring(buf) => {
                QUring::SocketSend(task, self.fd, self.queue.clone(), buf.clone(), srcs, self)?
            }
            UringSocketType::Loopback(loopback) => loopback.Writev(task, srcs)?,
            _ => {
                //return Err(Error::SysError(SysErr::ECONNREFUSED));
                return Err(Error::SysError(SysErr::ENOTCONN));
            }
        };

        return Ok(ret);
    }
}

impl Deref for UringSocketOperations {
    type Target = Arc<UringSocketOperationsIntern, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<UringSocketOperationsIntern, GuestHostSharedAllocator> {
        &self.0
    }
}

impl UringSocketOperations {
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

impl Waitable for UringSocketOperations {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        match self.SocketType() {
            UringSocketType::TCPConnecting => {
                let errno = self.ConnErrno();
                if errno != -SysErr::EINPROGRESS {
                    return EVENT_OUT & mask;
                }
                return 0;
            }
            UringSocketType::TCPUringlServer(q) => {
                return q.lock().Events() & mask;
            }
            UringSocketType::Uring(buf) => {
                return buf.Events() & mask;
            }
            UringSocketType::Loopback(loopback) => {
                return loopback.Events() & mask;
            }
            UringSocketType::TCPInit => (),
        }

        let fd = self.fd;
        return NonBlockingPoll(fd, mask);
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.queue.clone();
        queue.EventRegister(task, e, mask);
        let fd = self.fd;

        match self.SocketType() {
            UringSocketType::TCPConnecting => {
                UpdateFD(fd).unwrap();
            }
            UringSocketType::TCPUringlServer(_q) => (),
            UringSocketType::Uring(_buf) => (),
            UringSocketType::Loopback(_) => (),
            UringSocketType::TCPInit => {
                UpdateFD(fd).unwrap();
            }
        }
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.queue.clone();
        queue.EventUnregister(task, e);
        let fd = self.fd;
        match self.SocketType() {
            UringSocketType::TCPConnecting => {
                UpdateFD(fd).unwrap();
            }
            UringSocketType::TCPUringlServer(_q) => (),
            UringSocketType::Uring(_buf) => (),
            UringSocketType::Loopback(_) => (),
            UringSocketType::TCPInit => {
                UpdateFD(fd).unwrap();
            }
        }
    }
}

impl SpliceOperations for UringSocketOperations {}

impl FileOperations for UringSocketOperations {
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
        let sockBufType = self.socketType.lock().clone();
        match sockBufType {
            UringSocketType::Uring(socketBuf) => {
                /*if self.SocketBuf().RClosed() {
                    return Err(Error::SysError(SysErr::ESPIPE))
                }*/
                let ret = QUring::RingFileRead(
                    task,
                    self.fd,
                    self.queue.clone(),
                    socketBuf,
                    dsts,
                    true,
                    false,
                )?;
                return Ok(ret);
            }
            UringSocketType::Loopback(loopback) => {
                let count = loopback.Readv(task, dsts, false)?;
                return Ok(count);
            }
            _ => {
                return Ok(0);
            }
        }
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

        let sockBufType = self.socketType.lock().clone();
        match sockBufType {
            UringSocketType::Uring(buf) => {
                if buf.WClosed() {
                    return Err(Error::SysError(SysErr::EPIPE));
                }

                return QUring::SocketSend(task, self.fd, self.queue.clone(), buf, srcs, self);
            }
            UringSocketType::Loopback(loopback) => {
                let count = loopback.Writev(task, srcs)?;
                return Ok(count);
            }
            _ => {
                return Err(Error::SysError(SysErr::EPIPE));
            }
        }
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
                if self.SocketBufEnabled() {
                    let v = self.SocketBuf().readBuf.lock().AvailableDataSize() as i32;
                    task.CopyOutObj(&v, val)?;
                    return Ok(0);
                } else {
                    let tmp: i32 = 0;
                    let res = Kernel::HostSpace::IoCtl(self.fd, request, &tmp as *const _ as u64,core::mem::size_of::<i32>());
                    if res < 0 {
                        return Err(Error::SysError(-res as i32));
                    }
                    task.CopyOutObj(&tmp, val)?;
                    return Ok(0);
                }
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
        match self.SocketType() {
            UringSocketType::Uring(b) => {
                return Err(Error::ErrSocketMap(b));
            }
            _ => return Err(Error::SysError(SysErr::ENODEV)),
        }
    }
}

impl SockOperations for UringSocketOperations {
    fn Connect(&self, task: &Task, sockaddr: &[u8], blocking: bool) -> Result<i64> {
        let sockType = self.SocketType();
        match sockType {
            UringSocketType::TCPInit => {
                let addr = unsafe { &*(&sockaddr[0] as *const _ as u64 as *const SockAddrInet) };

                if addr.Family == AFType::AF_INET as u16 && addr.Addr == [127, 0, 0, 1] {
                    match TCP_SOCKET.Get(addr.Port) {
                        // in case for host network, there are process listen localhost outside of the container
                        None => (), //return Err(Error::SysError(SysErr::ECONNREFUSED)),
                        Some(q) => {
                            let serverQueue = Queue::default();
                            let (clientSock, serverSock) =
                                LoopbackSocketPair(self.queue.clone(), serverQueue.clone());
                            *self.socketType.lock() = UringSocketType::Loopback(clientSock);
                            let addr = SockAddrInet {
                                Family: AFType::AF_INET as u16,
                                Port: 0,
                                Addr: [127, 0, 0, 1],
                                ..Default::default()
                            };

                            let res = Kernel::HostSpace::Socket(
                                AFType::AF_INET,
                                SocketType::SOCK_STREAM | SocketFlags::SOCK_CLOEXEC,
                                0,
                            );
                            if res < 0 {
                                return Err(Error::SysError(-res as i32));
                            }

                            let fd = res as i32;

                            q.EnqSocket(
                                fd,
                                TcpSockAddr::NewFromInet(addr),
                                16,
                                serverSock.into(),
                                serverQueue,
                            );
                            return Ok(0);
                        }
                    };
                }

                QUring::AsyncConnect(self.fd, self, sockaddr)?;
                if !blocking {
                    return Err(Error::SysError(SysErr::EINPROGRESS));
                }
            }
            UringSocketType::TCPConnecting => {
                if !blocking {
                    return Err(Error::SysError(SysErr::EALREADY));
                }
            }
            UringSocketType::Uring(_) => {
                return Err(Error::SysError(SysErr::EISCONN));
            }
            _ => {
                return Err(Error::SysError(SysErr::EBADF));
            }
        }

        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_OUT);
        defer!(self.EventUnregister(task, &general));
        if self.Readiness(task, WRITEABLE_EVENT) == 0 {
            match task.blocker.BlockWithMonoTimer(true, None) {
                Err(Error::ErrInterrupted) => {
                    return Err(Error::SysError(SysErr::ERESTARTNOINTR));
                }
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }
        }

        let errno = self.ConnErrno();
        return Err(Error::SysError(-errno));
    }

    fn Accept(
        &self,
        task: &Task,
        addr: &mut [u8],
        addrlen: &mut u32,
        flags: i32,
        blocking: bool,
    ) -> Result<i64> {
        let acceptItem;
        if !blocking {
            let ai = self.AcceptData();

            match ai {
                Err(Error::SysError(SysErr::EAGAIN)) => {
                    return Err(Error::SysError(SysErr::EAGAIN));
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
                let ai = self.AcceptData();

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
        //let sockBuf = self.ConfigUringSocketType();
        let sockBuf = self.SocketType().Accept(acceptItem.sockBuf.clone());

        let file = newUringSocketFile(
            task,
            self.family,
            fd as i32,
            self.stype,
            flags & SocketFlags::SOCK_NONBLOCK != 0,
            acceptItem.queue.clone(),
            sockBuf,
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

        let addr = unsafe { &*(&socketaddr[0] as *const _ as u64 as *const SockAddrInet) };
        info!(
            "hostinet socket bind {:?}, addr is {:?}/{:?}",
            self.family, addr, socketaddr
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

        if addr.Family == AFType::AF_INET as u16
            && (addr.Addr == [0, 0, 0, 0] || addr.Addr == [127, 0, 0, 1])
        {
            self.loopbackPort.store(addr.Port, Ordering::Release);
        }

        return Ok(res);
    }

    fn Listen(&self, _task: &Task, backlog: i32) -> Result<i64> {
        let asyncAccept = SHARESPACE.config.read().AsyncAccept
            && (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            && self.stype == SockType::SOCK_STREAM;

        let len = if backlog <= 0 { 5 } else { backlog };

        let socketBuf = self.socketType.lock().clone();
        let acceptQueue = match socketBuf {
            UringSocketType::TCPUringlServer(q) => {
                q.lock().SetQueueLen(len as usize);
                let loopbackPort = self.loopbackPort.load(Ordering::Acquire);
                if loopbackPort > 0 {
                    TCP_SOCKET.Add(loopbackPort, q.clone())?;
                }
                return Ok(0);
            }
            UringSocketType::TCPInit => AcceptQueue::New(len as usize, self.queue.clone()),
            _ => panic!("uring socket listen on wrong type {:?}", socketBuf), // panic?
        };

        let res = Kernel::HostSpace::Listen(self.fd, backlog, asyncAccept);

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        *self.socketType.lock() = {
            if !self.AsyncAcceptEnabled() {
                IOURING.AcceptInit(self.fd, &self.queue, &acceptQueue)?;
                self.enableAsyncAccept.store(true, Ordering::Relaxed);
            }

            let loopbackPort = self.loopbackPort.load(Ordering::Acquire);
            if loopbackPort > 0 {
                TCP_SOCKET.Add(loopbackPort, acceptQueue.clone())?;
            }

            UringSocketType::TCPUringlServer(acceptQueue)
        };

        return Ok(res);
    }

    fn Shutdown(&self, task: &Task, how: i32) -> Result<i64> {
        let how = how as u64;

        let sockType = self.SocketType();
        match &sockType {
            UringSocketType::TCPInit => (),
            UringSocketType::TCPConnecting => (),
            UringSocketType::Uring(ref buf) => {
                if how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR {
                    if buf.HasWriteData() {
                        buf.SetPendingWriteShutdown();
                        let general = task.blocker.generalEntry.clone();
                        self.EventRegister(task, &general, EVENT_PENDING_SHUTDOWN);
                        defer!(self.EventUnregister(task, &general));

                        while buf.HasWriteData() {
                            task.blocker.BlockGeneralOnly();
                        }
                    }
                }
            }
            UringSocketType::Loopback(_) => (),
            UringSocketType::TCPUringlServer(_) => (),
        }

        if how == LibcConst::SHUT_RD || how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR {
            let res = Kernel::HostSpace::Shutdown(self.fd, how as i32);
            if res < 0 {
                return Err(Error::SysError(-res as i32));
            }

            match &sockType {
                UringSocketType::TCPInit => (),
                UringSocketType::TCPConnecting => (),
                UringSocketType::Uring(ref buf) => {
                    if how == LibcConst::SHUT_RD || how == LibcConst::SHUT_RDWR {
                        buf.SetRClosed();
                    }

                    if how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR {
                        buf.SetWClosed();
                    }
                }
                UringSocketType::Loopback(ref loopback) => {
                    if how == LibcConst::SHUT_RD || how == LibcConst::SHUT_RDWR {
                        loopback.SetRClosed();
                    }

                    if how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR {
                        loopback.SetWClosed();
                    }
                }
                UringSocketType::TCPUringlServer(_) => (),
            }

            self.queue.Notify(EventMaskFromLinux(EVENT_HUP as u32));
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

        match level as u64 {
            LibcConst::SOL_SOCKET => match name as u64 {
                LibcConst::SO_ERROR => {
                    if opt.len() < 4 {
                        return Err(Error::SysError(SysErr::EINVAL));
                    }

                    if self.ConnErrno() != 0 {
                        let errno = self.ConnErrno();
                        self.SetConnErrno(0);
                        unsafe {
                            *(&mut opt[0] as *mut _ as u64 as *mut i32) = -errno;
                        }
                        return Ok(4);
                    }
                }
                _ => (),
            },
            _ => (),
        };

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
        /*let optlen = match level as u64 {
            LibcConst::SOL_IPV6 => {
                match name as u64 {
                    LibcConst::IPV6_V6ONLY => SocketSize::SIZEOF_INT32,
                    _ => 0,
                }
            }
            LibcConst::SOL_SOCKET => {
                match name as u64 {
                    LibcConst::SO_SNDBUF
                    | LibcConst::SO_RCVBUF
                    | LibcConst::SO_REUSEADDR => {
                        SocketSize::SIZEOF_INT32
                    }
                    _ => 0,
                }
            }
            LibcConst::SOL_TCP => {
                match name as u64 {
                    LibcConst::TCP_NODELAY => SocketSize::SIZEOF_INT32,
                    _ => 0,
                }
            }
            _ => 0,
        };

        if optlen == 0 {
            return Err(Error::SysError(SysErr::ENOPROTOOPT))
        }

        if opt.len() < optlen {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let opt = &opt[..optlen];*/

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
            let val: i32 = task.CopyInObj::<i32>(&opt[0] as *const _ as u64)?;

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

        {
            let peerName = self.remoteAddr.lock();
            match &*peerName {
                None => (),
                Some(addr) => {
                    let v = addr.ToVec()?;
                    for i in 0..v.len() {
                        socketaddr[i] = v[i];
                    }

                    return Ok(v.len() as i64);
                }
            }
        }

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
        let buf = self.socketType.lock().clone();

        //todo: we don't support MSG_ERRQUEUE
        if flags & MsgType::MSG_ERRQUEUE != 0 {
            // Pretend we have an empty error queue.
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        if flags
            & !(MsgType::MSG_DONTWAIT
                | MsgType::MSG_PEEK
                | MsgType::MSG_TRUNC
                | MsgType::MSG_CTRUNC
                | MsgType::MSG_WAITALL)
            != 0
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let waitall = (flags & MsgType::MSG_WAITALL) != 0;
        let dontwait = (flags & MsgType::MSG_DONTWAIT) != 0;
        let trunc = (flags & MsgType::MSG_TRUNC) != 0;
        let peek = (flags & MsgType::MSG_PEEK) != 0;

        if buf.RClosed() {
            let senderAddr = if senderRequested {
                let addr = self.remoteAddr.lock().as_ref().unwrap().clone();
                let l = addr.Len();
                Some((addr, l))
            } else {
                None
            };

            let (retFlags, controlData) = self.prepareControlMessage(controlDataLen);
            return Ok((0 as i64, retFlags, senderAddr, controlData));
        }

        let len = IoVec::NumBytes(dsts);
        let data = if trunc { Some(Iovs(dsts).Data()) } else { None };

        let mut iovs = dsts;

        let mut count = 0;
        let mut tmp;

        if dontwait {
            match self.ReadFromBuf(task, &buf, iovs, peek) {
                Err(e) => return Err(e),
                Ok(count) => {
                    let senderAddr = if senderRequested {
                        let addr = self.remoteAddr.lock().as_ref().unwrap().clone();
                        let l = addr.Len();
                        Some((addr, l))
                    } else {
                        None
                    };

                    let (retFlags, controlData) = self.prepareControlMessage(controlDataLen);
                    return Ok((count, retFlags, senderAddr, controlData));
                }
            }
        }

        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_READ);
        defer!(self.EventUnregister(task, &general));

        'main: loop {
            loop {
                match self.ReadFromBuf(task, &buf, iovs, peek) {
                    Err(Error::SysError(SysErr::EWOULDBLOCK)) => {
                        if count > 0 {
                            if dontwait || !waitall {
                                break 'main;
                            }
                        }

                        if count == len as i64 {
                            break 'main;
                        }

                        if count == 0 && dontwait {
                            return Err(Error::SysError(SysErr::EWOULDBLOCK));
                        }

                        break;
                    }
                    Err(e) => {
                        if count > 0 {
                            break 'main;
                        }
                        return Err(e);
                    }
                    Ok(n) => {
                        if n == 0 {
                            break 'main;
                        }

                        count += n;
                        if count == len as i64 || peek {
                            break 'main;
                        }

                        tmp = Iovs(iovs).DropFirst(n as usize);
                        iovs = &mut tmp;
                    }
                };
            }

            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(e) => {
                    if count > 0 {
                        break 'main;
                    }
                    match e {
                        Error::SysError(SysErr::ETIMEDOUT) => {
                            return Err(Error::SysError(SysErr::EAGAIN));
                        }
                        Error::ErrInterrupted => {
                            return Err(Error::SysError(SysErr::ERESTARTSYS));
                        }
                        _ => {
                            return Err(e);
                        }
                    }
                }
                _ => (),
            }
        }

        let senderAddr = if senderRequested {
            let addr = self.remoteAddr.lock().as_ref().unwrap().clone();
            let l = addr.Len();
            Some((addr, l))
        } else {
            None
        };

        if trunc {
            task.mm
                .ZeroDataOutToIovs(task, &data.unwrap(), count as usize, false)?;
        }

        let (retFlags, controlData) = self.prepareControlMessage(controlDataLen);
        return Ok((count as i64, retFlags, senderAddr, controlData));
    }

    fn SendMsg(
        &self,
        task: &Task,
        srcs: &[IoVec],
        flags: i32,
        _msgHdr: &mut MsgHdr,
        deadline: Option<Time>,
    ) -> Result<i64> {
        let buf = self.SocketType();

        if buf.WClosed() {
            return Err(Error::SysError(SysErr::EPIPE));
        }

        /*if msgHdr.msgName != 0 || msgHdr.msgControl != 0 {
            error!("Hostnet Socketbuf doesn't support MsgHdr");
        }*/

        let dontwait = flags & MsgType::MSG_DONTWAIT != 0;

        let len = Iovs(srcs).Count();
        let mut count = 0;
        let mut srcs = srcs;
        let mut tmp;

        if dontwait {
            return self.WriteToBuf(task, &buf, srcs);
        }

        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_WRITE);
        defer!(self.EventUnregister(task, &general));

        loop {
            loop {
                match self.WriteToBuf(task, &buf, srcs) {
                    Err(Error::SysError(SysErr::EWOULDBLOCK)) => {
                        if flags & MsgType::MSG_DONTWAIT != 0 {
                            if count > 0 {
                                return Ok(count);
                            }
                            return Err(Error::SysError(SysErr::EWOULDBLOCK));
                        }

                        if count > 0 {
                            return Ok(count);
                        }

                        if flags & MsgType::MSG_DONTWAIT != 0 {
                            return Err(Error::SysError(SysErr::EWOULDBLOCK));
                        }

                        break;
                    }
                    Err(e) => {
                        if count > 0 {
                            return Ok(count);
                        }

                        return Err(e);
                    }
                    Ok(n) => {
                        count += n;
                        if count == len as i64 {
                            return Ok(count);
                        }
                        tmp = Iovs(srcs).DropFirst(n as usize);
                        srcs = &mut tmp;
                    }
                }
            }

            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    if count > 0 {
                        return Ok(count);
                    }
                    return Err(Error::SysError(SysErr::EWOULDBLOCK));
                }
                Err(e) => {
                    if count > 0 {
                        return Ok(count);
                    }
                    return Err(e);
                }
                _ => (),
            }
        }
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
            error!("Failed to get TCP socket info getsockopt(2) returned {} bytes, expecting {} bytes.", SocketSize::SIZEOF_TCPINFO, ret);
            return 0;
        }

        return info.State as u32;
    }

    fn Type(&self) -> (i32, i32, i32) {
        return (self.family, self.stype, -1);
    }
}
