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


use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt;
use core::ops::Deref;
use core::ptr;
use core::slice;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;

use crate::qlib::mutex::*;
use crate::qlib::rdma_share::*;
use crate::qlib::rdmasocket::*;
use super::super::super::super::common::*;
use super::super::super::super::fileinfo::*;
use super::super::super::super::linux::netdevice::*;
use super::super::super::super::linux::time::Timeval;
use super::super::super::super::linux_def::*;
use super::super::super::super::mem::block::*;
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
use super::super::super::quring::QUring;
use super::tsotsocket::NewTsotSocketFile;
use super::tsotsocket::TsotSocketType;
use crate::qlib::rdmasocket::RDMAServerSock;
// use super::super::super::rdmasocket::*;
use super::super::super::task::*;
use super::super::super::tcpip::tcpip::*;
use super::super::super::GlobalIOMgr;
use super::super::super::GlobalRDMASvcCli;
use super::super::super::Kernel;
use super::super::super::Kernel::HostSpace;
use super::super::super::IOURING;
use super::super::super::SHARESPACE;
use super::super::control::*;
use super::super::socket::*;
use super::super::unix::transport::unix::*;
use super::hostsocket::*;
use super::rdma_socket::*;
use super::uring_socket::*;
use crate::qlib::kernel::socket::hostinet::loopbacksocket::LoopbackSocket;
// use super::super::super::TSC;

lazy_static! {
    pub static ref DUMMY_HOST_SOCKET: DummyHostSocket = DummyHostSocket::New();
}

fn newSocketFile(
    task: &Task,
    family: i32,
    fd: i32,
    stype: i32,
    nonblock: bool,
    socketBuf: SocketBufType,
    addr: Option<Vec<u8>>,
) -> Result<File> {
    let dirent = NewSocketDirent(task, SOCKET_DEVICE.clone(), fd)?;
    let inode = dirent.Inode();
    let iops = inode.lock().InodeOp.clone();
    let hostiops = iops.HostInodeOp().unwrap();
    let s = SocketOperations::New(
        family,
        fd,
        stype,
        hostiops.Queue(),
        hostiops.clone(),
        socketBuf,
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
pub enum SocketBufType {
    Unknown,
    NoTCP,                        // Not TCP Socket
    TCPInit,                      // Init TCP Socket, no listen and no connect
    TCPNormalServer,              // Common TCP Server socket, when socket start to listen
    TCPUringlServer(AcceptQueue), // Uring TCP Server socket, when socket start to listen
    TCPRDMAServer(AcceptQueue),   // TCP Server socket over RDMA
    TCPNormalData,                // Common TCP socket
    Uring(SocketBuff),
    RDMA(SocketBuff),
    Loopback(LoopbackSocket),
}

impl fmt::Debug for SocketBufType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Unknown => write!(f, "SocketBufType::Unknown"),
            Self::NoTCP => write!(f, "SocketBufType::NoTCP"),
            Self::TCPInit => write!(f, "SocketBufType::TCPInit"),
            Self::TCPNormalServer => write!(f, "SocketBufType::TCPNormalServer"),
            Self::TCPUringlServer(_) => write!(f, "SocketBufType::TCPUringlServer"),
            Self::TCPRDMAServer(_) => write!(f, "SocketBufType::TCPRDMAServer"),
            Self::TCPNormalData => write!(f, "SocketBufType::TCPNormalData"),
            Self::Uring(_) => write!(f, "SocketBufType::Uring"),
            Self::RDMA(_) => write!(f, "SocketBufType::RDMA"),
            Self::Loopback(_) => write!(f, "SocketBufType::Loopback"),
        }
    }
}

impl SocketBufType {
    pub fn Accept(&self, accept: AcceptSocket) -> Self {
        match self {
            SocketBufType::TCPNormalServer => return SocketBufType::TCPNormalData,
            //SocketBufType::TCPUringlServer(_) => return SocketBufType::Uring(socketBuf),
            SocketBufType::TCPRDMAServer(_) => {
                match accept {
                    AcceptSocket::SocketBuff(socketBuf) => return SocketBufType::RDMA(socketBuf),
                    /*AcceptSocket::LoopbackSocket(loopback) => {
                        return Self::Loopback(loopback)
                    }*/
                    _ => {
                        panic!("UringSocketType::Accept unexpect AcceptSocket::None")
                    }
                }
            }
            _ => {
                panic!("SocketBufType::Accept unexpect type {:?}", self)
            }
        }
    }

    pub fn Connect(&self) -> Self {
        match self {
            Self::TCPInit => return self.ConnectType(),
            // in bazel, there is UDP socket also call connect
            Self::NoTCP => return Self::NoTCP,
            _ => {
                panic!("SocketBufType::Connect unexpect type {:?}", self)
            }
        }
    }

    fn ConnectType(&self) -> Self {
        if SHARESPACE.config.read().EnableRDMA {
            let socketBuf = SocketBuff(Arc::new_in(
                SocketBuffIntern::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT),
                crate::GUEST_HOST_SHARED_ALLOCATOR,
            ));
            return Self::RDMA(socketBuf);
        } else if SHARESPACE.config.read().UringIO {
            let socketBuf = SocketBuff(Arc::new_in(
                SocketBuffIntern::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT),
                crate::GUEST_HOST_SHARED_ALLOCATOR,
            ));
            return Self::Uring(socketBuf);
        } else {
            return Self::TCPNormalData;
        }
    }
}

pub struct SocketOperationsIntern {
    pub send: AtomicI64,
    pub recv: AtomicI64,
    pub family: i32,
    pub stype: i32,
    pub fd: i32,
    pub queue: Queue,
    pub remoteAddr: QMutex<Option<SockAddr>>,
    pub socketBuf: QMutex<SocketBufType>,
    pub enableAsyncAccept: AtomicBool,
    pub hostops: HostInodeOp,
    passInq: AtomicBool,
    pub tcpRDMA: bool,
    pub udpRDMA: bool,
}

#[derive(Clone)]
pub struct SocketOperations(Arc<SocketOperationsIntern>);

impl SocketOperations {
    pub fn New(
        family: i32,
        fd: i32,
        stype: i32,
        queue: Queue,
        hostops: HostInodeOp,
        socketBuf: SocketBufType,
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

        match &socketBuf {
            SocketBufType::Uring(ref buf) => {
                QUring::BufSockInit(fd, queue.clone(), buf.clone(), true).unwrap();
            }
            _ => (),
        }

        // Only enable RDMA for IPv4 now. (Open IPv6 for nodejs, need a TODO here)
        let tcpRDMA = SHARESPACE.config.read().EnableRDMA
            && (family == AFType::AF_INET || family == AFType::AF_INET6)
            //&& family == AFType::AF_INET
            && (stype == SockType::SOCK_STREAM);
        let udpRDMA = SHARESPACE.config.read().EnableRDMA
            && (family == AFType::AF_INET || family == AFType::AF_INET6)
            // && family == AFType::AF_INET
            && (stype == SockType::SOCK_DGRAM);

        let ret = SocketOperationsIntern {
            send: AtomicI64::new(0),
            recv: AtomicI64::new(0),
            family,
            stype,
            fd,
            queue,
            remoteAddr: QMutex::new(addr),
            socketBuf: QMutex::new(socketBuf.clone()),
            enableAsyncAccept: AtomicBool::new(false),
            hostops: hostops,
            passInq: AtomicBool::new(false),
            tcpRDMA,
            udpRDMA,
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

    pub fn SocketBufType(&self) -> SocketBufType {
        return self.socketBuf.lock().clone();
    }

    pub fn SocketBuf(&self) -> SocketBuff {
        match self.SocketBufType() {
            SocketBufType::Uring(b) => return b,
            SocketBufType::RDMA(b) => return b,
            _ => panic!(
                "SocketBufType::None has no SockBuff {:?}",
                self.SocketBufType()
            ),
        }
    }

    pub fn SocketBufEnabled(&self) -> bool {
        match self.SocketBufType() {
            SocketBufType::Uring(_) => return true,
            SocketBufType::RDMA(_) => return true,
            _ => false,
        }
    }

    pub fn AcceptQueue(&self) -> Option<AcceptQueue> {
        match self.SocketBufType() {
            SocketBufType::TCPUringlServer(q) => return Some(q.clone()),
            SocketBufType::TCPRDMAServer(q) => return Some(q.clone()),
            _ => return None,
        }
    }

    pub fn PostConnect(&self, _task: &Task) {
        let socketBuf;
        if self.tcpRDMA {
            socketBuf = self.socketBuf.lock().clone();
        } else {
            socketBuf = self.SocketBufType().Connect();
            *self.socketBuf.lock() = socketBuf.clone();
        }

        match socketBuf {
            SocketBufType::RDMA(_buf) => {
                assert!(
                    (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
                        && self.stype == SockType::SOCK_STREAM,
                    "family {}, stype {}",
                    self.family,
                    self.stype
                );
                // HostSpace::PostRDMAConnect(task, self.fd, buf);
                let sockInfo = GlobalIOMgr()
                    .GetByHost(self.fd)
                    .unwrap()
                    .lock()
                    .sockInfo
                    .lock()
                    .clone();
                match sockInfo {
                    SockInfo::RDMADataSocket(rdmaSocket) => {
                        *self.socketBuf.lock() = SocketBufType::RDMA(rdmaSocket.socketBuf.clone());
                    }
                    _ => {
                        panic!(
                            "PostConnect, Incorrect sockInfo: {:?}, fd: {}",
                            sockInfo, self.fd
                        );
                    }
                }
            }
            SocketBufType::Uring(buf) => {
                assert!(
                    (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
                        && self.stype == SockType::SOCK_STREAM,
                    "family {}, stype {}",
                    self.family,
                    self.stype
                );
                QUring::BufSockInit(self.fd, self.queue.clone(), buf, true).unwrap();
            }
            _ => (),
        }

        /*assert!((self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            && self.stype == SockType::SOCK_STREAM, "family {}, stype {}", self.family, self.stype);

        let socketBuf = Arc::new(SocketBuff::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT));
        *self.socketBuf.lock() = Some(socketBuf);
        self.enableSocketBuf.store(true, Ordering::Relaxed);
        QUring::BufSockInit(self.fd, self.queue.clone(), self.SocketBuf(), true).unwrap();*/
    }

    pub fn Notify(&self, mask: EventMask) {
        self.queue.Notify(EventMaskFromLinux(mask as u32));
    }

    pub fn AcceptData(&self) -> Result<AcceptItem> {
        let sockBufType = self.socketBuf.lock().clone();
        match sockBufType {
            SocketBufType::TCPNormalServer => return self.IOAccept(),
            SocketBufType::TCPUringlServer(ref queue) => {
                return IOURING.Accept(self.fd, &self.queue, queue)
            }
            SocketBufType::TCPRDMAServer(ref queue) => return RDMA::Accept(self.fd, queue),
            _ => {
                error!("SocketBufType invalid accept {:?}", sockBufType);
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }
    }

    pub fn ReadFromBuf(
        &self,
        task: &Task,
        sockBufType: SocketBufType,
        dsts: &mut [IoVec],
        peek: bool,
    ) -> Result<i64> {
        match sockBufType {
            SocketBufType::Uring(socketBuf) => {
                let ret = QUring::RingFileRead(
                    task,
                    self.fd,
                    self.queue.clone(),
                    socketBuf,
                    dsts,
                    true,
                    peek,
                )?;
                return Ok(ret);
            }
            SocketBufType::RDMA(socketBuf) => {
                let ret = RDMA::Read(task, self.fd, socketBuf, dsts, peek);
                return ret;
            }
            t => {
                panic!("ReadFromBuf get type {:?}", t);
            }
        }
    }

    pub fn WriteToBuf(
        &self,
        task: &Task,
        sockBufType: SocketBufType,
        srcs: &[IoVec],
    ) -> Result<i64> {
        match sockBufType {
            /*SocketBufType::Uring(socketBuf) => {
                let ret =
                    QUring::SocketSend(task, self.fd, self.queue.clone(), socketBuf, srcs, self)?;
                return Ok(ret);
            }*/
            SocketBufType::RDMA(socketBuf) => {
                let ret = RDMA::Write(task, self.fd, socketBuf, srcs);
                return ret;
            }
            t => {
                panic!("ReadFromBuf get type {:?}", t);
            }
        }
    }
}

impl Deref for SocketOperations {
    type Target = Arc<SocketOperationsIntern>;

    fn deref(&self) -> &Arc<SocketOperationsIntern> {
        &self.0
    }
}

impl SocketOperations {
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

    fn UDPEventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = GlobalRDMASvcCli().udpSentBufferAllocator.lock().Queue();
        queue.EventRegister(task, e, mask);
    }

    fn UDPEventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = GlobalRDMASvcCli().udpSentBufferAllocator.lock().Queue();
        queue.EventUnregister(task, e);
    }
}

pub const SIZEOF_SOCKADDR: usize = SocketSize::SIZEOF_SOCKADDR_INET6;

impl Waitable for SocketOperations {
    fn AsyncReadiness(&self, _task: &Task, mask: EventMask, wait: &MultiWait) -> Future<EventMask> {
        if self.SocketBufEnabled() {
            let future = Future::New(0 as EventMask);
            let ret = self.SocketBuf().Events() & mask;
            future.Set(Ok(ret));
            //wait.Done();
            return future;
        };

        let fd = self.fd;
        let future = IOURING.UnblockPollAdd(fd, mask as u32, wait);
        return future;
    }

    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        if self.SocketBufEnabled() {
            if self.tcpRDMA {
                let sockInfo = GlobalIOMgr()
                    .GetByHost(self.fd)
                    .unwrap()
                    .lock()
                    .sockInfo
                    .lock()
                    .clone();

                match sockInfo {
                    SockInfo::RDMADataSocket(dataSock) => {
                        return dataSock.socketBuf.Events() & mask;
                    }
                    _ => {
                        return 0;
                    }
                }
            }
            return self.SocketBuf().Events() & mask;
        };

        if self.udpRDMA {
            let sockInfo = GlobalIOMgr()
                .GetByHost(self.fd)
                .unwrap()
                .lock()
                .sockInfo
                .lock()
                .clone();
            match sockInfo {
                SockInfo::RDMAUDPSocket(sock) => {
                    return (sock.recvQueue.lock().Events() | WRITEABLE_EVENT) & mask;
                }
                _ => {
                    return 0;
                }
            }
        }

        match self.AcceptQueue() {
            Some(q) => return q.lock().Events() & mask,
            None => (),
        }

        let fd = self.fd;
        return NonBlockingPoll(fd, mask);
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.queue.clone();
        queue.EventRegister(task, e, mask);
        let fd = self.fd;
        if !self.SocketBufEnabled() && self.AcceptQueue().is_none() {
            UpdateFD(fd).unwrap();
        };
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.queue.clone();
        queue.EventUnregister(task, e);
        let fd = self.fd;
        if !self.SocketBufEnabled() && self.AcceptQueue().is_none() {
            UpdateFD(fd).unwrap();
        };
    }
}

pub struct DummyHostSocket {
    pub socket: QMutex<i32>,
}

impl DummyHostSocket {
    pub fn New() -> Self {
        return Self {
            socket: QMutex::new(-1),
        };
    }

    pub fn Socket(&self) -> i32 {
        let mut s = self.socket.lock();
        if *s == -1 {
            let fd = HostSpace::Socket(AFType::AF_UNIX, SockType::SOCK_DGRAM, 0);
            if fd < 0 {
                panic!("HostSocket create socket fail with error {}", fd);
            }

            *s = fd as i32;
        };

        return *s;
    }

    pub fn HostIoctlIFConf(&self, task: &Task, request: u64, addr: u64) -> Result<()> {
        return HostIoctlIFConf(task, self.Socket(), request, addr);
    }
}

// pass the ioctl to the shadow hostfd
pub fn HostIoctlIFReq(task: &Task, hostfd: i32, request: u64, addr: u64) -> Result<()> {
    let mut ifr: IFReq = task.CopyInObj(addr)?;
    let res = HostSpace::IoCtl(hostfd, request, &mut ifr as *const _ as u64,core::mem::size_of::<IFReq>());
    if res < 0 {
        return Err(Error::SysError(-res as i32));
    }

    task.CopyOutObj(&ifr, addr)?;
    return Ok(());
}

pub fn HostIoctlIFConf(task: &Task, hostfd: i32, request: u64, addr: u64) -> Result<()> {
    let mut ifc: IFConf = task.CopyInObj(addr)?;

    const MAX_LEN: usize = 64 * 0x1000; // 256 KB

    // todo: how to handle very large ifconf?
    let len = if MAX_LEN > ifc.Len as usize {
        ifc.Len as usize
    } else {
        MAX_LEN
    };

    let buf = DataBuff::New(len);

    let mut ifr = IFConf {
        Len: len as i32,
        ..Default::default()
    };

    if ifc.Ptr != 0 {
        ifr.Ptr = buf.Ptr();
    }

    let res = HostSpace::IoCtl(hostfd, request, &mut ifr as *const _ as u64, core::mem::size_of::<IFConf>());
    if res < 0 {
        return Err(Error::SysError(-res as i32));
    }

    if ifc.Ptr > 0 {
        task.mm
            .CopyDataOut(task, ifr.Ptr, ifc.Ptr, ifr.Len as usize, false)?;
    }

    ifc.Len = ifr.Len;

    task.CopyOutObj(&ifc, addr)?;
    return Ok(());
}

impl SpliceOperations for SocketOperations {}

impl FileOperations for SocketOperations {
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
        let sockBufType = self.socketBuf.lock().clone();
        match sockBufType {
            SocketBufType::Uring(socketBuf) => {
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
            SocketBufType::RDMA(socketBuf) => {
                let ret = RDMA::Read(task, self.fd, socketBuf, dsts, false);
                return ret;
            }
            _ => {
                let size = IoVec::NumBytes(dsts);
                let buf = DataBuff::New(size);
                let iovs = buf.Iovs(size);
                let ret = IORead(self.fd, &iovs)?;

                // handle partial memcopy
                task.CopyDataOutToIovs(&buf.buf[0..ret as usize], dsts, false)?;
                return Ok(ret);
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

        let sockBufType = self.socketBuf.lock().clone();
        match sockBufType {
            /*SocketBufType::Uring(socketBuf) => {
                if self.SocketBuf().WClosed() {
                    return Err(Error::SysError(SysErr::ESPIPE));
                }

                return QUring::SocketSend(
                    task,
                    self.fd,
                    self.queue.clone(),
                    socketBuf,
                    srcs,
                    self,
                );
            }*/
            SocketBufType::RDMA(socketBuf) => {
                let ret = RDMA::Write(task, self.fd, socketBuf, srcs)?;
                return Ok(ret);
            }
            _ => {
                let size = IoVec::NumBytes(srcs);
                let mut buf = DataBuff::New(size);
                let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
                let iovs = buf.Iovs(len);
                return IOWrite(self.fd, &iovs);
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
                } else if self.udpRDMA {
                    let sockInfo = GlobalIOMgr()
                        .GetByHost(self.fd)
                        .unwrap()
                        .lock()
                        .sockInfo
                        .lock()
                        .clone();
                    match sockInfo {
                        SockInfo::RDMAUDPSocket(udpSock) => {
                            let mut v = udpSock.recvQueue.lock().udpRecvQueue.len();
                            if v != 0 {
                                let idx = udpSock
                                    .recvQueue
                                    .lock()
                                    .udpRecvQueue
                                    .get(0)
                                    .unwrap()
                                    .udpBuffIdx;
                                let udpPacket =
                                    &GlobalRDMASvcCli().cliShareRegion.lock().udpBufRecv
                                        [idx as usize];
                                v = udpPacket.length as usize;
                            }

                            task.CopyOutObj(&v, val)?;
                            return Ok(0);
                        }
                        _ => {
                            return Err(Error::SysError(SysErr::EINVAL));
                        }
                    }
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
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SocketOperations {
    //pub fn ConnectIntern(fd: i32, addr: u64, addrlen: u32) -> i64 {}
}

impl SockOperations for SocketOperations {
    fn Connect(&self, task: &Task, sockaddr: &[u8], blocking: bool) -> Result<i64> {
        // GlobalRDMASvcCli().timestamp.lock().push(TSC.Rdtsc()); //1
        let mut socketaddr = sockaddr;

        if (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            && socketaddr.len() > SIZEOF_SOCKADDR
        {
            socketaddr = &socketaddr[..SIZEOF_SOCKADDR]
        }

        if self.udpRDMA {
            self.SetRemoteAddr(socketaddr.to_vec())?;
            return Ok(0);
        }

        let res;
        if self.tcpRDMA {
            let sockAddr = GetAddr(sockaddr[0] as i16, &sockaddr[0..sockaddr.len()])?;
            match sockAddr {
                SockAddr::Inet(ipv4) => {
                    let ipAddr = u32::from_be_bytes(ipv4.Addr);
                    let port = ipv4.Port.to_le();

                    // TODO: handle port used up!!
                    let srcPort = (GlobalRDMASvcCli()
                        .tcpPortAllocator
                        .lock()
                        .AllocFromCurrent()
                        .unwrap() as u16)
                        .to_be();
                    let rdmaId = GlobalRDMASvcCli()
                        .nextRDMAId
                        .fetch_add(1, Ordering::Release);
                    GlobalRDMASvcCli()
                        .rdmaIdToSocketMappings
                        .lock()
                        .insert(rdmaId, self.fd);
                    let _ret = GlobalRDMASvcCli().connectUsingPodId(rdmaId, ipAddr, port, srcPort);
                    let socketBuf = self.SocketBufType().Connect();
                    *self.socketBuf.lock() = socketBuf.clone();
                    // GlobalRDMASvcCli().timestamp.lock().push(TSC.Rdtsc()); // 3 (38)
                    res = -SysErr::EINPROGRESS;
                }
                _ => {
                    panic!("sockAddr: {:?} can't enable RDMA!", sockAddr);
                }
            }
        } else {
            res = Kernel::HostSpace::IOConnect(
                self.fd,
                &socketaddr[0] as *const _ as u64,
                socketaddr.len() as u32,
            ) as i32;
            if res == 0 {
                self.SetRemoteAddr(socketaddr.to_vec())?;
                if self.stype == SockType::SOCK_STREAM {
                    self.PostConnect(task);
                }

                return Ok(0);
            }
        }

        let blocking = if blocking {
            true
        } else {
            // in order to enable uring buff, have to do block accept
            if SHARESPACE.config.read().UringIO
                && (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
                && self.stype == SockType::SOCK_STREAM
            {
                true
            } else {
                false
            }

            //false
        };

        // // let t2 = Timestamp();
        // GlobalRDMASvcCli().timestamp.lock().push(TSC.Rdtsc()); // 4 (1)

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
                        return Err(Error::SysError(SysErr::EINTR));
                    }
                    Err(e) => {
                        error!("connect error {:?}", &e);
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
            if val == SysErr::ECONNREFUSED {
                return Err(Error::SysError(SysErr::EINPROGRESS));
            }
            return Err(Error::SysError(val as i32));
        }

        self.SetRemoteAddr(socketaddr.to_vec())?;
        if self.stype == SockType::SOCK_STREAM {
            self.PostConnect(task);
        }
        // GlobalRDMASvcCli().timestamp.lock().push(TSC.Rdtsc()); // 7
        // let mut len = GlobalRDMASvcCli().timestamp.lock().len();
        // let mut i = 1;
        // let mut v1 = GlobalRDMASvcCli().timestamp.lock()[0];
        // let mut v2 = GlobalRDMASvcCli().timestamp.lock()[1];
        // error!("qq, Connect time is: {}", GlobalRDMASvcCli().timestamp.lock()[len - 1] - v1);
        // loop {
        //     error!("{}", v2 - v1);
        //     i += 1;
        //     if i == len {
        //         break;
        //     }
        //     v1 = v2;
        //     v2 = GlobalRDMASvcCli().timestamp.lock()[i];
        // }

        // GlobalRDMASvcCli().timestamp.lock().clear();

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
            let ai = self.AcceptData();

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
        //let sockBuf = self.ConfigSocketBufType();
        let sockBuf = self.SocketBufType().Accept(acceptItem.sockBuf.clone());

        let file = newSocketFile(
            task,
            self.family,
            fd as i32,
            self.stype,
            flags & SocketFlags::SOCK_NONBLOCK != 0,
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

        //TODO: remember ipAddr and port, hardcoded for now.
        if self.tcpRDMA || self.udpRDMA {
            let sockAddr = GetAddr(sockaddr[0] as i16, &sockaddr[0..sockaddr.len()])?;
            match sockAddr {
                SockAddr::Inet(ipv4) => {
                    let port = ipv4.Port.to_le();
                    let fdInfo = GlobalIOMgr().GetByHost(self.fd).unwrap();
                    if self.tcpRDMA {
                        *fdInfo.lock().sockInfo.lock() = SockInfo::Socket(SocketInfo {
                            ipAddr: u32::from_be_bytes(ipv4.Addr), //u32::from_be_bytes([192, 168, 6, 8]), //ipAddr: u32::from_be_bytes(ipv4.Addr), // ipAddr: 3232237064,
                            port,                                  // port: 58433,
                        }); //192.168.6.8:16868
                    } else if self.udpRDMA {
                        debug!("SocketOperations::Bind, port: {}", port);
                        *fdInfo.lock().sockInfo.lock() =
                            SockInfo::RDMAUDPSocket(RDMAUDPSock::New(self.queue.clone(), port));
                        GlobalRDMASvcCli()
                            .portToFdInfoMappings
                            .lock()
                            .insert(port, fdInfo.clone());
                    }
                }
                SockAddr::Inet6(ipv6) => {
                    let port = ipv6.Port.to_le();
                    let fdInfo = GlobalIOMgr().GetByHost(self.fd).unwrap();
                    if self.tcpRDMA {
                        *fdInfo.lock().sockInfo.lock() = SockInfo::Socket(SocketInfo {
                            //ipAddr: u32::from_be_bytes(ipv6.Addr), //u32::from_be_bytes([192, 168, 6, 8]), //ipAddr: u32::from_be_bytes(ipv4.Addr), // ipAddr: 3232237064,
                            ipAddr: u32::from_be_bytes([0, 0, 0, 0]), //TODO: this is a temp workaround for nodejs
                            port,                                     // port: 58433,
                        }); //192.168.6.8:16868
                    }
                    if self.udpRDMA {
                        *fdInfo.lock().sockInfo.lock() =
                            SockInfo::RDMAUDPSocket(RDMAUDPSock::New(self.queue.clone(), port));
                        GlobalRDMASvcCli()
                            .portToFdInfoMappings
                            .lock()
                            .insert(port, fdInfo.clone());
                    }
                }
                _ => {
                    panic!("sockAddr: {:?} can't enable RDMA!", sockAddr);
                }
            }
        }

        return Ok(res);
    }

    fn Listen(&self, _task: &Task, backlog: i32) -> Result<i64> {
        let asyncAccept = SHARESPACE.config.read().AsyncAccept
            && (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            && self.stype == SockType::SOCK_STREAM;

        let len = if backlog <= 0 { 5 } else { backlog };

        let socketBuf = self.socketBuf.lock().clone();
        let acceptQueue = match socketBuf {
            SocketBufType::TCPUringlServer(q) => {
                q.lock().SetQueueLen(len as usize);
                return Ok(0);
            }
            SocketBufType::TCPRDMAServer(q) => {
                q.lock().SetQueueLen(len as usize);
                return Ok(0);
            }
            SocketBufType::TCPInit => AcceptQueue::New(len as usize, self.queue.clone()),
            _ => panic!("socket::listen unexpect buf type {:?}", socketBuf), // panic?
        };

        let res = if self.tcpRDMA {
            // Kernel::HostSpace::RDMAListen(self.fd, backlog, asyncAccept, acceptQueue.clone())
            let fdInfo = GlobalIOMgr().GetByHost(self.fd).unwrap();
            let socketInfo = fdInfo.lock().sockInfo.lock().clone();

            let port;
            match socketInfo {
                SockInfo::Socket(info) => {
                    port = info.port;
                    let rdmaId = GlobalRDMASvcCli()
                        .nextRDMAId
                        .fetch_add(1, Ordering::Release);
                    GlobalRDMASvcCli()
                        .rdmaIdToSocketMappings
                        .lock()
                        .insert(rdmaId, self.fd);
                    let rdmaSocket =
                        RDMAServerSock::New(rdmaId, acceptQueue.clone(), info.ipAddr, info.port);
                    *fdInfo.lock().sockInfo.lock() = SockInfo::RDMAServerSocket(rdmaSocket);
                    debug!("Listen, rdmaId: {}, serverSockFd: {}", rdmaId, self.fd);
                    let _ret = GlobalRDMASvcCli().listenUsingPodId(rdmaId, port, backlog);
                }
                _ => {
                    panic!("RDMA Listen with wrong state");
                }
            }
            0
        } else {
            Kernel::HostSpace::Listen(self.fd, backlog, asyncAccept)
        };

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        *self.socketBuf.lock() = if self.tcpRDMA {
            SocketBufType::TCPRDMAServer(acceptQueue)
        } else if asyncAccept {
            if !self.AsyncAcceptEnabled() {
                IOURING.AcceptInit(self.fd, &self.queue, &acceptQueue)?;
                self.enableAsyncAccept.store(true, Ordering::Relaxed);
            }

            SocketBufType::TCPUringlServer(acceptQueue)
        } else {
            SocketBufType::TCPNormalServer
        };

        return Ok(res);
    }

    fn Shutdown(&self, task: &Task, how: i32) -> Result<i64> {
        let how = how as u64;

        if self.stype == SockType::SOCK_STREAM
            && (how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR)
        {
            if self.tcpRDMA {
                //TODO:
                let fdInfo = GlobalIOMgr().GetByHost(self.fd).unwrap();
                let socketInfo = fdInfo.lock().sockInfo.lock().clone();
                match socketInfo {
                    SockInfo::RDMADataSocket(dataSocket) => {
                        let _res = GlobalRDMASvcCli().shutdown(dataSocket.channelId, how as u8);
                    }
                    _ => {
                        error!("Shutdown with sockInfo: {:?}", socketInfo);
                    }
                }
            }
            if self.SocketBuf().HasWriteData() {
                self.SocketBuf().SetPendingWriteShutdown();
                let general = task.blocker.generalEntry.clone();
                self.EventRegister(task, &general, EVENT_PENDING_SHUTDOWN);
                defer!(self.EventUnregister(task, &general));

                if self.tcpRDMA {
                    let fdInfo = GlobalIOMgr().GetByHost(self.fd).unwrap();
                    let socketInfo = fdInfo.lock().sockInfo.lock().clone();
                    match socketInfo {
                        SockInfo::RDMADataSocket(dataSocket) => {
                            let _res = GlobalRDMASvcCli().pendingshutdown(dataSocket.channelId);
                        }
                        _ => {
                            panic!("Shutdown with sockInfo: {:?}", socketInfo);
                        }
                    }
                }
                while self.SocketBuf().HasWriteData() {
                    task.blocker.BlockGeneralOnly();
                }
            }
        }

        if how == LibcConst::SHUT_RD || how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR {
            let res = 0;

            if !self.tcpRDMA {
                let res = Kernel::HostSpace::Shutdown(self.fd, how as i32);
                if res < 0 {
                    return Err(Error::SysError(-res as i32));
                }
            }

            if self.stype == SockType::SOCK_STREAM
                && (how == LibcConst::SHUT_RD || how == LibcConst::SHUT_RDWR)
            {
                self.SocketBuf().SetRClosed();
                self.queue.Notify(EventMaskFromLinux(EVENT_HUP as u32));
            }

            if self.stype == SockType::SOCK_STREAM
                && (how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR)
            {
                self.SocketBuf().SetWClosed();
                self.queue.Notify(EventMaskFromLinux(EVENT_HUP as u32));
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
        // if self.tcpRDMA {
        //     return Ok(0);
        // }
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
        if self.tcpRDMA || self.udpRDMA {
            let fdInfo = GlobalIOMgr().GetByHost(self.fd).unwrap();
            let fdInfoLock = fdInfo.lock();
            let sockInfo = fdInfoLock.sockInfo.lock().clone();
            let ipAddr;
            let port;
            match sockInfo {
                SockInfo::RDMADataSocket(sock) => {
                    ipAddr = sock.localIpAddr;
                    port = sock.localPort;
                }
                SockInfo::RDMAServerSocket(sock) => {
                    ipAddr = sock.ipAddr;
                    port = sock.port;
                }
                SockInfo::Socket(sock) => {
                    ipAddr = sock.ipAddr;
                    port = sock.port;
                }
                SockInfo::RDMAUDPSocket(sock) => {
                    ipAddr = 0;
                    port = sock.port;
                }
                _ => {
                    panic!("Incorrect sockInfo")
                }
            }

            let sockAddr;
            let addrSlice = ipAddr.to_be_bytes();
            if self.family == AFType::AF_INET {
                sockAddr = SockAddr::Inet(SockAddrInet {
                    Family: AFType::AF_INET as u16,
                    Port: port,
                    Addr: addrSlice,
                    Zero: [0; 8],
                });
            } else {
                sockAddr = SockAddr::Inet6(SocketAddrInet6 {
                    Family: AFType::AF_INET6 as u16,
                    Port: port,
                    Flowinfo: 0,
                    Addr: [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0xFF,
                        0xFF,
                        addrSlice[0],
                        addrSlice[1],
                        addrSlice[2],
                        addrSlice[3],
                    ],
                    Scope_id: 0,
                });
            }
            let len = socketaddr.len() as usize;
            sockAddr.Marsh(socketaddr, len)?;

            //TODO: handle unhappy case
            return Ok(len as i64);
        }
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
        if self.tcpRDMA {
            let fdInfo = GlobalIOMgr().GetByHost(self.fd).unwrap();
            let fdInfoLock = fdInfo.lock();
            let sockInfo = fdInfoLock.sockInfo.lock().clone();
            let ipAddr;
            let port;
            match sockInfo {
                SockInfo::RDMADataSocket(sock) => {
                    ipAddr = sock.peerIpAddr;
                    port = sock.peerPort;
                }
                SockInfo::RDMAServerSocket(_sock) => {
                    return Err(Error::SysError(SysErr::ENOTCONN));
                }
                SockInfo::Socket(_sock) => {
                    return Err(Error::SysError(SysErr::ENOTCONN));
                }
                _ => {
                    panic!("Incorrect sockInfo")
                }
            }
            debug!("GetPeerName, ipAddr: {}, port: {}", ipAddr, port);
            let sockAddr = SockAddr::Inet(SockAddrInet {
                Family: AFType::AF_INET as u16,
                Port: port,
                Addr: ipAddr.to_be_bytes(),
                Zero: [0; 8],
            });
            let len = sockAddr.Len();
            sockAddr.Marsh(socketaddr, len)?;
            //TODO: handle unhappy case
            return Ok(len as i64);
        }
        if self.udpRDMA {
            return Err(Error::SysError(SysErr::ENOTCONN));
        }

        // Should not come to this point!
        return Err(Error::SysError(SysErr::EINVAL));
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

        if self.SocketBufEnabled() {
            if self.SocketBuf().RClosed() {
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
            let socketType = self.SocketBufType();

            let general = task.blocker.generalEntry.clone();
            self.EventRegister(task, &general, EVENT_READ);
            defer!(self.EventUnregister(task, &general));

            'main: loop {
                loop {
                    match self.ReadFromBuf(task, socketType.clone(), iovs, peek) {
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

        /*
        if IoVec::NumBytes(dsts) == 0 {
            return Ok((0, 0, None, SCMControlMessages::default()))
        }
        */

        /*defer!(task.GetPtr().iovs.clear());
        task.V2PIovs(dsts, true, &mut task.GetPtr().iovs)?;
        let iovs = &mut task.GetPtr().iovs;*/

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

        if self.tcpRDMA {
            //TODO: this needs revisit for TCP over RDMA
            let mut res = if msgHdr.msgControlLen != 0 {
                Kernel::HostSpace::IORecvMsg(
                    self.fd,
                    &mut msgHdr as *mut _ as u64,
                    flags | MsgType::MSG_DONTWAIT,
                    false,
                ) as i32
            } else {
                Kernel::HostSpace::IORecvfrom(
                    self.fd,
                    buf.Ptr(),
                    size,
                    flags | MsgType::MSG_DONTWAIT,
                    msgHdr.msgName,
                    &msgHdr.nameLen as *const _ as u64,
                ) as i32
            };

            while res == -SysErr::EWOULDBLOCK && flags & MsgType::MSG_DONTWAIT == 0 {
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

                res = if msgHdr.msgControlLen != 0 {
                    Kernel::HostSpace::IORecvMsg(
                        self.fd,
                        &mut msgHdr as *mut _ as u64,
                        flags | MsgType::MSG_DONTWAIT,
                        false,
                    ) as i32
                } else {
                    Kernel::HostSpace::IORecvfrom(
                        self.fd,
                        buf.Ptr(),
                        size,
                        flags | MsgType::MSG_DONTWAIT,
                        msgHdr.msgName,
                        &msgHdr.nameLen as *const _ as u64,
                    ) as i32
                };
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
        } else {
            // if msgHdr.msgControlLen != 0 {
            //     panic!("TODO: UDP over RDMA doesn't support control msg yet!");
            // }
            let sockInfo = GlobalIOMgr()
                .GetByHost(self.fd)
                .unwrap()
                .lock()
                .sockInfo
                .lock()
                .clone();
            match sockInfo {
                SockInfo::RDMAUDPSocket(udpSock) => {
                    let mut recvUdpItem = UDPRecvItem::default();
                    if flags & MsgType::MSG_DONTWAIT != 0 {
                        let (_trigger, recvItem) = udpSock.recvQueue.lock().DeqSocket();
                        match recvItem {
                            Err(Error::SysError(SysErr::EAGAIN)) => {
                                if flags & MsgType::MSG_DONTWAIT != 0 {
                                    return Err(Error::SysError(SysErr::EAGAIN));
                                }
                            }
                            Err(e) => return Err(e),
                            Ok(item) => {
                                recvUdpItem = item;
                            }
                        }
                    } else {
                        //blocking
                        loop {
                            let (_trigger, recvItem) = udpSock.recvQueue.lock().DeqSocket();
                            match recvItem {
                                Err(Error::SysError(SysErr::EAGAIN)) => (),
                                Err(e) => return Err(e),
                                Ok(item) => {
                                    recvUdpItem = item;
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

                    let srcPort;
                    let srcIpAddr;
                    let len;
                    {
                        let udpPacket = &GlobalRDMASvcCli().cliShareRegion.lock().udpBufRecv
                            [recvUdpItem.udpBuffIdx as usize];
                        let buf = &udpPacket.buf[0..udpPacket.length as usize];
                        len = task.CopyDataOutToIovs(buf, dsts, false)?;
                        srcPort = udpPacket.srcPort.clone();
                        srcIpAddr = udpPacket.srcIpAddr.clone();
                        debug!(
                            "qq, RecvMsg 4.1 fd: {}, len: {}, buf {:?} ",
                            self.fd, len, buf
                        );
                    }
                    let _res = GlobalRDMASvcCli().returnUDPBuff(recvUdpItem.udpBuffIdx);
                    let senderAddr = if senderRequested {
                        let addr;
                        if self.remoteAddr.lock().is_some() {
                            addr = self.remoteAddr.lock().as_ref().unwrap().clone();
                        } else {
                            if self.family == AFType::AF_INET {
                                addr = SockAddr::Inet(SockAddrInet {
                                    Family: AFType::AF_INET as u16,
                                    Port: srcPort,
                                    Addr: srcIpAddr.to_be_bytes(),
                                    Zero: [0; 8],
                                });
                            } else {
                                // if self.family == AFType::AF_INET6 {
                                let srcIp = srcIpAddr.to_be_bytes();
                                addr = SockAddr::Inet6(SocketAddrInet6 {
                                    Family: AFType::AF_INET6 as u16,
                                    Port: srcPort,
                                    Flowinfo: 0,
                                    Addr: [
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, srcIp[0],
                                        srcIp[1], srcIp[2], srcIp[3],
                                    ],
                                    Scope_id: 0,
                                });
                            }
                        }

                        let l = addr.Len();
                        Some((addr, l))
                    } else {
                        None
                    };
                    let msgFlags = msgHdr.msgFlags & !MsgType::MSG_CTRUNC;
                    return Ok((len as i64, msgFlags, senderAddr, controlVec));
                }
                _ => {
                    panic!("SockInfo: {:?} is not expected for UDP", sockInfo);
                }
            }
        }
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

        if msgHdr.msgControlLen > 0 {
            panic!("TCP/UDP over RDMA doesn't support SendMsg with ancillary data");
        }
        if self.SocketBufEnabled() {
            if self.SocketBuf().WClosed() {
                return Err(Error::SysError(SysErr::EPIPE));
            }

            if msgHdr.msgName != 0 || msgHdr.msgControl != 0 {
                panic!("Hostnet Socketbuf doesn't supprot MsgHdr");
            }

            let len = Iovs(srcs).Count();
            let mut count = 0;
            let mut srcs = srcs;
            let mut tmp;
            let socketType = self.SocketBufType();
            let general = task.blocker.generalEntry.clone();
            self.EventRegister(task, &general, EVENT_WRITE);
            defer!(self.EventUnregister(task, &general));

            loop {
                loop {
                    match self.WriteToBuf(task, socketType.clone(), srcs) {
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

        // Handle UDP over RDMA
        if self.udpRDMA {
            let totalLen = Iovs(srcs).Count();
            if totalLen > UDP_BUF_COUNT {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            let sockInfo = GlobalIOMgr()
                .GetByHost(self.fd)
                .unwrap()
                .lock()
                .sockInfo
                .lock()
                .clone();
            let port;
            match sockInfo {
                SockInfo::RDMAUDPSocket(sockInfo) => {
                    if sockInfo.port == 0 {
                        // TODO: handle port used up!!
                        port = (GlobalRDMASvcCli()
                            .udpPortAllocator
                            .lock()
                            .AllocFromCurrent()
                            .unwrap() as u16)
                            .to_be();
                        *GlobalIOMgr()
                            .GetByHost(self.fd)
                            .unwrap()
                            .lock()
                            .sockInfo
                            .lock() =
                            SockInfo::RDMAUDPSocket(RDMAUDPSock::New(self.queue.clone(), port));
                        GlobalRDMASvcCli()
                            .portToFdInfoMappings
                            .lock()
                            .insert(port, GlobalIOMgr().GetByHost(self.fd).unwrap().clone());
                    } else {
                        port = sockInfo.port;
                    }
                }
                _ => {
                    panic!("SockInfo: {:?} is not allowed for UDP RDMA", sockInfo);
                }
            }
            // wait for buffer available.
            let general = task.blocker.generalEntry.clone();
            self.UDPEventRegister(task, &general, EVENT_WRITE);
            defer!(self.UDPEventUnregister(task, &general));

            let mut udpBuffAddr: u64;
            let mut udpBuffIdx;
            loop {
                (udpBuffAddr, udpBuffIdx) = GlobalRDMASvcCli()
                    .udpSentBufferAllocator
                    .lock()
                    .GetFreeBuffer();
                if udpBuffAddr == 0 {
                    match task.blocker.BlockWithMonoTimer(true, None) {
                        Err(Error::ErrInterrupted) => {
                            return Err(Error::SysError(SysErr::ERESTARTSYS));
                        }
                        Err(e) => {
                            return Err(e);
                        }
                        _ => {}
                    }
                } else {
                    break;
                }
            }

            let udpPacket = unsafe { &mut (*(udpBuffAddr as *mut UDPPacket)) };
            udpPacket.srcPort = port;
            udpPacket.length = totalLen as u16;

            let dstAddr;
            if msgHdr.msgName != 0 {
                dstAddr = unsafe {
                    GetAddr(
                        *(msgHdr.msgName as *const i16),
                        slice::from_raw_parts(msgHdr.msgName as *const u8, msgHdr.nameLen as usize),
                    )
                }
                .unwrap();
            } else if self.remoteAddr.lock().is_some() {
                dstAddr = self.remoteAddr.lock().as_ref().unwrap().clone();
            } else {
                return Err(Error::SysError(SysErr::ENETUNREACH));
            }

            match dstAddr {
                SockAddr::Inet(ipv4) => {
                    udpPacket.dstIpAddr = u32::from_be_bytes(ipv4.Addr);
                    udpPacket.dstPort = ipv4.Port.to_le();
                }
                SockAddr::Inet6(ipv6) => {
                    assert!(
                        ipv6.Addr[10] == 0xFF && ipv6.Addr[11] == 0xFF,
                        "UDP over RDMA only support send to IPv4 address, current address is {:?}",
                        ipv6
                    );
                    let ipv4Bytes = [ipv6.Addr[12], ipv6.Addr[13], ipv6.Addr[14], ipv6.Addr[15]];
                    udpPacket.dstIpAddr = u32::from_be_bytes(ipv4Bytes);
                    udpPacket.dstPort = ipv6.Port.to_le();
                }
                _ => {
                    panic!("dstAddr: {:?} can't enable RDMA!", dstAddr);
                }
            }

            let dstIovs = [IoVec {
                start: udpBuffAddr + UDP_BUFF_OFFSET as u64,
                len: totalLen,
            }];

            // copy mm
            let cnt = task.mm.CopyIovsInFromIovs(task, srcs, &dstIovs, false)?;
            let _res = GlobalRDMASvcCli().sendUDPPacket(udpBuffIdx);
            return Ok(cnt as i64);
        }

        panic!("SendMsg should not come to here!");

        // //TODO: Should impelement RDMA for TCP SendMsg
        // let size = IoVec::NumBytes(srcs);
        // let mut buf = DataBuff::New(size);
        // let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
        // let iovs = buf.Iovs(len);

        // msgHdr.iov = &iovs[0] as *const _ as u64;
        // msgHdr.iovLen = iovs.len();
        // msgHdr.msgFlags = 0;

        // let mut res = if msgHdr.msgControlLen > 0 {
        //     Kernel::HostSpace::IOSendMsg(
        //         self.fd,
        //         msgHdr as *const _ as u64,
        //         flags | MsgType::MSG_DONTWAIT,
        //         false,
        //     ) as i32
        // } else {
        //     Kernel::HostSpace::IOSendto(
        //         self.fd,
        //         buf.Ptr(),
        //         len,
        //         flags | MsgType::MSG_DONTWAIT,
        //         msgHdr.msgName,
        //         msgHdr.nameLen,
        //     ) as i32
        // };

        // while res == -SysErr::EWOULDBLOCK && flags & MsgType::MSG_DONTWAIT == 0 {
        //     let general = task.blocker.generalEntry.clone();

        //     self.EventRegister(task, &general, EVENT_WRITE);
        //     defer!(self.EventUnregister(task, &general));
        //     match task.blocker.BlockWithMonoTimer(true, deadline) {
        //         Err(Error::SysError(SysErr::ETIMEDOUT)) => {
        //             return Err(Error::SysError(SysErr::EAGAIN))
        //         }
        //         Err(e) => {
        //             return Err(e);
        //         }
        //         _ => (),
        //     }

        //     res = if msgHdr.msgControlLen > 0 {
        //         Kernel::HostSpace::IOSendMsg(
        //             self.fd,
        //             msgHdr as *const _ as u64,
        //             flags | MsgType::MSG_DONTWAIT,
        //             false,
        //         ) as i32
        //     } else {
        //         Kernel::HostSpace::IOSendto(
        //             self.fd,
        //             buf.Ptr(),
        //             len,
        //             flags | MsgType::MSG_DONTWAIT,
        //             msgHdr.msgName,
        //             msgHdr.nameLen,
        //         ) as i32
        //     };
        // }

        // if res < 0 {
        //     return Err(Error::SysError(-res as i32));
        // }

        // return Ok(res as i64);
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

pub struct SocketProvider {
    pub family: i32,
}

impl Provider for SocketProvider {
    fn Socket(&self, task: &Task, stype: i32, protocol: i32) -> Result<Option<Arc<File>>> {
        let nonblocking = stype & SocketFlags::SOCK_NONBLOCK != 0;
        let stype = stype & SocketType::SOCK_TYPE_MASK;

        let fd = if SHARESPACE.config.read().EnableTsot && stype == SockType::SOCK_STREAM {
            if self.family == AFType::AF_INET && stype == SockType::SOCK_STREAM {
                let general = task.blocker.generalEntry.clone();
                SHARESPACE.tsotSocketMgr.CreateSocket()?;
                
                SHARESPACE.tsotSocketMgr.EventRegister(task, &general, EVENT_IN);
                defer!(SHARESPACE.tsotSocketMgr.EventUnregister(task, &general));
                let fd;
                loop {
                    match SHARESPACE.tsotSocketMgr.GetSocket() {
                        None => {
                            match task.blocker.BlockWithMonoTimer(true, None) {
                                Err(e) => {
                                    return Err(e);
                                }
                                _ => (),
                            }
                        },
                        Some(socket) => {
                            fd = socket;
                            break;
                        }
                    }
                }

                fd
            } else {
                // tsot only support IPv4 tcp
                return Err(Error::SysError(SysErr::ESOCKTNOSUPPORT));
            }
        } else {
            let res = Kernel::HostSpace::Socket(self.family, stype | SocketFlags::SOCK_CLOEXEC, protocol);
            if res < 0 {
                return Err(Error::SysError(-res as i32));
            }

            let fd = res as i32;
            fd
        };

        let file;
        let tcpRDMA = SHARESPACE.config.read().EnableRDMA
            && (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            //&& family == AFType::AF_INET
            && (stype == SockType::SOCK_STREAM);
        let udpRDMA = SHARESPACE.config.read().EnableRDMA
            && (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            // && self.family == AFType::AF_INET
            && (stype == SockType::SOCK_DGRAM);

        // // tsot only support TCP
        if SHARESPACE.config.read().EnableTsot && stype == SockType::SOCK_STREAM {
            let socketType = TsotSocketType::Init;
            // if stype != SockType::SOCK_STREAM && stype != SockType::SOCK_DGRAM {
            //     // tsot only support TCP/UDP
            //     return Err(Error::SysError(SysErr::ENOTSUP));
            // }

            file = NewTsotSocketFile(
                task, 
                self.family,
                fd,
                stype & SocketType::SOCK_TYPE_MASK,
                nonblocking,
                Queue::default(),
                socketType,
                None,
            )?;
        } else if tcpRDMA || udpRDMA {
            let socketType = SocketBufType::TCPInit;

            file = newSocketFile(
                task,
                self.family,
                fd,
                stype & SocketType::SOCK_TYPE_MASK,
                nonblocking,
                socketType,
                None,
            )?;
            //TODO: UDP
            if (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
                && (stype == SockType::SOCK_DGRAM)
            {
                match &file.FileOp {
                    FileOps::SocketOperations(sockOp) => {
                        *GlobalIOMgr().GetByHost(fd).unwrap().lock().sockInfo.lock() =
                            SockInfo::RDMAUDPSocket(RDMAUDPSock::New(sockOp.queue.clone(), 0));
                    }
                    _ => {
                        panic!("SocketProvider::Socket, fileOp is not valid for UDP");
                    }
                }
            }
        } else if SHARESPACE.config.read().UringIO
            && (self.family == AFType::AF_INET || self.family == AFType::AF_INET6)
            && stype == SockType::SOCK_STREAM
        {
            let socketType = UringSocketType::TCPInit;

            file = newUringSocketFile(
                task,
                self.family,
                fd,
                stype & SocketType::SOCK_TYPE_MASK,
                nonblocking,
                Queue::default(),
                socketType,
                None,
            )?;
        } else {
            file = newHostSocketFile(
                task,
                self.family,
                fd,
                stype & SocketType::SOCK_TYPE_MASK,
                nonblocking,
                None,
            )?;
        }

        return Ok(Some(Arc::new(file)));
    }

    fn Pair(
        &self,
        _task: &Task,
        _stype: i32,
        _protocol: i32,
    ) -> Result<Option<(Arc<File>, Arc<File>)>> {
        /*if self.family == AFType::AF_UNIX {
            let fds: [i32; 2] = [0; 2];

            let res = Kernel::HostSpace::SocketPair(self.family, stype | SocketFlags::SOCK_CLOEXEC, protocol, &fds[0] as *const _ as u64);
            if res < 0 {
                return Err(Error::SysError(-res as i32))
            }

            let file0 = newSocketFile(task, self.family, fds[0], stype & SocketFlags::SOCK_NONBLOCK != 0)?;
            let file1 = newSocketFile(task, self.family, fds[1], stype & SocketFlags::SOCK_NONBLOCK != 0)?;

            return Ok(Some((Arc::new(file0), Arc::new(file1))));
        }*/

        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }
}

pub fn Init() {
    for family in [AFType::AF_INET, AFType::AF_INET6, AFType::AF_NETLINK].iter() {
        FAMILIAES
            .write()
            .RegisterProvider(*family, Box::new(SocketProvider { family: *family }))
    }
}
