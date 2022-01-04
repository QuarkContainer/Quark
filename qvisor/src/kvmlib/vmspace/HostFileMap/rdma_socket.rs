use alloc::sync::Arc;
use libc::*;
use core::ops::Deref;
use super::super::super::qlib::mutex::*;
use core::mem;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::super::super::IO_MGR;
use super::super::super::URING_MGR;
//use super::super::super::SHARE_SPACE;
use super::super::super::qlib::linux_def::*;
use super::super::super::vmspace::singleton::*;
use super::super::super::qlib::common::*;
//use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::socket_buf::*;
use super::super::super::qlib::qmsg::qcall::*;
//use super::super::super::qlib::qmsg::input::*;
use super::fdinfo::*;
use super::socket_info::*;
use super::rdma::*;

pub static RDMA : Singleton<RDMAContext> = Singleton::<RDMAContext>::New();


#[derive(Clone)]
pub struct RDMAServerSock(Arc<QMutex<RDMAServerSockIntern>>);

impl Deref for RDMAServerSock {
    type Target = Arc<QMutex<RDMAServerSockIntern>>;

    fn deref(&self) -> &Arc<QMutex<RDMAServerSockIntern>> {
        &self.0
    }
}

impl RDMAServerSock {
    pub fn New(fd: i32, acceptQueue: AcceptQueue) -> Self {
        return Self (
            Arc::new(QMutex::new(RDMAServerSockIntern{
                fd: fd,
                acceptQueue: acceptQueue
            }))
        )
    }

    pub fn Notify(&self, _eventmask: EventMask) {
        self.Accept();
    }

    pub fn Accept(&self) {
        let minefd = self.lock().fd;
        let acceptQueue = self.lock().acceptQueue.clone();
        if acceptQueue.lock().Err() != 0 {
            FdNotify(minefd, EVENT_ERR | EVENT_IN);
            return
        }

        let mut hasSpace = acceptQueue.lock().HasSpace();

        while hasSpace {
            let tcpAddr = TcpSockAddr::default();
            let mut len : u32 = TCP_ADDR_LEN as _;

            let ret = unsafe{
                accept4(minefd, tcpAddr.Addr() as  *mut sockaddr, &mut len as  *mut socklen_t, SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC)
            };

            if ret < 0 {
                let errno = errno::errno().0;
                if errno == SysErr::EAGAIN {
                    return
                }

                FdNotify(minefd, EVENT_ERR | EVENT_IN);
                acceptQueue.lock().SetErr(errno);
                return
            }

            let fd = ret;

            IO_MGR.AddSocket(fd);
            let socketBuf = Arc::new(SocketBuff::default());

            let rdmaType = if super::rdma_socket::RDMA_ENABLE {
                let sockInfo = RDMAServerSocketInfo {
                    sock: self.clone(),
                    fd: fd,
                    addr: tcpAddr,
                    len: len,
                    sockBuf: socketBuf.clone()
                };
                RDMAType::Server(sockInfo)
            } else {
                RDMAType::None
            };

            let rdmaSocket = RDMADataSock::New(fd, socketBuf.clone(), rdmaType);
            let fdInfo = IO_MGR.GetByHost(fd).unwrap();
            *fdInfo.lock().sockInfo.lock() = SockInfo::RDMADataSocket(rdmaSocket);

            URING_MGR.lock().Addfd(fd).unwrap();
            IO_MGR.AddWait(fd, EVENT_READ | EVENT_WRITE);

            if !super::rdma_socket::RDMA_ENABLE {
                let (trigger, tmp) = acceptQueue.lock().EnqSocket(fd, tcpAddr, len, socketBuf);
                hasSpace = tmp;

                if trigger {
                    FdNotify(minefd, EVENT_IN);
                }
            } else {
                // todo: how to handle the accept queue len?
                hasSpace = true;
            }
        }
    }
}

pub struct RDMAServerSocketInfo {
    pub sock: RDMAServerSock,
    pub fd: i32,
    pub addr: TcpSockAddr,
    pub len: u32,
    pub sockBuf: Arc<SocketBuff>
}

pub struct RDMAServerSockIntern {
    pub fd: i32,
    pub acceptQueue: AcceptQueue,
}

pub struct RDMADataSockIntern {
    pub fd: i32,
    pub socketBuf: Arc<SocketBuff>,
    pub readLock: QMutex<()>,
    pub writeLock: QMutex<()>,
    pub qp: QMutex<QueuePair>,
    pub peerInfo: QMutex<RDMAInfo>,
    pub socketState: AtomicU64,
    pub localRDMAInfo: RDMAInfo,
    pub remoteRDMAInfo: QMutex<RDMAInfo>,
    pub mr: MemoryRegion,
    pub rdmaType: RDMAType,
}

#[derive(Clone, Default)]
#[repr(C)]
pub struct RDMAInfo {
    raddr: u64,     /* Read Buffer address */
    rlen: u32,      /* Read Buffer len */
    rkey: u32,     /* Read Buffer Remote key */
    qp_num: u32,   /* QP number */
    lid: u16,      /* LID of the IB port */
    offset: u32,    //read buffer offset
    freeSpace: u32, //read buffer free space size
    gid: Gid, /* gid */
}


impl RDMAInfo {
    pub fn Size() -> usize {
        return mem::size_of::<Self>();
    }
}

#[derive(Debug)]
#[repr(u64)]
pub enum SocketState {
    Init,
    Connect,
    WaitingForRemoteMeta,
    WaitingForRemoteReady,
    Ready,
    Error,
}

pub enum RDMAType {
    Client(&'static mut PostRDMAConnect),
    Server(RDMAServerSocketInfo),
    None,
}

#[derive(Clone)]
pub struct RDMADataSock(Arc<RDMADataSockIntern>);

impl Deref for RDMADataSock {
    type Target = Arc<RDMADataSockIntern>;

    fn deref(&self) -> &Arc<RDMADataSockIntern> {
        &self.0
    }
}

pub const RDMA_ENABLE : bool= false;

impl RDMADataSock {
    pub fn New(fd: i32, socketBuf: Arc<SocketBuff>, rdmaType: RDMAType) -> Self {
        let (addr, len) = socketBuf.ReadBuf();

        if RDMA_ENABLE {
            let mr = RDMA.CreateMemoryRegion(addr, len).expect("RDMADataSock CreateMemoryRegion fail");
            let qp = RDMA.CreateQueuePair().expect("RDMADataSock create QP fail");

            let localRDMAInfo = RDMAInfo {
                raddr: addr,
                rlen: len as _,
                rkey: mr.RKey(),
                qp_num: qp.qpNum(),
                lid: RDMA.Lid(),
                offset: 0,
                freeSpace: len as u32,
                gid: RDMA.Gid()
            };

            return Self (
                Arc::new(RDMADataSockIntern{
                    fd: fd,
                    socketBuf: socketBuf,
                    readLock: QMutex::new(()),
                    writeLock: QMutex::new(()),
                    qp: QMutex::new(qp),
                    peerInfo: QMutex::new(RDMAInfo::default()),
                    socketState: AtomicU64::new(0),
                    localRDMAInfo: localRDMAInfo,
                    remoteRDMAInfo: QMutex::new(RDMAInfo::default()),
                    mr: mr,
                    rdmaType: rdmaType
                })
            )
        } else {
            let mr = MemoryRegion::default();
            let qp = QueuePair::default();

            let localRDMAInfo = RDMAInfo::default();

            return Self (
                Arc::new(RDMADataSockIntern{
                    fd: fd,
                    socketBuf: socketBuf,
                    readLock: QMutex::new(()),
                    writeLock: QMutex::new(()),
                    qp: QMutex::new(qp),
                    peerInfo: QMutex::new(RDMAInfo::default()),
                    socketState: AtomicU64::new(0),
                    localRDMAInfo: localRDMAInfo,
                    remoteRDMAInfo: QMutex::new(RDMAInfo::default()),
                    mr: mr,
                    rdmaType: rdmaType,
                })
            )
        }

    }

    pub fn SendLocalRDMAInfo(&self) -> Result<()> {
        let ret = unsafe {
            write(self.fd, &self.localRDMAInfo as * const _ as u64 as _, RDMAInfo::Size())
        };

        if ret < 0 {
            let errno = errno::errno().0;
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno))
        }

        assert!(ret == RDMAInfo::Size() as isize, "SendLocalRDMAInfo fail ret is {}, expect {}", ret, RDMAInfo::Size());
        return Ok(())
    }

    pub fn RecvRemoteRDMAInfo(&self) -> Result<()> {
        let mut data = RDMAInfo::default();
        let ret = unsafe {
            read(self.fd, &mut data as * mut _ as u64 as _, RDMAInfo::Size())
        };

        if ret < 0 {
            let errno = errno::errno().0;
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno))
        }

        assert!(ret == RDMAInfo::Size() as isize, "SendLocalRDMAInfo fail ret is {}, expect {}", ret, RDMAInfo::Size());

        *self.remoteRDMAInfo.lock() = data;

        return Ok(())
    }

    pub const ACK_DATA : u64 = 0x1234567890;
    pub fn SendAck(&self) -> Result<()> {
        let data : u64 = Self::ACK_DATA;
        let ret = unsafe {
            write(self.fd, &data as * const _ as u64 as _, 8)
        };

        if ret < 0 {
            let errno = errno::errno().0;
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno))
        }

        assert!(ret == 8, "SendAck fail ret is {}, expect {}", ret, 8);
        return Ok(())
    }

    pub fn RecvAck(&self) -> Result<()> {
        let mut data = 0;
        let ret = unsafe {
            read(self.fd, &mut data as * mut _ as u64 as _, 8)
        };

        if ret < 0 {
            let errno = errno::errno().0;
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno))
        }

        assert!(ret == 8 as isize, "RecvAck fail ret is {}, expect {}", ret, 8);
        assert!(data == Self::ACK_DATA, "RecvAck fail data is {:x}, expect {:x}", ret, Self::ACK_DATA);

        return Ok(())
    }

    pub fn SocketState(&self) -> SocketState {
        let state = self.socketState.load(Ordering::Relaxed);
        assert!(state <= SocketState::Ready as u64);
        let state: SocketState = unsafe { mem::transmute(state) };
        return state
    }

    pub fn SetSocketState(&self, state: SocketState) {
        self.socketState.store(state as u64, Ordering::SeqCst)
    }

    // after get remote peer's RDMA metadata and need to setup RDMA
    pub fn SetupUpUring(&self) {

    }

    pub fn SetReady(&self) {
        match &self.rdmaType {
            RDMAType::Client(ref msg) => {
                let addr = msg as *const _ as u64;
                let msg = PostRDMAConnect::ToRef(addr);
                msg.Finish(0)
            }
            RDMAType::Server(ref serverSock) => {
                let minefd = serverSock.sock.lock().fd;
                let acceptQueue = serverSock.sock.lock().acceptQueue.clone();
                let (trigger, _tmp) = acceptQueue.lock().EnqSocket(serverSock.fd, serverSock.addr, serverSock.len, serverSock.sockBuf.clone());

                if trigger {
                    FdNotify(minefd, EVENT_IN);
                }
            }
            RDMAType::None => {
                panic!("RDMADataSock setready fail ...");
            }
        }

        self.SetSocketState(SocketState::Ready);
    }

    pub fn Read(&self) {
        if !RDMA_ENABLE {
            self.ReadData();
        } else {
            let _readlock = self.readLock.lock();
            match self.SocketState() {
                SocketState::WaitingForRemoteMeta => {
                    self.RecvRemoteRDMAInfo().unwrap();
                    self.SetupUpUring();
                    self.SendAck().unwrap(); // assume the socket is ready for send
                    self.SetSocketState(SocketState::WaitingForRemoteReady);

                    match self.RecvAck() {
                        Ok(()) => {
                            self.SetReady();
                        }
                        _ => ()
                    }
                }
                SocketState::WaitingForRemoteReady => {
                    self.RecvAck().unwrap();
                    self.SetReady();
                }
                SocketState::Ready => {
                    self.ReadData();
                }
                _ => {
                    panic!("RDMA socket read state error with state {:?}", self.SocketState())
                }
            }
        }
    }

    pub fn ReadData(&self) {
        let _readlock = self.readLock.lock();

        let fd = self.fd;
        let socketBuf = self.socketBuf.clone();

        let (mut addr, mut count) = socketBuf.GetFreeReadBuf();
        if count == 0 { // no more space
            return
        }

        loop {
            let len = unsafe {
                read(fd, addr as _, count as _)
            };

            // closed
            if len == 0 {
                socketBuf.SetRClosed();
                if socketBuf.HasReadData() {
                    FdNotify(fd, EVENT_IN);
                } else {
                    FdNotify(fd, EVENT_HUP);
                }
                return
            }

            if len < 0 {
                let errno = errno::errno().0;
                if errno == SysErr::EAGAIN {
                    return
                }

                socketBuf.SetErr(errno);
                FdNotify(fd, EVENT_ERR | EVENT_IN);
                return
            }

            let (trigger, addrTmp, countTmp) = socketBuf.ProduceAndGetFreeReadBuf(len as _);
            if trigger {
                FdNotify(fd, EVENT_IN);
            }

            if len < count as _ { // have clean the read buffer
                return
            }

            if countTmp == 0 {  // no more space
                return
            }

            addr = addrTmp;
            count = countTmp;
        }
    }

    pub fn Write(&self) {
        if !RDMA_ENABLE {
            self.WriteData();
        } else {
            let _writelock = self.writeLock.lock();
            match self.SocketState() {
                SocketState::Init => {
                    self.SendLocalRDMAInfo().unwrap();
                    self.SetSocketState(SocketState::WaitingForRemoteMeta);
                }
                SocketState::Ready => {
                    self.WriteData();
                }
                _ => {
                    panic!("RDMA socket Write state error with state {:?}", self.SocketState())
                }
            }
        }
    }

    pub fn WriteData(&self) {
        let _writelock = self.writeLock.lock();

        let fd = self.fd;
        let socketBuf = self.socketBuf.clone();

        let (mut addr, mut count) = socketBuf.GetAvailableWriteBuf();
        if count == 0 { // no data
            return
        }

        loop {
            let len = unsafe {
                write(fd, addr as _, count as _)
            };

            // closed
            if len == 0 {
                socketBuf.SetWClosed();
                if socketBuf.HasWriteData() {
                    FdNotify(fd, EVENT_OUT);
                } else {
                    FdNotify(fd, EVENT_HUP);
                }
                return
            }

            if len < 0 {
                let errno = errno::errno().0;
                if errno == SysErr::EAGAIN {
                    return
                }

                socketBuf.SetErr(errno);
                FdNotify(fd, EVENT_ERR | EVENT_IN);
                return
            }

            let (trigger, addrTmp, countTmp) = socketBuf.ConsumeAndGetAvailableWriteBuf(len as _);
            if trigger {
                FdNotify(fd, EVENT_OUT);
            }

            if len < count as _ { // have fill the write buffer
                return
            }

            if countTmp == 0 {
                if socketBuf.PendingWriteShutdown() {
                    FdNotify(fd, EVENT_PENDING_SHUTDOWN);
                }

                return;
            }

            addr = addrTmp;
            count = countTmp;
        }
    }

    pub fn Notify(&self, eventmask: EventMask) {
        let fd = self.fd;
        let socketBuf = self.socketBuf.clone();

        if socketBuf.Error() != 0 {
            FdNotify(fd, EVENT_ERR | EVENT_IN);
            return
        }

        if eventmask & EVENT_WRITE != 0 {
            self.Write()
        }

        if eventmask & EVENT_READ != 0 {
            self.Read()
        }
    }
}