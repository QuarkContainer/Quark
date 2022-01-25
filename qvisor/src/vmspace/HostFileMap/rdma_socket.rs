use super::super::super::qlib::mutex::*;
use alloc::sync::Arc;
use core::mem;
use core::ops::Deref;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use libc::*;

use super::super::super::qlib::common::*;
use super::super::super::qlib::kernel::guestfdnotifier::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::qmsg::qcall::*;
use super::super::super::qlib::socket_buf::*;
use super::super::super::IO_MGR;
use super::super::super::URING_MGR;
use super::rdma::*;
use super::socket_info::*;

pub struct RDMAServerSockIntern {
    pub fd: i32,
    pub acceptQueue: AcceptQueue,
}

#[derive(Clone)]
pub struct RDMAServerSock(Arc<RDMAServerSockIntern>);

impl Deref for RDMAServerSock {
    type Target = Arc<RDMAServerSockIntern>;

    fn deref(&self) -> &Arc<RDMAServerSockIntern> {
        &self.0
    }
}

impl RDMAServerSock {
    pub fn New(fd: i32, acceptQueue: AcceptQueue) -> Self {
        return Self(Arc::new(RDMAServerSockIntern {
            fd: fd,
            acceptQueue: acceptQueue,
        }));
    }

    pub fn Notify(&self, _eventmask: EventMask, waitinfo: FdWaitInfo) {
        self.Accept(waitinfo);
    }

    pub fn Accept(&self, waitinfo: FdWaitInfo) {
        error!("ServerSock::Accept 1");
        let minefd = self.fd;
        let acceptQueue = self.acceptQueue.clone();
        if acceptQueue.lock().Err() != 0 {
            waitinfo.Notify(EVENT_ERR | EVENT_IN);
            return;
        }

        let mut hasSpace = acceptQueue.lock().HasSpace();

        while hasSpace {
            let tcpAddr = TcpSockAddr::default();
            let mut len: u32 = TCP_ADDR_LEN as _;
            error!("ServerSock::Accept 2, fd: {}", minefd);
            let ret = unsafe {
                accept4(
                    minefd,
                    tcpAddr.Addr() as *mut sockaddr,
                    &mut len as *mut socklen_t,
                    SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC,
                )
            };

            error!("ServerSock::Accept 3, ret: {}", ret);

            if ret < 0 {
                let errno = errno::errno().0;
                if errno == SysErr::EAGAIN {
                    return;
                }

                waitinfo.Notify(EVENT_ERR | EVENT_IN);
                acceptQueue.lock().SetErr(errno);
                return;
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
                    sockBuf: socketBuf.clone(),
                    waitInfo: waitinfo.clone(),
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
                    waitinfo.Notify(EVENT_IN);
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
    pub sockBuf: Arc<SocketBuff>,
    pub waitInfo: FdWaitInfo,
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
    pub readMemoryRegion: MemoryRegion,
    pub writeMemoryRegion: MemoryRegion,
    pub rdmaType: RDMAType,
    pub writeCount: AtomicUsize, //when run the writeimm, save the write bytes count here
}

#[derive(Clone, Default)]
#[repr(C)]
pub struct RDMAInfo {
    raddr: u64,     /* Read Buffer address */
    rlen: u32,      /* Read Buffer len */
    rkey: u32,      /* Read Buffer Remote key */
    qp_num: u32,    /* QP number */
    lid: u16,       /* LID of the IB port */
    offset: u32,    //read buffer offset
    freespace: u32, //read buffer free space size
    gid: Gid,       /* gid */
    sending: bool,  // the writeimmediately is ongoing
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
    Client(u64),
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

impl RDMADataSock {
    pub fn New(fd: i32, socketBuf: Arc<SocketBuff>, rdmaType: RDMAType) -> Self {
        if RDMA_ENABLE {
            let (addr, len) = socketBuf.ReadBuf();
            let readMR = RDMA
                .CreateMemoryRegion(addr, len)
                .expect("RDMADataSock CreateMemoryRegion fail");
            let qp = RDMA.CreateQueuePair().expect("RDMADataSock create QP fail");

            let localRDMAInfo = RDMAInfo {
                raddr: addr,
                rlen: len as _,
                rkey: readMR.RKey(),
                qp_num: qp.qpNum(),
                lid: RDMA.Lid(),
                offset: 0,
                freespace: len as u32,
                gid: RDMA.Gid(),
                sending: false,
            };

            let (waddr, wlen) = socketBuf.WriteBuf();
            let writeMR = RDMA
                .CreateMemoryRegion(waddr, wlen)
                .expect("RDMADataSock CreateMemoryRegion fail");

            return Self(Arc::new(RDMADataSockIntern {
                fd: fd,
                socketBuf: socketBuf,
                readLock: QMutex::new(()),
                writeLock: QMutex::new(()),
                qp: QMutex::new(qp),
                peerInfo: QMutex::new(RDMAInfo::default()),
                socketState: AtomicU64::new(0),
                localRDMAInfo: localRDMAInfo,
                remoteRDMAInfo: QMutex::new(RDMAInfo::default()),
                readMemoryRegion: readMR,
                writeMemoryRegion: writeMR,
                rdmaType: rdmaType,
                writeCount: AtomicUsize::new(0),
            }));
        } else {
            let readMR = MemoryRegion::default();
            let writeMR = MemoryRegion::default();
            let qp = QueuePair::default();

            let localRDMAInfo = RDMAInfo::default();

            return Self(Arc::new(RDMADataSockIntern {
                fd: fd,
                socketBuf: socketBuf,
                readLock: QMutex::new(()),
                writeLock: QMutex::new(()),
                qp: QMutex::new(qp),
                peerInfo: QMutex::new(RDMAInfo::default()),
                socketState: AtomicU64::new(0),
                localRDMAInfo: localRDMAInfo,
                remoteRDMAInfo: QMutex::new(RDMAInfo::default()),
                readMemoryRegion: readMR,
                writeMemoryRegion: writeMR,
                rdmaType: rdmaType,
                writeCount: AtomicUsize::new(0),
            }));
        }
    }

    pub fn SendLocalRDMAInfo(&self) -> Result<()> {
        let ret = unsafe {
            write(
                self.fd,
                &self.localRDMAInfo as *const _ as u64 as _,
                RDMAInfo::Size(),
            )
        };

        if ret < 0 {
            let errno = errno::errno().0;
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno));
        }

        assert!(
            ret == RDMAInfo::Size() as isize,
            "SendLocalRDMAInfo fail ret is {}, expect {}",
            ret,
            RDMAInfo::Size()
        );
        return Ok(());
    }

    pub fn RecvRemoteRDMAInfo(&self) -> Result<()> {
        error!("RDMADataSock::RecvRemoteRDMAInfo 1");
        let mut data = RDMAInfo::default();
        let ret = unsafe { read(self.fd, &mut data as *mut _ as u64 as _, RDMAInfo::Size()) };
        error!("RDMADataSock::RecvRemoteRDMAInfo 2 ret: {}", ret);

        if ret < 0 {
            let errno = errno::errno().0;
            error!(
                "RDMADataSock::RecvRemoteRDMAInfo 3 ret: {}, errno: {:x}",
                ret, errno
            );
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno));
        }

        assert!(
            ret == RDMAInfo::Size() as isize,
            "SendLocalRDMAInfo fail ret is {}, expect {}",
            ret,
            RDMAInfo::Size()
        );

        error!(
            "RDMADataSock::RecvRemoteRDMAInfo 3 ret: {}, data.qp_num {}",
            ret, &data.qp_num
        );

        *self.remoteRDMAInfo.lock() = data;

        return Ok(());
    }

    pub const ACK_DATA: u64 = 0x1234567890;
    pub fn SendAck(&self) -> Result<()> {
        error!("SendAck 1");
        let data: u64 = Self::ACK_DATA;
        let ret = unsafe { write(self.fd, &data as *const _ as u64 as _, 8) };
        error!("SendAck 2, ret: {}", ret);
        if ret < 0 {
            let errno = errno::errno().0;
            error!("SendAck 3, ret: {}, errno: {:x}", ret, errno);
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno));
        }

        assert!(ret == 8, "SendAck fail ret is {}, expect {}", ret, 8);
        error!("SendAck 4, ret: {}", ret);
        return Ok(());
    }

    pub fn RecvAck(&self) -> Result<()> {
        let mut data = 0;
        error!("RecvAck 1");
        let ret = unsafe { read(self.fd, &mut data as *mut _ as u64 as _, 8) };
        error!("RecvAck 2, ret: {}", ret);

        if ret < 0 {
            let errno = errno::errno().0;
            error!("RecvAck 3, ret: {}, errno: {:x}", ret, errno);
            if errno == SysErr::EAGAIN {
                return Err(Error::SysError(errno));
            }
            self.socketBuf.SetErr(errno);
            return Err(Error::SysError(errno));
        }

        assert!(
            ret == 8 as isize,
            "RecvAck fail ret is {}, expect {}",
            ret,
            8
        );
        assert!(
            data == Self::ACK_DATA,
            "RecvAck fail data is {:x}, expect {:x}",
            ret,
            Self::ACK_DATA
        );

        error!("RecvAck 4, ret: {}", ret);
        return Ok(());
    }

    pub fn SocketState(&self) -> SocketState {
        let state = self.socketState.load(Ordering::Relaxed);
        assert!(state <= SocketState::Ready as u64);
        let state: SocketState = unsafe { mem::transmute(state) };
        return state;
    }

    pub fn SetSocketState(&self, state: SocketState) {
        self.socketState.store(state as u64, Ordering::SeqCst)
    }

    /************************************ rdma integration ****************************/
    // after get remote peer's RDMA metadata and need to setup RDMA
    pub fn SetupRDMA(&self) {
        let remoteInfo = self.remoteRDMAInfo.lock();
        error!("SetupRDMA 1");
        self.qp
            .lock()
            .Setup(&RDMA, remoteInfo.qp_num, remoteInfo.lid, remoteInfo.gid)
            .expect("SetupRDMA fail...");
        error!("SetupRDMA 2");
        for _i in 0..MAX_RECV_WR {
            let wr = WorkRequestId::New(self.fd);
            error!("Setup RDMA 2.1, create wr_id: {}", wr.0);
            self.qp
                .lock()
                .PostRecv(wr.0, self.localRDMAInfo.raddr, self.localRDMAInfo.rkey)
                .expect("SetupRDMA PostRecv fail");
        }
        error!("SetupRDMA 3");
    }

    pub fn RDMAWriteImm(
        &self,
        localAddr: u64,
        remoteAddr: u64,
        writeCount: usize,
        readCount: usize,
        remoteInfo: &QMutexGuard<RDMAInfo>,
    ) -> Result<()> {
        let wrid = WorkRequestId::New(self.fd);
        let immData = ImmData::New(writeCount as u16, readCount as u16);
        error!(
            "RDMAWriteImm 1, writeCount: {}, readCount: {}, wr_id: {}",
            writeCount, readCount, wrid.0
        );
        let rkey = remoteInfo.rkey;
        error!("RDMAWriteImm 2");

        self.qp.lock().WriteImm(
            wrid.0,
            localAddr,
            writeCount as u32,
            self.writeMemoryRegion.LKey(),
            remoteAddr,
            rkey,
            immData.0,
        )?;
        error!("RDMAWriteImm 3");
        self.writeCount.store(writeCount, QOrdering::RELEASE);
        return Ok(());
    }

    // need to be called when the self.writeLock is locked
    pub fn RDMASend(&self) {
        error!("RDMASend: 1");
        let remoteInfo = self.remoteRDMAInfo.lock();
        error!("RDMASend, remoteInfo.sending: {}", remoteInfo.sending);
        if remoteInfo.sending == true {
            return; // the sending is ongoing
        }

        self.RDMASendLocked(remoteInfo);
    }

    pub fn RDMASendLocked(&self, mut remoteInfo: QMutexGuard<RDMAInfo>) {
        error!("RDMASendLocked:0");
        let readCount = self.socketBuf.GetAndClearConsumeReadData();
        let buf = self.socketBuf.writeBuf.lock();
        let (addr, mut len) = buf.GetDataBuf();
        error!("RDMASendLocked:1 readCount: {}, len: {}, remote freesapce: {}", readCount, len, remoteInfo.freespace);
        if readCount > 0 || len > 0 {
            if len > remoteInfo.freespace as usize {
                len = remoteInfo.freespace as usize;
            }

            error!("RDMASendLocked:2 readCount: {}, len: {}", readCount, len);

            if len != 0 || readCount > 0 {
                self.RDMAWriteImm(
                    addr,
                    remoteInfo.raddr + remoteInfo.offset as u64,
                    len,
                    readCount as usize,
                    &remoteInfo,
                )
                .expect("RDMAWriteImm fail...");
                remoteInfo.freespace -= len as u32;
                remoteInfo.offset = (remoteInfo.offset + len as u32) % remoteInfo.rlen;
                remoteInfo.sending = true;
                error!("RDMASendLocked:3 readCount: {}, len: {}, remote freesapce: {}", readCount, len, remoteInfo.freespace);
            }
            error!("RDMASendLocked:4 readCount: {}, len: {}, remote freesapce: {}", readCount, len, remoteInfo.freespace);
        }
    }

    // triggered by the RDMAWriteImmediately finish
    pub fn ProcessRDMAWriteImmFinish(&self, waitinfo: FdWaitInfo) {
        error!("writelock1 start");
        defer!(error!("writelock1 finish"));
        let _writelock = self.writeLock.lock();
        let mut remoteInfo = self.remoteRDMAInfo.lock();
        remoteInfo.sending = false;

        error!("ProcessRDMAWriteImmFinish::1");
        // let wr = WorkRequestId::New(self.fd);
        // error!("ProcessRDMAWriteImmFinish::2, create wr_id: {}", wr.0);

        // let res = self
        //     .qp
        //     .lock()
        //     .PostRecv(wr.0, self.localRDMAInfo.raddr, self.localRDMAInfo.rkey);

        // match res {
        //     Ok(()) => {
        //         error!("ProcessRDMAWriteImmFinish::3, Create RR successfully");
        //     }
        //     _ => {
        //         error!("ProcessRDMAWriteImmFinish::4, fail to create RR");
        //     }
        // }

        let writeCount = self.writeCount.load(QOrdering::ACQUIRE);

        let (trigger, addr, _len) = self
            .socketBuf
            .ConsumeAndGetAvailableWriteBuf(writeCount as usize);
        error!(
            "ProcessRDMAWriteImmFinish::3 trigger: {}, addr: {}, _len: {}",
            trigger, addr, _len
        );
        if trigger {
            waitinfo.Notify(EVENT_OUT);
        }

        if addr != 0 {
            self.RDMASendLocked(remoteInfo)
        }
        else{
            error!("ProcessRDMAWriteImmFinish::3 send finished");
        }
    }

    // triggered when remote's writeimmedate reach local
    pub fn ProcessRDMARecvWriteImm(
        &self,
        recvCount: u64,
        writeConsumeCount: u64,
        waitinfo: FdWaitInfo,
    ) {
        error!(
            "ProcessRDMARecvWriteImm: 1, recvCount: {}, writeConsumeCount: {}, waitinfo: {:?}",
            recvCount, writeConsumeCount, &waitinfo
        );
        let wr = WorkRequestId::New(self.fd);
        error!("ProcessRDMARecvWriteImm: 2, create wr_id: {}", wr.0);

        // self.qp
        //     .lock()
        //     .PostRecv(wr.0, self.localRDMAInfo.raddr, self.localRDMAInfo.rkey)
        //     .expect("ProcessRDMARecvWriteImm PostRecv fail");

        let res = self
            .qp
            .lock()
            .PostRecv(wr.0, self.localRDMAInfo.raddr, self.localRDMAInfo.rkey);

        match res {
            Ok(()) => {
                error!("ProcessRDMARecvWriteImm::3, Create RR successfully");
            }
            _ => {
                error!("ProcessRDMARecvWriteImm::4, fail to create RR");
            }
        }

        if recvCount > 0 {
            let (trigger, _addr, _len) =
                self.socketBuf.ProduceAndGetFreeReadBuf(recvCount as usize);
            error!(
                "ProcessRDMARecvWriteImm: 5, trigger: {}, _addr: {}, _len: {}",
                trigger, _addr, _len
            );
            if trigger {
                waitinfo.Notify(EVENT_IN);
            }
        }

        if writeConsumeCount > 0 {
            let mut remoteInfo = self.remoteRDMAInfo.lock();
            let trigger = remoteInfo.freespace == 0;
            error!("ProcessRDMARecvWriteImm::6, freespace: {}, trigger: {}, sending: {}", remoteInfo.freespace, trigger, remoteInfo.sending);
            remoteInfo.freespace += writeConsumeCount as u32;
            
            if trigger && !remoteInfo.sending {
                self.RDMASendLocked(remoteInfo);
            }
            
            
        }
    }

    /*********************************** end of rdma integration ****************************/

    pub fn SetReady(&self, waitinfo: FdWaitInfo) {
        error!("SetReady: 1");
        match &self.rdmaType {
            RDMAType::Client(ref addr) => {
                //let addr = msg as *const _ as u64;
                error!("SetReady: 2, addr: {:x}", *addr);
                let msg = PostRDMAConnect::ToRef(*addr);
                error!("SetReady: 2.1, addr: {:x}, msg: {:x?}", *addr, &msg);
                msg.Finish(0);
                error!("SetReady: 3");
            }
            RDMAType::Server(ref serverSock) => {
                error!("SetReady: 4");
                let acceptQueue = serverSock.sock.acceptQueue.clone();
                let (trigger, _tmp) = acceptQueue.lock().EnqSocket(
                    serverSock.fd,
                    serverSock.addr,
                    serverSock.len,
                    serverSock.sockBuf.clone(),
                );

                error!("SetReady: 5");

                if trigger {
                    error!("SetReady: 6");
                    waitinfo.Notify(EVENT_IN);
                    error!("SetReady: 7");
                }
            }
            RDMAType::None => {
                panic!("RDMADataSock setready fail ...");
            }
        }

        error!("SetReady: 8");
        self.SetSocketState(SocketState::Ready);
        error!("SetReady: 9");
    }

    pub fn Read(&self, waitinfo: FdWaitInfo) {
        if !RDMA_ENABLE {
            error!("RDMADataSock::Read 1, fd: {}", self.fd);
            self.ReadData(waitinfo);
        } else {
            match self.SocketState() {
                SocketState::WaitingForRemoteMeta => {
                    let _readlock = self.readLock.lock();
                    error!("RDMADataSock::Read 2, fd: {}", self.fd);
                    self.RecvRemoteRDMAInfo().unwrap();
                    error!("RDMADataSock::Read 2.1, fd: {}", self.fd);
                    self.SetupRDMA();
                    error!("RDMADataSock::Read 3, fd: {}", self.fd);
                    self.SendAck().unwrap(); // assume the socket is ready for send
                    self.SetSocketState(SocketState::WaitingForRemoteReady);
                    error!("RDMADataSock::Read 4, fd: {}", self.fd);

                    match self.RecvAck() {
                        Ok(()) => {
                            error!("RDMADataSock::Read 5, fd: {}", self.fd);
                            let waitinfo = match &self.rdmaType {
                                RDMAType::Client(_) => waitinfo,
                                RDMAType::Server(ref serverSock) => serverSock.waitInfo.clone(),
                                _ => {
                                    panic!("Not right RDMAType");
                                }
                            };
                            self.SetReady(waitinfo);
                            error!("RDMADataSock::Read 6, fd: {}", self.fd);
                        }
                        _ => (),
                    }
                }
                SocketState::WaitingForRemoteReady => {
                    let _readlock = self.readLock.lock();
                    error!("RDMADataSock::Read 7, fd: {}", self.fd);
                    self.RecvAck().unwrap();
                    error!("RDMADataSock::Read 8, fd: {}", self.fd);
                    self.SetReady(waitinfo);
                    error!("RDMADataSock::Read 9, fd: {}", self.fd);
                }
                SocketState::Ready => {
                    error!("RDMADataSock::Read 10, fd: {}", self.fd);
                    self.ReadData(waitinfo);
                    error!("RDMADataSock::Read 11, fd: {}", self.fd);
                }
                _ => {
                    panic!(
                        "RDMA socket read state error with state {:?}",
                        self.SocketState()
                    )
                }
            }
        }
    }

    //notify rdmadatasocket to sync read buff freespace with peer
    pub fn RDMARead(&self) {
        error!("writelock2 start");
        defer!(error!("writelock2 finish"));
        let _writelock = self.writeLock.lock();
        self.RDMASend();
    }

    pub fn RDMAWrite(&self) {
        error!("RDMAWrite: 1");
        error!("writelock3 start");
        defer!(error!("writelock3 finish"));
        let _writelock = self.writeLock.lock();
        error!("RDMAWrite: 2");
        self.RDMASend();
        error!("RDMAWrite: 3");
    }

    pub fn ReadData(&self, waitinfo: FdWaitInfo) {
        let _readlock = self.readLock.lock();

        let fd = self.fd;
        let socketBuf = self.socketBuf.clone();

        let (mut addr, mut count) = socketBuf.GetFreeReadBuf();
        error!("RDMADataSock::ReadData: 1, count {}", count);
        if count == 0 {
            // no more space
            return;
        }

        loop {
            let len = unsafe { read(fd, addr as _, count as _) };

            // closed
            if len == 0 {
                socketBuf.SetRClosed();
                if socketBuf.HasReadData() {
                    waitinfo.Notify(EVENT_IN);
                } else {
                    waitinfo.Notify(EVENT_HUP);
                }
                return;
            }

            if len < 0 {
                let errno = errno::errno().0;
                if errno == SysErr::EAGAIN {
                    return;
                }

                socketBuf.SetErr(errno);
                waitinfo.Notify(EVENT_ERR | EVENT_IN);
                return;
            }

            let (trigger, addrTmp, countTmp) = socketBuf.ProduceAndGetFreeReadBuf(len as _);
            if trigger {
                waitinfo.Notify(EVENT_IN);
            }

            if len < count as _ {
                // have clean the read buffer
                return;
            }

            if countTmp == 0 {
                // no more space
                return;
            }

            addr = addrTmp;
            count = countTmp;
        }
    }

    pub fn Write(&self, waitinfo: FdWaitInfo) {
        if !RDMA_ENABLE {
            error!("RDMADataSock::Write 1, fd: {}", self.fd);
            self.WriteData(waitinfo);
        } else {
            error!("writelock5 start");
            defer!(error!("writelock5 finish"));
            let _writelock = self.writeLock.lock();
            match self.SocketState() {
                SocketState::Init => {
                    error!("RDMADataSock::Write 2, fd: {}", self.fd);
                    self.SendLocalRDMAInfo().unwrap();
                    error!("RDMADataSock::Write 3, fd: {}", self.fd);
                    self.SetSocketState(SocketState::WaitingForRemoteMeta);
                    error!("RDMADataSock::Write 4, fd: {}", self.fd);
                }
                SocketState::WaitingForRemoteMeta => {
                    //TODO: server side received 4(W) first and 5 (R|W) afterwards. Need more investigation to see why it's different.
                    error!("RDMADataSock::Write 4.1, fd: {}", self.fd);
                }
                SocketState::WaitingForRemoteReady => {
                    //TODO: server side received 4(W) first and 5 (R|W) afterwards. Need more investigation to see why it's different.
                    error!("RDMADataSock::Write 4.2, fd: {}", self.fd);
                }
                SocketState::Ready => {
                    error!("RDMADataSock::Write 5, fd: {}", self.fd);
                    self.WriteDataLocked(waitinfo);
                    error!("RDMADataSock::Write 6, fd: {}", self.fd);
                }
                _ => {
                    panic!(
                        "RDMA socket Write state error with state {:?}",
                        self.SocketState()
                    )
                }
            }
        }
    }

    pub fn WriteData(&self, waitinfo: FdWaitInfo) {
        error!("RDMADataSock::WriteData: 0");
        error!("writelock6 start");
        defer!(error!("writelock6 finish"));
        let _writelock = self.writeLock.lock();
        error!("RDMADataSock::WriteData: 1");
        self.WriteDataLocked(waitinfo);
    }
    pub fn WriteDataLocked(&self, waitinfo: FdWaitInfo) {
        error!("RDMADataSock::WriteData: 0");
        error!("writelock7 start");
        defer!(error!("writelock7 finish"));
        //let _writelock = self.writeLock.lock();
        error!("RDMADataSock::WriteData: 1");
        let fd = self.fd;
        let socketBuf = self.socketBuf.clone();
        error!("RDMADataSock::WriteData: 2");
        let (mut addr, mut count) = socketBuf.GetAvailableWriteBuf();
        error!("RDMADataSock::WriteData: 3, count {}", count);
        if count == 0 {
            // no data
            return;
        }

        loop {
            let len = unsafe { write(fd, addr as _, count as _) };

            // closed
            if len == 0 {
                socketBuf.SetWClosed();
                if socketBuf.HasWriteData() {
                    waitinfo.Notify(EVENT_OUT);
                } else {
                    waitinfo.Notify(EVENT_HUP);
                }
                return;
            }

            if len < 0 {
                let errno = errno::errno().0;
                if errno == SysErr::EAGAIN {
                    return;
                }

                socketBuf.SetErr(errno);
                waitinfo.Notify(EVENT_ERR | EVENT_IN);
                return;
            }

            let (trigger, addrTmp, countTmp) = socketBuf.ConsumeAndGetAvailableWriteBuf(len as _);
            if trigger {
                waitinfo.Notify(EVENT_OUT);
            }

            if len < count as _ {
                // have fill the write buffer
                return;
            }

            if countTmp == 0 {
                if socketBuf.PendingWriteShutdown() {
                    waitinfo.Notify(EVENT_PENDING_SHUTDOWN);
                }

                return;
            }

            addr = addrTmp;
            count = countTmp;
        }
    }

    pub fn Notify(&self, eventmask: EventMask, waitinfo: FdWaitInfo) {
        error!(
            "RDMASocket:: Notify 1 eventmask: {:x}, fd: {}",
            eventmask, self.fd
        );
        let socketBuf = self.socketBuf.clone();

        if socketBuf.Error() != 0 {
            error!(
                "RDMASocket:: Notify 2 eventmask: {:x}, fd: {}, socketBuff.Error: {:x}",
                eventmask,
                self.fd,
                socketBuf.Error()
            );
            waitinfo.Notify(EVENT_ERR | EVENT_IN);
            error!(
                "RDMASocket:: Notify 3 eventmask: {:x}, fd: {}",
                eventmask, self.fd
            );
            return;
        }

        if eventmask & EVENT_WRITE != 0 {
            error!(
                "RDMASocket:: Notify 4 eventmask: {:x}, fd: {}",
                eventmask, self.fd
            );
            self.Write(waitinfo.clone());
            error!(
                "RDMASocket:: Notify 5 eventmask: {:x}, fd: {}",
                eventmask, self.fd
            );
        }

        if eventmask & EVENT_READ != 0 {
            error!(
                "RDMASocket:: Notify 6 eventmask: {:x}, fd: {}",
                eventmask, self.fd
            );
            self.Read(waitinfo);
            error!(
                "RDMASocket:: Notify 7 eventmask: {:x}, fd: {}",
                eventmask, self.fd
            );
        }
    }
}
