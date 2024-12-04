use alloc::collections::BTreeMap;
use alloc::sync::Arc;
// use alloc::vec::Vec;
use core::ops::Deref;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::Ordering;
use spin::Mutex;

use super::common::*;
use super::fileinfo::*;
use super::idallocator::IdAllocator;
use super::kernel::tcpip::tcpip::*;
use super::kernel::GlobalIOMgr;
use super::kernel::GlobalRDMASvcCli;
use super::linux_def::*;
use super::rdma_share::*;
use super::rdmasocket::*;
use super::socket_buf::*;
use super::unix_socket::UnixSocket;
use crate::qlib::kernel::kernel::waiter::Queue;
// use super::kernel::TSC;

pub struct RDMASvcCliIntern {
    // agent id
    pub agentId: u32,

    // the unix socket fd between rdma client and RDMASrv
    pub cliSock: UnixSocket,

    // the memfd share memory with rdma client
    pub cliMemFd: i32,

    // the memfd share memory with rdma server
    pub srvMemFd: i32,

    // the eventfd which send notification to client
    pub cliEventFd: i32,

    // the eventfd which send notification to client
    pub srvEventFd: i32,

    // the memory region shared with client
    pub cliMemRegion: MemRegion,

    pub cliShareRegion: Mutex<&'static mut ClientShareRegion>,

    // srv memory region shared with all RDMAClient
    pub srvMemRegion: MemRegion,

    // the bitmap to expedite ready container search
    pub srvShareRegion: Mutex<&'static mut ShareRegion>,

    //TODO: rename, it's the channelId to RDMAId's mapping
    pub channelToSocketMappings: Mutex<BTreeMap<u32, u32>>,

    pub rdmaIdToSocketMappings: Mutex<BTreeMap<u32, i32>>,

    pub nextRDMAId: AtomicU32,

    pub podId: [u8; 64],

    pub udpSentBufferAllocator: Mutex<UDPBufferAllocator>,

    pub portToFdInfoMappings: Mutex<BTreeMap<u16, FdInfo>>,

    pub tcpPortAllocator: Mutex<IdAllocator>,
    pub udpPortAllocator: Mutex<IdAllocator>,
    // pub timestamp: Mutex<Vec<i64>>,
}

impl Deref for RDMASvcClient {
    type Target = Arc<RDMASvcCliIntern>;

    fn deref(&self) -> &Arc<RDMASvcCliIntern> {
        &self.intern
    }
}

pub struct RDMASvcClient {
    pub intern: Arc<RDMASvcCliIntern>,
}

impl Default for RDMASvcClient {
    fn default() -> Self {
        Self {
            intern: Arc::new(RDMASvcCliIntern {
                agentId: 0,
                cliSock: UnixSocket { fd: -1 },
                cliMemFd: 0,
                srvMemFd: 0,
                srvEventFd: 0,
                cliEventFd: 0,
                cliMemRegion: MemRegion { addr: 0, len: 0 },
                cliShareRegion: unsafe { Mutex::new(&mut (*(0 as *mut ClientShareRegion))) },
                srvMemRegion: MemRegion { addr: 0, len: 0 },
                srvShareRegion: unsafe { Mutex::new(&mut (*(0 as *mut ShareRegion))) },
                channelToSocketMappings: Mutex::new(BTreeMap::new()),
                rdmaIdToSocketMappings: Mutex::new(BTreeMap::new()),
                nextRDMAId: AtomicU32::new(0), //AtomicU64::new((i32::MAX + 1) as u64), //2147483647 + 1 = 2147483648
                podId: [0; 64],
                udpSentBufferAllocator: Mutex::new(UDPBufferAllocator::default()),
                portToFdInfoMappings: Mutex::new(BTreeMap::new()),
                tcpPortAllocator: Mutex::new(IdAllocator::default()),
                udpPortAllocator: Mutex::new(IdAllocator::default()),
                // timestamp: Mutex::new(Vec::with_capacity(0)),
            }),
        }
    }
}

impl RDMASvcClient {
    pub fn listen(&self, sockfd: u32, endpoint: &Endpoint, waitingLen: i32) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAListen(RDMAListenReq {
            sockfd: sockfd,
            ipAddr: endpoint.ipAddr,
            port: endpoint.port,
            waitingLen,
        }));
        res
    }

    pub fn listenUsingPodId(&self, sockfd: u32, port: u16, waitingLen: i32) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAListenUsingPodId(RDMAListenReqUsingPodId {
            sockfd: sockfd,
            podId: self.podId,
            port,
            waitingLen,
        }));
        res
    }

    pub fn connect(
        &self,
        sockfd: u32,
        dstIpAddr: u32,
        dstPort: u16,
        srcIpAddr: u32,
        srcPort: u16,
    ) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAConnect(RDMAConnectReq {
            sockfd,
            dstIpAddr,
            dstPort,
            srcIpAddr, //101099712, //u32::from(Ipv4Addr::from_str("192.168.6.6").unwrap()).to_be(),
            srcPort,   //16866u16.to_be(),
        }));
        res
    }

    pub fn connectUsingPodId(
        &self,
        sockfd: u32,
        dstIpAddr: u32,
        dstPort: u16,
        srcPort: u16,
    ) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAConnectUsingPodId(
            RDMAConnectReqUsingPodId {
                sockfd,
                dstIpAddr,
                dstPort,
                podId: self.podId, //101099712, //u32::from(Ipv4Addr::from_str("192.168.6.6").unwrap()).to_be(),
                srcPort,           //16866u16.to_be(),
            },
        ));
        res
    }

    pub fn read(&self, channelId: u32) -> Result<()> {
        // println!("rdmaSvcCli::read 1");
        if self.cliShareRegion.lock().sq.Push(RDMAReq {
            user_data: 0,
            msg: RDMAReqMsg::RDMARead(RDMAReadReq {
                channelId: channelId,
            }),
        }) {
            // println!("rdmaSvcCli::read 2");
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            error!("rdmaSvcCli::read 3");
            return Err(Error::NoEnoughSpace);
        }
    }

    pub fn write(&self, channelId: u32) -> Result<()> {
        // println!("rdmaSvcCli::write 1");
        if self.cliShareRegion.lock().sq.Push(RDMAReq {
            user_data: 0,
            msg: RDMAReqMsg::RDMAWrite(RDMAWriteReq {
                channelId: channelId,
            }),
        }) {
            // error!("rdmaSvcCli::write 2");
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            error!("rdmaSvcCli::write 3");
            return Err(Error::NoEnoughSpace);
        }
    }

    pub fn shutdown(&self, channelId: u32, howto: u8) -> Result<()> {
        // println!(
        //     "rdmaSvcCli::shutdown 1, channelId: {}, howto: {}",
        //     channelId, howto
        // );
        if self.cliShareRegion.lock().sq.Push(RDMAReq {
            user_data: 0,
            msg: RDMAReqMsg::RDMAShutdown(RDMAShutdownReq {
                channelId: channelId,
                howto,
            }),
        }) {
            // println!("rdmaSvcCli::shutdown 2");
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            // println!("rdmaSvcCli::shutdown 3");
            return Err(Error::NoEnoughSpace);
        }
    }

    pub fn pendingshutdown(&self, channelId: u32) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAPendingShutdown(RDMAPendingShutdownReq {
            channelId,
        }));
        res
    }

    pub fn close(&self, channelId: u32) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAClose(RDMACloseReq { channelId }));
        res
    }

    pub fn sendUDPPacket(&self, udpBuffIndex: u32) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMASendUDPPacket(RDMASendUDPPacket {
            podId: self.podId,
            udpBuffIdx: udpBuffIndex,
        }));
        res
    }

    pub fn returnUDPBuff(&self, udpBuffIndex: u32) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAReturnUDPBuff(RDMAReturnUDPBuff {
            udpBuffIdx: udpBuffIndex,
        }));
        res
    }

    pub fn updateBitmapAndWakeUpServerIfNecessary(&self) {
        // error!("updateBitmapAndWakeUpServerIfNecessary 1 ");
        let mut srvShareRegion = self.srvShareRegion.lock();
        // error!("updateBitmapAndWakeUpServerIfNecessary 2 ");
        srvShareRegion.updateBitmap(self.agentId);
        if srvShareRegion.srvBitmap.load(Ordering::Acquire) == 1 {
            self.wakeupSvc();
        } else {
            // error!("server is not sleeping");
            // self.updateBitmapAndWakeUpServerIfNecessary();
        }
    }

    pub fn SentMsgToSvc(&self, msg: RDMAReqMsg) -> Result<()> {
        if self
            .cliShareRegion
            .lock()
            .sq
            .Push(RDMAReq { user_data: 0, msg })
        {
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            error!("SentMsgToSvc, no space");
            return Err(Error::NoEnoughSpace);
        }
    }

    pub fn DrainCompletionQueue(&self) -> usize {
        self.cliShareRegion
            .lock()
            .clientBitmap
            .store(0, Ordering::Release);
        let mut count = 0;
        count += self.ProcessRDMASvcMessage();
        self.cliShareRegion
            .lock()
            .clientBitmap
            .store(1, Ordering::Release);
        count += self.ProcessRDMASvcMessage();
        count
    }

    pub fn ProcessRDMASvcMessage(&self) -> usize {
        let mut count = 0;
        loop {
            let request = self.cliShareRegion.lock().cq.Pop();
            count += 1;
            match request {
                Some(cq) => {
                    match cq.msg {
                        RDMARespMsg::RDMAConnect(response) => {
                            // GlobalRDMASvcCli().timestamp.lock().push(TSC.Rdtsc()); // 5 (219/184)
                            let sockfd = match self
                                .rdmaIdToSocketMappings
                                .lock()
                                .get(&response.sockfd)
                            {
                                Some(sockFdVal) => *sockFdVal,
                                None => {
                                    panic!("RDMARespMsg::RDMAConnect, Can't find sockfd based on rdmaId: {}", response.sockfd);
                                }
                            };

                            let sockInfo = GlobalIOMgr()
                                .GetByHost(sockfd)
                                .unwrap()
                                .lock()
                                .sockInfo
                                .lock()
                                .clone();
                            match sockInfo {
                                SockInfo::Socket(_) => {
                                    let ioBufIndex = response.ioBufIndex as usize;
                                    let shareRegion = self.cliShareRegion.lock();
                                    let sockBuf = SocketBuff(Arc::new_in(
                                        SocketBuffIntern::InitWithShareMemory(
                                            MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                            &shareRegion.ioMetas[ioBufIndex].readBufAtoms
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].readBufWaitingRW
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].writeBufAtoms
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].writeBufWaitingRW
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].consumeReadData
                                                as *const _
                                                as u64,
                                            &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                            &shareRegion.iobufs[ioBufIndex].write as *const _
                                                as u64,
                                            false,
                                        ),
                                        crate::GUEST_HOST_SHARED_ALLOCATOR,
                                    ));

                                    let dataSock = RDMADataSock::New(
                                        response.sockfd,
                                        sockBuf.clone(),
                                        response.channelId,
                                        response.srcIpAddr,
                                        response.srcPort,
                                        response.dstIpAddr,
                                        response.dstPort,
                                    );
                                    self.channelToSocketMappings
                                        .lock()
                                        .insert(response.channelId, response.sockfd);

                                    *GlobalIOMgr()
                                        .GetByHost(sockfd)
                                        .unwrap()
                                        .lock()
                                        .sockInfo
                                        .lock() = SockInfo::RDMADataSocket(dataSock);
                                    GlobalIOMgr()
                                        .GetByHost(sockfd)
                                        .unwrap()
                                        .lock()
                                        .waitInfo
                                        .Notify(EVENT_OUT);
                                    // GlobalRDMASvcCli().timestamp.lock().push(TSC.Rdtsc()); // 6 (23/6)
                                }
                                _ => {
                                    panic!("RDMARespMsg::RDMAConnect, SockInfo is not correct type: {:?}", sockInfo);
                                }
                            }
                        }
                        RDMARespMsg::RDMAAccept(response) => {
                            let sockfd = match self
                                .rdmaIdToSocketMappings
                                .lock()
                                .get(&response.sockfd)
                            {
                                Some(sockFdVal) => *sockFdVal,
                                None => {
                                    debug!("RDMARespMsg::RDMAAccept, Can't find sockfd based on rdmaId: {}", response.sockfd);
                                    break;
                                }
                            };

                            let sockInfo = GlobalIOMgr()
                                .GetByHost(sockfd)
                                .unwrap()
                                .lock()
                                .sockInfo
                                .lock()
                                .clone();
                            match sockInfo {
                                SockInfo::RDMAServerSocket(rdmaServerSock) => {
                                    // let fd = unsafe { libc::socket(AFType::AF_INET, SOCK_STREAM, 0) };
                                    // let c1 = TSC.Rdtsc();
                                    let fd = self.CreateSocket() as i32;
                                    // let c2 = TSC.Rdtsc();
                                    // error!("Create socket time used: {}", c2 - c1);
                                    let rdmaId = GlobalRDMASvcCli()
                                        .nextRDMAId
                                        .fetch_add(1, Ordering::Release);
                                    self.rdmaIdToSocketMappings.lock().insert(rdmaId, fd);
                                    let ioBufIndex = response.ioBufIndex as usize;
                                    let shareRegion = self.cliShareRegion.lock();
                                    let sockBuf = SocketBuff(Arc::new_in(
                                        SocketBuffIntern::InitWithShareMemory(
                                            MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                            &shareRegion.ioMetas[ioBufIndex].readBufAtoms
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].readBufWaitingRW
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].writeBufAtoms
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].writeBufWaitingRW
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].consumeReadData
                                                as *const _
                                                as u64,
                                            &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                            &shareRegion.iobufs[ioBufIndex].write as *const _
                                                as u64,
                                            false,
                                        ),
                                        crate::GUEST_HOST_SHARED_ALLOCATOR,
                                    ));

                                    let dataSock = RDMADataSock::New(
                                        rdmaId,
                                        sockBuf.clone(),
                                        response.channelId,
                                        response.srcIpAddr,
                                        response.srcPort,
                                        response.dstIpAddr,
                                        response.dstPort,
                                    );

                                    *GlobalIOMgr()
                                        .GetByHost(fd as i32)
                                        .unwrap()
                                        .lock()
                                        .sockInfo
                                        .lock() = SockInfo::RDMADataSocket(dataSock);

                                    let sockAddr = SockAddr::Inet(SockAddrInet {
                                        Family: AFType::AF_INET as u16,
                                        Port: response.dstPort,
                                        Addr: response.dstIpAddr.to_be_bytes(),
                                        Zero: [0; 8],
                                    });
                                    let mut tcpSockAddr = TcpSockAddr::default();
                                    let len = sockAddr.Len();
                                    let _res = sockAddr.Marsh(&mut tcpSockAddr.data, len);
                                    let _tmp = rdmaServerSock.acceptQueue.EnqSocket(
                                        fd,
                                        tcpSockAddr,
                                        len as u32,
                                        sockBuf.into(),
                                        Queue::default(),
                                    );

                                    self.channelToSocketMappings
                                        .lock()
                                        .insert(response.channelId, rdmaId);

                                    /*if trigger {
                                        GlobalIOMgr()
                                            .GetByHost(sockfd)
                                            .unwrap()
                                            .lock()
                                            .waitInfo
                                            .Notify(EVENT_IN);
                                    }*/
                                }
                                _ => {
                                    panic!("RDMARespMsg::RDMAAccept, SockInfo is not correct type: {:?}", sockInfo);
                                }
                            }
                        }
                        RDMARespMsg::RDMANotify(response) => {
                            let rdmaId = match self
                                .channelToSocketMappings
                                .lock()
                                .get_mut(&response.channelId)
                            {
                                Some(rdmaIdVal) => *rdmaIdVal,
                                None => {
                                    //     debug!(
                                    //     "RDMARespMsg::RDMANotify, Can't find rdmaId based on channelId: {}",
                                    //     response.channelId
                                    // );
                                    break;
                                }
                            };

                            let sockFd = match self.rdmaIdToSocketMappings.lock().get(&rdmaId) {
                                Some(sockFdVal) => *sockFdVal,
                                None => {
                                    debug!("RDMARespMsg::RDMANotify, Can't find sockfd based on rdmaId: {}", rdmaId);
                                    break;
                                }
                            };

                            if response.event & EVENT_IN != 0 {
                                let waitInfo = GlobalIOMgr()
                                    .GetByHost(sockFd)
                                    .unwrap()
                                    .lock()
                                    .waitInfo
                                    .clone();

                                waitInfo.Notify(EVENT_IN);
                            }
                            if response.event & EVENT_OUT != 0 {
                                let waitInfo = GlobalIOMgr()
                                    .GetByHost(sockFd)
                                    .unwrap()
                                    .lock()
                                    .waitInfo
                                    .clone();
                                waitInfo.Notify(EVENT_OUT);
                            }
                            if response.event & EVENT_PENDING_SHUTDOWN != 0 {
                                let sockInfo = GlobalIOMgr()
                                    .GetByHost(sockFd)
                                    .unwrap()
                                    .lock()
                                    .sockInfo
                                    .lock()
                                    .clone();
                                match sockInfo {
                                    SockInfo::RDMADataSocket(dataSock) => {
                                        if dataSock.socketBuf.PendingWriteShutdown() {
                                            let waitInfo = GlobalIOMgr()
                                                .GetByHost(sockFd)
                                                .unwrap()
                                                .lock()
                                                .waitInfo
                                                .clone();
                                            waitInfo.Notify(EVENT_PENDING_SHUTDOWN);
                                        }
                                    }
                                    _ => {
                                        panic!("RDMARespMsg::RDMAFinNotify, Unexpected sockInfo type: {:?}", sockInfo);
                                    }
                                }
                            }
                        }
                        RDMARespMsg::RDMAFinNotify(response) => {
                            // debug!("RDMARespMsg::RDMAFinNotify, response: {:?}", response);
                            let rdmaId = match self
                                .channelToSocketMappings
                                .lock()
                                .get_mut(&response.channelId)
                            {
                                Some(rdmaIdVal) => *rdmaIdVal,
                                None => {
                                    break;
                                }
                            };

                            let sockFd = match self.rdmaIdToSocketMappings.lock().get(&rdmaId) {
                                Some(sockFdVal) => *sockFdVal,
                                None => {
                                    debug!("RDMARespMsg::RDMAFinNotify, Can't find sockfd based on rdmaId: {}", rdmaId);
                                    break;
                                }
                            };

                            let fdInfoOpt = GlobalIOMgr().GetByHost(sockFd);
                            if fdInfoOpt.is_some() {
                                let fdInfo = fdInfoOpt.unwrap().clone();
                                let sockInfo = fdInfo.lock().sockInfo.lock().clone();
                                match sockInfo {
                                    SockInfo::RDMADataSocket(dataSock) => {
                                        dataSock.socketBuf.SetRClosed();
                                        let waitInfo = fdInfo.lock().waitInfo.clone();
                                        waitInfo.Notify(EVENT_IN);
                                    }
                                    _ => {
                                        panic!("RDMARespMsg::RDMAFinNotify, Unexpected sockInfo type: {:?}", sockInfo);
                                    }
                                }
                            }
                        }
                        RDMARespMsg::RDMAReturnUDPBuff(response) => {
                            // debug!("RDMARespMsg::RDMAReturnUDPBuff, response: {:?}", response);
                            GlobalRDMASvcCli()
                                .udpSentBufferAllocator
                                .lock()
                                .ReturnBuffer(response.udpBuffIdx);
                        }
                        RDMARespMsg::RDMARecvUDPPacket(response) => {
                            // debug!("RDMARespMsg::RDMARecvUDPPacket, response: {:?}", response);
                            let udpPacket = &GlobalRDMASvcCli().cliShareRegion.lock().udpBufRecv
                                [response.udpBuffIdx as usize];
                            // + wrId * (mem::size_of::<UDPPacket>() + 40) as u64
                            // + 40;
                            // debug!(
                            //     "RDMARespMsg::RDMARecvUDPPacket, 1 udpPacket: {:?}",
                            //     udpPacket
                            // );
                            match GlobalRDMASvcCli()
                                .portToFdInfoMappings
                                .lock()
                                .get(&udpPacket.dstPort)
                            {
                                Some(fdInfo) => {
                                    let sockInfo = fdInfo.lock().sockInfo.lock().clone();
                                    match sockInfo {
                                        SockInfo::RDMAUDPSocket(udpSock) => {
                                            udpSock
                                                .recvQueue
                                                .EnqSocket(response.udpBuffIdx, Queue::default());
                                            fdInfo.lock().waitInfo.Notify(EVENT_IN);
                                        }
                                        _ => {
                                            panic!("RDMARespMsg::RDMARecvUDPPacket, sockInfo: {:?} is not expected for UDP over RDMA", sockInfo);
                                        }
                                    }
                                }
                                None => {
                                    error!("RDMARespMsg::RDMARecvUDPPacket, no FdInfo found for port: {}, ipAddr: {}", udpPacket.dstPort, udpPacket.dstIpAddr);
                                }
                            }
                        }
                    }
                }
                None => {
                    count -= 1;
                    break;
                }
            }
        }
        count
    }
}
