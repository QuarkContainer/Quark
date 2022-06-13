use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::ops::Deref;
use core::sync::atomic::Ordering;
use spin::Mutex;

use super::common::*;
use super::fileinfo::*;
use super::kernel::GlobalIOMgr;
use super::linux_def::*;
use super::rdma_share::*;
use super::rdmasocket::*;
use super::socket_buf::*;
use super::unix_socket::UnixSocket;

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

    pub channelToSocketMappings: Mutex<BTreeMap<u32, i32>>,
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

    pub fn connect(&self, sockfd: u32, ipAddr: u32, port: u16) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAConnect(RDMAConnectReq {
            sockfd,
            dstIpAddr: ipAddr,
            dstPort: port,
            srcIpAddr: 101099712, //u32::from(Ipv4Addr::from_str("192.168.6.6").unwrap()).to_be(),
            srcPort: 16866u16.to_be(),
        }));
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
            // println!("rdmaSvcCli::read 3");
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
            // println!("rdmaSvcCli::write 2");
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            // println!("rdmaSvcCli::write 3");
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

    pub fn updateBitmapAndWakeUpServerIfNecessary(&self) {
        // println!("updateBitmapAndWakeUpServerIfNecessary 1 ");
        let mut srvShareRegion = self.srvShareRegion.lock();
        // println!("updateBitmapAndWakeUpServerIfNecessary 2 ");
        srvShareRegion.updateBitmap(self.agentId);
        if srvShareRegion.srvBitmap.load(Ordering::Relaxed) == 1 {
            // println!("before write srvEventFd");
            self.wakeupSvc();
        } else {
            // println!("server is not sleeping");
            self.updateBitmapAndWakeUpServerIfNecessary();
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
                Some(cq) => match cq.msg {
                    RDMARespMsg::RDMAConnect(response) => {
                        // debug!("RDMARespMsg::RDMAConnect, response: {:?}", response);
                        let fdInfo = GlobalIOMgr().GetByHost(response.sockfd as i32).unwrap();
                        let fdInfoLock = fdInfo.lock();
                        let sockInfo = fdInfoLock.sockInfo.lock().clone();

                        match sockInfo {
                            SockInfo::Socket(_) => {
                                let ioBufIndex = response.ioBufIndex as usize;
                                let shareRegion = self.cliShareRegion.lock();
                                let sockBuf = Arc::new(SocketBuff::InitWithShareMemory(
                                    MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                    &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _
                                        as u64,
                                    &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _
                                        as u64,
                                    &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _
                                        as u64,
                                    &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                    &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
                                    false,
                                ));

                                let dataSock = RDMADataSock::New(
                                    response.sockfd as i32, //Allocate fd
                                    // 0, //TODO: need udpate
                                    // 0, //TODO: need update
                                    // response.dstIpAddr,
                                    // response.dstPort,
                                    // SockStatus::ESTABLISHED,
                                    // response.channelId,
                                    sockBuf.clone(),
                                    response.channelId,
                                );
                                self.channelToSocketMappings
                                    .lock()
                                    .insert(response.channelId, response.sockfd as i32);

                                *fdInfoLock.sockInfo.lock() = SockInfo::RDMADataSocket(dataSock);
                                fdInfoLock.waitInfo.Notify(EVENT_OUT);
                            }
                            _ => {
                                panic!("SockInfo is not correct type");
                            }
                        }
                    }
                    RDMARespMsg::RDMAAccept(response) => {
                        // debug!("RDMARespMsg::RDMAAccept, response: {:?}", response);

                        let fdInfo = GlobalIOMgr().GetByHost(response.sockfd as i32).unwrap();
                        let fdInfoLock = fdInfo.lock();
                        let sockInfo = fdInfoLock.sockInfo.lock().clone();

                        match sockInfo {
                            SockInfo::RDMAServerSocket(rdmaServerSock) => {
                                // let fd = unsafe { libc::socket(AFType::AF_INET, SOCK_STREAM, 0) };
                                let fd = self.CreateSocket() as i32;
                                let ioBufIndex = response.ioBufIndex as usize;
                                let shareRegion = self.cliShareRegion.lock();
                                let sockBuf = Arc::new(SocketBuff::InitWithShareMemory(
                                    MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                    &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _
                                        as u64,
                                    &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _
                                        as u64,
                                    &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _
                                        as u64,
                                    &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                    &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
                                    false,
                                ));

                                let dataSock = RDMADataSock::New(
                                    fd, //Allocate fd
                                    // 0, //TODO: need udpate
                                    // 0, //TODO: need update
                                    // response.dstIpAddr,
                                    // response.dstPort,
                                    // SockStatus::ESTABLISHED,
                                    // response.channelId,
                                    sockBuf.clone(),
                                    response.channelId,
                                );

                                // GlobalIOMgr().AddSocket(fd);
                                let fdInfo = GlobalIOMgr().GetByHost(fd as i32).unwrap();
                                let fdInfoLock1 = fdInfo.lock();
                                *fdInfoLock1.sockInfo.lock() = SockInfo::RDMADataSocket(dataSock);

                                let (trigger, _tmp) = rdmaServerSock.acceptQueue.lock().EnqSocket(
                                    fd,
                                    TcpSockAddr::default(),
                                    0, //TCP_ADDR_LEN as _,
                                    sockBuf,
                                );
                                self.channelToSocketMappings
                                    .lock()
                                    .insert(response.channelId, fd);
                                if trigger {
                                    fdInfoLock.waitInfo.Notify(EVENT_IN);
                                }
                            }
                            _ => {
                                panic!("SockInfo is not correct type");
                            }
                        }
                    }
                    RDMARespMsg::RDMANotify(response) => {
                        // debug!("RDMARespMsg::RDMANotify, response: {:?}", response);
                        if response.event & EVENT_IN != 0 {
                            let mut channelToSocketMappings = self.channelToSocketMappings.lock();
                            let sockFd = channelToSocketMappings.get_mut(&response.channelId);
                            match sockFd {
                                Some(fd) => {
                                    GlobalIOMgr()
                                        .GetByHost(*fd)
                                        .unwrap()
                                        .lock()
                                        .waitInfo
                                        .Notify(EVENT_IN);
                                }
                                None => {
                                    info!("channelId: {} is not found", response.channelId);
                                }
                            }

                            // let shareRegion = self.cliShareRegion.lock();
                            // let readBufAddr = &shareRegion.iobufs as *const _ as u64;
                            // let mut readBufHeadTailAddr = &shareRegion.ioMetas as *const _ as u64 - 24;
                            // debug!(
                            //     "RDMARespMsg::RDMANotify readBufAddr: {:x}, first byte: {}",
                            //     readBufAddr,
                            //     unsafe { *(readBufAddr as *const u8) }
                            // );
                            // loop {
                            //     debug!(
                            //         "RDMARespMsg::RDMANotify, readBufHeadTailAddr: {:x}, readHead: {}, readTail: {}, writehead: {}, writeTail: {}, consumedData: {}",
                            //         readBufHeadTailAddr,
                            //         unsafe { *(readBufHeadTailAddr as *const u32) },
                            //         unsafe { *((readBufHeadTailAddr + 4) as *const u32) },
                            //         unsafe { *((readBufHeadTailAddr + 8) as *const u32) },
                            //         unsafe { *((readBufHeadTailAddr + 12) as *const u32) },
                            //         unsafe { *((readBufHeadTailAddr + 16) as *const u64) }
                            //     );
                            //     readBufHeadTailAddr += 24;
                            //     if readBufHeadTailAddr > (&shareRegion.iobufs as *const _ as u64) {
                            //         break;
                            //     }
                            // }
                            // let mut i = 0;
                            // readBufHeadTailAddr = &shareRegion.iobufs as *const _ as u64;
                            // loop {
                            //     debug!(
                            //         "RDMARespMsg::RDMANotify, buf: {:x}, val: {}",
                            //         readBufHeadTailAddr,
                            //         unsafe { *((readBufHeadTailAddr + i) as *const u8) },
                            //     );
                            //     i += 1;
                            //     if i > 16 {
                            //         break;
                            //     }
                            // }
                        }
                        if response.event & EVENT_OUT != 0 {
                            let mut channelToSocketMappings = self.channelToSocketMappings.lock();
                            let sockFd = channelToSocketMappings
                                .get_mut(&response.channelId)
                                .unwrap();
                            GlobalIOMgr()
                                .GetByHost(*sockFd)
                                .unwrap()
                                .lock()
                                .waitInfo
                                .Notify(EVENT_OUT);
                        }
                    }
                    RDMARespMsg::RDMAFinNotify(response) => {
                        // debug!("RDMARespMsg::RDMAFinNotify, response: {:?}", response);
                        // let mut channelToSockInfos = gatewayCli.channelToSockInfos.lock();
                        // let sockInfo = channelToSockInfos.get_mut(&response.channelId).unwrap();
                        // if response.event & FIN_RECEIVED_FROM_PEER != 0 {
                        //     *sockInfo.finReceived.lock() = true;
                        //     gatewayCli.WriteToSocket(sockInfo, &sockFdMappings);
                        // }
                    }
                },
                None => {
                    count -= 1;
                    break;
                }
            }
        }
        count
    }
}
