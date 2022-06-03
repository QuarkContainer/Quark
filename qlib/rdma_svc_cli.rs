use alloc::sync::Arc;
use core::ops::Deref;
use core::sync::atomic::Ordering;
use spin::Mutex;

use super::common::*;
use super::rdma_share::*;
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
}
