use alloc::collections::vec_deque::VecDeque;
use alloc::sync::Arc;
use core::fmt;
use core::ops::Deref;
// use core::sync::atomic::AtomicU64;
// use core::sync::atomic::AtomicUsize;
// use core::sync::atomic::Ordering;
// use libc::*;

use super::common::*;
use super::fileinfo::*;
use super::mutex::*;
use crate::qlib::kernel::kernel::waiter::Queue;
// use super::super::super::qlib::kernel::guestfdnotifier::*;
// use super::kernel::GlobalIOMgr;
// use super::super::super::qlib::kernel::TSC;
use super::linux_def::*;
// use super::super::super::qlib::qmsg::qcall::*;
use super::socket_buf::*;
// use super::URING_MGR;
// use super::socket_info::*;

pub struct RDMAServerSockIntern {
    pub rdmaId: u32,
    pub acceptQueue: AcceptQueue,
    pub ipAddr: u32,
    pub port: u16,
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
    pub fn New(rdmaId: u32, acceptQueue: AcceptQueue, ipAddr: u32, port: u16) -> Self {
        return Self(Arc::new(RDMAServerSockIntern {
            rdmaId,
            acceptQueue: acceptQueue,
            ipAddr,
            port,
        }));
    }

    pub fn Notify(&self, _eventmask: EventMask, waitinfo: FdWaitInfo) {
        self.Accept(waitinfo);
    }

    pub fn Accept(&self, _waitinfo: FdWaitInfo) {
        error!("RDMAServerSock::Accept");
    }
}

pub struct RDMAServerSocketInfo {
    pub sock: RDMAServerSock,
    pub fd: i32,
    pub addr: TcpSockAddr,
    pub len: u32,
    pub sockBuf: SocketBuff,
    pub waitInfo: FdWaitInfo,
}

pub struct RDMADataSockIntern {
    pub rdmaId: u32,
    pub socketBuf: SocketBuff,
    // pub rdmaType: RDMAType,
    pub channelId: u32,
    pub localIpAddr: u32,
    pub localPort: u16,
    pub peerIpAddr: u32,
    pub peerPort: u16,
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
    // pub fn New(fd: i32, socketBuf: Arc<SocketBuff>, rdmaType: RDMAType, channelId: u32) -> Self {
    pub fn New(
        rdmaId: u32,
        socketBuf: SocketBuff,
        channelId: u32,
        localIpAddr: u32,
        localPort: u16,
        peerIpAddr: u32,
        peerPort: u16,
    ) -> Self {
        if RDMA_ENABLE {
            return Self(Arc::new(RDMADataSockIntern {
                rdmaId,
                socketBuf: socketBuf,
                // rdmaType: rdmaType,
                channelId,
                localIpAddr,
                localPort,
                peerIpAddr,
                peerPort,
            }));
        } else {
            return Self(Arc::new(RDMADataSockIntern {
                rdmaId,
                socketBuf: socketBuf,
                // rdmaType: rdmaType,
                channelId,
                localIpAddr,
                localPort,
                peerIpAddr,
                peerPort,
            }));
        }
    }

    /************************************ rdma integration ****************************/
    /*********************************** end of rdma integration ****************************/

    // pub fn SetReady(&self, _waitinfo: FdWaitInfo) {
    //     match &self.rdmaType {
    //         RDMAType::Client(ref addr) => {
    //             //let addr = msg as *const _ as u64;
    //             let msg = PostRDMAConnect::ToRef(*addr);
    //             msg.Finish(0);
    //         }
    //         RDMAType::Server(ref serverSock) => {
    //             let acceptQueue = serverSock.sock.acceptQueue.clone();
    //             let (trigger, _tmp) = acceptQueue.lock().EnqSocket(
    //                 serverSock.fd,
    //                 serverSock.addr,
    //                 serverSock.len,
    //                 serverSock.sockBuf.clone(),
    //             );

    //             if trigger {
    //                 serverSock.waitInfo.Notify(EVENT_IN);
    //             }
    //         }
    //         RDMAType::None => {
    //             panic!("RDMADataSock setready fail ...");
    //         }
    //     }

    //     self.SetSocketState(SocketState::Ready);
    // }

    pub fn Read(&self, _waitinfo: FdWaitInfo) {
        if !RDMA_ENABLE {
            error!("RDMADataSock::Read");
        } else {
            error!("RDMADataSock::Read");
        }
    }

    pub fn Write(&self, _waitinfo: FdWaitInfo) {
        if !RDMA_ENABLE {
            error!("RDMADataSock::Write");
        } else {
            error!("RDMADataSock::Write");
        }
    }

    //notify rdmadatasocket to sync read buff freespace with peer
    pub fn RDMARead(&self) {
        error!("RDMADataSock::RDMARead");
    }

    pub fn RDMAWrite(&self) {
        self.RDMARead();
    }

    pub fn ProcessRDMAWriteImmFinish(&self, _waitinfo: FdWaitInfo) {
        error!("RDMADataSock::ProcessRDMAWriteImmFinish");
    }

    pub fn ProcessRDMARecvWriteImm(
        &self,
        _recvCount: u64,
        _writeConsumeCount: u64,
        _waitinfo: FdWaitInfo,
    ) {
        error!("RDMADataSock::ProcessRDMARecvWriteImm");
    }

    pub fn Notify(&self, eventmask: EventMask, waitinfo: FdWaitInfo) {
        let socketBuf = self.socketBuf.clone();

        if socketBuf.Error() != 0 {
            waitinfo.Notify(EVENT_ERR | EVENT_IN);
            return;
        }

        if eventmask & EVENT_WRITE != 0 {
            self.Write(waitinfo.clone());
        }

        if eventmask & EVENT_READ != 0 {
            self.Read(waitinfo);
        }
    }
}

pub struct RDMAUDPSockIntern {
    pub recvQueue: UDPRecvQueue,
    pub port: u16,
}

#[derive(Clone)]
pub struct RDMAUDPSock(Arc<RDMAUDPSockIntern>);

impl Deref for RDMAUDPSock {
    type Target = Arc<RDMAUDPSockIntern>;

    fn deref(&self) -> &Arc<RDMAUDPSockIntern> {
        &self.0
    }
}

pub const UDP_RECV_QUEUE_LEN: usize = 5;

impl RDMAUDPSock {
    pub fn New(queue: Queue, port: u16) -> Self {
        Self(Arc::new(RDMAUDPSockIntern {
            recvQueue: UDPRecvQueue::New(UDP_RECV_QUEUE_LEN, queue),
            port,
        }))
    }
    pub fn Notify(&self, eventmask: EventMask, _waitinfo: FdWaitInfo) {
        error!("RDMAUDPSock::Notify, eventmask: {:x}", eventmask);
    }
}

#[derive(Default, Debug)]
pub struct UDPRecvItem {
    pub udpBuffIdx: u32,
    // pub addr: TcpSockAddr,
    // pub len: u32,
    // pub sockBuf: AcceptSocket,
    pub queue: Queue,
}

#[derive(Clone, Debug)]
pub struct UDPRecvQueue(Arc<QMutex<UDPRecvQueueIntern>>);

impl Deref for UDPRecvQueue {
    type Target = Arc<QMutex<UDPRecvQueueIntern>>;

    fn deref(&self) -> &Arc<QMutex<UDPRecvQueueIntern>> {
        &self.0
    }
}

impl UDPRecvQueue {
    pub fn New(len: usize, queue: Queue) -> Self {
        let inner = UDPRecvQueueIntern {
            udpRecvQueue: VecDeque::new(),
            queueLen: len,
            error: 0,
            total: 0,
            queue: queue,
        };

        return Self(Arc::new(QMutex::new(inner)));
    }

    pub fn EnqSocket(&self, udpBuffIdx: u32, queue: Queue) -> bool {
        let (trigger, hasSpace) = {
            let mut inner = self.lock();
            let item = UDPRecvItem {
                udpBuffIdx,
                queue: queue,
            };

            inner.udpRecvQueue.push_back(item);
            inner.total += 1;
            let trigger = inner.udpRecvQueue.len() == 1;

            (trigger, inner.udpRecvQueue.len() < inner.queueLen)
        };

        if trigger {
            let queue = self.lock().queue.clone();
            queue.Notify(READABLE_EVENT)
        }

        return hasSpace;
    }
}

pub struct UDPRecvQueueIntern {
    pub udpRecvQueue: VecDeque<UDPRecvItem>,
    pub queueLen: usize,
    pub error: i32,
    pub total: u64,
    pub queue: Queue,
}

impl fmt::Debug for UDPRecvQueueIntern {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "UDPRecvQueueIntern aiQueue {:x?}", self.udpRecvQueue)
    }
}

impl Drop for UDPRecvQueueIntern {
    fn drop(&mut self) {}
}

impl UDPRecvQueueIntern {
    pub fn SetErr(&mut self, error: i32) {
        self.error = error
    }

    pub fn Err(&self) -> i32 {
        return self.error;
    }

    pub fn SetQueueLen(&mut self, len: usize) {
        self.queueLen = len;
    }

    pub fn HasSpace(&self) -> bool {
        return self.udpRecvQueue.len() < self.queueLen;
    }

    pub fn DeqSocket(&mut self) -> (bool, Result<UDPRecvItem>) {
        let trigger = self.udpRecvQueue.len() == self.queueLen;

        match self.udpRecvQueue.pop_front() {
            None => {
                if self.error != 0 {
                    return (false, Err(Error::SysError(self.error)));
                }
                return (trigger, Err(Error::SysError(SysErr::EAGAIN)));
            }
            Some(item) => return (trigger, Ok(item)),
        }
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.udpRecvQueue.len() > 0 {
            event |= READABLE_EVENT;
        }

        if self.error != 0 {
            event |= EVENT_ERR;
        }

        return event;
    }
}
