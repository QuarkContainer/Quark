// use super::super::super::qlib::mutex::*;
use alloc::sync::Arc;
// use core::mem;
use core::ops::Deref;
// use core::sync::atomic::AtomicU64;
// use core::sync::atomic::AtomicUsize;
// use core::sync::atomic::Ordering;
// use libc::*;

use super::common::*;
use super::fileinfo::*;
// use super::super::super::qlib::kernel::guestfdnotifier::*;
// use super::kernel::GlobalIOMgr;
// use super::super::super::qlib::kernel::TSC;
use super::linux_def::*;
// use super::super::super::qlib::qmsg::qcall::*;
use super::socket_buf::*;
// use super::URING_MGR;
// use super::socket_info::*;

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

    pub fn Accept(&self, _waitinfo: FdWaitInfo) {
        error!("RDMAServerSock::Accept");
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
    pub rdmaType: RDMAType,
    pub channelId: u32,
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
    pub fn New(fd: i32, socketBuf: Arc<SocketBuff>, rdmaType: RDMAType, channelId: u32) -> Self {
        if RDMA_ENABLE {
            return Self(Arc::new(RDMADataSockIntern {
                fd: fd,
                socketBuf: socketBuf,
                rdmaType: rdmaType,
                channelId,
            }));
        } else {
            return Self(Arc::new(RDMADataSockIntern {
                fd: fd,
                socketBuf: socketBuf,
                rdmaType: rdmaType,
                channelId,
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
