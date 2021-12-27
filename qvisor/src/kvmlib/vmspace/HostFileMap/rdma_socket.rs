use alloc::sync::Arc;
use libc::*;
use core::ops::Deref;
use super::super::super::qlib::mutex::*;

use super::super::super::IO_MGR;
use super::super::super::URING_MGR;
//use super::super::super::SHARE_SPACE;
use super::super::super::qlib::linux_def::*;
//use super::super::super::qlib::common::*;
//use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::socket_buf::*;
//use super::super::super::qlib::qmsg::qcall::*;
//use super::super::super::qlib::qmsg::input::*;
use super::fdinfo::*;

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
                if ret == -SysErr::EAGAIN {
                    return
                }

                FdNotify(minefd, EVENT_ERR | EVENT_IN);
                acceptQueue.lock().SetErr(-ret);
                return
            }

            let fd = ret;

            IO_MGR.AddSocket(fd);
            URING_MGR.lock().Addfd(fd).unwrap();

            let (trigger, tmp) = acceptQueue.lock().EnqSocket(fd, tcpAddr, len, Arc::new(SocketBuff::default()));
            hasSpace = tmp;

            if trigger {
                FdNotify(minefd, EVENT_IN);
            }
        }
    }
}

pub struct RDMAServerSockIntern {
    pub fd: i32,
    pub acceptQueue: AcceptQueue,
}
