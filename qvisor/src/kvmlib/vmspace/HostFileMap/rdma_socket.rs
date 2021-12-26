use alloc::sync::Arc;
use libc::*;

use super::super::super::IO_MGR;
use super::super::super::URING_MGR;
use super::super::super::SHARE_SPACE;
use super::super::super::qlib::linux_def::*;
//use super::super::super::qlib::common::*;
//use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::socket_buf::*;
//use super::super::super::qlib::qmsg::qcall::*;
use super::super::super::qlib::qmsg::input::*;


pub struct RDMAServerSocket {
    pub fd: i32,
    pub acceptQueue: AcceptQueue,
}

impl RDMAServerSocket {
    pub fn TryAccept(&mut self) {
        if self.acceptQueue.lock().Err() != 0 {
            Self::FdNotify(self.fd, EVENT_ERR | EVENT_IN);
            return
        }

        let mut hasSpace = self.acceptQueue.lock().HasSpace();

        while hasSpace {
            let tcpAddr = TcpSockAddr::default();
            let mut len : u32 = TCP_ADDR_LEN as _;

            let ret = unsafe{
                accept4(self.fd, tcpAddr.Addr() as  *mut sockaddr, &mut len as  *mut socklen_t, SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC)
            };

            if ret < 0 {
                if ret == -SysErr::EAGAIN {
                    return
                }

                Self::FdNotify(self.fd, EVENT_ERR | EVENT_IN);
                self.acceptQueue.lock().SetErr(-ret);
                return
            }

            let fd = ret;

            IO_MGR.lock().AddFd(fd, true);
            URING_MGR.lock().Addfd(fd).unwrap();

            let (trigger, tmp) = self.acceptQueue.lock().EnqSocket(fd, tcpAddr, len, Arc::new(SocketBuff::default()));
            hasSpace = tmp;

            if trigger {
                Self::FdNotify(self.fd, EVENT_IN);
            }
        }
    }

    pub fn FdNotify(fd: i32, mask: EventMask) {
        SHARE_SPACE.AQHostInputCall(&HostInputMsg::FdNotify(FdNotify{
            fd: fd,
            mask: mask,
        }));
    }

    pub fn Trigger(&mut self, _eventmask: EventMask) {
        self.TryAccept();
    }
}