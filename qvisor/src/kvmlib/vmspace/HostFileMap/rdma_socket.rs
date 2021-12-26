use alloc::sync::Arc;

use std::collections::VecDeque;

use super::super::super::qlib::linux_def::*;
//use super::super::super::qlib::common::*;
//use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::socket_buf::*;
use super::super::super::qlib::qmsg::qcall::*;

pub struct AcceptStruct {
    pub addr: TcpSockAddr,
    pub addrlen: u32,
    pub sockBuf: Arc<SocketBuff>,
}

pub struct RDMAServerSocket {
    pub fd: i32,
    pub queueSize: usize,
    pub waitQueue: VecDeque<u64>,
    pub error: Option<i32>,
    pub acceptQueue: VecDeque<AcceptStruct>,
}

impl RDMAServerSocket {
    pub fn Accept(&mut self, acceptAddr: u64) -> i64 {
        if self.acceptQueue.len() > 0 {
            let wait = RDMAAcceptStruct::FromAddr(acceptAddr);
            let accept = self.acceptQueue.pop_front().unwrap();
            for i in 0..accept.addrlen as usize {
                wait.addr.data[i] = accept.addr.data[i];
            }

            wait.addrlen = accept.addrlen;
            wait.ret = 0;

            if self.acceptQueue.len() == self.queueSize - 1 {
                self.TryAccept();
            }
            return 0
        }

        if self.error.is_some() {
            return self.error.unwrap() as i64;
        }

        self.waitQueue.push_back(acceptAddr);
        return SysErr::EAGAIN as i64;
    }

    pub fn TryAccept(&mut self) {

    }

    /*pub fn Trigger(&mut self, eventmask: EventMask) {
        if eventmask
    }*/
}