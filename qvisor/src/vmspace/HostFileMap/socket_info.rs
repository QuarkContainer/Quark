use core::fmt;

use super::rdma_socket::*;
use super::fdinfo::*;
use super::super::super::qlib::linux_def::*;

#[derive(Clone)]
pub enum SockInfo {
    File, // it is not socket
    Socket, // normal socket
    RDMAServerSocket(RDMAServerSock), //
    RDMADataSocket(RDMADataSock), //
    RDMAContext,
}

impl fmt::Debug for SockInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => write!(f, "SockInfo::File"),
            Self::Socket => write!(f, "SockInfo::Socket"),
            Self::RDMAServerSocket(_) => write!(f, "SockInfo::RDMAServerSocket"),
            Self::RDMADataSocket(_) => write!(f, "SockInfo::RDMADataSocket"),
            Self::RDMAContext => write!(f, "SockInfo::RDMAContext"),
        }
    }
}

impl SockInfo {
    pub fn Notify(&self, fd: i32, eventmask: EventMask) {
        match self {
            Self::File => {
                FdNotify(fd, eventmask)
            }
            Self::Socket => {
                FdNotify(fd, eventmask)
            }
            Self::RDMAServerSocket(ref sock) => {
                sock.Notify(eventmask)
            }
            Self::RDMADataSocket(ref sock) => {
                sock.Notify(eventmask)
            }
            Self::RDMAContext => {
                RDMA.PollCompletion().expect("RDMA.PollCompletion fail");
            }
        }
    }
}

