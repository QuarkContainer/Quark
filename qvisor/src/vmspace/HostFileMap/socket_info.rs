use core::fmt;

use super::rdma_socket::*;
use super::rdma::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::kernel::guestfdnotifier::*;

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
    pub fn Notify(&self, eventmask: EventMask, waitinfo: FdWaitInfo) {
        match self {
            Self::File => {
                waitinfo.Notify(eventmask);
            }
            Self::Socket => {
                waitinfo.Notify(eventmask);
            }
            Self::RDMAServerSocket(ref sock) => {
                sock.Notify(eventmask, waitinfo)
            }
            Self::RDMADataSocket(ref sock) => {
                sock.Notify(eventmask, waitinfo)
            }
            Self::RDMAContext => {
                // RDMA.PollCompletion().expect("RDMA.PollCompletion fail");
            }
        }
    }
}

