use core::fmt;

use super::super::super::qlib::fileinfo::*;
// use super::super::super::qlib::kernel::guestfdnotifier::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::rdmasocket::*;

#[derive(Clone)]
pub enum SockInfo {
    File,                             // it is not socket
    Socket(SocketInfo),               // normal socket
    RDMAServerSocket(RDMAServerSock), //
    RDMADataSocket(RDMADataSock),     //
    RDMAContext,
}

impl fmt::Debug for SockInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => write!(f, "SockInfo::File"),
            Self::Socket(_) => write!(f, "SockInfo::Socket"),
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
            Self::Socket(_) => {
                waitinfo.Notify(eventmask);
            }
            Self::RDMAServerSocket(ref sock) => sock.Notify(eventmask, waitinfo),
            Self::RDMADataSocket(ref sock) => sock.Notify(eventmask, waitinfo),
            Self::RDMAContext => {
                //RDMA.PollCompletion().expect("RDMA.PollCompletion fail");
                //error!("RDMAContextEpoll");
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct SocketInfo {
    pub ipAddr: u32,
    pub port: u16,
}
