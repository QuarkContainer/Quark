use core::fmt;

use super::rdma_socket::*;

#[derive(Clone)]
pub enum SockInfo {
    File, // it is not socket
    Socket, // normal socket
    RDMAServerSocket(RDMAServerSock), //
    RDMADataSocket, //
}

impl fmt::Debug for SockInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => write!(f, "SockInfo::File"),
            Self::Socket => write!(f, "SockInfo::Socket"),
            Self::RDMAServerSocket(_) => write!(f, "SockInfo::RDMAServerSocket"),
            Self::RDMADataSocket => write!(f, "SockInfo::RDMADataSocket"),
        }
    }
}