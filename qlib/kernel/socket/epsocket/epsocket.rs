use super::super::super::super::common::*;
use super::super::super::super::linux::socket::*;
use super::super::super::super::linux::time::*;
use super::super::super::super::linux_def::*;
use super::super::super::task::*;
use super::super::super::tcpip::tcpip::*;
use super::super::unix::transport::unix::*;

use super::super::hostinet::socket::DUMMY_HOST_SOCKET;

pub fn Ioctl(task: &Task, ep: &BoundEndpoint, _fd: i32, request: u64, val: u64) -> Result<()> {
    let flags = request as i32;

    match flags as u64 {
        /*LibcConst::SIOCGIFFLAGS
        | LibcConst::SIOCGIFBRDADDR
        | LibcConst::SIOCGIFDSTADDR
        | LibcConst::SIOCGIFHWADDR
        | LibcConst::SIOCGIFINDEX
        | LibcConst::SIOCGIFMAP
        | LibcConst::SIOCGIFMETRIC
        | LibcConst::SIOCGIFMTU
        | LibcConst::SIOCGIFNAME
        | LibcConst::SIOCGIFNETMASK
        | LibcConst::SIOCGIFTXQLEN => {
            let addr = val;
            let bep = ep.BaseEndpoint();
            bep.HostIoctlIFReq(task, request, addr)?;

            return Ok(());
        }*/
        LibcConst::SIOCGIFINDEX => {
            return Err(Error::SysError(SysErr::ENODEV))
        }
        LibcConst::SIOCGIFCONF => {
            let addr = val;
            DUMMY_HOST_SOCKET.HostIoctlIFConf(task, request, addr)?;

            return Ok(());
        }
        LibcConst::TIOCINQ => {
            let mut v =  ep.BaseEndpoint().GetSockRecvQueueSize()?;

            if v > i32::MAX {
                v = i32::MAX;
            }
            let addr = val;
            task.CopyOutObj(&v, addr)?;
            return Ok(());
        }
        LibcConst::TIOCOUTQ => {
            let mut v =  ep.BaseEndpoint().GetSockSendQueueSize()?;

            if v > i32::MAX {
                v = i32::MAX;
            }
            let addr = val;
            task.CopyOutObj(&v, addr)?;
            return Ok(());
        }
        SIOCGIFMEM | SIOCGIFPFLAGS | SIOCGMIIPHY | SIOCGMIIREG => {
            info!("Ioctl: not implment flags is {}", flags);
        }
        _ => (),
    }

    return Err(Error::SysError(SysErr::ENOTTY));
}

#[derive(Debug, Copy, Clone)]
pub enum SockOptResult {
    I32(i32),
    Ucred(Ucred),
    Linger(Linger),
    Timeval(Timeval),
}

impl SockOptResult {
    pub fn Marsh(&self, buf: &mut [u8]) -> Result<usize> {
        match self {
            SockOptResult::I32(v) => {
                if buf.len() < SIZEOF_I32 {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                let t: *mut i32 = &mut buf[0] as *mut u8 as u64 as *mut i32;
                unsafe {
                    *t = *v;
                }
                return Ok(core::mem::size_of::<i32>());
            }
            SockOptResult::Ucred(v) => {
                if buf.len() < SIZEOF_UCRED {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                let t: *mut Ucred = &mut buf[0] as *mut u8 as u64 as *mut Ucred;

                unsafe {
                    *t = *v;
                }
                return Ok(core::mem::size_of::<Ucred>());
            }
            SockOptResult::Linger(v) => {
                if buf.len() < SIZEOF_LINGER {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                let t: *mut Linger = &mut buf[0] as *mut u8 as u64 as *mut Linger;

                unsafe {
                    *t = *v;
                }
                return Ok(core::mem::size_of::<Linger>());
            }
            SockOptResult::Timeval(v) => {
                if buf.len() < SIZE_OF_TIMEVAL {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                let t: *mut Timeval = &mut buf[0] as *mut u8 as u64 as *mut Timeval;

                unsafe {
                    *t = *v;
                }
                return Ok(core::mem::size_of::<Timeval>());
            }
        }
    }
}

pub const SIZEOF_I32: usize = 4;
pub const SIZEOF_SOCKADDR_INET4: usize = 0x10;
pub const SIZEOF_SOCKADDR_INET6: usize = 0x1c;
pub const SIZEOF_SOCKADDR_ANY: usize = 0x70;
pub const SIZEOF_SOCKADDR_UNIX: usize = 0x6e;
pub const SIZEOF_SOCKADDR_LINKLAYER: usize = 0x14;
pub const SIZEOF_SOCKADDR_NETLINK: usize = 0xc;
pub const SIZEOF_LINGER: usize = 0x8;
pub const SIZEOF_TIMEVAL: usize = 0x10;
pub const SIZEOF_IPMREQ: usize = 0x8;
pub const SIZEOF_IPMREQN: usize = 0xc;
pub const SIZEOF_IPV6_MREQ: usize = 0x14;
pub const SIZEOF_MSGHDR: usize = 0x38;
pub const SIZEOF_CMSGHDR: usize = 0x10;
pub const SIZEOF_INET4_PKTINFO: usize = 0xc;
pub const SIZEOF_INET6_PKTINFO: usize = 0x14;
pub const SIZEOF_IPV6_MTUINFO: usize = 0x20;
pub const SIZEOF_ICMPV6_FILTER: usize = 0x20;
pub const SIZEOF_UCRED: usize = 0xc;
pub const SIZEOF_TCPINFO: usize = 0x68;

#[repr(C)]
#[derive(Default, Copy, Clone, Debug)]
pub struct Ucred {
    pub Pid: i32,
    pub Uid: u32,
    pub Gid: u32,
}

// Linger is struct linger, from include/linux/socket.h.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Linger {
    pub OnOff: i32,
    pub Linger: i32,
}

pub fn ConvertShutdown(how: i32) -> Result<ShutdownFlags> {
    let f = match how {
        SHUT_RD => SHUTDOWN_READ,
        SHUT_WR => SHUTDOWN_WRITE,
        SHUT_RDWR => SHUTDOWN_READ | SHUTDOWN_WRITE,
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    };

    return Ok(f);
}

// The minimum size of the send/receive buffers.
pub const MINIMUM_BUFFER_SIZE : usize = 4 << 10; // 4 KiB (match default in linux)

// The default size of the send/receive buffers.
pub const DEFAULT_BUFFER_SIZE : usize = 208 << 10; // 208 KiB  (default in linux for net.core.wmem_default)

// The maximum permitted size for the send/receive buffers.
pub const MAX_BUFFER_SIZE : usize = 4 << 20; // 4 MiB 4 MiB (default in linux for net.core.wmem_max)

pub struct SendBufferSizeOption {
    pub Min: usize,
    pub Default: usize,
    pub Max: usize
}

// getSendBufferLimits implements tcpip.GetSendBufferLimits.
//
// AF_UNIX sockets buffer sizes are not tied to the networking stack/namespace
// in linux but are bound by net.core.(wmem|rmem)_(max|default).
//
// In gVisor net.core sysctls today are not exposed or if exposed are currently
// tied to the networking stack in use. This makes it complicated for AF_UNIX
// when we are in a new namespace w/ no networking stack. As a result for now we
// define default/max values here in the unix socket implementation itself.
pub fn GetSendBufferLimits() -> SendBufferSizeOption {
    return SendBufferSizeOption {
        Min: MINIMUM_BUFFER_SIZE,
        Default: DEFAULT_BUFFER_SIZE,
        Max: MAX_BUFFER_SIZE
    }
}

// getReceiveBufferLimits implements tcpip.GetReceiveBufferLimits.
//
// We define min, max and default values for unix socket implementation. Unix
// sockets do not use receive buffer.
pub fn GetReceiveBufferLimits() -> SendBufferSizeOption {
    return SendBufferSizeOption {
        Min: MINIMUM_BUFFER_SIZE,
        Default: DEFAULT_BUFFER_SIZE,
        Max: MAX_BUFFER_SIZE
    }
}

pub fn SendBufferLimits() -> (usize, usize) {
    let opt = GetSendBufferLimits();
    return (opt.Min, opt.Max)
}

pub fn ReceiveBufferLimits() -> (usize, usize) {
    let opt = GetReceiveBufferLimits();
    return (opt.Min, opt.Max)
}

pub fn clampBufSize(newsz: usize, min: usize, max: usize, ignoreMax: bool) -> usize {
    // packetOverheadFactor is used to multiply the value provided by the user on
    // a setsockopt(2) for setting the send/receive buffer sizes sockets.
    const PACKET_OVERHEAD_FACTOR : usize = 2;

    let mut newsz = newsz;
    if !ignoreMax && newsz > max {
        newsz = max;
    }

    if newsz < i32::MAX as usize / PACKET_OVERHEAD_FACTOR {
        newsz *= PACKET_OVERHEAD_FACTOR;
        if newsz < min {
            newsz = min
        }
    } else {
        newsz = i32::MAX as usize
    }

    return newsz
}
