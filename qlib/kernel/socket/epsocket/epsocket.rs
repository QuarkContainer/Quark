use super::super::super::super::common::*;
use super::super::super::super::linux::socket::*;
use super::super::super::super::linux::time::*;
use super::super::super::super::linux_def::*;
use super::super::super::fs::file::*;
use super::super::super::task::*;
use super::super::super::tcpip::tcpip::*;
use super::super::unix::transport::unix::*;

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
        }
        LibcConst::SIOCGIFCONF => {
            let addr = val;
            let bep = ep.BaseEndpoint();
            bep.HostIoctlIFConf(task, request, addr)?;

            return Ok(());
        }*/
        LibcConst::TIOCINQ => {
            let mut v = SockOpt::ReceiveQueueSizeOption(0);
            ep.GetSockOpt(&mut v)?;

            /*if v > core::I32::Max {
                v = core::I32::Max;
            }*/
            let addr = val;
            if let SockOpt::ReceiveQueueSizeOption(res) = v {
                //*task.GetTypeMut(addr)? = res;
                task.CopyOutObj(&res, addr)?;
            }

            return Ok(());
        }
        LibcConst::TIOCOUTQ => {
            let mut v = SockOpt::SendQueueSizeOption(0);
            ep.GetSockOpt(&mut v)?;

            /*if v > core::I32::Max {
                v = core::I32::Max;
            }*/
            let addr = val;
            if let SockOpt::SendQueueSizeOption(res) = v {
                //*task.GetTypeMut(addr)? = res;
                task.CopyOutObj(&res, addr)?;
            }

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

// getSockOptSocket implements GetSockOpt when level is SOL_SOCKET.
pub fn GetSockOptSocket(
    task: &Task,
    s: &FileOperations,
    ep: &BoundEndpoint,
    family: i32,
    skType: i32,
    name: i32,
    outlen: usize,
) -> Result<SockOptResult> {
    match name as u64 {
        LibcConst::SO_TYPE => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            return Ok(SockOptResult::I32(skType));
        }
        LibcConst::SO_ERROR => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            // Get the last error and convert it.
            match ep.GetSockOpt(&mut SockOpt::ErrorOption) {
                Ok(()) => return Ok(SockOptResult::I32(0)),
                Err(Error::SysError(syserr)) => return Ok(SockOptResult::I32(syserr)),
                Err(e) => {
                    panic!("GetSockOptSocket::ErrorOption get unpexected error {:?}", e)
                }
            }
        }
        LibcConst::SO_PEERCRED => {
            if family != AFType::AF_UNIX || outlen < SIZEOF_UCRED {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let tcred = task.Creds();

            let pid = task.Thread().ThreadGroup().ID();
            let userns = tcred.lock().UserNamespace.clone();
            let uid = tcred.lock().EffectiveKUID.In(&userns).OrOverflow();
            let gid = tcred.lock().EffectiveKGID.In(&userns).OrOverflow();

            let ucred = Ucred {
                Pid: pid,
                Uid: uid.0,
                Gid: gid.0,
            };

            return Ok(SockOptResult::Ucred(ucred));
        }
        LibcConst::SO_PASSCRED => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut opt = SockOpt::PasscredOption(0);
            ep.GetSockOpt(&mut opt)?;

            match opt {
                SockOpt::PasscredOption(v) => return Ok(SockOptResult::I32(v)),
                _ => (),
            }
        }
        LibcConst::SO_SNDBUF => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut opt = SockOpt::SendBufferSizeOption(0);
            ep.GetSockOpt(&mut opt)?;

            match opt {
                SockOpt::SendBufferSizeOption(v) => {
                    return Ok(SockOptResult::I32(v))
                },
                _ => (),
            }
        }
        LibcConst::SO_RCVBUF => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut opt = SockOpt::ReceiveBufferSizeOption(0);
            ep.GetSockOpt(&mut opt)?;

            match opt {
                SockOpt::ReceiveBufferSizeOption(v) => return Ok(SockOptResult::I32(v)),
                _ => (),
            }
        }
        LibcConst::SO_REUSEADDR => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut opt = SockOpt::ReusePortOption(0);
            ep.GetSockOpt(&mut opt)?;

            match opt {
                SockOpt::ReusePortOption(v) => return Ok(SockOptResult::I32(v)),
                _ => (),
            }
        }
        LibcConst::SO_BROADCAST => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut opt = SockOpt::BroadcastOption(0);
            ep.GetSockOpt(&mut opt)?;

            match opt {
                SockOpt::BroadcastOption(v) => return Ok(SockOptResult::I32(v)),
                _ => (),
            }
        }
        LibcConst::SO_KEEPALIVE => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut opt = SockOpt::KeepaliveEnabledOption(0);
            ep.GetSockOpt(&mut opt)?;

            match opt {
                SockOpt::KeepaliveEnabledOption(v) => return Ok(SockOptResult::I32(v)),
                _ => (),
            }
        }
        LibcConst::SO_LINGER => {
            if outlen < SIZEOF_LINGER {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            return Ok(SockOptResult::Linger(Linger::default()));
        }
        LibcConst::SO_SNDTIMEO => {
            if outlen < SIZE_OF_TIMEVAL {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let tv = Timeval::FromNs(s.SendTimeout());
            return Ok(SockOptResult::Timeval(tv));
        }
        LibcConst::SO_RCVTIMEO => {
            if outlen < SIZE_OF_TIMEVAL {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let tv = Timeval::FromNs(s.RecvTimeout());
            return Ok(SockOptResult::Timeval(tv));
        }
        LibcConst::SO_OOBINLINE => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            let mut opt = SockOpt::OutOfBandInlineOption(0);
            ep.GetSockOpt(&mut opt)?;

            match opt {
                SockOpt::OutOfBandInlineOption(v) => return Ok(SockOptResult::I32(v)),
                _ => (),
            }
        }
        LibcConst::SO_ACCEPTCONN => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            // This flag is only viable for TCP endpoints
            return Ok(SockOptResult::I32(0));
        }
        LibcConst::SO_DOMAIN => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            return Ok(SockOptResult::I32(AFType::AF_UNIX));
        }
        LibcConst::SO_PROTOCOL => {
            if outlen < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            // there is only one supported protocol for UNIX socket
            return Ok(SockOptResult::I32(0));
        }
        _ => (),
    }

    return Err(Error::SysError(SysErr::ENOPROTOOPT));
}

pub fn GetSockOpt(
    task: &Task,
    s: &FileOperations,
    ep: &BoundEndpoint,
    family: i32,
    skType: i32,
    level: i32,
    name: i32,
    outlen: usize,
) -> Result<SockOptResult> {
    match level {
        SOL_SOCKET => return GetSockOptSocket(task, s, ep, family, skType, name, outlen),
        SOL_TCP | SOL_IPV6 | SOL_IP | SOL_UDP => return Err(Error::SysError(SysErr::ENOPROTOOPT)),
        _ => return Err(Error::SysError(SysErr::ENOPROTOOPT)),
    }
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

// setSockOptSocket implements SetSockOpt when level is SOL_SOCKET.
pub fn SetSockOptSocket(
    _task: &Task,
    s: &FileOperations,
    ep: &BoundEndpoint,
    name: i32,
    optVal: &[u8],
) -> Result<()> {
    match name {
        SO_SNDBUF => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };
            let (min, max) = SendBufferLimits();
            let clamped = clampBufSize(val as usize, min, max, false) as i32;
            return ep.SetSockOpt(&SockOpt::SendBufferSizeOption(clamped));
        }
        SO_RCVBUF => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };
            let (min, max) = ReceiveBufferLimits();
            let clamped = clampBufSize(val as usize, min, max, false) as i32;

            return ep.SetSockOpt(&SockOpt::ReceiveBufferSizeOption(clamped));
        }
        SO_REUSEADDR => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };

            return ep.SetSockOpt(&SockOpt::ReuseAddressOption(val));
        }
        SO_REUSEPORT => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };

            return ep.SetSockOpt(&SockOpt::ReusePortOption(val));
        }
        SO_BROADCAST => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };

            return ep.SetSockOpt(&SockOpt::BroadcastOption(val));
        }
        SO_PASSCRED => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };

            return ep.SetSockOpt(&SockOpt::PasscredOption(val));
        }
        SO_KEEPALIVE => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };

            return ep.SetSockOpt(&SockOpt::KeepaliveEnabledOption(val));
        }
        SO_SNDTIMEO => {
            if optVal.len() < SIZE_OF_TIMEVAL {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= core::mem::size_of::<Timeval>());
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const Timeval) };

            if val.Usec < 0 || val.Usec >= SECOND / MICROSECOND {
                return Err(Error::SysError(SysErr::EDOM));
            }

            s.SetSendTimeout(val.ToDuration());
            return Ok(());
        }
        SO_RCVTIMEO => {
            if optVal.len() < SIZE_OF_TIMEVAL {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= core::mem::size_of::<Timeval>());
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const Timeval) };

            if val.Usec < 0 || val.Usec >= SECOND / MICROSECOND {
                return Err(Error::SysError(SysErr::EDOM));
            }

            s.SetRecvTimeout(val.ToDuration());
            return Ok(());
        }
        SO_OOBINLINE => {
            if optVal.len() < SIZEOF_I32 {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= 4);
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const i32) };

            if val == 0 {
                info!("setSockOptSocket::SO_RCVTIMEO unimplement");
            }

            return ep.SetSockOpt(&SockOpt::OutOfBandInlineOption(val));
        }
        SO_LINGER => {
            if optVal.len() < SIZEOF_LINGER {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            assert!(optVal.len() >= core::mem::size_of::<Linger>());
            let val = unsafe { *(&optVal[0] as *const _ as u64 as *const Linger) };

            if val != Linger::default() {
                info!("setSockOptSocket::SO_LINGER unimplement");
            }

            return Ok(());
        }
        _ => {}
    }

    return Err(Error::SysError(SysErr::ENOPROTOOPT));
}

// SetSockOpt can be used to implement the linux syscall setsockopt(2) for
// sockets backed by a commonEndpoint.
pub fn SetSockOpt(
    task: &Task,
    s: &FileOperations,
    ep: &BoundEndpoint,
    level: i32,
    name: i32,
    optVal: &[u8],
) -> Result<()> {
    match level {
        SOL_SOCKET => return SetSockOptSocket(task, s, ep, name, optVal),
        SOL_TCP | SOL_IPV6 | SOL_IP | SOL_UDP => return Err(Error::SysError(SysErr::ENOPROTOOPT)),
        _ => return Err(Error::SysError(SysErr::ENOPROTOOPT)),
    }
}
