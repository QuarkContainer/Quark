// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use core::slice;

use super::super::super::common::*;
use super::super::super::linux_def::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SockOpt {
    // ErrorOption is used in GetSockOpt to specify that the last error reported by
    // the endpoint should be cleared and returned.
    ErrorOption,

    // SendBufferSizeOption is used by SetSockOpt/GetSockOpt to specify the send
    // buffer size option.
    SendBufferSizeOption(i32),

    // ReceiveBufferSizeOption is used by SetSockOpt/GetSockOpt to specify the
    // receive buffer size option.
    ReceiveBufferSizeOption(i32),

    // SendQueueSizeOption is used in GetSockOpt to specify that the number of
    // unread bytes in the output buffer should be returned.
    SendQueueSizeOption(i32),

    // ReceiveQueueSizeOption is used in GetSockOpt to specify that the number of
    // unread bytes in the input buffer should be returned.
    ReceiveQueueSizeOption(i32),

    // V6OnlyOption is used by SetSockOpt/GetSockOpt to specify whether an IPv6
    // socket is to be restricted to sending and receiving IPv6 packets only.
    V6OnlyOption(i32),

    // DelayOption is used by SetSockOpt/GetSockOpt to specify if data should be
    // sent out immediately by the transport protocol. For TCP, it determines if the
    // Nagle algorithm is on or off.
    DelayOption(i32),

    // CorkOption is used by SetSockOpt/GetSockOpt to specify if data should be
    // held until segments are full by the TCP transport protocol.
    CorkOption(i32),

    // ReuseAddressOption is used by SetSockOpt/GetSockOpt to specify whether Bind()
    // should allow reuse of local address.
    ReuseAddressOption(i32),

    // ReusePortOption is used by SetSockOpt/GetSockOpt to permit multiple sockets
    // to be bound to an identical socket address.
    ReusePortOption(i32),

    // QuickAckOption is stubbed out in SetSockOpt/GetSockOpt.
    QuickAckOption(i32),

    // PasscredOption is used by SetSockOpt/GetSockOpt to specify whether
    // SCM_CREDENTIALS socket control messages are enabled.
    //
    // Only supported on Unix sockets.
    PasscredOption(i32),

    // KeepaliveEnabledOption is used by SetSockOpt/GetSockOpt to specify whether
    // TCP keepalive is enabled for this socket.
    KeepaliveEnabledOption(i32),

    // KeepaliveIdleOption is used by SetSockOpt/GetSockOpt to specify the time a
    // connection must remain idle before the first TCP keepalive packet is sent.
    // Once this time is reached, KeepaliveIntervalOption is used instead.
    KeepaliveIdleOption(i64),

    // KeepaliveIntervalOption is used by SetSockOpt/GetSockOpt to specify the
    // interval between sending TCP keepalive packets.
    KeepaliveIntervalOption(i64),

    // KeepaliveCountOption is used by SetSockOpt/GetSockOpt to specify the number
    // of un-ACKed TCP keepalives that will be sent before the connection is
    // closed.
    KeepaliveCountOption(i32),

    // MulticastTTLOption is used by SetSockOpt/GetSockOpt to control the default
    // TTL value for multicast messages. The default is 1.
    MulticastTTLOption(u8),

    // AddMembershipOption is used by SetSockOpt/GetSockOpt to join a multicast
    // group identified by the given multicast address, on the interface matching
    // the given interface address.
    //type AddMembershipOption MembershipOption

    // RemoveMembershipOption is used by SetSockOpt/GetSockOpt to leave a multicast
    // group identified by the given multicast address, on the interface matching
    // the given interface address.
    //type RemoveMembershipOption MembershipOption

    // OutOfBandInlineOption is used by SetSockOpt/GetSockOpt to specify whether
    // TCP out-of-band data is delivered along with the normal in-band data.
    OutOfBandInlineOption(i32),

    // BroadcastOption is used by SetSockOpt/GetSockOpt to specify whether
    // datagram sockets are allowed to send packets to a broadcast address.
    BroadcastOption(i32),
}

// NICID is a number that uniquely identifies a NIC.
pub type NICID = i32;

// ShutdownFlags represents flags that can be passed to the Shutdown() method
// of the Endpoint interface.
pub type ShutdownFlags = i32;

// Values of the flags that can be passed to the Shutdown() method. They can
// be OR'ed together.
pub const SHUTDOWN_READ: ShutdownFlags = 1;
pub const SHUTDOWN_WRITE: ShutdownFlags = 2;

// Address is a byte slice cast as a string that represents the address of a
// network node. Or, in the case of unix endpoints, it may represent a path.
pub type Address = String;

// AddressMask is a bitmask for an address.
pub type AddressMask = String;

#[derive(Default, Debug)]
pub struct FullAddrInternal {
    // NIC is the ID of the NIC this address refers to.
    //
    // This may not be used by all endpoint types.
    pub NIC: NICID,

    // Addr is the network address.
    pub Addr: Vec<u8>,

    // Port is the transport port.
    //
    // This may not be used by all endpoint types.
    pub Port: u16,
}

#[derive(Default, Clone, Debug)]
pub struct FullAddr(Arc<FullAddrInternal>);

impl Deref for FullAddr {
    type Target = Arc<FullAddrInternal>;

    fn deref(&self) -> &Arc<FullAddrInternal> {
        &self.0
    }
}

impl FullAddr {
    pub fn New(internal: FullAddrInternal) -> Self {
        return Self(Arc::new(internal));
    }

    pub fn NewWithAddr(addr: &[u8]) -> Self {
        let internal = FullAddrInternal {
            Addr: addr.to_vec(),
            ..Default::default()
        };

        return Self::New(internal);
    }
}

// GetAddress reads an sockaddr struct from the given address and converts it
// to the FullAddress format. It supports AF_UNIX, AF_INET and AF_INET6
// addresses.
pub fn GetAddr(sfamily: i16, addr: &[u8]) -> Result<SockAddr> {
    // Make sure we have at least 2 bytes for the address family.
    if addr.len() < 2 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let family = unsafe { &*(&addr[0] as *const u8 as *const i16) };

    if *family != sfamily {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // Get the rest of the fields based on the address family.
    match sfamily as i32 {
        AFType::AF_UNIX => {
            let mut path = &addr[2..];
            if path.len() > UNIX_PATH_MAX {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            // Drop the terminating NUL (if one exists) and everything after
            // it for filesystem (non-abstract) addresses.
            if path.len() > 0 && path[0] != 0 {
                let mut idx = 0;
                for i in 1..path.len() {
                    if path[i] == 0 {
                        idx = i;
                        break;
                    }
                }

                if idx != 0 {
                    path = &path[..idx];
                }
            }

            let unix = SockAddrUnix {
                Family: sfamily as u16,
                Path: match String::from_utf8(path.to_vec()) {
                    Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
                    Ok(s) => s,
                },
            };

            return Ok(SockAddr::Unix(unix));
        }
        AFType::AF_INET => {
            if addr.len() < SOCK_ADDR_INET_SIZE {
                return Err(Error::SysError(SysErr::EFAULT));
            }

            let a = unsafe { &*((&addr[0]) as *const _ as *const SockAddrInet) };

            return Ok(SockAddr::Inet(*a));
        }
        AFType::AF_INET6 => {
            if addr.len() < SOCK_ADDR_INET6_SIZE {
                return Err(Error::SysError(SysErr::EFAULT));
            }

            let a = unsafe { &*((&addr[0]) as *const _ as *const SocketAddrInet6) };

            return Ok(SockAddr::Inet6(*a));
        }
        AFType::AF_NETLINK => {
            if addr.len() < SockAddrNetlink::SOCK_ADDR_NETLINK_SIZE {
                return Err(Error::SysError(SysErr::EFAULT));
            }

            let a = unsafe { &*((&addr[0]) as *const _ as *const SockAddrNetlink) };

            return Ok(SockAddr::Netlink(*a));
        }
        _ => (),
    }

    return Err(Error::SysError(SysErr::EAFNOSUPPORT));
}

// isLinkLocal determines if the given IPv6 address is link-local. This is the
// case when it has the fe80::/10 prefix. This check is used to determine when
// the NICID is relevant for a given IPv6 address.
pub fn IsLinkLocal(addr: &[u8]) -> bool {
    return addr.len() >= 2 && addr[0] == 0xfe && addr[1] & 0xc0 == 0x0;
}

pub const SOCK_ADDR_INET_SIZE: usize = 16;
pub const SOCK_ADDR_INET6_SIZE: usize = 28;

// ntohs converts a 16-bit number from network byte order to host byte order. It
// assumes that the host is little endian.
pub fn ntohs(v: u16) -> u16 {
    return v << 8 | v >> 8;
}

// htons converts a 16-bit number from host byte order to network byte order. It
// assumes that the host is little endian.
pub fn htons(v: u16) -> u16 {
    return ntohs(v);
}

#[derive(Clone, Debug)]
pub enum SockAddr {
    Inet(SockAddrInet),
    Inet6(SocketAddrInet6),
    Unix(SockAddrUnix),
    Netlink(SockAddrNetlink),
    None,
}

impl SockAddr {
    pub fn Len(&self) -> usize {
        match self {
            SockAddr::Inet(addr) => addr.Len(),
            SockAddr::Inet6(addr) => addr.Len(),
            SockAddr::Unix(addr) => addr.Len(),
            SockAddr::Netlink(addr) => addr.Len(),
            SockAddr::None => 0,
        }
    }

    pub fn ToVec(&self) -> Result<Vec<u8>> {
        let len = self.Len();
        let mut buf = Vec::with_capacity(len);
        buf.resize(len, 0);
        self.Marsh(&mut buf, len)?;
        return Ok(buf);
    }

    pub fn Marsh(&self, buf: &mut [u8], len: usize) -> Result<()> {
        if buf.len() < len {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        match self {
            SockAddr::Inet(addr) => {
                let ptr = addr as *const _ as u64 as *const u8;
                let slice = unsafe { slice::from_raw_parts(ptr, len) };

                for i in 0..len {
                    buf[i] = slice[i];
                }
                return Ok(());
            }
            SockAddr::Inet6(addr) => {
                let ptr = addr as *const _ as u64 as *const u8;
                let slice = unsafe { slice::from_raw_parts(ptr, len) };

                for i in 0..len {
                    buf[i] = slice[i];
                }
                return Ok(());
            }
            SockAddr::Unix(addr) => {
                let native = addr.ToNative();
                let ptr = &native as *const _ as u64 as *const u8;
                let slice = unsafe { slice::from_raw_parts(ptr, len) };

                for i in 0..len {
                    buf[i] = slice[i];
                }
                return Ok(());
            }
            SockAddr::Netlink(addr) => {
                let ptr = addr as *const _ as u64 as *const u8;
                let slice = unsafe { slice::from_raw_parts(ptr, len) };

                for i in 0..len {
                    buf[i] = slice[i];
                }
                return Ok(());
            }
            SockAddr::None => return Err(Error::SysError(SysErr::EINVAL)),
        }
    }
}

// SockAddrInet is struct sockaddr_in, from uapi/linux/in.h.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SockAddrInet {
    pub Family: u16,
    pub Port: u16,
    pub Addr: [u8; 4],
    pub Zero: [u8; 8], // pad to sizeof(struct sockaddr).
}

impl SockAddrInet {
    pub fn Len(&self) -> usize {
        return core::mem::size_of::<SockAddrInet>();
    }
}

// SockAddrInet6 is struct sockaddr_in6, from uapi/linux/in6.h.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SocketAddrInet6 {
    pub Family: u16,
    pub Port: u16,
    pub Flowinfo: u32,
    pub Addr: [u8; 16],
    pub Scope_id: u32,
}

impl SocketAddrInet6 {
    pub fn Len(&self) -> usize {
        return core::mem::size_of::<SocketAddrInet6>();
    }
}

// SockAddrUnix is struct sockaddr_un, from uapi/linux/un.h.
#[repr(C)]
#[derive(Clone)]
pub struct SockAddrUnixNative {
    pub Family: u16,
    pub Path: [u8; UNIX_PATH_MAX],
}

#[derive(Clone, Debug)]
pub struct SockAddrUnix {
    pub Family: u16,
    pub Path: String,
}

impl Default for SockAddrUnix {
    fn default() -> Self {
        return Self {
            Family: 0,
            Path: "".to_string(),
        };
    }
}

impl SockAddrUnix {
    pub fn New(str: &str) -> Self {
        let path = if str.len() > UNIX_PATH_MAX {
            str[0..UNIX_PATH_MAX].to_string()
        } else {
            str.to_string()
        };

        // todo: what if the size is larger than max len
        let ret = Self {
            Family: AFType::AF_UNIX as u16,
            Path: path,
        };

        return ret;
    }

    pub fn Len(&self) -> usize {
        // Linux returns the used length of the address struct (including the
        // null terminator) for filesystem paths. The Family field is 2 bytes.
        // It is sometimes allowed to exclude the null terminator if the
        // address length is the max. Abstract and empty paths always return
        // the full exact length.
        let l = self.Path.len();
        if l == 0 || self.Path.as_bytes()[0] == 0 || l == UNIX_PATH_MAX {
            return l + 2;
        }

        return l + 3;
    }

    pub fn ToNative(&self) -> SockAddrUnixNative {
        let mut ret = SockAddrUnixNative {
            Family: AFType::AF_UNIX as u16,
            Path: [0; UNIX_PATH_MAX],
        };

        let arr = self.Path.as_bytes();

        for i in 0..arr.len() {
            ret.Path[i] = arr[i];
        }

        return ret;
    }
}

// SockAddrNetlink is struct sockaddr_nl, from uapi/linux/netlink.h.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SockAddrNetlink {
    pub Family: u16,
    pub Padding: u16,
    pub PortID: u32,
    pub Groups: u32,
}

impl SockAddrNetlink {
    pub const SOCK_ADDR_NETLINK_SIZE: usize = 12;

    pub fn Len(&self) -> usize {
        return Self::SOCK_ADDR_NETLINK_SIZE;
    }
}
