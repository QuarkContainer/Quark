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

use super::super::super::mutex::*;
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::collections::vec_deque::VecDeque;
use core::ops::Deref;

use super::super::super::common::*;
use super::super::super::linux::socket::*;
use super::super::tcpip::tcpip::*;

pub trait SocketOptionsHandler {
    // OnReuseAddressSet is invoked when SO_REUSEADDR is set for an endpoint.
    fn OnReuseAddressSet(&self, _v: bool) {}

    // OnReusePortSet is invoked when SO_REUSEPORT is set for an endpoint.
    fn OnReusePortSet(&self, _v: bool) {}

    // OnKeepAliveSet is invoked when SO_KEEPALIVE is set for an endpoint.
    fn OnKeepAliveSet(&self, _v: bool) {}

    // OnDelayOptionSet is invoked when TCP_NODELAY is set for an endpoint.
    // Note that v will be the inverse of TCP_NODELAY option.
    fn OnDelayOptionSet(&self, _v: bool) {}

    // OnCorkOptionSet is invoked when TCP_CORK is set for an endpoint.
    fn OnCorkOptionSet(&self, _v: bool) {}

    // LastError is invoked when SO_ERROR is read for an endpoint.
    fn LastError(&self) -> Option<Error> {
        return None;
    }

    // UpdateLastError updates the endpoint specific last error field.
    fn UpdateLastError(&self, _err : Error) {}

    // HasNIC is invoked to check if the NIC is valid for SO_BINDTODEVICE.
    fn HasNIC(&self, _err : Error) -> bool {
        return false;
    }

    // OnSetSendBufferSize is invoked when the send buffer size for an endpoint is
    // changed. The handler is invoked with the new value for the socket send
    // buffer size. It also returns the newly set value.
    fn OnSetSendBufferSize(&self, v: i64) -> i64 {
        return v;
    }

    // OnSetReceiveBufferSize is invoked by SO_RCVBUF and SO_RCVBUFFORCE. The
    // handler can optionally return a callback which will be called after
    // the buffer size is updated to newSz.
    fn OnSetReceiveBufferSize(&self, v: i64, _oldSz: i64) -> i64 {
        return v;
    }

    // WakeupWriters is invoked when the send buffer size for an endpoint is
    // changed. The handler notifies the writers if the send buffer size is
    // increased with setsockopt(2) for TCP endpoints.
    fn WakeupWriters(&self) {}
}

pub struct SocketOptions (Arc<QMutex<SocketOptionsInternal>>);

impl Deref for SocketOptions {
    type Target = Arc<QMutex<SocketOptionsInternal>>;

    fn deref(&self) -> &Arc<QMutex<SocketOptionsInternal>> {
        &self.0
    }
}

impl SocketOptionsHandler for SocketOptions {}

impl SocketOptions {
    pub fn Boolval(v: bool) -> u32 {
        if v {
            1
        } else {
            0
        }
    }

    // SetLastError sets the last error for a socket.
    pub fn SocketOptions(&self, err: Error) {
        self.UpdateLastError(err)
    }

    // GetBroadcast gets value for SO_BROADCAST option.
    pub fn GetBroadcast(&self) -> bool {
        return self.lock().broadcastEnabled != 0;
    }

    // SetBroadcast sets value for SO_BROADCAST option.
    pub fn SetBroadcast(&self, v: bool) {
        self.lock().broadcastEnabled = Self::Boolval(v);
    }

    // SetPassCred sets value for SO_PASSCRED option.
    pub fn GetPassCred(&self) -> bool {
        return self.lock().passCredEnabled != 0;
    }

    // SetPassCred sets value for SO_PASSCRED option.
    pub fn SetPassCred(&self, v: bool) {
        self.lock().passCredEnabled = Self::Boolval(v);
    }

    // GetNoChecksum gets value for SO_NO_CHECK option.
    pub fn GetNoChecksum(&self) -> bool {
        return self.lock().noChecksumEnabled != 0;
    }

    // SetNoChecksum sets value for SO_NO_CHECK option.
    pub fn SetNoChecksum(&self, v: bool) {
        self.lock().noChecksumEnabled = Self::Boolval(v);
    }

    // GetReuseAddress gets value for SO_REUSEADDR option.
    pub fn GetReuseAddress(&self) -> bool {
        return self.lock().reuseAddressEnabled != 0;
    }

    // SetReuseAddress sets value for SO_REUSEADDR option.
    pub fn SetReuseAddress(&self, v: bool) {
        self.lock().reuseAddressEnabled = Self::Boolval(v);
        self.OnReuseAddressSet(v);
    }

    // GetReusePort gets value for SO_REUSEPORT option.
    pub fn GetReusePort(&self) -> bool {
        return self.lock().reusePortEnabled != 0;
    }

    // SetReusePort sets value for SO_REUSEPORT option.
    pub fn SetReusePort(&self, v: bool) {
        self.lock().reusePortEnabled = Self::Boolval(v);
        self.OnReusePortSet(v);
    }

    // GetKeepAlive gets value for SO_KEEPALIVE option.
    pub fn GetKeepAlive(&self) -> bool {
        return self.lock().keepAliveEnabled != 0;
    }

    // SetKeepAlive sets value for SO_KEEPALIVE option.
    pub fn SetKeepAlive(&self, v: bool) {
        self.lock().keepAliveEnabled = Self::Boolval(v);
        self.OnKeepAliveSet(v);
    }

    // GetMulticastLoop gets value for IP_MULTICAST_LOOP option.
    pub fn GetMulticastLoop(&self) -> bool {
        return self.lock().multicastLoopEnabled != 0;
    }

    // SetMulticastLoop sets value for IP_MULTICAST_LOOP option.
    pub fn SetMulticastLoop(&self, v: bool) {
        self.lock().multicastLoopEnabled = Self::Boolval(v);
    }

    // GetReceiveTOS gets value for IP_RECVTOS option.
    pub fn GetReceiveTOS(&self) -> bool {
        return self.lock().receiveTOSEnabled != 0;
    }

    // SetReceiveTOS sets value for IP_RECVTOS option.
    pub fn SetReceiveTOS(&self, v: bool) {
        self.lock().receiveTOSEnabled = Self::Boolval(v);
    }

}

pub struct SocketOptionsInternal {
    // broadcastEnabled determines whether datagram sockets are allowed to
    // send packets to a broadcast address.
    pub broadcastEnabled : u32,

    // passCredEnabled determines whether SCM_CREDENTIALS socket control
    // messages are enabled.
    pub passCredEnabled : u32,

    // noChecksumEnabled determines whether UDP checksum is disabled while
    // transmitting for this socket.
    pub noChecksumEnabled : u32,

    // reuseAddressEnabled determines whether Bind() should allow reuse of
    // local address.
    pub reuseAddressEnabled : u32,

    // reusePortEnabled determines whether to permit multiple sockets to be
    // bound to an identical socket address.
    pub reusePortEnabled : u32,

    // keepAliveEnabled determines whether TCP keepalive is enabled for this
    // socket.
    pub keepAliveEnabled : u32,

    // multicastLoopEnabled determines whether multicast packets sent over a
    // non-loopback interface will be looped back.
    pub multicastLoopEnabled : u32,

    // receiveTOSEnabled is used to specify if the TOS ancillary message is
    // passed with incoming packets.
    pub receiveTOSEnabled : u32,

    // receiveTTLEnabled is used to specify if the TTL ancillary message is passed
    // with incoming packets.
    pub receiveTTLEnabled : u32,

    // receiveHopLimitEnabled is used to specify if the HopLimit ancillary message
    // is passed with incoming packets.
    pub receiveHopLimitEnabled : u32,

    // receiveTClassEnabled is used to specify if the IPV6_TCLASS ancillary
    // message is passed with incoming packets.
    pub receiveTClassEnabled : u32,

    // receivePacketInfoEnabled is used to specify if more information is
    // provided with incoming IPv4 packets.
    pub receivePacketInfoEnabled : u32,

    // receivePacketInfoEnabled is used to specify if more information is
    // provided with incoming IPv6 packets.
    pub receiveIPv6PacketInfoEnabled : u32,

    // hdrIncludeEnabled is used to indicate for a raw endpoint that all packets
    // being written have an IP header and the endpoint should not attach an IP
    // header.
    pub hdrIncludedEnabled : u32,

    // v6OnlyEnabled is used to determine whether an IPv6 socket is to be
    // restricted to sending and receiving IPv6 packets only.
    pub v6OnlyEnabled : u32,

    // quickAckEnabled is used to represent the value of TCP_QUICKACK option.
    // It currently does not have any effect on the TCP endpoint.
    pub quickAckEnabled : u32,

    // delayOptionEnabled is used to specify if data should be sent out immediately
    // by the transport protocol. For TCP, it determines if the Nagle algorithm
    // is on or off.
    pub delayOptionEnabled : u32,

    // corkOptionEnabled is used to specify if data should be held until segments
    // are full by the TCP transport protocol.
    pub corkOptionEnabled : u32,

    // receiveOriginalDstAddress is used to specify if the original destination of
    // the incoming packet should be returned as an ancillary message.
    pub receiveOriginalDstAddress : u32,

    // ipv4RecvErrEnabled determines whether extended reliable error message
    // passing is enabled for IPv4.
    pub ipv4RecvErrEnabled : u32,

    // ipv6RecvErrEnabled determines whether extended reliable error message
    // passing is enabled for IPv6.
    pub ipv6RecvErrEnabled : u32,

    // errQueue is the per-socket error queue. It is protected by errQueueMu.
    pub errQueue  : VecDeque<SockError>,

    // bindToDevice determines the device to which the socket is bound.
    pub bindToDevice : i32,

    // getSendBufferLimits provides the handler to get the min, default and max
    // size for send buffer. It is initialized at the creation time and will not
    // change.
    //pub getSendBufferLimits GetSendBufferLimits `state:"manual"`

    // sendBufferSize determines the send buffer size for this socket.
    pub sendBufferSize : i64,

    // getReceiveBufferLimits provides the handler to get the min, default and
    // max size for receive buffer. It is initialized at the creation time and
    // will not change.
    //pub getReceiveBufferLimits GetReceiveBufferLimits `state:"manual"`

    // receiveBufferSize determines the receive buffer size for this socket.
    pub receiveBufferSize : i64,

    // linger determines the amount of time the socket should linger before
    // close. We currently implement this option for TCP socket only.
    pub linger : LingerOption,

    // rcvlowat specifies the minimum number of bytes which should be
    // received to indicate the socket as readable.
    pub rcvlowat : i32,
}

pub enum SockErrorCause {
    None,
}

pub struct SockError {
    // Err is the error caused by the errant packet.
    pub Err: Error,

    // Cause is the detailed cause of the error.
    pub cause: SockErrorCause,

    // Payload is the errant packet's payload.
    pub Payload: Vec<u8>,

    // Dst is the original destination address of the errant packet.
    pub Dst: FullAddr,

    // Offender is the original sender address of the errant packet.
    pub Offender: FullAddr,

    // NetProto is the network protocol being used to transmit the packet.
    pub NetProto: u32,
}
