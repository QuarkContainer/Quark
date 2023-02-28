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
use alloc::collections::vec_deque::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use super::super::super::common::*;
use super::super::super::linux::socket::*;
use super::super::tcpip::tcpip::*;

// The minimum size of the send/receive buffers.
pub const MINIMUM_BUFFER_SIZE: usize = 4 << 10; // 4 KiB (match default in linux)

// The default size of the send/receive buffers.
pub const DEFAULT_BUFFER_SIZE: usize = 208 << 10; // 208 KiB  (default in linux for net.core.wmem_default)

// The maximum permitted size for the send/receive buffers.
pub const MAX_BUFFER_SIZE: usize = 4 << 20; // 4 MiB 4 MiB (default in linux for net.core.wmem_max)

#[derive(Default, Clone, Copy)]
pub struct BufferSizeOption {
    pub Min: usize,
    pub Default: usize,
    pub Max: usize,
}

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
    fn UpdateLastError(&self, _err: Error) {}

    // HasNIC is invoked to check if the NIC is valid for SO_BINDTODEVICE.
    fn HasNIC(&self, _v: i32) -> bool {
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

#[derive(Default, Clone)]
pub struct SocketOptions(Arc<QMutex<SocketOptionsInternal>>);

impl Deref for SocketOptions {
    type Target = Arc<QMutex<SocketOptionsInternal>>;

    fn deref(&self) -> &Arc<QMutex<SocketOptionsInternal>> {
        &self.0
    }
}

impl SocketOptionsHandler for SocketOptions {}

impl SocketOptions {
    pub fn InitLimit(
        &self,
        SendBufferLimits: BufferSizeOption,
        ReceiveBufferLimits: BufferSizeOption,
    ) {
        let mut intern = self.lock();
        intern.SendBufferLimits = SendBufferLimits;
        intern.ReceiveBufferLimits = ReceiveBufferLimits;
    }

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

    // GetReceiveTTL gets value for IP_RECVTTL option.
    pub fn GetReceiveTTL(&self) -> bool {
        return self.lock().receiveTTLEnabled != 0;
    }

    // SetReceiveTTL sets value for IP_RECVTTL option.
    pub fn SetReceiveTTL(&self, v: bool) {
        self.lock().receiveTTLEnabled = Self::Boolval(v);
    }

    // GetReceiveHopLimit gets value for IP_RECVHOPLIMIT option.
    pub fn GetReceiveHopLimit(&self) -> bool {
        return self.lock().receiveHopLimitEnabled != 0;
    }

    // SetReceiveHopLimit sets value for IP_RECVHOPLIMIT option.
    pub fn SetReceiveHopLimit(&self, v: bool) {
        self.lock().receiveHopLimitEnabled = Self::Boolval(v);
    }

    // GetReceiveTClass gets value for IPV6_RECVTCLASS option.
    pub fn GetReceiveTClass(&self) -> bool {
        return self.lock().receiveTClassEnabled != 0;
    }

    // SetReceiveTClass sets value for IPV6_RECVTCLASS option.
    pub fn SetReceiveTClass(&self, v: bool) {
        self.lock().receiveTClassEnabled = Self::Boolval(v);
    }

    // GetReceivePacketInfo gets value for IP_PKTINFO option.
    pub fn GetReceivePacketInfo(&self) -> bool {
        return self.lock().receivePacketInfoEnabled != 0;
    }

    // SetReceivePacketInfo sets value for IP_PKTINFO option.
    pub fn SetReceivePacketInfo(&self, v: bool) {
        self.lock().receivePacketInfoEnabled = Self::Boolval(v);
    }

    // GetIPv6ReceivePacketInfo gets value for IPV6_RECVPKTINFO option.
    pub fn GetIPv6ReceivePacketInfo(&self) -> bool {
        return self.lock().receiveIPv6PacketInfoEnabled != 0;
    }

    // SetIPv6ReceivePacketInfo sets value for IPV6_RECVPKTINFO option.
    pub fn SetIPv6ReceivePacketInfo(&self, v: bool) {
        self.lock().receiveIPv6PacketInfoEnabled = Self::Boolval(v);
    }

    // GetHeaderIncluded gets value for IP_HDRINCL option.
    pub fn GetHeaderIncluded(&self) -> bool {
        return self.lock().hdrIncludedEnabled != 0;
    }

    // SetHeaderIncluded sets value for IP_HDRINCL option.
    pub fn SetHeaderIncluded(&self, v: bool) {
        self.lock().hdrIncludedEnabled = Self::Boolval(v);
    }

    // GetV6Only gets value for IPV6_V6ONLY option.
    pub fn GetV6Only(&self) -> bool {
        return self.lock().v6OnlyEnabled != 0;
    }

    // SetV6Only sets value for IPV6_V6ONLY option.
    //
    // Preconditions: the backing TCP or UDP endpoint must be in initial state.
    pub fn SetV6Only(&self, v: bool) {
        self.lock().v6OnlyEnabled = Self::Boolval(v);
    }

    // GetQuickAck gets value for TCP_QUICKACK option.
    pub fn GetQuickAck(&self) -> bool {
        return self.lock().quickAckEnabled != 0;
    }

    // SetQuickAck sets value for TCP_QUICKACK option.
    pub fn SetQuickAck(&self, v: bool) {
        self.lock().quickAckEnabled = Self::Boolval(v);
    }

    // GetDelayOption gets inverted value for TCP_NODELAY option.
    pub fn GetDelayOption(&self) -> bool {
        return self.lock().delayOptionEnabled != 0;
    }

    // SetDelayOption sets inverted value for TCP_NODELAY option.
    pub fn SetDelayOption(&self, v: bool) {
        self.lock().delayOptionEnabled = Self::Boolval(v);
    }

    // GetCorkOption gets value for TCP_CORK option.
    pub fn GetCorkOption(&self) -> bool {
        return self.lock().corkOptionEnabled != 0;
    }

    // SetCorkOption sets value for TCP_CORK option.
    pub fn SetCorkOption(&self, v: bool) {
        self.lock().corkOptionEnabled = Self::Boolval(v);
    }

    // GetReceiveOriginalDstAddress gets value for IP(V6)_RECVORIGDSTADDR option.
    pub fn GetReceiveOriginalDstAddress(&self) -> bool {
        return self.lock().receiveOriginalDstAddress != 0;
    }

    // SetReceiveOriginalDstAddress sets value for IP(V6)_RECVORIGDSTADDR option.
    pub fn SetReceiveOriginalDstAddress(&self, v: bool) {
        self.lock().receiveOriginalDstAddress = Self::Boolval(v);
    }

    // GetIPv4RecvError gets value for IP_RECVERR option.
    pub fn GetIPv4RecvError(&self) -> bool {
        return self.lock().ipv4RecvErrEnabled != 0;
    }

    // SetIPv4RecvError sets value for IP_RECVERR option.
    pub fn SetIPv4RecvError(&self, v: bool) {
        self.lock().ipv4RecvErrEnabled = Self::Boolval(v);
        if !v {
            self.pruneErrQueue();
        }
    }

    // GetIPv6RecvError gets value for IPV6_RECVERR option.
    pub fn GetIPv6RecvError(&self) -> bool {
        return self.lock().ipv6RecvErrEnabled != 0;
    }

    // SetIPv6RecvError sets value for IPV6_RECVERR option.
    pub fn SetIPv6RecvError(&self, v: bool) {
        self.lock().ipv6RecvErrEnabled = Self::Boolval(v);
        if !v {
            self.pruneErrQueue();
        }
    }

    // GetLastError gets value for SO_ERROR option.
    pub fn GetLastError(&self) -> Option<Error> {
        return self.LastError();
    }

    // GetOutOfBandInline gets value for SO_OOBINLINE option.
    pub fn GetOutOfBandInline(&self) -> bool {
        return true;
    }

    // SetOutOfBandInline sets value for SO_OOBINLINE option. We currently do not
    // support disabling this option.
    pub fn SetOutOfBandInline(&self, _: bool) {}

    // GetLinger gets value for SO_LINGER option.
    pub fn GetLinger(&self) -> LingerOption {
        return self.lock().linger.clone();
    }

    // SetLinger sets value for SO_LINGER option.
    pub fn SetLinger(&self, linger: &LingerOption) {
        self.lock().linger = *linger;
    }

    // pruneErrQueue resets the queue.
    pub fn pruneErrQueue(&self) {
        self.lock().errQueue.clear();
    }

    // DequeueErr dequeues a socket extended error from the error queue and returns
    // it. Returns nil if queue is empty.
    pub fn DequeueErr(&self) -> Option<SockError> {
        let mut l = self.lock();
        return l.errQueue.pop_front();
    }

    // PeekErr returns the error in the front of the error queue. Returns nil if
    // the error queue is empty.
    pub fn PeekErr(&self) -> Option<SockError> {
        let mut l = self.lock();
        match l.errQueue.pop_front() {
            None => return None,
            Some(e) => {
                l.errQueue.push_front(e.Clone());
                return Some(e);
            }
        }
    }

    // QueueErr inserts the error at the back of the error queue.
    //
    // Preconditions: so.GetIPv4RecvError() or so.GetIPv6RecvError() is true.
    pub fn QueueErr(&self, err: SockError) {
        self.lock().errQueue.push_back(err)
    }

    // QueueLocalErr queues a local error onto the local queue.
    pub fn QueueLocalErr(&self, err: Error, net: u32, info: u32, dst: FullAddr, payload: Vec<u8>) {
        self.QueueErr(SockError {
            Err: err,
            Cause: SockErrorCause::LocalSockError(info),
            Payload: payload,
            Dst: dst,
            Offender: FullAddr::default(),
            NetProto: net,
        })
    }

    // GetBindToDevice gets value for SO_BINDTODEVICE option.
    pub fn GetBindToDevice(&self) -> i32 {
        return self.lock().bindToDevice;
    }

    // SetBindToDevice sets value for SO_BINDTODEVICE option. If bindToDevice is
    // zero, the socket device binding is removed.
    pub fn SetBindToDevice(&self, bindToDevice: i32) -> Result<()> {
        if bindToDevice != 0 && !self.HasNIC(bindToDevice) {
            return Err(Error::UnknownDevice);
        }

        let mut l = self.lock();
        l.bindToDevice = bindToDevice;
        return Ok(());
    }

    // GetSendBufferSize gets value for SO_SNDBUF option.
    pub fn GetSendBufferSize(&self) -> i64 {
        return self.lock().sendBufferSize;
    }

    // SetSendBufferSize sets value for SO_SNDBUF option. notify indicates if the
    // stack handler should be invoked to set the send buffer size.
    pub fn SetSendBufferSize(&self, sendBufferSize: i64, notify: bool) {
        let mut sendBufferSize = sendBufferSize;
        if notify {
            sendBufferSize = self.OnSetSendBufferSize(sendBufferSize);
        }
        self.lock().sendBufferSize = sendBufferSize;
        if notify {
            self.WakeupWriters();
        }
    }

    // SendBufferLimits returns the [min, max) range of allowable send buffer
    // sizes.
    pub fn SendBufferLimits(&self) -> (i64, i64) {
        let limits = self.lock().SendBufferLimits;
        return (limits.Min as i64, limits.Max as i64);
    }

    // GetReceiveBufferSize gets value for SO_RCVBUF option.
    pub fn GetReceiveBufferSize(&self) -> i64 {
        return self.lock().receiveBufferSize;
    }

    // SetReceiveBufferSize sets the value of the SO_RCVBUF option, optionally
    // notifying the owning endpoint.
    pub fn SetReceiveBufferSize(&self, receiveBufferSize: i64, notify: bool) {
        let mut receiveBufferSize = receiveBufferSize;
        if notify {
            let oldsz = self.lock().receiveBufferSize;
            receiveBufferSize = self.OnSetReceiveBufferSize(receiveBufferSize, oldsz);
        }
        self.lock().receiveBufferSize = receiveBufferSize;
    }

    // ReceiveBufferLimits returns the [min, max) range of allowable receive buffer
    // sizes.
    pub fn ReceiveBufferLimits(&self) -> (i64, i64) {
        let limits = self.lock().ReceiveBufferLimits;
        return (limits.Min as i64, limits.Max as i64);
    }

    // GetRcvlowat gets value for SO_RCVLOWAT option.
    pub fn GetRcvlowat(&self) -> i32 {
        // todo: fix this later
        let defaultRcvlowat = 1;
        return defaultRcvlowat;
    }

    // SetRcvlowat sets value for SO_RCVLOWAT option.
    pub fn SetRcvlowat(&self, rcvlowat: i32) -> Result<()> {
        self.lock().rcvlowat = rcvlowat;
        return Ok(());
    }
}

#[derive(Default)]
pub struct SocketOptionsInternal {
    pub SendBufferLimits: BufferSizeOption,
    pub ReceiveBufferLimits: BufferSizeOption,

    // broadcastEnabled determines whether datagram sockets are allowed to
    // send packets to a broadcast address.
    pub broadcastEnabled: u32,

    // passCredEnabled determines whether SCM_CREDENTIALS socket control
    // messages are enabled.
    pub passCredEnabled: u32,

    // noChecksumEnabled determines whether UDP checksum is disabled while
    // transmitting for this socket.
    pub noChecksumEnabled: u32,

    // reuseAddressEnabled determines whether Bind() should allow reuse of
    // local address.
    pub reuseAddressEnabled: u32,

    // reusePortEnabled determines whether to permit multiple sockets to be
    // bound to an identical socket address.
    pub reusePortEnabled: u32,

    // keepAliveEnabled determines whether TCP keepalive is enabled for this
    // socket.
    pub keepAliveEnabled: u32,

    // multicastLoopEnabled determines whether multicast packets sent over a
    // non-loopback interface will be looped back.
    pub multicastLoopEnabled: u32,

    // receiveTOSEnabled is used to specify if the TOS ancillary message is
    // passed with incoming packets.
    pub receiveTOSEnabled: u32,

    // receiveTTLEnabled is used to specify if the TTL ancillary message is passed
    // with incoming packets.
    pub receiveTTLEnabled: u32,

    // receiveHopLimitEnabled is used to specify if the HopLimit ancillary message
    // is passed with incoming packets.
    pub receiveHopLimitEnabled: u32,

    // receiveTClassEnabled is used to specify if the IPV6_TCLASS ancillary
    // message is passed with incoming packets.
    pub receiveTClassEnabled: u32,

    // receivePacketInfoEnabled is used to specify if more information is
    // provided with incoming IPv4 packets.
    pub receivePacketInfoEnabled: u32,

    // receivePacketInfoEnabled is used to specify if more information is
    // provided with incoming IPv6 packets.
    pub receiveIPv6PacketInfoEnabled: u32,

    // hdrIncludeEnabled is used to indicate for a raw endpoint that all packets
    // being written have an IP header and the endpoint should not attach an IP
    // header.
    pub hdrIncludedEnabled: u32,

    // v6OnlyEnabled is used to determine whether an IPv6 socket is to be
    // restricted to sending and receiving IPv6 packets only.
    pub v6OnlyEnabled: u32,

    // quickAckEnabled is used to represent the value of TCP_QUICKACK option.
    // It currently does not have any effect on the TCP endpoint.
    pub quickAckEnabled: u32,

    // delayOptionEnabled is used to specify if data should be sent out immediately
    // by the transport protocol. For TCP, it determines if the Nagle algorithm
    // is on or off.
    pub delayOptionEnabled: u32,

    // corkOptionEnabled is used to specify if data should be held until segments
    // are full by the TCP transport protocol.
    pub corkOptionEnabled: u32,

    // receiveOriginalDstAddress is used to specify if the original destination of
    // the incoming packet should be returned as an ancillary message.
    pub receiveOriginalDstAddress: u32,

    // ipv4RecvErrEnabled determines whether extended reliable error message
    // passing is enabled for IPv4.
    pub ipv4RecvErrEnabled: u32,

    // ipv6RecvErrEnabled determines whether extended reliable error message
    // passing is enabled for IPv6.
    pub ipv6RecvErrEnabled: u32,

    // errQueue is the per-socket error queue. It is protected by errQueueMu.
    pub errQueue: VecDeque<SockError>,

    // bindToDevice determines the device to which the socket is bound.
    pub bindToDevice: i32,

    // getSendBufferLimits provides the handler to get the min, default and max
    // size for send buffer. It is initialized at the creation time and will not
    // change.
    //pub getSendBufferLimits GetSendBufferLimits `state:"manual"`

    // sendBufferSize determines the send buffer size for this socket.
    pub sendBufferSize: i64,

    // getReceiveBufferLimits provides the handler to get the min, default and
    // max size for receive buffer. It is initialized at the creation time and
    // will not change.
    //pub getReceiveBufferLimits GetReceiveBufferLimits `state:"manual"`

    // receiveBufferSize determines the receive buffer size for this socket.
    pub receiveBufferSize: i64,

    // linger determines the amount of time the socket should linger before
    // close. We currently implement this option for TCP socket only.
    pub linger: LingerOption,

    // rcvlowat specifies the minimum number of bytes which should be
    // received to indicate the socket as readable.
    pub rcvlowat: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SockErrOrigin {
    // SockExtErrorOriginNone represents an unknown error origin.
    SockExtErrorOriginNone,

    // SockExtErrorOriginLocal indicates a local error.
    SockExtErrorOriginLocal,

    // SockExtErrorOriginICMP indicates an IPv4 ICMP error.
    SockExtErrorOriginICMP,

    // SockExtErrorOriginICMP6 indicates an IPv6 ICMP error.
    SockExtErrorOriginICMP6,
}

impl SockErrOrigin {
    pub fn IsICMPErr(&self) -> bool {
        return *self == Self::SockExtErrorOriginICMP || *self == Self::SockExtErrorOriginICMP6;
    }
}

#[derive(Clone)]
pub enum SockErrorCause {
    LocalSockError(u32),
}

impl SockErrorCause {
    // Origin implements SockErrorCause.
    pub fn Origin(&self) -> SockErrOrigin {
        match self {
            SockErrorCause::LocalSockError(_) => return SockErrOrigin::SockExtErrorOriginLocal,
        }
    }

    // Type implements SockErrorCause.
    pub fn Type(&self) -> u8 {
        match self {
            SockErrorCause::LocalSockError(_) => return 0,
        }
    }

    // Code implements SockErrorCause.
    pub fn Code(&self) -> u8 {
        match self {
            SockErrorCause::LocalSockError(_) => return 0,
        }
    }

    // Info implements SockErrorCause.
    pub fn Info(&self) -> u32 {
        match self {
            SockErrorCause::LocalSockError(ref info) => return *info,
        }
    }
}

pub struct SockError {
    // Err is the error caused by the errant packet.
    pub Err: Error,

    // Cause is the detailed cause of the error.
    pub Cause: SockErrorCause,

    // Payload is the errant packet's payload.
    pub Payload: Vec<u8>,

    // Dst is the original destination address of the errant packet.
    pub Dst: FullAddr,

    // Offender is the original sender address of the errant packet.
    pub Offender: FullAddr,

    // NetProto is the network protocol being used to transmit the packet.
    pub NetProto: u32,
}

impl SockError {
    pub fn Clone(&self) -> Self {
        return Self {
            Err: self.Err.clone(),
            Cause: self.Cause.clone(),
            Payload: self.Payload.to_vec(),
            Dst: self.Dst.clone(),
            Offender: self.Offender.clone(),
            NetProto: self.NetProto.clone(),
        };
    }
}
