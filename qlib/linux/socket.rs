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

// Set/get socket option levels, from socket.h.
pub const SOL_IP      :i32 = 0;
pub const SOL_SOCKET  :i32 = 1;
pub const SOL_TCP     :i32 = 6;
pub const SOL_UDP     :i32 = 17;
pub const SOL_IPV6    :i32 = 41;
pub const SOL_ICMPV6  :i32 = 58;
pub const SOL_RAW     :i32 = 255;
pub const SOL_PACKET  :i32 = 263;
pub const SOL_NETLINK :i32 = 270;

// Socket options from socket.h.
pub const SO_DEBUG                 :i32 = 1;
pub const SO_REUSEADDR             :i32 = 2;
pub const SO_TYPE                  :i32 = 3;
pub const SO_ERROR                 :i32 = 4;
pub const SO_DONTROUTE             :i32 = 5;
pub const SO_BROADCAST             :i32 = 6;
pub const SO_SNDBUF                :i32 = 7;
pub const SO_RCVBUF                :i32 = 8;
pub const SO_KEEPALIVE             :i32 = 9;
pub const SO_OOBINLINE             :i32 = 10;
pub const SO_NO_CHECK              :i32 = 11;
pub const SO_PRIORITY              :i32 = 12;
pub const SO_LINGER                :i32 = 13;
pub const SO_BSDCOMPAT             :i32 = 14;
pub const SO_REUSEPORT             :i32 = 15;
pub const SO_PASSCRED              :i32 = 16;
pub const SO_PEERCRED              :i32 = 17;
pub const SO_RCVLOWAT              :i32 = 18;
pub const SO_SNDLOWAT              :i32 = 19;
pub const SO_RCVTIMEO              :i32 = 20;
pub const SO_SNDTIMEO              :i32 = 21;
pub const SO_BINDTODEVICE          :i32 = 25;
pub const SO_ATTACH_FILTER         :i32 = 26;
pub const SO_DETACH_FILTER         :i32 = 27;
pub const SO_GET_FILTER            :i32 = SO_ATTACH_FILTER;
pub const SO_PEERNAME              :i32 = 28;
pub const SO_TIMESTAMP             :i32 = 29;
pub const SO_ACCEPTCONN            :i32 = 30;
pub const SO_PEERSEC               :i32 = 31;
pub const SO_SNDBUFFORCE           :i32 = 32;
pub const SO_RCVBUFFORCE           :i32 = 33;
pub const SO_PASSSEC               :i32 = 34;
pub const SO_TIMESTAMPNS           :i32 = 35;
pub const SO_MARK                  :i32 = 36;
pub const SO_TIMESTAMPING          :i32 = 37;
pub const SO_PROTOCOL              :i32 = 38;
pub const SO_DOMAIN                :i32 = 39;
pub const SO_RXQ_OVFL              :i32 = 40;
pub const SO_WIFI_STATUS           :i32 = 41;
pub const SO_PEEK_OFF              :i32 = 42;
pub const SO_NOFCS                 :i32 = 43;
pub const SO_LOCK_FILTER           :i32 = 44;
pub const SO_SELECT_ERR_QUEUE      :i32 = 45;
pub const SO_BUSY_POLL             :i32 = 46;
pub const SO_MAX_PACING_RATE       :i32 = 47;
pub const SO_BPF_EXTENSIONS        :i32 = 48;
pub const SO_INCOMING_CPU          :i32 = 49;
pub const SO_ATTACH_BPF            :i32 = 50;
pub const SO_ATTACH_REUSEPORT_CBPF :i32 = 51;
pub const SO_ATTACH_REUSEPORT_EBPF :i32 = 52;
pub const SO_CNX_ADVICE            :i32 = 53;
pub const SO_MEMINFO               :i32 = 55;
pub const SO_INCOMING_NAPI_ID      :i32 = 56;
pub const SO_COOKIE                :i32 = 57;
pub const SO_PEERGROUPS            :i32 = 59;
pub const SO_ZEROCOPY              :i32 = 60;
pub const SO_TXTIME                :i32 = 61;

// shutdown(2) how commands, from <linux/net.h>.
pub const SHUT_RD   :i32 = 0;
pub const SHUT_WR   :i32 = 1;
pub const SHUT_RDWR :i32 = 2;

// enum socket_state, from uapi/linux/net.h.
pub const SS_FREE          :i32 = 0; // Not allocated.
pub const SS_UNCONNECTED   :i32 = 1; // Unconnected to any socket.
pub const SS_CONNECTING    :i32 = 2; // In process of connecting.
pub const SS_CONNECTED     :i32 = 3; // Connected to socket.
pub const SS_DISCONNECTING :i32 = 4; // In process of disconnecting.

// LingerOption is used by SetSockOpt/GetSockOpt to set/get the
// duration for which a socket lingers before returning from Close.
//
// +stateify savable
#[derive(Default, Debug)]
pub struct LingerOption {
    Enabled: bool,
    Timeout: i64,
}