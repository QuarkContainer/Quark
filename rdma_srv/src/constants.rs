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

// rdma_cli needs same file from rdma_srv. Crating a soft link.
// cd rdma_cli/src;ln -s ../../rdma_srv/src/constants.rs constants.rs

pub const EVENT_TYPE_SET: &str = "set";
pub const EVENT_TYPE_DELETE: &str = "delete";

pub const GRPC_SERVER_ADDRESS: &str = "http://[::1]:51051";

pub const SO_ORIGINAL_DST: i32 = 80;
pub const SOL_IP: i32 = 0;
pub const INCLUSTER_INGRESS_PORT: u16 = 7981;

pub const PROTOCOL_TCP: &str = "TCP";
pub const PROTOCOL_UDP: &str = "UDP";
