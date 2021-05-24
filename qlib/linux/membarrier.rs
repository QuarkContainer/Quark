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


// membarrier(2) commands, from include/uapi/linux/membarrier.h.
pub const MEMBARRIER_CMD_QUERY                                : i32 = 0;
pub const MEMBARRIER_CMD_GLOBAL                               : i32 = 1 << 0;
pub const MEMBARRIER_CMD_GLOBAL_EXPEDITED                     : i32 = 1 << 1;
pub const MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED            : i32 = 1 << 2;
pub const MEMBARRIER_CMD_PRIVATE_EXPEDITED                    : i32 = 1 << 3;
pub const MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED           : i32 = 1 << 4;
pub const MEMBARRIER_CMD_PRIVATE_EXPEDITED_SYNC_CORE          : i32 = 1 << 5;
pub const MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_SYNC_CORE : i32 = 1 << 6;
pub const MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ               : i32 = 1 << 7;
pub const MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ      : i32 = 1 << 8;

// membarrier(2) flags, from include/uapi/linux/membarrier.h.
pub const MEMBARRIER_CMD_FLAG_CPU : u32 = 1 << 0;