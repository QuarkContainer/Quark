// Copyright (c) 2021 Quark Container Authors
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

use super::time::Timeval;


// Flags that may be used with wait4(2) and getrusage(2).

// wait4(2) uses this to aggregate RUSAGE_SELF and RUSAGE_CHILDREN.
pub const RUSAGE_BOTH       :i32 = -0x2;

// getrusage(2) flags.
pub const RUSAGE_CHILDREN   :i32 = -0x1;
pub const RUSAGE_SELF       :i32 = 0x0;
pub const RUSAGE_THREAD     :i32 = 0x1;

#[repr(C)]
#[derive(Default, Debug)]
pub struct Rusage {
    pub UTime    : Timeval,
    pub STime    : Timeval,
    pub MaxRSS   : i64,
    pub IXRSS    : i64,
    pub IDRSS    : i64,
    pub ISRSS    : i64,
    pub MinFlt   : i64,
    pub MajFlt   : i64,
    pub NSwap    : i64,
    pub InBlock  : i64,
    pub OuBlock  : i64,
    pub MsgSnd   : i64,
    pub MsgRcv   : i64,
    pub NSignals : i64,
    pub NVCSw    : i64,
    pub NIvCSw   : i64,
}