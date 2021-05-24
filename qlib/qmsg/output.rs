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

use super::super::config::*;
use super::super::linux_def::*;

// kernel to host
#[derive(Debug, Copy, Clone)]
#[repr(align(128))]
pub enum HostOutputMsg {
    QCall(u64),
    WaitFD(WaitFD),
    Close(Close),
    //WriteBuffTrigger(WriteBuffTrigger),
    //ReadBuffTrigger(ReadBuffTrigger),
    MUnmap(MUnmap),
    IOBufWrite(IOBufWrite),
    PrintStr(PrintStr),
    WakeVCPU(WakeVCPU)
}

#[derive(Clone, Default, Debug, Copy)]
pub struct WaitFD {
    pub fd: i32,
    pub mask: EventMask,
}

#[derive(Clone, Default, Debug, Copy)]
pub struct Close {
    pub fd: i32,
}

//write buff and find write buff is empty, notify host to write it to os, async call
#[derive(Clone, Default, Debug, Copy)]
pub struct WriteBuffTrigger {
    pub fd: i32,
}
//read buff and find read buff full, notify host to read more from os, async call
#[derive(Clone, Default, Debug, Copy)]
pub struct ReadBuffTrigger {
    pub fd: i32,
}

#[derive(Clone, Default, Debug, Copy)]
pub struct MUnmap {
    pub addr: u64,
    pub len: u64,
}

#[derive(Clone, Default, Debug, Copy)]
pub struct IOBufWrite {
    pub fd: i32,
    pub addr: u64,
    pub len: usize,
    pub offset: isize,
}

#[derive(Clone, Debug, Copy)]
pub struct PrintStr {
    pub level: DebugLevel,
    pub addr: u64,
    pub len: usize,
}

#[derive(Clone, Debug, Copy)]
pub struct WakeVCPU {
    pub vcpuId: usize,
}