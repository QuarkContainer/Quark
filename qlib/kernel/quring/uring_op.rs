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

use super::super::super::linux_def::EpollEvent;
use super::super::super::task_mgr::*;

pub static DEFAULT_MSG: UringOp = UringOp::None;

#[derive(Clone, Debug, Copy)]
pub struct UringCall {
    pub taskId: TaskId,
    pub ret: i32,
    pub msg: UringOp,
}

impl Default for UringCall {
    fn default() -> Self {
        return Self {
            taskId: TaskId::default(),
            ret: 0,
            msg: DEFAULT_MSG,
        };
    }
}

pub enum UringCallRetType {
    Normal,
    Block,
}

impl UringCall {
    pub fn Ptr(&self) -> u64 {
        return self as *const _ as u64;
    }
}

#[derive(Clone, Debug, Copy)]
pub enum UringOp {
    None,
    Read(ReadOp),
    Write(WriteOp),
    Statx(StatxOp),
    Fsync(FsyncOp),
    Splice(SpliceOp),
    Accept(AcceptOp),
}

impl Default for UringOp {
    fn default() -> Self {
        return UringOp::None;
    }
}

#[derive(Clone, Debug, Copy)]
pub struct TimerRemoveOp {
    pub userData: u64,
}

#[derive(Clone, Debug, Copy)]
pub struct ReadOp {
    pub fd: i32,
    pub addr: u64,
    pub len: u32,
    pub offset: i64,
}

#[derive(Clone, Debug, Copy)]
pub struct WriteOp {
    pub fd: i32,
    pub addr: u64,
    pub len: u32,
    pub offset: i64,
}

#[derive(Clone, Debug, Copy)]
pub struct StatxOp {
    pub dirfd: i32,
    pub pathname: u64,
    pub statxBuf: u64,
    pub flags: i32,
    pub mask: u32,
}

#[derive(Clone, Debug, Copy)]
pub struct FsyncOp {
    pub fd: i32,
    pub dataSyncOnly: bool,
}

#[derive(Clone, Debug, Copy)]
pub struct SpliceOp {
    pub fdIn: i32,
    pub offsetIn: i64,
    pub fdOut: i32,
    pub offsetOut: i64,
    pub len: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, Copy)]
pub struct EpollCtlOp {
    pub epollfd: i32,
    pub fd: i32,
    pub op: i32,
    pub ev: EpollEvent,
}

#[derive(Clone, Debug, Copy)]
pub struct AcceptOp {
    pub fd: i32,
}

