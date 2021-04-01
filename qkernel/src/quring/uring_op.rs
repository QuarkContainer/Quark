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

use super::super::qlib::task_mgr::*;
use super::super::qlib::uring::squeue;
use super::super::qlib::uring::opcode::*;

pub static DEFAULT_MSG : UringOp = UringOp::None;

#[derive(Clone, Debug, Copy)]
pub struct UringCall {
    pub taskId: TaskIdQ,
    pub ret: i32,
    pub msg: UringOp,
}

impl Default for UringCall {
    fn default() -> Self {
        return Self {
            taskId : TaskIdQ::default(),
            ret: 0,
            msg: DEFAULT_MSG,
        }
    }
}

pub enum UringCallRetType {
    Normal,
    Block,
}

impl UringCall {
    pub fn Ptr(&self) -> u64 {
        return self as * const _ as u64;
    }

    pub fn SEntry(&self) -> squeue::Entry {
        match self.msg {
            UringOp::None => (),
            UringOp::TimerRemove(ref msg) => return msg.SEntry(),
            UringOp::Read(ref msg) => return msg.SEntry(),
            UringOp::Write(ref msg) => return msg.SEntry(),
            UringOp::Statx(ref msg) => return msg.SEntry(),
            UringOp::Fsync(ref msg) => return msg.SEntry(),
        };

        panic!("UringCall SEntry UringOp::None")
    }
}

#[derive(Clone, Debug, Copy)]
pub enum UringOp {
    None,
    TimerRemove(TimerRemoveOp),
    Read(ReadOp),
    Write(WriteOp),
    Statx(StatxOp),
    Fsync(FsyncOp),
}

impl Default for UringOp {
    fn default() -> Self {
        return UringOp::None;
    }
}

#[derive(Clone, Debug, Copy)]
pub struct TimerRemoveOp {
    pub userData: u64
}

impl TimerRemoveOp {
    pub fn SEntry(&self) -> squeue::Entry {
        let op = TimeoutRemove::new(self.userData);

        return op.build();
    }
}

#[derive(Clone, Debug, Copy)]
pub struct ReadOp {
    pub fd: i32,
    pub addr: u64,
    pub cnt: u32,
    pub offset: i64,
}

impl ReadOp {
    pub fn SEntry(&self) -> squeue::Entry {
        let op = Readv::new(types::Fd(self.fd), self.addr as * const _, self.cnt)
            .offset(self.offset);

        return op.build()
            .flags(squeue::Flags::FIXED_FILE);
    }
}

#[derive(Clone, Debug, Copy)]
pub struct WriteOp {
    pub fd: i32,
    pub addr: u64,
    pub cnt: u32,
    pub offset: i64,
}

impl WriteOp {
    pub fn SEntry(&self) -> squeue::Entry {
        let op = Writev::new(types::Fd(self.fd), self.addr as * const _, self.cnt)
            .offset(self.offset);

        return op.build()
            .flags(squeue::Flags::FIXED_FILE);
    }
}

#[derive(Clone, Debug, Copy)]
pub struct StatxOp {
    pub dirfd: i32,
    pub pathname: u64,
    pub statxBuf: u64,
    pub flags: i32,
    pub mask: u32,
}

impl StatxOp {
    pub fn SEntry(&self) -> squeue::Entry {
        let op = Statx::new(types::Fd(self.dirfd), self.pathname as * const _, self.statxBuf as * mut types::statx)
            .flags(self.flags)
            .mask(self.mask);

        return op.build()
            .flags(squeue::Flags::FIXED_FILE);
    }
}

#[derive(Clone, Debug, Copy)]
pub struct FsyncOp {
    pub fd: i32,
    pub dataSyncOnly: bool
}

impl FsyncOp {
    pub fn SEntry(&self) -> squeue::Entry {
        let op = if self.dataSyncOnly {
            Fsync::new(types::Fd(self.fd))
                .flags(types::FsyncFlags::DATASYNC)
        } else {
            Fsync::new(types::Fd(self.fd))
        };

        return op.build()
            .flags(squeue::Flags::FIXED_FILE);
    }
}