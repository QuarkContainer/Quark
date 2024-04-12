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

use std::borrow::Borrow;
use std::ptr;
use enum_dispatch::enum_dispatch;

use io_uring::opcode;
use io_uring::squeue;
use io_uring::types;

use crate::qlib::kernel::quring::uring_op::*;
use crate::qlib::kernel::quring::uring_async::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::*;
use super::kernel::SHARESPACE;
use crate::URING;

pub fn CopyCompleteEntry() -> usize {
    let mut count = 0;

    let mut uring = URING.lock();
    {
        let mut cq = uring.completion();
        loop {
            let cqe = cq.next();

            match cqe {
                None => break,
                Some(cqe) => {
                    count += 1;
                    let entry = unsafe {
                        *(&cqe as * const _ as u64 as * const _)
                    };
                    match SHARESPACE.uringQueue.completeq.push(entry) {
                        Err(_) => {
                            panic!("CopyCompleteEntry fail ...");
                        }
                        _ => (),
                    }
                }
            }
        }
    }

    return count;
}

#[inline]
pub fn HostSubmit() -> Result<usize> {
    let _count = CopyCompleteEntry();
        
    let mut count = 0;

    {
        let mut uring = URING.lock();
        let mut sq = uring.submission();
        let submitq = SHARESPACE.uringQueue.submitq.borrow();

        if sq.dropped() != 0 {
            error!("uring fail dropped {}", sq.dropped());
        }

        if sq.cq_overflow() {
            error!("uring fail overflow")
        }
        assert!(sq.dropped() == 0, "dropped {}", sq.dropped());
        assert!(!sq.cq_overflow());

        while !sq.is_full() {
            let uringEntry = match submitq.pop() {
                None => break,
                Some(e) => e,
            };

            let entry = match &uringEntry.ops {
                UringOps::UringCall(call) => {
                    call.Entry()
                }
                UringOps::AsyncOps(ops) => {
                    ops.Entry()
                }
            };
    
            let entry = entry.user_data(uringEntry.userdata);
            let entry = if uringEntry.linked {
                entry.flags(squeue::Flags::IO_LINK)
            } else {
                entry
            };

            unsafe {
                match sq.push(&entry) {
                    Ok(_) => (),
                    Err(_) => panic!("AUringCall submission queue is full"),
                }
            }

            count += 1;
        }
    }

    if count > 0 {
        let ret = URING.lock().submit_and_wait(0)?;
        return Ok(ret);
    }

    return Ok(0);
}

impl UringEntry {
    pub fn Entry(&self) -> squeue::Entry {
        let entry = match &self.ops {
            UringOps::UringCall(call) => {
                error!("request is {:x?}/{:x}", call, self.userdata);
                call.Entry()
            }
            UringOps::AsyncOps(ops) => ops.Entry(),
        };

        let entry = entry.user_data(self.userdata);
        let entry = if self.linked {
            entry.flags(squeue::Flags::IO_LINK)
        } else {
            entry
        };

        return entry;
    }
}

impl TimerRemoveOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::TimeoutRemove::new(self.userData);

        return op.build();
    }
}

impl ReadOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Read::new(types::Fd(self.fd), self.addr as *mut _, self.len).offset(self.offset as u64);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl WriteOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(
            types::Fd(self.fd), 
            self.addr as *const _, 
            self.len
        )
        .offset(
            self.offset as u64
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl StatxOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Statx::new(
            types::Fd(self.dirfd),
            self.pathname as *const _,
            self.statxBuf as *mut types::statx,
        )
        .flags(self.flags)
        .mask(self.mask);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl FsyncOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = if self.dataSyncOnly {
            opcode::Fsync::new(types::Fd(self.fd)).flags(types::FsyncFlags::DATASYNC)
        } else {
            opcode::Fsync::new(types::Fd(self.fd))
        };

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl EpollCtlOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::EpollCtl::new(
            types::Fd(self.epollfd),
            types::Fd(self.fd),
            self.op,
            &self.ev as *const _ as u64 as _, //*const types::epoll_event,
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl AcceptOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Accept::new(
            types::Fd(self.fd),
            ptr::null_mut(),
            ptr::null_mut(),
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringCall {
    pub fn Entry(&self) -> squeue::Entry {
        match self.msg {
            UringOp::None => (),
            UringOp::Read(ref msg) => return msg.Entry(),
            UringOp::Write(ref msg) => return msg.Entry(),
            UringOp::Statx(ref msg) => return msg.Entry(),
            UringOp::Fsync(ref msg) => return msg.Entry(),
            UringOp::Accept(ref msg) => return msg.Entry(),
        };

        panic!("UringCall SEntry UringOp::None")
    }
}


impl UringAsyncOpsTrait for AsyncWritev {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(
            types::Fd(self.fd), 
            self.addr as *const u8, 
            self.len
        )
        .offset(
            self.offset as u64
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncBufWrite {
    fn Entry(&self) -> squeue::Entry {
        //let op = Write::new(types::Fd(self.fd), self.addr as * const u8, self.len as u32);
        let op = opcode::Write::new(
            types::Fd(self.fd),
            self.bufAddr as *const u8,
            self.bufLen as u32,
        )
        .offset(self.offset as _);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncLogFlush {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32); //.flags(MsgType::MSG_DONTWAIT);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncSend {
    fn Entry(&self) -> squeue::Entry {
        //let op = Write::new(types::Fd(self.fd), self.addr as * const u8, self.len as u32);
        let op = opcode::Send::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32);
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}


impl UringAsyncOpsTrait for AsyncFiletWrite {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncAccept {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Accept::new(
            types::Fd(self.fd),
            &self.addr.addr.data[0] as *const _ as u64 as *mut _,
            &self.addr.len as *const _ as u64 as *mut _,
        );
        
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncFileRead {
    fn Entry(&self) -> squeue::Entry {
        if self.isSocket {
            let op = opcode::Recv::new(types::Fd(self.fd), self.addr as *mut u8, self.len as u32);
            if SHARESPACE.config.read().UringFixedFile {
                return op.build().flags(squeue::Flags::FIXED_FILE);
            } else {
                return op.build();
            }
        }

        let op = opcode::Read::new(types::Fd(self.fd), self.addr as *mut u8, self.len as u32);
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}



impl UringAsyncOpsTrait for AIOWrite {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(
            types::Fd(self.fd),
            self.buf.Ptr() as *const u8,
            self.buf.Len() as u32,
        )
        .offset(self.offset as u64);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AIORead {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Read::new(
            types::Fd(self.fd),
            self.buf.Ptr() as *mut u8,
            self.buf.Len() as u32,
        )
        .offset(self.offset as u64);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AIOFsync {
    fn Entry(&self) -> squeue::Entry {
        let op = if self.dataSyncOnly {
            opcode::Fsync::new(types::Fd(self.fd)).flags(types::FsyncFlags::DATASYNC)
        } else {
            opcode::Fsync::new(types::Fd(self.fd))
        };

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncConnect {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Connect::new(
            types::Fd(self.fd), 
            &self.addr.data[0] as * const _ as u64 as *const _, 
            self.len
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}


impl UringAsyncOpsTrait for DNSRecv {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::RecvMsg::new(
            types::Fd(self.fd), 
            self.msgAddr as * mut _
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for DNSSend {
    fn Entry(&self) -> squeue::Entry {
        let intern = self.lock();
        let op = opcode::SendMsg::new(
            types::Fd(intern.fd), 
            &intern.msg as *const _ as *const _
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for PollHostEpollWait {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::PollAdd::new(types::Fd(self.fd), EVENT_READ as u32);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncEpollCtl {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::EpollCtl::new(
            types::Fd(self.epollfd),
            types::Fd(self.fd),
            self.op,
            &self.ev as *const _ as u64 as *const types::epoll_event,
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for TsotPoll {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::PollAdd::new(types::Fd(self.fd), EVENT_READ as u32);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for TsotAsyncSend {
    fn Entry(&self) -> squeue::Entry {
        //let op = Write::new(types::Fd(self.fd), self.addr as * const u8, self.len as u32);
        let op = opcode::Send::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32);
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}


impl UringAsyncOpsTrait for AsyncNone {}

#[enum_dispatch(AsyncOps)]
pub trait UringAsyncOpsTrait {
    fn Entry(&self) -> squeue::Entry {
        panic!("doesn't support AsyncOpsTrait::SEntry")
    }
}