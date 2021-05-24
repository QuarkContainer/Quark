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

use alloc::sync::Arc;
use spin::Mutex;

use super::super::super::task::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
//use super::super::super::mem::seq::*;
use super::super::host::tty::*;
use super::queue::*;
use super::utf8::*;

// canonMaxBytes is the number of bytes that fit into a single line of
// terminal input in canonical mode. This corresponds to N_TTY_BUF_SIZE
// in include/linux/tty.h.
pub const CANON_MAX_BYTES: usize = 4096;

// nonCanonMaxBytes is the maximum number of bytes that can be read at
// a time in noncanonical mode.
pub const NON_CANON_MAX_BYTES: usize = CANON_MAX_BYTES - 1;

pub const SPACES_PER_TAB: usize = 8;

pub struct LineDiscipline {
    pub size: WindowSize,
    pub inQueue: Arc<Mutex<Queue>>,
    pub outQueue: Arc<Mutex<Queue>>,
    pub termios: KernelTermios,
    pub column: i32,
}

impl LineDiscipline {
    pub fn New(termios: KernelTermios) -> Self {
        let ld = Self {
            size: WindowSize::default(),
            inQueue: Arc::new(Mutex::new(Queue::NewInputQueue())),
            outQueue: Arc::new(Mutex::new(Queue::NewOutputQueue())),
            termios: termios,
            column: 0,
        };

        return ld
    }

    pub fn GetTermios(&self, task: &Task, dstAddr: u64) -> Result<()> {
        let t = self.termios.ToTermios();
        task.CopyOutObj(&t, dstAddr)?;
        return Ok(())
    }

    pub fn SetTermios(&mut self, task: &Task, srcAddr: u64) -> Result<()> {
        let oldCanonEnabled = self.termios.LEnabled(LocalFlags::ICANON);

        let mut t: Termios = Termios::default();
        //task.CopyInObject(&t as * const _ as u64, srcAddr, size_of::<Termios>())?;
        task.CopyInObj(srcAddr, &mut t)?;
        self.termios.FromTermios(&t);

        if oldCanonEnabled && !self.termios.LEnabled(LocalFlags::ICANON) {
            //l.inQueue.pushWaitBuf(l)
            self.inQueue.lock().readable = true;
            //l.slaveWaiter.Notify(waiter.EventIn)
        }

        //todo: don't understand
        /*
        // If canonical mode is turned off, move bytes from inQueue's wait
	// buffer to its read buffer. Anything already in the read buffer is
	// now readable.
	if oldCanonEnabled && !l.termios.LEnabled(linux.ICANON) {
		l.inQueue.pushWaitBuf(l)
		l.inQueue.readable = true
		l.slaveWaiter.Notify(waiter.EventIn)
	}
        */

        return Ok(())
    }

    pub fn GetWindowSize(&self, task: &Task, dstAddr: u64) -> Result<()> {
        task.CopyOutObj(&self.size, dstAddr)?;
        return Ok(())
    }

    pub fn SetWindowSize(&mut self, task: &Task, srcAddr: u64) -> Result<()> {
        task.CopyInObj(srcAddr, &mut self.size)?;
        return Ok(())
    }

    pub fn InputQueueReadSize(&self, task: &Task, dstAddr: u64) -> Result<()> {
        return self.inQueue.lock().ReableSize(task, dstAddr);
    }

    pub fn InputQueueRead(&self, _task: &Task, dst: &mut [u8]) -> Result<i64> {
        let n = self.inQueue.lock().Read(dst)?;

        if n > 0 {
            return Ok(n)
        }

        return Err(Error::SysError(SysErr::EAGAIN))
    }

    pub fn InputQueueWrite(&mut self, _task: &Task, src: &mut [u8]) -> Result<i64> {
        let inQueue = self.inQueue.clone();
        let n = inQueue.lock().Write(src, self)?;

        if n > 0 {
            return Ok(n)
        }

        return Err(Error::SysError(SysErr::EAGAIN))
    }

    pub fn OutputQueueReadSize(&self, task: &Task, dstAddr: u64) -> Result<()> {
        return self.outQueue.lock().ReableSize(task, dstAddr);
    }

    pub fn OutputQueueRead(&self, _task: &Task, dst: &mut [u8]) -> Result<i64> {
        let n = self.outQueue.lock().Read(dst)?;

        if n > 0 {
            return Ok(n)
        }

        return Err(Error::SysError(SysErr::EAGAIN))
    }

    pub fn OutputQueueWrite(&mut self, _task: &Task, src: &mut [u8]) -> Result<i64> {
        let inQueue = self.outQueue.clone();
        let n = inQueue.lock().Write(src, self)?;

        if n > 0 {
            return Ok(n)
        }

        return Err(Error::SysError(SysErr::EAGAIN))
    }

    pub fn ShouldDiscard(&self, q: &Queue, cBytes: &[u8]) -> bool {
        return self.termios.LEnabled(LocalFlags::ICANON)
            && q.buf.AvailableDataSize() + cBytes.len() > CANON_MAX_BYTES
            && !self.termios.IsTerminating(cBytes)
    }

    pub fn Peek(&self, b: &[u8]) -> usize {
        let mut size = 1;

        if self.termios.IEnabled(InputFlags::IUTF8) {
            size = DecodeUtf8(b);
        }

        return size
    }
}