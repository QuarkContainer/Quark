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

use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;
use alloc::collections::linked_list::LinkedList;
use alloc::vec::Vec;
use spin::Mutex;
use spin::MutexGuard;
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::waiter::*;
use super::super::super::fs::attr::*;
use super::super::super::fs::dirent::*;
use super::super::super::fs::inode::*;
use super::super::super::fs::mount::*;
use super::super::super::fs::file::*;
use super::super::super::fs::flags::*;
use super::super::super::task::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::device::*;
use super::super::super::qlib::mem::seq::*;
use super::super::waiter::cond::*;
use super::buffer::*;
use super::node::*;
use super::reader_writer::*;
use super::reader::*;
use super::writer::*;

// MinimumPipeSize is a hard limit of the minimum size of a pipe.
pub const MINIMUM_PIPE_SIZE : usize = MemoryDef::PAGE_SIZE as usize;

// DefaultPipeSize is the system-wide default size of a pipe in bytes.
pub const DEFAULT_PIPE_SIZE : usize = 16 * MemoryDef::PAGE_SIZE as usize;

// MaximumPipeSize is a hard limit on the maximum size of a pipe.
// It corresponds to fs/pipe.c:pipe_max_size.
pub const MAXIMUM_PIPE_SIZE : usize = 1048576;

// atomicIOBytes is the maximum number of bytes that the pipe will
// guarantee atomic reads or writes atomically.
// It corresponds to limits.h:PIPE_BUF.
pub const ATOMIC_IO_BYTES : usize = 4096;

// NewConnectedPipe initializes a pipe and returns a pair of objects
// representing the read and write ends of the pipe.
pub fn NewConnectedPipe(task: &Task, sizeBytes: usize, atomicIOBytes: usize) -> (File, File) {
    let p = Pipe::New(task, false, sizeBytes, atomicIOBytes);
    let r = p.Open(task, &FileFlags {Read: true, ..Default::default()});
    let w = p.Open(task, &FileFlags {Write: true, ..Default::default()});

    return (r, w)
}

// Pipe is an encapsulation of a platform-independent pipe.
// It manages a buffered byte queue shared between a reader/writer
// pair.
#[derive(Default)]
pub struct PipeInternal {
    pub id: u64,

    // data is the buffer queue of pipe contents.
    pub data: LinkedList<Buffer>,

    // max is the maximum size of the pipe in bytes. When this max has been
    // reached, writers will get EWOULDBLOCK.
    pub max: usize,

    // size is the current size of the pipe in bytes.
    pub size: usize,

    // hadWriter indicates if this pipe ever had a writer. Note that this
    // does not necessarily indicate there is *currently* a writer, just
    // that there has been a writer at some point since the pipe was
    // created.
    pub hadWriter: bool,

    // The dirent backing this pipe. Shared by all readers and writers.
    //
    // This value is immutable.
    pub dirent: Option<Dirent>,
}

impl PipeInternal {
    pub fn Available(&self) -> usize {
        return (self.max - self.size) as usize;
    }

    pub fn Write(&mut self, _task: &Task, src: BlockSeq, _atomicIOBytes: usize) -> Result<usize> {
        let mut p = self;

        let mut src = src;

        // POSIX requires that a write smaller than atomicIOBytes (PIPE_BUF) be
        // atomic, but requires no atomicity for writes larger than this.
        let wanted = src.NumBytes() as usize;
        let avail = p.Available();
        //info!("pipe::write id is {} wanted is {}, avail is {}, atomicIOBytes is {}", p.id, wanted, avail, self.atomicIOBytes);
        if wanted > avail {
            // Is this needed? todo: confirm this
            // if this is must, Pipe::Readfrom needs redesign
            /*if wanted <= atomicIOBytes {
                return Err(Error::SysError(SysErr::EAGAIN))
            }*/

            // Limit to the available capacity.
            src = src.TakeFirst(avail as u64);
        }

        let mut done = 0;
        while src.NumBytes() > 0 {
            // Need a new buffer?
            if p.data.back().is_none() || p.data.back().as_ref().unwrap().borrow().Full() {
                p.data.push_back(NewBuff());
            }

            // Copy user data.
            let n = src.CopyInTo(*p.data.back_mut().as_mut().unwrap())?;
            done += n;
            p.size += n;
            src = src.DropFirst(n as u64);
        }

        if wanted > done {
            // Partial write due to full pipe.
            return Ok(done)
        }

        return Ok(done)
    }
}

pub struct PipeIn {
    pub queue: Queue,

    // isNamed indicates whether this is a named pipe.
    //
    // This value is immutable.
    pub isNamed: bool,


    // atomicIOBytes is the maximum number of bytes that the pipe will
    // guarantee atomic reads or writes atomically.
    //
    // This value is immutable.
    pub atomicIOBytes: usize,

    // The number of active readers for this pipe.
    //
    // Access atomically.
    pub readers: AtomicI64,

    // The number of active writes for this pipe.
    //
    // Access atomically.
    pub writers: AtomicI64,

    pub intern: Mutex<PipeInternal>,

    pub rWakeup: Cond,
    pub wWakeup: Cond,

}

#[derive(Clone)]
pub struct Pipe(Arc<PipeIn>);

impl Deref for Pipe {
    type Target =Arc<PipeIn>;

    fn deref(&self) -> &Arc<PipeIn> {
        &self.0
    }
}

// NewPipe initializes and returns a pipe.
impl Pipe {
    pub fn New(task: &Task, isNamed: bool, sizeBytes: usize, atomicIOBytes: usize) -> Self {
        let sizeBytes = if sizeBytes < MINIMUM_PIPE_SIZE {
            MINIMUM_PIPE_SIZE
        } else {
            sizeBytes
        };

        let mut atomicIOBytes = if atomicIOBytes == 0 {
            1
        } else {
            atomicIOBytes
        };

        if atomicIOBytes > sizeBytes {
            atomicIOBytes = sizeBytes;
        }

        let p = Self(Arc::new(PipeIn {
            queue: Queue::default(),
            isNamed: isNamed,
            atomicIOBytes: atomicIOBytes,
            readers: AtomicI64::new(0),
            writers: AtomicI64::new(0),
            rWakeup: Cond::default(),
            wWakeup: Cond::default(),
            intern: Mutex::new(PipeInternal{
                id : super::super::super::uid::NewUID(),
                max: sizeBytes,
                ..Default::default()
            }),
        }));

        // Build the fs.Dirent of this pipe, shared by all fs.Files associated
        // with this pipe.
        let perms = FilePermissions {
            User: PermMask {read: true, write: true, execute: false},
            ..Default::default()
        };

        let iops = Arc::new(NewPipeInodeOps(task, &perms, p.clone()));
        let deviceId = TMPFS_DEVICE.lock().DeviceID();
        let inodeId = TMPFS_DEVICE.lock().NextIno();
        let attr = StableAttr {
            Type: InodeType::Pipe,
            DeviceId: deviceId,
            InodeId: inodeId,
            BlockSize: atomicIOBytes as i64,
            DeviceFileMajor: 0,
            DeviceFileMinor: 0,
        };

        let ms = Arc::new(Mutex::new(MountSource::NewPseudoMountSource()));
        let inode = Inode::New(&iops, &ms, &attr);
        let dirent = Dirent::New(&inode, &format!("pipe:[{}]", inodeId));
        p.intern.lock().dirent = Some(dirent);

        return p
    }

    pub fn Notify(&self, mask: EventMask) {
        self.queue.Notify(mask)
    }

    pub fn Uid(&self) -> u64 {
        return self.intern.lock().id;
    }

    pub fn Readers(&self) -> i64 {
        self.readers.load(Ordering::SeqCst)
    }

    pub fn Writers(&self) -> i64 {
        self.writers.load(Ordering::SeqCst)
    }

    // Open opens the pipe and returns a new file.
    //
    // Precondition: at least one of flags.Read or flags.Write must be set.
    pub fn Open(&self, _task: &Task, flags: &FileFlags) -> File {
        if flags.Read && flags.Write {
            self.ROpen();
            self.WOpen();
            let rw = ReaderWriter {pipe: self.clone()};
            let dirent = self.intern.lock().dirent.clone().unwrap();
            return File::New(&dirent, flags, rw);
        } else if flags.Read {
            self.ROpen();
            let r = Reader {pipe: self.clone()};
            let dirent = self.intern.lock().dirent.clone().unwrap();
            return File::New(&dirent, flags, r);
        } else if flags.Write {
            self.WOpen();
            let w = Writer {pipe: self.clone()};
            let dirent = self.intern.lock().dirent.clone().unwrap();
            return File::New(&dirent, flags, w);
        } else {
            // Precondition violated.
            panic!("invalid pipe flags")
        }
    }

    // read reads data from the pipe into dst and returns the number of bytes
    // read, or returns ErrWouldBlock if the pipe is empty.
    //
    // Precondition: this pipe must have readers.
    pub fn Read(&self, _task: &Task, dst: BlockSeq) -> Result<usize> {
        // Don't block for a zero-length read even if the pipe is empty.
        if dst.NumBytes() == 0 {
            return Ok(0)
        }

        let mut p = self.intern.lock();
        let mut dst = dst;
        // Is the pipe empty?
        //info!("pipe::read id is {} p.size is {}, writers is {}", p.id, p.size, self.Writers());
        if p.size == 0 {
            if !self.HasWriters() {
                // There are no writers, return EOF.
                return Ok(0)
            }

            return Err(Error::SysError(SysErr::EAGAIN))
        }

        // Limit how much we consume.
        if dst.NumBytes() as usize > p.size {
            dst = dst.TakeFirst(p.size as u64);
        }

        let mut done = 0;
        while dst.NumBytes() > 0 {
            let mut needPop = false;
            let n;
            {
                // Pop the first buffer.
                let first = match p.data.front_mut() {
                    None => break,
                    Some(f) => f,
                };

                // Copy user data.
                n = dst.CopyOutFrom(first)?;
                done += n;
                dst = dst.DropFirst(n as u64);

                // Empty buffer?
                if first.borrow().Empty() {
                    needPop = true;
                }
            }

            p.size -= n;
            if needPop {
                // Push to the free list.
                let v = p.data.pop_front().unwrap();
                ReturnBuff(v);
            }
        }

        return Ok(done)
    }

    pub fn ReadFrom(&self, task: &Task, src: &File, opts: &SpliceOpts) -> Result<usize> {
        if opts.DstOffset {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if opts.SrcOffset && !src.FileOp.Seekable() {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let len = {
            let p = self.intern.lock();
            // Can't write to a pipe with no readers.
            if !self.HasReaders() {
                return Err(Error::SysError(SysErr::EPIPE))
            }

            let mut len = p.Available() as usize;

            if len == 0 {
                return Err(Error::SysError(SysErr::EAGAIN))
            }

            if len > opts.Length as usize {
                len = opts.Length as usize
            }

            len
        };


        let mut buf = Vec::with_capacity(len);
        buf.resize(len, 0);
        let dst = IoVec::New(&buf);
        let mut iovs = [dst];
        //let src = BlockSeq::New(&buf);

        let readCount = if opts.SrcOffset {
            src.Preadv(task, &mut iovs, opts.SrcStart)?
        } else {
            src.Readv(task, &mut iovs)?
        };

        let src = BlockSeq::New(&buf[0..readCount as usize]);
        let writeCount = self.intern.lock().Write(task, src, self.atomicIOBytes)? as usize;

        assert!(readCount as usize == writeCount);
        return Ok(writeCount)
    }

    // write writes data from sv into the pipe and returns the number of bytes
    // written. If no bytes are written because the pipe is full (or has less than
    // atomicIOBytes free capacity), write returns ErrWouldBlock.
    //
    // Precondition: this pipe must have writers.
    pub fn Write(&self, task: &Task, src: BlockSeq) -> Result<usize> {
        let mut p = self.intern.lock();

        // Can't write to a pipe with no readers.
        if !self.HasReaders() {
            return Err(Error::SysError(SysErr::EPIPE))
        }

        return p.Write(task, src, self.atomicIOBytes)
    }

    // rOpen signals a new reader of the pipe.
    pub fn ROpen(&self) {
        self.readers.fetch_add(1, Ordering::SeqCst);
    }

    // wOpen signals a new writer of the pipe.
    pub fn WOpen(&self) {
        self.intern.lock().hadWriter = true;
        self.writers.fetch_add(1, Ordering::SeqCst);
    }

    // rClose signals that a reader has closed their end of the pipe.
    pub fn RClose(&self) {
        let readers = self.readers.fetch_sub(1, Ordering::SeqCst);
        //error!("pipe [{}] rclose readers is {}", self.Uid(), readers);

        if readers <= 0 {
            panic!("Refcounting bug, pipe has negative readers: {}", readers-1)
        }

        if readers == 1 {
            self.rWakeup.Reset();
        }
    }

    // wClose signals that a writer has closed their end of the pipe.
    pub fn WClose(&self) {
        let writers = self.writers.fetch_sub(1, Ordering::SeqCst);
        //error!("pipe [{}] WClose readers is {}", self.Uid(), writers);
        if writers <= 0 {
            panic!("Refcounting bug, pipe has negative writers: {}", writers-1)
        }

        if writers == 1 {
            self.wWakeup.Reset();
        }
    }

    // HasReaders returns whether the pipe has any active readers.
    pub fn HasReaders(&self) -> bool {
        self.readers.load(Ordering::SeqCst) > 0
    }

    // HasWriters returns whether the pipe has any active writers.
    pub fn HasWriters(&self) -> bool {
        self.writers.load(Ordering::SeqCst) > 0
    }

    // rReadinessLocked calculates the read readiness.
    pub fn RReadinessLocked(&self, intern: &MutexGuard<PipeInternal>) -> EventMask {
        let mut ready = 0;

        if self.HasReaders() && intern.data.len() > 0 {
            ready |= EVENT_IN;
        }

        if !self.HasWriters() && intern.hadWriter {
            // POLLHUP must be suppressed until the pipe has had at least one writer
            // at some point. Otherwise a reader thread may poll and immediately get
            // a POLLHUP before the writer ever opens the pipe, which the reader may
            // interpret as the writer opening then closing the pipe.
            ready |= EVENT_HUP;
        }

        return ready;
    }

    // rReadiness returns a mask that states whether the read end of the pipe is
    // ready for reading.
    pub fn RReadiness(&self) -> EventMask {
        let intern = self.intern.lock();
        return self.RReadinessLocked(&intern)
    }

    // wReadinessLocked calculates the write readiness.
    pub fn WReadinessLocked(&self, intern: &MutexGuard<PipeInternal>) -> EventMask {
        let mut ready = 0;
        if self.HasWriters() && intern.size < intern.max {
            ready |= EVENT_OUT;
        }

        if !self.HasReaders() {
            ready |= EVENT_ERR;
        }

        return ready;
    }

    // rwReadiness returns a mask that states whether a read-write handle to the
    // pipe is ready for IO.
    pub fn RWReadiness(&self) -> EventMask {
        let intern = self.intern.lock();
        return self.RReadinessLocked(&intern) | self.WReadinessLocked(&intern)
    }

    // wReadiness returns a mask that states whether the write end of the pipe
    // is ready for writing.
    pub fn WReadiness(&self) -> EventMask {
        let intern = self.intern.lock();
        return self.WReadinessLocked(&intern)
    }

    // queued returns the amount of queued data.
    pub fn Queued(&self) -> usize {
        return self.intern.lock().size;
    }

    // PipeSize implements PipeSizer.PipeSize.
    pub fn PipeSize(&self) -> usize {
        return self.intern.lock().max;
    }

    // SetPipeSize implements PipeSize.SetPipeSize.
    pub fn SetPipeSize(&self, size: i64) -> Result<usize>  {
        if size < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mut size = size as usize;
        if size < MINIMUM_PIPE_SIZE {
            size = MINIMUM_PIPE_SIZE;
        }

        if size > MAXIMUM_PIPE_SIZE {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mut intern = self.intern.lock();
        if size < intern.size {
            return Err(Error::SysError(SysErr::EBUSY))
        }

        intern.max = size;
        return Ok(size)
    }
}
