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

use super::super::fs::attr::*;
use super::super::fs::file::*;
use super::super::kernel::waiter::qlock::*;
use super::super::kernel::waiter::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::mem::block::*;
use super::super::qlib::addr::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use kernel::pipe::node::PipeIops;
use qlib::mem::seq::BlockSeq;
use kernel::pipe::pipe::Pipe;

// Splice moves data to this file, directly from another.
//
// Offsets are updated only if DstOffset and SrcOffset are set.
pub fn Splice(task: &Task, dst: &File, src: &File, opts: &mut SpliceOpts) -> Result<i64> {
    // Verify basic file flag permissions.
    if !dst.Flags().Write || !src.Flags().Read {
        return Err(Error::SysError(SysErr::EBADF));
    }

    // Check whether or not the objects being sliced are stream-oriented
    // (i.e. pipes or sockets). If yes, we elide checks and offset locks.
    let srcInode = src.Dirent.Inode();
    let srcType = srcInode.StableAttr();
    let dstInode = dst.Dirent.Inode();
    let dstType = dstInode.StableAttr();

    let srcPipe = srcType.IsPipe() || srcType.IsSocket();
    let dstPipe = dstType.IsPipe() || dstType.IsSocket();

    let srcTmp = QLock::New(0);
    let dstTmp = QLock::New(0);

    let mut srcLock = srcTmp.Lock(task)?;
    let mut dstLock = dstTmp.Lock(task)?;

    if !dstPipe && !opts.DstOffset && !srcPipe && !opts.SrcOffset {
        if dst.UniqueId() < src.UniqueId() {
            dstLock = dst.offset.Lock(task)?;
            srcLock = src.offset.Lock(task)?;
            // Use both offsets (locked).
            opts.DstStart = *dstLock;
            opts.SrcStart = *srcLock;
        } else if dst.UniqueId() > src.UniqueId() {
            srcLock = src.offset.Lock(task)?;
            dstLock = dst.offset.Lock(task)?;
            // Use both offsets (locked).
            opts.DstStart = *dstLock;
            opts.SrcStart = *srcLock;
        } else {
            srcLock = src.offset.Lock(task)?;
            opts.DstStart = *srcLock;
            opts.SrcStart = *srcLock;
        }
    } else if !dstPipe && !opts.DstOffset {
        dstLock = dst.offset.Lock(task)?;
        opts.DstStart = *dstLock;
    } else if !srcPipe && !opts.SrcOffset {
        srcLock = src.offset.Lock(task)?;
        opts.SrcStart = *dstLock;
    }

    // Check append-only mode and the limit.
    if !dstPipe {
        if dst.Flags().Append {
            if opts.DstOffset {
                // We need to acquire the lock.
                dstLock = dst.offset.Lock(task)?;
            }

            // Figure out the appropriate offset to use.
            opts.DstStart = dst.offsetForAppend(task)?;
        }

        // Enforce file limits.
        let (limit, ok) = dst.checkLimit(opts.DstStart);
        if ok && limit == 0 {
            return Err(Error::ErrExceedsFileSizeLimit);
        } else if ok && limit < opts.DstStart {
            opts.Length = limit; // Cap the write.
        }
    }

    let n = match src.FileOp.WriteTo(task, src, dst, &opts) {
        Err(Error::SysError(SysErr::ENOSYS)) => {
            // Attempt as a ReadFrom. If a WriteTo, a ReadFrom may also
            // be more efficient than a copy if buffers are cached or readily
            // available. (It's unlikely that they can actually be donate
            match dst.FileOp.ReadFrom(task, dst, src, &opts) {
                Err(Error::SysError(SysErr::ENOSYS)) => {
                    // If we've failed up to here, and at least one of the sources
                    // is a pipe or socket, then we can't properly support dup.
                    // Return an error indicating that this operation is not
                    // supported.
                    if (srcPipe && dstPipe) || opts.Dup {
                        return Err(Error::SysError(SysErr::EINVAL));
                    }

                    // We failed to splice the files. But that's fine; we just fall
                    // back to a slow path in this case. This copies without doing
                    // any mode changes, so should still be more efficient.

                    let bufLen = if opts.Length > 2 * MemoryDef::ONE_MB as i64 {
                        2 * MemoryDef::ONE_MB as i64
                    } else {
                        opts.Length
                    };

                    let buf = DataBuff::New(bufLen as usize);
                    let mut copyLen = 0;
                    let srcStart = opts.SrcStart;
                    let dstStart = opts.DstStart;

                    while copyLen < opts.Length {
                        let mut iovs = buf.Iovs(bufLen as usize);
                        let readLen = match ReadAt(task, src, &mut iovs, srcStart + copyLen) {
                            Err(e) => {
                                if copyLen > 0 {
                                    return Ok(copyLen)
                                }
                                return Err(e)
                            }
                            Ok(n) => {
                                if n == 0 {
                                    break;
                                }
                                n
                            }
                        };

                        let iovs = Iovs(&iovs).First(readLen as usize);
                        match WriteAt(task, dst, &iovs, dstStart + copyLen) {
                            Err(e) => {
                                if copyLen > 0 {
                                    return Ok(copyLen)
                                }
                                return Err(e)
                            }
                            Ok(n) => {
                                copyLen += n;
                                if n == 0 {
                                    break;
                                }
                            }
                        };
                    }

                    copyLen
                }
                Err(e) => return Err(e),
                Ok(n) => n,
            }
        }
        Err(e) => return Err(e),
        Ok(n) => n,
    };

    if n > 0 {
        if !dstPipe && !opts.DstOffset {
            *dstLock += n;
        }

        if !srcPipe && !opts.SrcOffset {
            *srcLock += n;
        }
    }

    return Ok(n);
}

pub fn RepWriteAt(task: &Task, f: &File, srcs: &[IoVec], offset: i64) -> Result<i64> {
    let len = Iovs(srcs).Count();
    let mut count = 0;
    let mut srcs = srcs;
    let mut tmp;

    loop {
        match f.FileOp.WriteAt(task, f, &srcs, offset + count, false) {
            Err(e) => {
                if count > 0 {
                    return Ok(count);
                }

                return Err(e);
            }
            Ok(n) => {
                count += n;
                if count == len as i64 {
                    return Ok(count);
                }

                tmp = Iovs(srcs).DropFirst(n as usize);
                srcs = &tmp;
            }
        }
    }
}

pub fn WriteAt(task: &Task, f: &File, srcs: &[IoVec], offset: i64) -> Result<i64> {
    let wouldBlock = f.WouldBlock();
    if !wouldBlock {
        return RepWriteAt(task, f, srcs,  offset)
    }

    let general = task.blocker.generalEntry.clone();
    f.EventRegister(task, &general, EVENT_WRITE);
    defer!(f.EventUnregister(task, &general));

    let len = Iovs(srcs).Count();
    let mut count = 0;
    let mut srcs = srcs;
    let mut tmp;

    loop {
        match f.Writev(task, srcs) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
            Err(e) => {
                if count > 0 {
                    return Ok(count);
                }

                return Err(e);
            }
            Ok(n) => {
                count += n;
                if count == len as i64 {
                    return Ok(count);
                }

                tmp = Iovs(srcs).DropFirst(n as usize);
                srcs = &tmp;
            }
        }

        match task.blocker.BlockWithMonoTimer(true, None) {
            Err(e) => {
                if count > 0 {
                    return Ok(count);
                }
                return Err(e);
            }
            _ => (),
        }
    }
}

pub fn ReadAt(task: &Task, f: &File, dsts: &mut [IoVec], offset: i64) -> Result<i64> {
    let wouldBlock = f.WouldBlock();
    if !wouldBlock {
       return f
           .FileOp
           .ReadAt(task, f, &mut dsts[..], offset, false);
    }

    let general = task.blocker.generalEntry.clone();
    f.EventRegister(task, &general, EVENT_READ);
    defer!(f.EventUnregister(task, &general));

    loop {
        match f
            .FileOp
            .ReadAt(task, f, &mut dsts[..], offset, false) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
                return Ok(n)
            }
        }

        match task.blocker.BlockWithMonoTimer(true, None) {
            Err(Error::ErrInterrupted) => {
                return Err(Error::SysError(SysErr::ERESTARTSYS));
            }
            Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                return Err(Error::SysError(SysErr::EAGAIN));
            }
            Err(e) => {
                return Err(e);
            }
            _ => (),
        }
    }
}


// doSplice implements a blocking splice operation.
pub fn DoSplice(
    task: &Task,
    dstFile: &File,
    srcFile: &File,
    opts: &mut SpliceOpts,
    nonBlocking: bool,
) -> Result<i64> {
    let general = task.blocker.generalEntry.clone();

    loop {
        let mut inW = false;
        let mut outW = false;
        if !inW && srcFile.Readiness(task, EVENT_READ) == 0 && !srcFile.Flags().NonBlocking {
            srcFile.EventRegister(task, &general, EVENT_READ);
            inW = true;
        } else if !outW && dstFile.Readiness(task, EVENT_WRITE) == 0 && !dstFile.Flags().NonBlocking
        {
            dstFile.EventRegister(task, &general, EVENT_WRITE);
            outW = true;
        }

        defer!({
            if inW {
                srcFile.EventUnregister(task, &general)
            }

            if outW {
                dstFile.EventUnregister(task, &general)
            }
        });

        match Splice(task, dstFile, srcFile, opts) {
            Err(e) => {
                if e != Error::SysError(SysErr::EWOULDBLOCK) {
                    return Err(e);
                }

                if e == Error::SysError(SysErr::EWOULDBLOCK) && nonBlocking {
                    return Err(e);
                }
            }
            Ok(n) => {
                if n > 0 {
                    // On Linux, inotify behavior is not very consistent with splice(2). We try
                    // our best to emulate Linux for very basic calls to splice, where for some
                    // reason, events are generated for output files, but not input files.
                    srcFile.Dirent.InotifyEvent(InotifyEvent::IN_ACCESS, 0);
                    dstFile.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0);
                }
                return Ok(n)
            },
        }


        // Was anything registered? If no, everything is non-blocking.
        if !inW && !outW {
            return Err(Error::SysError(SysErr::EWOULDBLOCK));
        }

        // Block until there's data.
        match task.blocker.BlockWithMonoTimer(true, None) {
            Err(Error::ErrInterrupted) => {
                return Err(Error::SysError(SysErr::ERESTARTNOHAND));
            }
            Err(e) => {
                return Err(e);
            }
            _ => (),
        }
    }
}

pub const SPLICE_F_MOVE: i32 = 1 << 0;
pub const SPLICE_F_NONBLOCK: i32 = 1 << 1;
pub const SPLICE_F_MORE: i32 = 1 << 2;
pub const SPLICE_F_GIFT: i32 = 1 << 3;

pub fn SysSplice(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let inFD = args.arg0 as i32;
    let inOffset = args.arg1 as u64;
    let outFD = args.arg2 as i32;
    let outOffset = args.arg3 as u64;
    let count = args.arg4 as i64;
    let flags = args.arg5 as i32;

    // Check for invalid flags.
    if flags & !(SPLICE_F_MOVE | SPLICE_F_NONBLOCK | SPLICE_F_MORE | SPLICE_F_GIFT) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // Only non-blocking is meaningful. Note that unlike in Linux, this
    // flag is applied consistently. We will have either fully blocking or
    // non-blocking behavior below, regardless of the underlying files
    // being spliced to. It's unclear if this is a bug or not yet.
    let nonBlocking = (flags & SPLICE_F_NONBLOCK) != 0;

    let dst = task.GetFile(outFD)?;
    let src = task.GetFile(inFD)?;

    // Construct our options.
    //
    // Note that exactly one of the underlying buffers must be a pipe. We
    // don't actually have this constraint internally, but we enforce it
    // for the semantics of the call.
    let mut opts = SpliceOpts {
        Length: count,
        ..Default::default()
    };

    let srcInode = src.Dirent.Inode();
    let srcAttr = srcInode.StableAttr();
    let dstInode = dst.Dirent.Inode();
    let dstAttr = dstInode.StableAttr();

    let srcPipe = srcAttr.IsPipe() || srcAttr.IsSocket();
    let dstPipe = dstAttr.IsPipe() || dstAttr.IsSocket();

    if srcPipe && !dstPipe {
        if inOffset != 0 {
            return Err(Error::SysError(SysErr::ESPIPE));
        }

        if outOffset != 0 {
            let offset: i64 = if outOffset != 0 {
                opts.DstOffset = true;
                task.CopyInObj(outOffset)?
            } else {
                0
            };

            // Use the destination offset.
            opts.DstStart = offset;
        }
    } else if !srcPipe && dstPipe {
        if outOffset != 0 {
            return Err(Error::SysError(SysErr::ESPIPE));
        }

        let offset: i64 = if inOffset != 0 {
            opts.SrcOffset = true;
            task.CopyInObj(inOffset)?
        } else {
            0
        };

        // Use the source offset.
        opts.SrcStart = offset;
    } else if srcPipe && dstPipe {
        if inOffset != 0 || outOffset != 0 {
            return Err(Error::SysError(SysErr::ESPIPE));
        }
    } else {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // We may not refer to the same pipe; otherwise it's a continuous loop.
    if srcAttr.DeviceId == dstAttr.DeviceId && srcAttr.InodeId == dstAttr.InodeId {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    return DoSplice(task, &dst, &src, &mut opts, nonBlocking);
}

pub fn SysTee(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let inFD = args.arg0 as i32;
    let outFD = args.arg1 as i32;
    let count = args.arg2 as i64;
    let flags = args.arg3 as i32;

    let MAX_RW_COUNT = Addr(i32::MAX as u64).RoundDown().unwrap().0 as i64;

    if count == 0 {
        return Ok(0)
    }

    let count = if count > MAX_RW_COUNT {
        MAX_RW_COUNT
    } else {
        count
    };

    if count < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // Check for invalid flags.
    if flags & !(SPLICE_F_MOVE | SPLICE_F_NONBLOCK | SPLICE_F_MORE | SPLICE_F_GIFT) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let dst = task.GetFile(outFD)?;
    let src = task.GetFile(inFD)?;

    if !dst.Writable() || !src.Readable() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let srcInode = src.Dirent.Inode();
    let srcAttr = srcInode.StableAttr();
    let dstInode = dst.Dirent.Inode();
    let dstAttr = dstInode.StableAttr();

    if !srcAttr.IsPipe() || !dstAttr.IsPipe() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // The operation is non-blocking if anything is non-blocking.
    //
    // N.B. This is a rather simplistic heuristic that avoids some
    // poor edge case behavior since the exact semantics here are
    // underspecified and vary between versions of Linux itself.
    let nonblock = !dst.WouldBlock() || !src.WouldBlock() || flags & SPLICE_F_NONBLOCK != 0;

    let srcIops = srcInode.lock().InodeOp.clone();
    let dstIops = dstInode.lock().InodeOp.clone();

    let srcIops = srcIops.as_any().downcast_ref::<PipeIops>().unwrap();
    let dstIops = dstIops.as_any().downcast_ref::<PipeIops>().unwrap();

    let srcPipe = srcIops.lock().p.clone();
    let dstPipe = dstIops.lock().p.clone();

    if srcPipe == dstPipe {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let srcTmp = QLock::New(0);
    let dstTmp = QLock::New(0);

    let mut _srcLock = srcTmp.Lock(task)?;
    let mut _dstLock = dstTmp.Lock(task)?;
    if dst.UniqueId() < src.UniqueId() {
        _dstLock = dst.offset.Lock(task)?;
        _srcLock = src.offset.Lock(task)?;
    } else if dst.UniqueId() > src.UniqueId() {
        _srcLock = src.offset.Lock(task)?;
        _dstLock = dst.offset.Lock(task)?;
    } else {
        _srcLock = src.offset.Lock(task)?;
    }

    let bufLen = if count > 2 * MemoryDef::ONE_MB as i64 {
        2 * MemoryDef::ONE_MB as i64
    } else {
        count
    };

    let buf = DataBuff::New(bufLen as usize);
    let bs = BlockSeq::New(&buf.buf);

    let n = ReadWithoutConsume(task, &src, &srcPipe, bs, !nonblock)?;

    let iovs = buf.Iovs(n as usize);

    let count = WritePipe(task, &dst, &iovs, !nonblock)?;

    if count > 0 {
        // On Linux, inotify behavior is not very consistent with splice(2). We try
        // our best to emulate Linux for very basic calls to splice, where for some
        // reason, events are generated for output files, but not input files.
        src.Dirent.InotifyEvent(InotifyEvent::IN_ACCESS, 0);
        dst.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0);
    }
    return Ok(count)
}

pub fn ReadWithoutConsume(task: &Task, f: &File, p: &Pipe, bs: BlockSeq, blocking: bool) -> Result<i64> {
    let general = task.blocker.generalEntry.clone();
    f.EventRegister(task, &general, EVENT_READ);
    defer!(f.EventUnregister(task, &general));

    loop {
        match p
            .ReadWithoutConsume(task, bs) {
            Err(Error::SysError(SysErr::EAGAIN)) => {
                if !blocking {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
            },
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
                return Ok(n as i64)
            }
        }

        match task.blocker.BlockWithMonoTimer(true, None) {
            Err(Error::ErrInterrupted) => {
                return Err(Error::SysError(SysErr::ERESTARTSYS));
            }
            Err(e) => {
                return Err(e);
            }
            _ => (),
        }
    }
}

pub fn WritePipe(task: &Task, f: &File, srcs: &[IoVec], blocking: bool) -> Result<i64> {
    let general = task.blocker.generalEntry.clone();
    f.EventRegister(task, &general, EVENT_WRITE);
    defer!(f.EventUnregister(task, &general));

    loop {
        match f.Writev(task, srcs) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => {
                if !blocking {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
            },
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
                return Ok(n)
            }
        }

        match task.blocker.BlockWithMonoTimer(true, None) {
            Err(e) => {
                return Err(e);
            }
            _ => (),
        }
    }
}

pub fn SysSendfile(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let outFD = args.arg0 as i32;
    let inFD = args.arg1 as i32;
    let offsetAddr = args.arg2 as u64;
    let count = args.arg3 as i64;

    let inFile = task.GetFile(inFD)?;
    if !inFile.Flags().Read {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let outFile = task.GetFile(outFD)?;
    if !outFile.Flags().Write {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if outFile.Flags().Append {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let inodeSrc = inFile.Dirent.Inode();
    if inodeSrc.InodeType() != InodeType::RegularFile {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let n;

    if offsetAddr != 0 {
        if !inFile.Flags().Pread {
            return Err(Error::SysError(SysErr::ESPIPE));
        }

        let offset: i64 = task.CopyInObj(offsetAddr)?;

        n = DoSplice(
            task,
            &outFile,
            &inFile,
            &mut SpliceOpts {
                Length: count,
                SrcOffset: true,
                SrcStart: offset,
                Dup: false,
                DstOffset: false,
                DstStart: 0,
            },
            outFile.Flags().NonBlocking,
        )?;

        //*task.GetTypeMut(offsetAddr)? = offset + n;
        task.CopyOutObj(&(offset + n), offsetAddr)?;
    } else {
        n = DoSplice(
            task,
            &outFile,
            &inFile,
            &mut SpliceOpts {
                Length: count,
                SrcOffset: false,
                SrcStart: 0,
                Dup: false,
                DstOffset: false,
                DstStart: 0,
            },
            outFile.Flags().NonBlocking,
        )?;
    }

    return Ok(n);
}
