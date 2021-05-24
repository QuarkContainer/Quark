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

use alloc::slice;
use alloc::vec::Vec;

use super::super::common::*;
use super::super::linux_def::*;

pub trait BlockSeqReader {
    // ReadToBlocks reads up to dsts.NumBytes() bytes into dsts and returns the
    // number of bytes read. It may return a partial read without an error
    // (i.e. (n, nil) where 0 < n < dsts.NumBytes()). It should not return a
    // full read with an error (i.e. (dsts.NumBytes(), err) where err != nil);
    // note that this differs from io.Reader.Read (in particular, io.EOF should
    // not be returned if ReadToBlocks successfully reads dsts.NumBytes()
    // bytes.)
    fn ReadToBlocks(&mut self, dsts: BlockSeq) -> Result<usize>;
}

pub trait BlockSeqWriter {
    // WriteFromBlocks writes up to srcs.NumBytes() bytes from srcs and returns
    // the number of bytes written. It may return a partial write without an
    // error (i.e. (n, nil) where 0 < n < srcs.NumBytes()). It should not
    // return a full write with an error (i.e. srcs.NumBytes(), err) where err
    // != nil).
    fn WriteFromBlocks(&mut self, srcs: BlockSeq) -> Result<usize>;
}

#[derive(Clone, Default, Debug, Copy)]
pub struct BlockSeq {
    data: u64,
    len: i32,
    off: i32,
    limit: u64,
}

impl BlockSeq {
    pub fn New(buf: &[u8]) -> Self {
        let block = IoVec::NewFromSlice(&buf);
        return Self::NewFromBlock(block)
    }

    pub fn NewFromBlock(b: IoVec) -> Self {
        let bs = Self {
            data: b.start,
            len: -1,
            off: 0,
            limit: b.len as u64,
        };

        return bs;
    }

    pub fn CopyOutFrom(&self, src: &mut BlockSeqReader) -> Result<usize> {
        return src.ReadToBlocks(*self)
    }

    pub fn CopyInTo(&mut self, dst: &mut BlockSeqWriter) -> Result<usize> {
        return dst.WriteFromBlocks(*self)
    }

    pub fn ToVec(&self) -> Vec<u8> {
        let mut ret = Vec::with_capacity(self.limit as usize);
        ret.resize(self.limit as usize, 0);
        self.CopyIn(&mut ret);

        return ret;
    }

    pub fn NewFromSlice(slice: &[IoVec]) -> Self {
        let slice = SkipEmty(slice);
        let mut limit: u64 = 0;

        for b in slice {
            limit += b.len as u64;
        }

        return Self::blockSeqFromSliceLimited(slice, limit)
    }

    pub fn ToIoVecs(&self) -> Vec<IoVec> {
        let mut ret = Vec::new();
        let mut bs = *self;

        while !bs.IsEmpty() {
            let first = bs.Head();
            bs = bs.Tail();
            ret.push(first);
        }

        return ret;
    }

    pub fn ToBlocks(iovs: &[IoVec]) -> Vec<IoVec> {
        let mut blocks = Vec::new();
        for iov in iovs {
            let b = IoVec::NewFromAddr(iov.start, iov.len);
            blocks.push(b);
        }
        return blocks;
    }

    pub fn blockSeqFromSliceLimited(slice: &[IoVec], limit: u64) -> Self {
        let len = slice.len();

        if len == 0 {
            return Self::default();
        } else if len == 1 {
            return Self::NewFromBlock(slice[0].TakeFirst(limit as usize))
        } else {
            return Self {
                data: &slice[0] as *const _ as u64,
                len: len as i32,
                off: 0,
                limit: limit,
            }
        }
    }

    pub fn IsEmpty(&self) -> bool {
        return self.len == 0
    }

    pub fn Len(&self) -> usize {
        return self.limit as usize;
    }

    pub fn NumBlocks(&self) -> u64 {
        let mut bs = *self;

        let mut res = 0;
        while !bs.IsEmpty() {
            res += 1;
            bs = bs.Tail();
        }

        return res;
    }

    pub fn NumBytes(&self) -> u64 {
        return self.limit
    }

    pub fn Head(&self) -> IoVec {
        assert!(self.len != 0, "empty blockseq");

        if self.len < 0 {
            return self.InternalBlock()
        }

        let b = unsafe {
            *(self.data as *const IoVec)
        };

        return b.DropFirst(self.off as usize).TakeFirst(self.limit as usize)
    }

    fn InternalBlock(&self) -> IoVec {
        //assert!(self.len<0)
        return IoVec {
            start: self.data,
            len: self.limit as usize,
        }
    }

    pub fn Tail(&self) -> Self {
        assert!(self.len != 0, "empty BlockSeq");

        if self.len < 0 {
            return Self::default()
        }

        let ptr = self.data as *const IoVec;
        let slice = unsafe { slice::from_raw_parts(ptr, self.len as usize) };

        let head = slice[0].DropFirst(self.off as usize);
        let headLen = head.len as u64;
        if headLen > self.limit {
            return Self::default();
        }

        let tailSlice = SkipEmty(&slice[1..]);
        let tailLimit = self.limit - headLen;
        return Self::blockSeqFromSliceLimited(tailSlice, tailLimit)
    }

    pub fn DropFirst(&self, n: u64) -> Self {
        if n >= self.limit {
            return Self::default()
        }

        if self.len < 0 {
            return Self::NewFromBlock(self.InternalBlock().DropFirst(n as usize))
        }

        let ptr = self.data as *const IoVec;
        let slice = unsafe { slice::from_raw_parts(ptr, self.len as usize) };
        let mut idx = 0;
        let mut n = n;
        let limit = self.limit - n;

        loop {
            let b = slice[idx];
            if n < b.len as u64 {
                let mut res = Self::NewFromSlice(&slice[idx..]);
                res.limit = limit;
                res.off = n as i32;

                return res;
            }

            if idx == 0 {
                n = n - (b.len as u64 - self.off as u64);
            } else {
                n = n - b.len as u64;
            }

            idx += 1;
        }
    }

    pub fn TakeFirst(&self, n: u64) -> Self {
        if n == 0 {
            return Self::default();
        }

        let mut ret = *self;

        if self.limit > n {
            ret.limit = n
        }

        return ret;
    }

    pub fn CopyOut(&self, src: &[u8]) -> usize {
        let srcs = BlockSeq::New(src);
        return Self::Copy(*self, srcs) as usize
    }

    pub fn CopyIn(&self, dst: &mut [u8]) -> usize {
        let dsts = BlockSeq::New(dst);
        return Self::Copy(dsts, *self) as usize
    }

    pub fn Copy(dsts: Self, srcs: Self) -> i64 {
        let mut dsts = dsts;
        let mut srcs = srcs;

        let mut done = 0;

        if dsts.IsEmpty() || srcs.IsEmpty() {
            return 0;
        }

        loop {
            let mut dst = dsts.Head();
            let src = srcs.Head();

            let n = dst.CopyFrom(&src);

            done += n;

            dsts = dsts.DropFirst(n as u64);
            srcs = srcs.DropFirst(n as u64);

            if dsts.IsEmpty() || srcs.IsEmpty() {
                return done;
            }
        }
    }

    pub fn Zero(dsts: Self) -> i64 {
        let mut done = 0;
        let mut dsts = dsts;

        while !dsts.IsEmpty() {
            let n = dsts.Head().Zero();
            done += n;

            dsts = dsts.DropFirst(n as u64);
        }

        return done;
    }
}

pub fn BlockSeqToIoVecs(bs: BlockSeq) -> Vec<IoVec> {
    let mut iovs = Vec::with_capacity(bs.NumBlocks() as usize);

    let mut bs = bs;
    while !bs.IsEmpty() {
        let iov = bs.Head();

        iovs.push(iov);

        bs = bs.Tail();
    }

    return iovs;
}

fn SkipEmty(slice: &[IoVec]) -> &[IoVec] {
    for i in 0..slice.len() {
        let b = slice[i];
        if b.len != 0 {
            return &slice[i..]
        }
    }

    return &slice[..0]
}