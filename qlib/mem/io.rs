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

use super::seq::*;
use super::super::common::*;
use super::super::linux_def::*;

pub trait IOReader {
    fn Read(&mut self, buf: &mut [u8]) -> Result<i64>;
}

pub trait IORead {
    fn IORead(&mut self, buf: &mut [IoVec]) -> Result<i64>;
}

pub trait IOWriter {
    fn Write(&mut self, buf: &[u8]) -> Result<i64>;
}

pub trait IOWrite {
    fn IOWrite(&mut self, buf: &[IoVec]) -> Result<i64>;
}

pub trait IOReaderAt {
    fn ReadAt(&mut self, buf: &mut [u8], off: i64) -> Result<i64>;
}

pub trait IOReadAt {
    fn IOReadAt(&mut self, buf: &mut [IoVec], off: u64) -> Result<i64>;
}

pub trait IOWriterAt {
    fn WriteAt(&mut self, buf: &[u8], off: i64) -> Result<i64>;
}

pub trait IOWriteAt {
    fn IOWriteAt(&mut self, buf: &[IoVec], off: u64) -> Result<i64>;
}

pub trait BlockReader {
    fn ReadToBlocks(&mut self, dsts: BlockSeq) -> Result<i64>;
}

pub trait BlockWriter {
    fn WriteFromBlocks(&mut self, srcs: BlockSeq) -> Result<i64>;
}

pub fn ReadFullToBlocks(r: &mut BlockReader, dsts: BlockSeq) -> Result<i64> {
    let mut done = 0;
    let mut dsts = dsts;

    while !dsts.IsEmpty() {
        let n = r.ReadToBlocks(dsts)?;

        done += n;
        dsts = dsts.DropFirst(n as u64);
    }

    return Ok(done);
}

pub fn WriteFullFromBlocks(w: &mut BlockWriter, srcs: BlockSeq) -> Result<i64> {
    let mut done = 0;
    let mut srcs = srcs;

    while !srcs.IsEmpty() {
        let n = w.WriteFromBlocks(srcs)?;
        if n == 0 {
            return Ok(done);
        }
        done += n;

        srcs = srcs.DropFirst(n as u64);
    }

    return Ok(done);
}

pub struct BlockSeqReader(pub BlockSeq);

impl BlockReader for BlockSeqReader {
    fn ReadToBlocks(&mut self, dsts: BlockSeq) -> Result<i64> {
        let n = BlockSeq::Copy(dsts, self.0);
        self.0 = self.0.DropFirst(n as u64);
        return Ok(n);
    }
}

pub struct BlockSeqWriter(pub BlockSeq);

impl BlockWriter for BlockSeqWriter {
    fn WriteFromBlocks(&mut self, srcs: BlockSeq) -> Result<i64> {
        let n = BlockSeq::Copy(self.0, srcs);
        self.0 = self.0.DropFirst(n as u64);
        return Ok(n);
    }
}

pub struct ToIOReader<'a> {
    pub reader: &'a mut BlockReader,
}

impl<'a> IOReader for ToIOReader<'a> {
    fn Read(&mut self, dst: &mut [u8]) -> Result<i64> {
        let b = IoVec::NewFromSlice(dst);
        let seq = BlockSeq::NewFromBlock(b);
        let n = self.reader.ReadToBlocks(seq)?;
        return Ok(n)
    }
}

pub struct ToIOWriter<'a> {
    pub writer: &'a mut BlockWriter,
}

impl<'a> IOWriter for ToIOWriter<'a> {
    fn Write(&mut self, src: &[u8]) -> Result<i64> {
        let b = IoVec::NewFromSlice(src);
        let seq = BlockSeq::NewFromBlock(b);
        let n = WriteFullFromBlocks(self.writer, seq)?;
        return Ok(n)
    }
}

pub struct FromIOReader<'a> {
    pub reader: &'a mut IOReader
}

impl<'a> BlockReader for FromIOReader<'a> {
    fn ReadToBlocks(&mut self, dsts: BlockSeq) -> Result<i64> {
        let mut done = 0;
        let mut dsts = dsts;

        while !dsts.IsEmpty() {
            let mut dst = dsts.Head();
            dsts = dsts.Tail();

            let slice = dst.ToSliceMut();

            if slice.len() == 0 {
                continue;
            }

            let cnt = self.reader.Read(slice)?;

            if cnt == 0 {
                return Ok(done)
            }

            done += cnt;
        }

        Ok(done)
    }
}

pub struct FromIOWriter<'a> {
    pub writer: &'a mut IOWriter
}

impl<'a> BlockWriter for FromIOWriter<'a> {
    fn WriteFromBlocks(&mut self, srcs: BlockSeq) -> Result<i64> {
        let mut done = 0;
        let mut srcs = srcs;

        while !srcs.IsEmpty() {
            let src = srcs.Head();
            srcs = srcs.Tail();
            let slice = src.ToSlice();

            if slice.len() == 0 {
                continue;
            }

            let cnt = self.writer.Write(slice)?;
            if cnt == 0 {
                return Ok(done)
            }

            done += cnt;
        }

        Ok(done)
    }
}

pub struct FromIOReaderAt<'a> {
    pub reader: &'a mut IOReaderAt,
    pub offset: i64,
}

impl<'a> BlockReader for FromIOReaderAt<'a> {
    fn ReadToBlocks(&mut self, dsts: BlockSeq) -> Result<i64> {
        let mut done = 0;
        let mut dsts = dsts;

        while !dsts.IsEmpty() {
            let mut dst = dsts.Head();
            dsts = dsts.Tail();

            let slice = dst.ToSliceMut();

            if slice.len() == 0 {
                continue;
            }

            let cnt = self.reader.ReadAt(slice, self.offset)?;
            self.offset += cnt;

            if cnt == 0 {
                return Ok(done)
            }

            done += cnt;
        }

        Ok(done)
    }
}

pub struct FromIOWriterAt<'a> {
    pub writer: &'a mut IOWriterAt,
    pub offset: i64,
}

impl<'a> BlockWriter for FromIOWriterAt<'a> {
    fn WriteFromBlocks(&mut self, srcs: BlockSeq) -> Result<i64> {
        let mut done = 0;
        let mut srcs = srcs;

        while !srcs.IsEmpty() {
            let src = srcs.Head();
            srcs = srcs.Tail();
            let slice = src.ToSlice();

            if slice.len() == 0 {
                continue;
            }

            let cnt = self.writer.WriteAt(slice, self.offset)?;
            self.offset += cnt;

            if cnt == 0 {
                return Ok(done)
            }

            done += cnt;
        }

        Ok(done)
    }
}