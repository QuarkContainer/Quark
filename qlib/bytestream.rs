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

use alloc::alloc::{alloc, dealloc, Layout};
use alloc::slice;
use alloc::vec::Vec;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::Ordering;

use super::common::*;
use super::linux_def::*;

#[derive(Default, Debug)]
pub struct SocketBufIovs {
    pub iovs: [IoVec; 2],
    pub cnt: usize,
}

impl SocketBufIovs {
    pub fn Address(&self) -> (u64, usize) {
        return (&self.iovs[0] as *const _ as u64, self.cnt);
    }

    pub fn Iovs(&self) -> Vec<IoVec> {
        let mut iovs = Vec::with_capacity(self.cnt);
        for i in 0..self.cnt {
            iovs.push(self.iovs[i])
        }

        return iovs;
    }
}

pub enum RingeBufAllocator {
    HeapAllocator,
    ShareAllocator,
}

impl RingeBufAllocator {
    pub fn AllocHeadTail(&self) -> &'static [AtomicU32] {
        match self {
            Self::HeapAllocator => return HeapAllocator::AllocHeadTail(),
            _ => {
                panic!("ShareAllocator call ...");
            }
        }
    }

    pub fn FreeHeadTail(&self, data: &'static [AtomicU32]) {
        match self {
            Self::HeapAllocator => return HeapAllocator::FreeHeadTail(data),
            _ => {
                panic!("ShareAllocator call ...");
            }
        }
    }

    pub fn AlllocBuf(&self, pageCount: usize) -> u64 {
        match self {
            Self::HeapAllocator => return HeapAllocator::AlllocBuf(pageCount),
            _ => {
                panic!("ShareAllocator call ...");
            }
        }
    }

    pub fn FreeBuf(&self, addr: u64, size: usize) {
        match self {
            Self::HeapAllocator => return HeapAllocator::FreeBuf(addr, size),
            _ => {
                panic!("ShareAllocator call ...");
            }
        }
    }
}

pub fn IsPowerOfTwo(x: usize) -> bool {
    return (x & (x - 1)) == 0;
}

pub struct HeapAllocator {}

unsafe impl Send for HeapAllocator {}
unsafe impl Sync for HeapAllocator {}

impl HeapAllocator {
    pub fn AllocHeadTail() -> &'static [AtomicU32] {
        let layout = Layout::from_size_align(8, 8)
            .expect("RingeBufAllocator::AllocHeadTail can't allocate memory");
        let addr = unsafe { alloc(layout) };

        let ptr = addr as *mut AtomicU32;
        let slice = unsafe { slice::from_raw_parts(ptr, 2 as usize) };
        slice[0].store(0, Ordering::Release);
        slice[1].store(0, Ordering::Release);
        return slice;
    }

    pub fn FreeHeadTail(data: &'static [AtomicU32]) {
        assert!(data.len() == 2);
        let addr = &data[0] as *const _ as u64;
        let layout = Layout::from_size_align(8, 8)
            .expect("RingeBufAllocator::FreeHeadTail can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }

    pub fn AlllocBuf(pageCount: usize) -> u64 {
        assert!(IsPowerOfTwo(pageCount));
        let layout = Layout::from_size_align(
            pageCount * MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize,
        )
        .expect("RingeBufAllocator::AlllocBuf can't allocate memory");
        let addr = unsafe { alloc(layout) };

        return addr as u64;
    }

    pub fn FreeBuf(addr: u64, size: usize) {
        assert!(IsPowerOfTwo(size) && addr % MemoryDef::PAGE_SIZE == 0);
        let layout = Layout::from_size_align(size, MemoryDef::PAGE_SIZE as usize)
            .expect("RingeBufAllocator::FreeBuf can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }
}

pub struct RingBuf {
    pub buf: u64,
    pub ringMask: u32,
    pub headtail: &'static [AtomicU32],
    pub allocator: RingeBufAllocator,
}

impl Drop for RingBuf {
    fn drop(&mut self) {
        self.allocator.FreeHeadTail(self.headtail);
        self.allocator.FreeBuf(self.buf, self.Len());
    }
}

impl RingBuf {
    pub fn IsPowerOfTwo(x: usize) -> bool {
        return (x & (x - 1)) == 0;
    }

    pub fn New(pagecount: usize, allocator: RingeBufAllocator) -> Self {
        let headtail = allocator.AllocHeadTail();
        assert!(headtail.len() == 2);
        let buf = allocator.AlllocBuf(pagecount);

        return Self {
            buf: buf,
            ringMask: (pagecount * MemoryDef::PAGE_SIZE as usize - 1) as u32,
            headtail: headtail,
            allocator: allocator,
        };
    }

    //return (bufAddr, bufSize)
    pub fn GetRawBuf(&self) -> (u64, usize) {
        return (self.buf, self.Len());
    }

    #[inline]
    pub fn Len(&self) -> usize {
        return (self.ringMask + 1) as usize;
    }

    #[inline]
    pub fn Buf(&self) -> &'static mut [u8] {
        let ptr = self.buf as *mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, self.Len() as usize) };
        return slice;
    }

    pub fn AvailableDataSize(&self) -> usize {
        let head = self.headtail[0].load(Ordering::Acquire);
        let tail = self.headtail[1].load(Ordering::Acquire);
        return tail.wrapping_sub(head) as usize;
    }

    pub fn AvailableSpace(&self) -> usize {
        return self.Len() - self.AvailableDataSize();
    }

    /****************************************** read *********************************************************/
    //return (initial size is full, how much read)
    pub fn read(&self, buf: &mut [u8]) -> Result<(bool, usize)> {
        let head = self.headtail[0].load(Ordering::Relaxed);
        let tail = self.headtail[1].load(Ordering::Acquire);

        let mut available = tail.wrapping_sub(head) as usize;
        let full = available == self.Len();

        if available > buf.len() {
            available = buf.len();
        }

        let readpos = (head & self.ringMask) as usize;
        let (firstLen, hasSecond) = {
            let toEnd = self.Len() - readpos;
            if toEnd < available {
                (toEnd, true)
            } else {
                (available, false)
            }
        };

        buf[0..firstLen].clone_from_slice(&self.Buf()[readpos..readpos + firstLen]);

        if hasSecond {
            let secondLen = available - firstLen;
            buf[firstLen..firstLen + secondLen].clone_from_slice(&self.Buf()[0..secondLen])
        }

        self.headtail[0].store(head.wrapping_add(available as u32), Ordering::Release);
        return Ok((full, available));
    }

    pub fn readViaAddr(&self, buf: u64, count: u64) -> (bool, usize) {
        let ptr = buf as *mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, count as usize) };
        let res = self.read(slice).expect("readViaAddr get error");
        return res;
    }

    //return addr, len, whethere there is more space
    pub fn GetReadBuf(&self) -> Option<(u64, usize, bool)> {
        let head = self.headtail[0].load(Ordering::Relaxed);
        let tail = self.headtail[1].load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;

        if available == 0 {
            return None;
        }

        let readpos = (head & self.ringMask) as usize;
        let toEnd = self.Len() - readpos;
        if toEnd < available {
            return Some((self.buf + readpos as u64, toEnd, true));
        } else {
            return Some((self.buf + readpos as u64, available, false));
        }
    }

    pub fn GetDataBuf(&self) -> (u64, usize) {
        let head = self.headtail[0].load(Ordering::Relaxed);
        let tail = self.headtail[1].load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;

        if available == 0 {
            return (0, 0);
        }

        let readpos = (head & self.ringMask) as usize;
        let toEnd = self.Len() - readpos;
        if toEnd < available {
            return (self.buf + readpos as u64, toEnd);
        } else {
            return (self.buf + readpos as u64, available);
        }
    }

    pub fn PrepareDataIovs(&self, data: &mut SocketBufIovs) {
        let mut iovs = &mut data.iovs;

        let head = self.headtail[0].load(Ordering::Relaxed);
        let tail = self.headtail[1].load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;

        if available == 0 {
            data.cnt = 0;
            return;
        }

        assert!(iovs.len() >= 2);
        let readPos = (head & self.ringMask) as usize;
        let toEnd = self.Len() - readPos;
        if toEnd < available {
            iovs[0].start = &self.Buf()[readPos as usize] as *const _ as u64;
            iovs[0].len = toEnd as usize;

            iovs[1].start = &self.Buf()[0] as *const _ as u64;
            iovs[1].len = available - toEnd;

            data.cnt = 2;
        } else {
            iovs[0].start = &self.Buf()[readPos as usize] as *const _ as u64;
            iovs[0].len = available as usize;

            data.cnt = 1;
        }
    }

    //consume count data
    pub fn Consume(&self, count: usize) -> bool {
        let head = self.headtail[0].load(Ordering::Relaxed);
        let tail = self.headtail[1].load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;
        let trigger = available == self.Len();

        self.headtail[0].store(head.wrapping_add(count as u32), Ordering::Release);
        return trigger;
    }

    /****************************************** write *********************************************************/

    pub fn GetWriteBuf(&self) -> Option<(u64, usize, bool)> {
        let head = self.headtail[0].load(Ordering::Acquire);
        let tail = self.headtail[1].load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;
        if available == self.Len() {
            return None;
        }

        let writePos = (tail & self.ringMask) as usize;
        let writeSize = self.Len() - available;

        let toEnd = self.Len() - writePos;
        if toEnd < writeSize {
            return Some((self.buf + writePos as u64, toEnd, true));
        } else {
            return Some((self.buf + writePos as u64, writeSize, false));
        }
    }

    pub fn PrepareSpaceIovs(&self, data: &mut SocketBufIovs) {
        let mut iovs = &mut data.iovs;

        let head = self.headtail[0].load(Ordering::Acquire);
        let tail = self.headtail[1].load(Ordering::Relaxed);
        let available = tail.wrapping_sub(head) as usize;

        if available == self.Len() {
            data.cnt = 0;
            return;
        }

        //error!("GetSpaceIovs available is {}", self.available);
        assert!(iovs.len() >= 2);
        let writePos = (tail & self.ringMask) as usize;
        let writeSize = self.Len() - available;

        let toEnd = self.Len() - writePos;

        //error!("GetSpaceIovs available is {}, toEnd is {}", self.available, toEnd);
        if toEnd < writeSize {
            iovs[0].start = &self.Buf()[writePos as usize] as *const _ as u64;
            iovs[0].len = toEnd as usize;

            iovs[1].start = &self.Buf()[0] as *const _ as u64;
            iovs[1].len = writeSize - toEnd;

            data.cnt = 2;
        } else {
            iovs[0].start = &self.Buf()[writePos as usize] as *const _ as u64;
            iovs[0].len = writeSize as usize;

            data.cnt = 1;
        }
    }

    pub fn GetSpaceBuf(&self) -> (u64, usize) {
        let head = self.headtail[0].load(Ordering::Acquire);
        let tail = self.headtail[1].load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;
        if available == self.Len() {
            return (0, 0);
        }

        let writePos = (tail & self.ringMask) as usize;
        let writeSize = self.Len() - available;

        let toEnd = self.Len() - writePos;
        if toEnd < writeSize {
            return (self.buf + writePos as u64, toEnd);
        } else {
            return (self.buf + writePos as u64, writeSize);
        }
    }

    pub fn Produce(&self, count: usize) -> bool {
        let head = self.headtail[0].load(Ordering::Acquire);
        let tail = self.headtail[1].load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;

        let trigger = available == 0;
        self.headtail[1].store(tail.wrapping_add(count as u32), Ordering::Release);
        return trigger;
    }

    /// return: write user buffer to socket bytestream and determine whether to trigger async socket ops
    pub fn write(&mut self, buf: &[u8]) -> Result<(bool, usize)> {
        let head = self.headtail[0].load(Ordering::Acquire);
        let tail = self.headtail[1].load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;

        let empty = available == 0;

        let writePos = (tail & self.ringMask) as usize;
        let mut writeSize = self.Len() - available;

        if writeSize > buf.len() {
            writeSize = buf.len();
        }

        let (firstLen, hasSecond) = {
            let toEnd = self.Len() - writePos;
            if toEnd < writeSize {
                (toEnd, true)
            } else {
                (writeSize, false)
            }
        };

        self.Buf()[writePos..writePos + firstLen].clone_from_slice(&buf[0..firstLen]);

        if hasSecond {
            let secondLen = writeSize - firstLen;
            self.Buf()[0..secondLen].clone_from_slice(&buf[firstLen..firstLen + secondLen]);
        }

        self.headtail[1].store(tail.wrapping_add(writeSize as u32), Ordering::Release);
        return Ok((empty, writeSize));
    }

    pub fn writeFull(&mut self, buf: &[u8]) -> Result<(bool, usize)> {
        let head = self.headtail[0].load(Ordering::Acquire);
        let tail = self.headtail[1].load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;
        let space = self.Len() - available;

        if available < buf.len() {
            let str = alloc::str::from_utf8(buf).unwrap();
            print!("write full {}/{}/{}", space, buf.len(), str);
            return Err(Error::QueueFull);
        }

        return self.write(buf);
    }

    pub fn writeViaAddr(&mut self, buf: u64, count: u64) -> (bool, usize) {
        let ptr = buf as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, count as usize) };
        self.write(slice).expect("writeViaAddr fail")
    }
}

/// ByteStream serves as a buffer for socket operations, backed by a fixed size byte slice.
///
///

#[repr(align(128))]
pub struct ByteStream {
    pub buf: RingBuf,
    pub dataIovs: SocketBufIovs,
    pub spaceiovs: SocketBufIovs,
}

impl ByteStream {
    pub fn IsPowerOfTwo(x: u64) -> bool {
        return (x & (x - 1)) == 0;
    }

    //allocate page from heap
    pub fn Init(pageCount: u64) -> Self {
        assert!(
            Self::IsPowerOfTwo(pageCount),
            "Bytetream pagecount is not power of two: {}",
            pageCount
        );
        let allocator = RingeBufAllocator::HeapAllocator;
        let buf = RingBuf::New(pageCount as usize, allocator);

        return Self {
            buf: buf,
            dataIovs: SocketBufIovs::default(),
            spaceiovs: SocketBufIovs::default(),
        };
    }

    //return (bufAddr, bufSize)
    pub fn GetRawBuf(&self) -> (u64, usize) {
        return self.buf.GetRawBuf();
    }

    pub fn AvailableSpace(&self) -> usize {
        return self.buf.AvailableSpace();
    }

    pub fn AvailableDataSize(&self) -> usize {
        return self.buf.AvailableDataSize();
    }

    pub fn BufSize(&self) -> usize {
        return self.buf.Len();
    }

    /****************************************** read *********************************************************/
    //return (initial size is full, how much read)
    pub fn read(&mut self, buf: &mut [u8]) -> Result<(bool, usize)> {
        return self.buf.read(buf);
    }

    pub fn readViaAddr(&mut self, buf: u64, count: u64) -> (bool, usize) {
        return self.buf.readViaAddr(buf, count);
    }

    //return addr, len, whethere there is more space
    pub fn GetReadBuf(&mut self) -> Option<(u64, usize, bool)> {
        return self.buf.GetReadBuf();
    }

    pub fn GetDataBuf(&self) -> (u64, usize) {
        return self.buf.GetDataBuf();
    }

    pub fn PrepareDataIovs(&mut self) {
        self.buf.PrepareDataIovs(&mut self.dataIovs);
    }

    pub fn GetDataIovs(&mut self) -> (u64, usize) {
        self.PrepareDataIovs();
        return self.dataIovs.Address();
    }

    pub fn GetDataIovsVec(&mut self) -> Vec<IoVec> {
        self.PrepareDataIovs();
        return self.dataIovs.Iovs();
    }

    //consume count data
    pub fn Consume(&mut self, count: usize) -> bool {
        return self.buf.Consume(count);
    }

    /****************************************** write *********************************************************/

    pub fn GetWriteBuf(&mut self) -> Option<(u64, usize, bool)> {
        return self.buf.GetWriteBuf();
    }

    pub fn PrepareSpaceIovs(&mut self) {
        self.buf.PrepareSpaceIovs(&mut self.spaceiovs)
    }

    pub fn GetSpaceIovs(&mut self) -> (u64, usize) {
        self.PrepareSpaceIovs();
        return self.spaceiovs.Address();
    }

    pub fn GetSpaceIovsVec(&mut self) -> Vec<IoVec> {
        self.PrepareSpaceIovs();
        return self.spaceiovs.Iovs();
    }

    pub fn GetSpaceBuf(&mut self) -> (u64, usize) {
        return self.buf.GetSpaceBuf();
    }

    pub fn Produce(&mut self, count: usize) -> bool {
        return self.buf.Produce(count);
    }

    /// return: write user buffer to socket bytestream and determine whether to trigger async socket ops
    pub fn write(&mut self, buf: &[u8]) -> Result<(bool, usize)> {
        return self.buf.write(buf);
    }

    pub fn writeFull(&mut self, buf: &[u8]) -> Result<(bool, usize)> {
        return self.buf.writeFull(buf);
    }

    pub fn writeViaAddr(&mut self, buf: u64, count: u64) -> (bool, usize) {
        return self.buf.writeViaAddr(buf, count);
    }
}
