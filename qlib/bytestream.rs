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

use alloc::slice;
use alloc::vec::Vec;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::Ordering;

use super::common::*;
use super::pagetable::AlignedAllocator;
use super::linux_def::*;
use super::mem::io::*;

#[derive(Default, Debug)]
pub struct SocketBufIovs {
    pub iovs: [IoVec; 2],
    pub cnt: usize,
}

impl SocketBufIovs {
    pub fn Address(&self) -> (u64, usize) {
        return (&self.iovs[0] as * const _ as u64, self.cnt)
    }

    pub fn Iovs(&self) -> Vec<IoVec> {
        let mut iovs = Vec::with_capacity(self.cnt);
        for i in 0..self.cnt {
            iovs.push(self.iovs[i])
        }

        return iovs;
    }
}
/// ByteStream serves as a buffer for socket operations, backed by a fixed size byte slice. 
/// 
///

#[repr(align(128))]
pub struct ByteStream {
    pub buf: &'static mut [u8],
    /// size of data available to consume by consumer
    pub head: AtomicU32,
    pub tail: AtomicU32,
    pub ringMask: u32,
    pub dataIovs: SocketBufIovs,
    pub spaceiovs: SocketBufIovs,
    pub allocator: AlignedAllocator,
}

impl Drop for ByteStream {
    fn drop(&mut self) {
        let (addr, size) = self.GetRawBuf();
        let size = size as u64;
        assert!(size & MemoryDef::PAGE_MASK == 0);
        self.allocator.Free(addr).expect("ByteStream::drop fail");
    }
}

impl IOReader for ByteStream {
    fn Read(&mut self, buf: &mut [u8]) -> Result<i64> {
        let (_, cnt) = self.read(buf)?;
        return Ok(cnt as i64)
    }
}

impl IOWriter for ByteStream {
    fn Write(&mut self, buf: &[u8]) -> Result<i64> {
        let (_, cnt) = self.write(buf)?;
        return Ok(cnt as i64)
    }
}

impl ByteStream {
    pub fn IsPowerOfTwo(x: u64) -> bool {
        return (x & (x - 1)) == 0;
    }

    //allocate page from heap
    pub fn Init(pageCount: u64) -> Self {
        assert!(Self::IsPowerOfTwo(pageCount), "Bytetream pagecount is not power of two: {}", pageCount);
        let size = pageCount * MemoryDef::PAGE_SIZE;
        let allocator = AlignedAllocator::New(size as usize, MemoryDef::PAGE_SIZE as usize);
        let addr = allocator.Allocate().expect("ByteStream can't allocate memory");
        let ptr = addr as *mut u8;
        let buf = unsafe { slice::from_raw_parts_mut(ptr, size as usize) };

        return Self {
            buf: buf,
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            ringMask: (pageCount as u32 * MemoryDef::PAGE_SIZE as u32) - 1,
            dataIovs: SocketBufIovs::default(),
            spaceiovs: SocketBufIovs::default(),
            allocator: allocator,
        }
    }

    //return (bufAddr, bufSize)
    pub fn GetRawBuf(&self) -> (u64, usize) {
        return (&self.buf[0] as *const _ as u64, self.buf.len())
    }

    pub fn AvailableSpace(&self) -> usize {
        return self.buf.len() - self.AvailableDataSize();
    }

    pub fn AvailableDataSize(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        return tail.wrapping_sub(head) as usize
    }

    pub fn BufSize(&self) -> usize {
        return self.buf.len();
    }

    /****************************************** read *********************************************************/
    //return (initial size is full, how much read)
    pub fn read(&mut self, buf: &mut [u8]) -> Result<(bool, usize)> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let mut available = tail.wrapping_sub(head) as usize;
        let full = available == self.buf.len();

        if available > buf.len() {
            available = buf.len();
        }

        let readpos = (head & self.ringMask) as usize;
        let (firstLen, hasSecond) = {
            let toEnd = self.buf.len() - readpos;
            if toEnd < available {
                (toEnd, true)
            } else {
                (available, false)
            }
        };

        buf[0..firstLen].clone_from_slice(&self.buf[readpos..readpos + firstLen]);

        if hasSecond {
            let secondLen = available - firstLen;
            buf[firstLen..firstLen + secondLen].clone_from_slice(&self.buf[0..secondLen])
        }

        self.head.store(head.wrapping_add(available as u32), Ordering::Release);
        return Ok((full, available))
    }

    pub fn readViaAddr(&mut self, buf: u64, count: u64) -> (bool, usize) {
        let ptr = buf as *mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, count as usize) };
        let res = self.read(slice).expect("readViaAddr get error");
        return res;
    }

    //return addr, len, whethere there is more space
    pub fn GetReadBuf(&mut self) -> Option<(u64, usize, bool)> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;

        if available == 0 {
            return None
        }

        let readpos = (head & self.ringMask) as usize;
        let toEnd = self.buf.len() - readpos;
        if toEnd < available {
            return Some((&self.buf[0] as *const _ as u64 + readpos as u64, toEnd, true))
        } else {
            return Some((&self.buf[0] as *const _ as u64 + readpos as u64, available, false))
        }
    }

    pub fn GetDataBuf(&self) -> (u64, usize) {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;

        if available == 0 {
            return (0, 0)
        }

        let readpos = (head & self.ringMask) as usize;
        let toEnd = self.buf.len() - readpos;
        if toEnd < available {
            return (&self.buf[0] as *const _ as u64 + readpos as u64, toEnd)
        } else {
            return (&self.buf[0] as *const _ as u64 + readpos as u64, available)
        }
    }

    pub fn PrepareDataIovs(&mut self) {
        let mut iovs = &mut self.dataIovs.iovs;

        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;

        if available == 0 {
            self.dataIovs.cnt = 0;
            return
        }

        assert!(iovs.len()>=2);
        let readPos = (head & self.ringMask) as usize;
        let toEnd = self.buf.len() - readPos;
        if toEnd < available {
            iovs[0].start = &self.buf[readPos as usize] as * const _ as u64;
            iovs[0].len = toEnd as usize;

            iovs[1].start = &self.buf[0] as *const _ as u64;
            iovs[1].len = available - toEnd;

            self.dataIovs.cnt = 2;
        } else {
            iovs[0].start = &self.buf[readPos as usize] as * const _ as u64;
            iovs[0].len = available as usize;

            self.dataIovs.cnt = 1;
        }
    }

    pub fn GetDataIovs(&mut self) -> (u64, usize) {
        self.PrepareDataIovs();
        return self.dataIovs.Address()
    }

    pub fn GetDataIovsVec(&mut self) -> Vec<IoVec> {
        self.PrepareDataIovs();
        return self.dataIovs.Iovs()
    }

    //consume count data
    pub fn Consume(&mut self, count: usize) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let available = tail.wrapping_sub(head) as usize;
        let trigger = available == self.buf.len();

        self.head.store(head.wrapping_add(count as u32), Ordering::Release);
        return trigger
    }


    /****************************************** write *********************************************************/

    pub fn GetWriteBuf(&mut self) -> Option<(u64, usize, bool)> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;
        if available == self.buf.len() {
            return None
        }

        let writePos = (tail & self.ringMask) as usize;
        let writeSize = self.buf.len() - available;

        let toEnd = self.buf.len() - writePos;
        if toEnd < writeSize {
            return Some((&self.buf[0] as *const _ as u64 + writePos as u64, toEnd, true))
        } else {
            return Some((&self.buf[0] as *const _ as u64 + writePos as u64, writeSize, false))
        }
    }

    pub fn PrepareSpaceIovs(&mut self) {
        let mut iovs = &mut self.spaceiovs.iovs;

        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        let available = tail.wrapping_sub(head) as usize;

        if available == self.buf.len() {
            self.spaceiovs.cnt = 0;
            return
        }

        //error!("GetSpaceIovs available is {}", self.available);
        assert!(iovs.len()>=2);
        let writePos = (tail & self.ringMask) as usize;
        let writeSize = self.buf.len() - available;

        let toEnd = self.buf.len() - writePos;

        //error!("GetSpaceIovs available is {}, toEnd is {}", self.available, toEnd);
        if toEnd < writeSize {
            iovs[0].start = &self.buf[writePos as usize] as * const _ as u64;
            iovs[0].len = toEnd as usize;

            iovs[1].start = &self.buf[0] as *const _ as u64;
            iovs[1].len = writeSize - toEnd;

            self.spaceiovs.cnt = 2;
        } else {
            iovs[0].start = &self.buf[writePos as usize] as * const _ as u64;
            iovs[0].len = writeSize as usize;

            self.spaceiovs.cnt = 1;
        }
    }

    pub fn GetSpaceIovs(&mut self) -> (u64, usize) {
        self.PrepareSpaceIovs();
        return self.spaceiovs.Address()
    }

    pub fn GetSpaceIovsVec(&mut self) -> Vec<IoVec> {
        self.PrepareSpaceIovs();
        return self.spaceiovs.Iovs();
    }

    pub fn GetSpaceBuf(&mut self) -> (u64, usize) {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;
        if available == self.buf.len() {
            return (0, 0)
        }

        let writePos = (tail & self.ringMask) as usize;
        let writeSize = self.buf.len() - available;

        let toEnd = self.buf.len() - writePos;
        if toEnd < writeSize {
            return (&self.buf[0] as *const _ as u64 + writePos as u64, toEnd)
        } else {
            return (&self.buf[0] as *const _ as u64 + writePos as u64, writeSize)
        }
    }

    pub fn Produce(&mut self, count: usize) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;

        let trigger = available == 0;
        self.tail.store(tail.wrapping_add(count as u32), Ordering::Release);
        return trigger
    }

    /// return: write user buffer to socket bytestream and determine whether to trigger async socket ops
    pub fn write(&mut self, buf: &[u8]) -> Result<(bool, usize)> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;

        let empty = available == 0;

        let writePos = (tail & self.ringMask) as usize;
        let mut writeSize = self.buf.len() - available;

        if writeSize > buf.len() {
            writeSize = buf.len();
        }

        let (firstLen, hasSecond) = {
            let toEnd = self.buf.len() - writePos;
            if toEnd < writeSize {
                (toEnd, true)
            } else {
                (writeSize, false)
            }
        };

        self.buf[writePos..writePos + firstLen].clone_from_slice(&buf[0..firstLen]);

        if hasSecond {
            let secondLen = writeSize - firstLen;
            self.buf[0..secondLen].clone_from_slice(&buf[firstLen..firstLen + secondLen]);
        }

        self.tail.store(tail.wrapping_add(writeSize as u32), Ordering::Release);
        return Ok((empty, writeSize))
    }

    pub fn writeFull(&mut self, buf: &[u8]) -> Result<(bool, usize)> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        let available = tail.wrapping_sub(head) as usize;
        let space = self.buf.len() - available;

        if available < buf.len() {
            let str = alloc::str::from_utf8(buf).unwrap();
            print!("write full {}/{}/{}", space, buf.len(), str);
            return Err(Error::QueueFull)
        }

        return self.write(buf);
    }

    pub fn writeViaAddr(&mut self, buf: u64, count: u64) -> (bool, usize) {
        let ptr = buf as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, count as usize) };
        self.write(slice).expect("writeViaAddr fail")
    }
}
