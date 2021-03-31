// Copyright (c) 2021 QuarkSoft LLC
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

use core::sync::atomic;
use alloc::slice;

use super::linux_def::*;
use super::pagetable::*;

// the lockfree bytestream for single writer and single read
pub struct LFByteStream {
    pub buf: &'static mut [u8],
    pub head: atomic::AtomicUsize,
    pub tail: atomic::AtomicUsize,
    pub capacity: usize,
    pub ringMask: usize,
    pub allocator: AlignedAllocator,
}

impl Drop for LFByteStream {
    fn drop(&mut self) {
        let (addr, size) = self.GetRawBuf();
        let size = size as u64;
        assert!(size & MemoryDef::PAGE_MASK == 0);
        self.allocator.Free(addr).expect("LFByteStream::drop fail");
    }
}

impl LFByteStream {
    // allocate 1<<ord bytes buffer
    pub fn Init(ord: usize) -> Self {
        assert!(ord > 12, "LFByteStream ord must be large than 12, i.e. one 4KB Page");
        let size = 1 << ord;
        let allocator = AlignedAllocator::New(size as usize, MemoryDef::PAGE_SIZE as usize);
        let addr = allocator.Allocate().expect("ByteStream can't allocate memory");
        let ptr = addr as *mut u8;
        let buf = unsafe { slice::from_raw_parts_mut(ptr, size as usize) };

        return Self {
            buf: buf,
            head: atomic::AtomicUsize::new(0),
            tail: atomic::AtomicUsize::new(0),
            capacity: size,
            ringMask: size - 1,
            allocator: allocator
        }
    }

    pub fn GetRawBuf(&self) -> (u64, usize) {
        return (&self.buf[0] as *const _ as u64, self.buf.len())
    }

    pub fn AvailableSpace(&self) -> usize {
        return self.buf.len() - self.AvailableDataSize();
    }

    pub fn AvailableDataSize(&self) -> usize {
        let head = self.head.load(atomic::Ordering::Acquire);
        let tail = self.tail.load(atomic::Ordering::Acquire);

        return tail.wrapping_sub(head);
    }

    pub fn IsFull(&self) -> bool {
        return self.AvailableDataSize() == self.capacity;
    }

    pub fn Consume(&self, size: usize) {
        self.head.fetch_add(size, atomic::Ordering::SeqCst);
    }

    //return (addr, len, whethere there is more space)
    pub fn ConsumeAndGetReadBuf(&self, size: usize) -> Option<(u64, usize, bool)> {
        let head = self.head.fetch_add(size, atomic::Ordering::SeqCst) + size;
        let tail = self.tail.load(atomic::Ordering::Acquire);

        let available = tail.wrapping_sub(head);

        if available == 0 {
            return None
        }

        let readpos = head & self.ringMask;

        let toEnd = self.capacity - readpos;
        if toEnd < available {
            return Some((&self.buf[0] as *const _ as u64 + readpos as u64, toEnd, true))
        } else {
            return Some((&self.buf[0] as *const _ as u64 + readpos as u64, available, false))
        }
    }

    pub fn Produce(&self, size: usize) {
        self.tail.fetch_add(size, atomic::Ordering::SeqCst);
    }

    // product size bytes, and return the new writebuf(addr, size, whether there is left buf)
    pub fn ProduceAndGetWriteBuf(&self, size: usize) -> Option<(u64, usize, bool)> {
        let head = self.head.load(atomic::Ordering::Acquire);
        let tail = self.tail.fetch_add(size, atomic::Ordering::SeqCst) + size;

        let available = tail.wrapping_sub(head);

        if available == self.capacity {
            return None
        }

        let writePos = tail & self.ringMask;
        let writeSize = self.capacity - available;

        let toEnd = self.capacity - writePos;
        if toEnd < writeSize {
            return Some((&self.buf[0] as *const _ as u64 + writePos as u64, toEnd, true))
        } else {
            return Some((&self.buf[0] as *const _ as u64 + writePos as u64, writeSize, false))
        }
    }

    // return (whether is full, read size)
    pub fn Read(&mut self, buf: &mut [u8]) -> (bool, usize) {
        let head = self.head.load(atomic::Ordering::Acquire);
        let tail = self.tail.load(atomic::Ordering::Acquire);

        let available = tail.wrapping_sub(head);

        let full = available == self.capacity;

        let mut readSize = available;
        let readpos = head & self.ringMask;

        if readSize > buf.len() {
            readSize = buf.len();
        }

        let (firstLen, hasSecond) = {
            let toEnd = self.capacity - readpos;
            if toEnd < readSize {
                (toEnd, true)
            } else {
                (readSize, false)
            }
        };

        buf[0..firstLen].clone_from_slice(&self.buf[readpos..readpos + firstLen]);

        if hasSecond {
            let secondLen = readSize - firstLen;
            buf[firstLen..firstLen + secondLen].clone_from_slice(&self.buf[0..secondLen])
        }

        self.head.store(readpos + readSize, atomic::Ordering::Release);
        return (full, readSize)
    }

    pub fn Write(&mut self, buf: &[u8]) -> (bool, usize) {
        let head = self.head.load(atomic::Ordering::Acquire);
        let tail = self.tail.load(atomic::Ordering::Acquire);

        let available = tail.wrapping_sub(head);
        let empty = available == 0;

        let writePos = tail & self.ringMask;
        let mut writeSize = self.capacity - available;

        if writeSize > buf.len() {
            writeSize = buf.len();
        }

        let (firstLen, hasSecond) = {
            let toEnd = self.capacity - writePos;
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

        self.tail.store(tail + writeSize, atomic::Ordering::Release);
        return (empty, writeSize)
    }
}