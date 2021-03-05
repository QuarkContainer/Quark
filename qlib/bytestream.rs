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

use alloc::slice;

use super::common::*;
use super::super::*;
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
}

pub struct ByteStream {
    pub buf: &'static mut [u8],
    pub available: usize,
    //available data
    pub readpos: usize,
    pub dataIovs: SocketBufIovs,
    pub spaceiovs: SocketBufIovs,
}

impl Drop for ByteStream {
    fn drop(&mut self) {
        let (addr, size) = self.GetRawBuf();
        let size = size as u64;
        assert!(size & MemoryDef::PAGE_MASK == 0);
        (*PAGE_ALLOCATOR).Free(addr, size >> 12).expect("ByteStream:: free memory fail");
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
    pub fn New(addr: u64, len: usize) -> Self {
        let ptr = addr as *mut u8;
        let buf = unsafe { slice::from_raw_parts_mut(ptr, len) };

        return Self {
            buf: buf,
            available: 0,
            readpos: 0,
            dataIovs: SocketBufIovs::default(),
            spaceiovs: SocketBufIovs::default(),
        }
    }

    //allocate page from memory allocator
    pub fn Init(pageCount: u64) -> Self {
        let addr = (*PAGE_ALLOCATOR).Alloc(pageCount).expect("ByteStream can't allocate memory");
        return Self::New(addr, (pageCount << 12) as usize);
    }

    //return (bufAddr, bufSize)
    pub fn GetRawBuf(&self) -> (u64, usize) {
        return (&self.buf[0] as *const _ as u64, self.buf.len())
    }

    pub fn AvailableSpace(&self) -> usize {
        return self.buf.len() - self.available;
    }

    pub fn AvailableDataSize(&self) -> usize {
        return self.available
    }

    //return (initial size is full, how much read)
    pub fn read(&mut self, buf: &mut [u8]) -> Result<(bool, usize)> {
        let full = self.available == self.buf.len();

        let mut readSize = self.available;

        if readSize > buf.len() {
            readSize = buf.len();
        }

        let (firstLen, hasSecond) = {
            let toEnd = self.buf.len() - self.readpos;
            if toEnd < readSize {
                (toEnd, true)
            } else {
                (readSize, false)
            }
        };

        buf[0..firstLen].clone_from_slice(&self.buf[self.readpos..self.readpos + firstLen]);

        if hasSecond {
            let secondLen = readSize - firstLen;
            buf[firstLen..firstLen + secondLen].clone_from_slice(&self.buf[0..secondLen])
        }

        self.readpos = (self.readpos + readSize) % self.buf.len();
        self.available -= readSize;

        return Ok((full, readSize))
    }

    pub fn readViaAddr(&mut self, buf: u64, count: u64) -> (bool, usize) {
        let ptr = buf as *mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, count as usize) };
        let res = self.read(slice).expect("readViaAddr get error");
        return res;
    }

    //return addr, len, whethere there is more space
    pub fn GetReadBuf(&mut self) -> Option<(u64, usize, bool)> {
        let left = self.available;

        if self.available == 0 {
            return None
        }

        let toEnd = self.buf.len() - self.readpos;
        if toEnd < left {
            return Some((&self.buf[0] as *const _ as u64 + self.readpos as u64, toEnd, true))
        } else {
            return Some((&self.buf[0] as *const _ as u64 + self.readpos as u64, left, false))
        }
    }


    pub fn GetWriteBuf(&mut self) -> Option<(u64, usize, bool)> {
        if self.available == self.buf.len() {
            return None
        }

        let writePos = (self.readpos + self.available) % self.buf.len();
        let writeSize = self.buf.len() - self.available;

        let toEnd = self.buf.len() - writePos;
        if toEnd < writeSize {
            return Some((&self.buf[0] as *const _ as u64 + writePos as u64, toEnd, true))
        } else {
            return Some((&self.buf[0] as *const _ as u64 + writePos as u64, writeSize, false))
        }
    }

    pub fn GetSpaceIovs(&mut self) -> (u64, usize) {
        let mut iovs = &mut self.spaceiovs.iovs;

        if self.available == self.buf.len() {
            self.spaceiovs.cnt = 0;
            return self.spaceiovs.Address()
        }

        //error!("GetSpaceIovs available is {}", self.available);
        assert!(iovs.len()>=2);
        let writePos = (self.readpos + self.available) % self.buf.len();
        let writeSize = self.buf.len() - self.available;

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

        return self.spaceiovs.Address()
    }

    pub fn GetDataBuf(&mut self) -> (u64, usize) {
        let left = self.available;

        if self.available == 0 {
            return (0, 0)
        }

        let toEnd = self.buf.len() - self.readpos;
        if toEnd < left {
            return (&self.buf[0] as *const _ as u64 + self.readpos as u64, toEnd)
        } else {
            return (&self.buf[0] as *const _ as u64 + self.readpos as u64, left)
        }
    }

    pub fn GetSpaceBuf(&mut self) -> (u64, usize) {
        if self.available == self.buf.len() {
            return (0, 0)
        }

        let writePos = (self.readpos + self.available) % self.buf.len();
        let writeSize = self.buf.len() - self.available;

        let toEnd = self.buf.len() - writePos;
        if toEnd < writeSize {
            return (&self.buf[0] as *const _ as u64 + writePos as u64, toEnd)
        } else {
            return (&self.buf[0] as *const _ as u64 + writePos as u64, writeSize)
        }
    }

    pub fn GetDataIovs(&mut self) -> (u64, usize) {
        let mut iovs = &mut self.dataIovs.iovs;

        if self.available == 0 {
            self.dataIovs.cnt = 0;
            return self.dataIovs.Address()
        }

        assert!(iovs.len()>=2);
        let readPos = self.readpos;
        let readSize = self.available;

        let toEnd = self.buf.len() - readPos;
        if toEnd < readSize {
            iovs[0].start = &self.buf[readPos as usize] as * const _ as u64;
            iovs[0].len = toEnd as usize;

            iovs[1].start = &self.buf[0] as *const _ as u64;
            iovs[1].len = readSize - toEnd;

            self.dataIovs.cnt = 2;
        } else {
            iovs[0].start = &self.buf[readPos as usize] as * const _ as u64;
            iovs[0].len = readSize as usize;

            self.dataIovs.cnt = 1;
        }

        return self.dataIovs.Address()
    }

    pub fn Produce(&mut self, count: usize) -> bool {
        //error!("bytesteam produce available is {}", self.available);
        let trigger = self.available == 0;
        self.available += count;
        return trigger
    }

    //consume count data
    pub fn Consume(&mut self, count: usize) -> bool {
        let trigger = self.available == self.buf.len();
        self.available -= count;

        //self.readpos = (self.readpos + count) % self.buf.len();
        if self.available == 0 {
            self.readpos = 0;
        } else {
            self.readpos = (self.readpos + count) % self.buf.len();
        }

        return trigger
    }

    pub fn write(&mut self, buf: &[u8]) -> Result<(bool, usize)> {
        let empty = self.available == 0;
        let writePos = (self.readpos + self.available) % self.buf.len();
        let mut writeSize = self.buf.len() - self.available;

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

        self.available += writeSize;
        /*if writeSize == 0 {
            error!("write: available = {:x}, self.readpos = {:x}, len is {}",
                self.available, self.readpos, buf.len());
        }*/
        return Ok((empty, writeSize))
    }

    pub fn writeViaAddr(&mut self, buf: u64, count: u64) -> (bool, usize) {
        let ptr = buf as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, count as usize) };
        self.write(slice).expect("writeViaAddr fail")
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ByteStream1() {
        let buf : [u8; 10] = [0; 10];
        let addr = &buf[0] as * const _ as u64;
        let len = buf.len();

        let mut bs = ByteStream::New(addr, len);

        let mut read : [u8; 10] = [0; 10];

        assert!(bs.GetReadBuf()==None);

        assert!(bs.write(&vec![0, 1, 2, 3]).unwrap() == 4);
        assert!(bs.GetReadBuf()==Some((addr, 4, false)));
        assert!(bs.GetWriteBuf()==Some((addr+4, 6, false)));

        assert!(bs.read(&mut read[0..3]).unwrap() == 3);
        assert!(read[0..3] == vec![0, 1, 2][..]);
        assert!(bs.GetReadBuf()==Some((addr+3, 1, false)));
        assert!(bs.GetWriteBuf()==Some((addr+4, 6, true)));

        assert!(bs.write(&vec![0, 1, 2, 3]).unwrap() == 4);
        assert!(bs.GetReadBuf()==Some((addr+3, 5, false)));
        assert!(bs.GetWriteBuf()==Some((addr+8, 2, true)));

        assert!(bs.write(&vec![0, 1, 2, 3]).unwrap() == 4);
        assert!(bs.GetReadBuf()==Some((addr+3, 7, true)));
        assert!(bs.GetWriteBuf()==Some((addr+2, 1, false)));

        assert!(bs.write(&vec![0, 1, 2, 3]).unwrap() == 1);
        assert!(bs.read(&mut read[0..3]).unwrap() == 3);

        assert!(read[0..3] == vec![3, 0, 1][..]);
        assert!(bs.read(&mut read[0..10]).unwrap() == 7);

        assert!(read[0..7] == vec![2,3,0,1,2,3,0][..]);
    }
}*/