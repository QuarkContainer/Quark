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

use super::super::linux_def::IoVec;

#[derive(Debug)]
pub struct IOVecs {
    pub data: Vec<IoVec>,
    pub len: usize,
}

impl IOVecs {
    pub fn NewWithIoVec(iov: IoVec) -> Self {
        return Self::New(vec![iov]);
    }

    pub fn New(data: Vec<IoVec>) -> Self {
        let len = {
            let mut count = 0;
            for iov in &data {
                count += iov.Len();
            }
            count
        };
        return Self {
            data: data,
            len: len 
        }
    }

    pub fn Split(&mut self, offset: usize) -> Option<Self> {
        if self.len <= offset {
            return None;
        }

        let mut count = 0;

        for i in 0..self.data.len() {
            if count + self.data[i].Len() >= offset {
                let splitOffset = offset - count; 
                
                if splitOffset == 0 {
                    let remain = self.data.split_off(i+1);
                    self.len = offset;
                    return Some(Self {
                        data: remain,
                        len: self.len - offset,
                    });
                } else {
                    let mut remain = self.data.split_off(i);
                    self.data.push(IoVec {
                        start: remain[0].start ,
                        len: splitOffset,
                    });
                    
                    let remainLen = self.len - offset;
                    self.len = offset;

                    remain[0].start += splitOffset as u64;
                    remain[0].len -= splitOffset;
                    return Some(Self {
                        data: remain,
                        len: remainLen,
                    });
                }
            }

            count += self.data[i].Len()
        }

        panic!("IOVecs split faill with {:x?} offset {:x?}", self, offset);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IOVecsRef <'a> {
    pub iovs: &'a [IoVec],
    pub skip: usize,
    pub len: usize,
}

impl <'a> IOVecsRef <'a> {
    pub fn Split(&self, offset: usize) -> (IOVecsRef,  IOVecsRef){
        assert!(self.iovs[0].Len() > self.skip);
        let offset = offset + self.skip;
        let mut count = 0;
        if self.len as usize <= offset {
            let first = *self;
            let second = IOVecsRef {
                iovs: self.iovs,
                skip: 0,
                len: 0,
            };

            return (first, second)
        }

        for i in 0..self.iovs.len() {
            if count + self.iovs[i].Len() >= offset {
                let first = IOVecsRef {
                    iovs: &self.iovs[0..i+1],
                    skip: self.skip,
                    len: offset + self.skip,
                };

                let skip = count + self.iovs[i].Len() - offset;
                let second;
                if skip == 0 {
                    if i == self.iovs.len()-1 {
                        second = IOVecsRef {
                            iovs: self.iovs,
                            skip: 0,
                            len: 0,
                        };
                    } else {
                        second = IOVecsRef {
                            iovs: &self.iovs[i+1..],
                            skip: 0,
                            len: 0,
                        };
                    }
                } else {
                    second = IOVecsRef {
                        iovs: &self.iovs[i+1..],
                        skip: skip,
                        len: self.len - offset,
                    };
                }

                return (first, second);
            }

            count += self.iovs[i].Len()
        }

        panic!("IOVecsRef split faill with {:x?} offset {:x?}", self, offset);
    }  
}

pub struct Iovs<'a>(pub &'a [IoVec]);

impl<'a> Iovs<'a> {
    pub fn Count(&self) -> usize {
        let mut count = 0;
        for iov in self.0 {
            count += iov.Len();
        }

        return count;
    }

    pub fn Data(&self) -> Vec<IoVec> {
        let mut res = Vec::new();

        for i in 0..self.0.len() {
            res.push(self.0[i]);
        }

        return res;
    }

    pub fn DropFirst(&self, n: usize) -> Vec<IoVec> {
        let mut n = n;
        let mut res = Vec::new();

        for i in 0..self.0.len() {
            let src = self.0[i];
            if src.Len() <= n {
                n -= self.0[i].Len()
            } else {
                if n > 0 {
                    res.push(IoVec::NewFromAddr(src.Start() + n as u64, src.Len() - n));
                    n = 0;
                } else {
                    res.push(src);
                }
            }
        }

        return res;
    }

    pub fn First(&self, n: usize) -> Vec<IoVec> {
        let mut n = n;
        let mut res = Vec::new();

        for i in 0..self.0.len() {
            let src = self.0[i];
            if src.Len() < n {
                res.push(IoVec::NewFromAddr(src.Start(), src.Len()));
                n -= self.0[i].Len()
            } else {
                res.push(IoVec::NewFromAddr(src.Start(), n));
                break
            }
        }

        return res;
    }
}

impl IoVec {
    pub fn New(buf: &[u8]) -> Self {
        return IoVec {
            start: &buf[0] as *const _ as u64,
            len: buf.len(),
        };
    }

    pub fn NewFromSlice(slice: &[u8]) -> Self {
        return Self {
            start: slice.as_ptr() as u64,
            len: slice.len(),
        };
    }

    pub fn NewFromAddr(start: u64, len: usize) -> Self {
        return Self { start, len };
    }

    pub fn Start(&self) -> u64 {
        return self.start;
    }

    pub fn Len(&self) -> usize {
        return self.len;
    }

    pub fn DropFirst(&self, n: usize) -> Self {
        if n > self.len {
            return Self::default();
        }

        return Self {
            start: self.start + n as u64,
            len: self.len - n,
        };
    }

    pub fn TakeFirst(&self, n: usize) -> Self {
        if n == 0 {
            return Self::default();
        }

        if n > self.len {
            return *self;
        }

        return Self {
            start: self.start,
            len: n,
        };
    }

    pub fn ToSliceMut<'a>(&self) -> &'a mut [u8] {
        let ptr = self.start as *mut u8;
        return unsafe { slice::from_raw_parts_mut(ptr, self.len) };
    }

    pub fn ToSlice<'a>(&self) -> &'a [u8] {
        let ptr = self.start as *const u8;
        return unsafe { slice::from_raw_parts(ptr, self.len) };
    }

    pub fn CopyFrom(&mut self, src: &IoVec) -> i64 {
        let dst = self.ToSliceMut();

        let src = src.ToSlice();
        let mut len = dst.len();
        if len > src.len() {
            len = src.len();
        }

        dst[0..len].copy_from_slice(&src[0..len]);

        return len as i64;
    }

    pub fn Zero(&mut self) -> i64 {
        let slice = self.ToSliceMut();

        unsafe {
            core::ptr::write_bytes(slice.as_mut_ptr() as *mut u8, 0, slice.len());
        }

        return slice.len() as i64;
    }

    pub fn NumBytes(iovs: &[IoVec]) -> usize {
        let mut res = 0;
        for iov in iovs {
            res += iov.len
        }

        return res;
    }

    pub fn End(&self) -> u64 {
        return self.start + self.len as u64;
    }

    pub fn Copy(from: &[IoVec], to: u64, size: usize) {
        let ptr = to as *mut u8;
        let mut toSlice = unsafe { slice::from_raw_parts_mut(ptr, size) };

        for iov in from {
            let fromSlice: &[u8] = iov.ToSlice();
            toSlice[0..iov.len].copy_from_slice(fromSlice);
            toSlice = &mut toSlice[iov.len..];
        }
    }

    pub fn CopySlice(src: &[u8], to: &[IoVec]) {
        let mut offset = 0;
        let mut idx = 0;
        while offset < src.len() && idx < to.len() {
            let ptr = to[idx].start as *mut u8;
            let toSlice = unsafe { slice::from_raw_parts_mut(ptr, to[idx].len) };
            idx += 1;

            let mut len = src.len() - offset;
            if len > toSlice.len() {
                len = toSlice.len()
            }

            toSlice[0..len].copy_from_slice(&src[offset..offset + len]);
            offset += len;
        }
    }
}
