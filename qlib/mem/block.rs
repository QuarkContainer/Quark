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

use super::super::linux_def::IoVec;

pub struct Iovs<'a>(pub &'a [IoVec]);

impl <'a> Iovs <'a> {
    pub fn Count(&self) -> usize {
        let mut count = 0;
        for iov in self.0 {
            count += iov.Len();
        }

        return count;
    }

    pub fn DropFirst(&self, n: usize) -> Vec<IoVec> {
        let mut n = n;
        let mut res = Vec::new();

        for i in 0..self.0.len() {
            let src = self.0[i];
            if src.Len() < n {
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
}

impl IoVec {
    pub fn New(buf: &[u8]) -> Self {
        return IoVec {
            start: &buf[0] as *const _ as u64,
            len: buf.len(),
        }
    }

    pub fn NewFromSlice(slice: &[u8]) -> Self {
        return Self {
            start: slice.as_ptr() as u64,
            len: slice.len(),
        }
    }

    pub fn NewFromAddr(start: u64, len: usize) -> Self {
        return Self {
            start,
            len,
        }
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
        }
    }

    pub fn TakeFirst(&self, n: usize) -> Self {
        if n == 0 {
            return Self::default();
        }

        if n > self.len {
            return *self
        }

        return Self {
            start: self.start,
            len: n,
        }
    }

    pub fn ToSliceMut<'a>(&mut self) -> &'a mut [u8] {
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

        dst[0..len].clone_from_slice(&src[0..len]);

        return len as i64
    }

    pub fn Zero(&mut self) -> i64 {
        let slice = self.ToSliceMut();

        unsafe {
            core::ptr::write_bytes(slice.as_mut_ptr() as * mut u8, 0, slice.len());
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

    pub fn Size(iovs: &[IoVec]) -> usize {
        let mut ret = 0;
        for iov in iovs {
            ret += iov.len;
        }

        return ret;
    }

    pub fn Copy(from: &[IoVec], to: u64, size: usize) {
        let ptr = to as * mut u8;
        let mut toSlice = unsafe {
            slice::from_raw_parts_mut (ptr, size)
        };

        for iov in from {
            let fromSlice : &[u8] = iov.ToSlice();
            toSlice[0..iov.len].copy_from_slice(fromSlice);
            toSlice = &mut toSlice[iov.len..];
        }
    }
}
