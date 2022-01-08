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

use core::slice;
use core::mem;
use alloc::string::String;
use alloc::vec::Vec;

use super::super::super::Common::*;

pub struct TransBuf<'a> {
    pub buf: &'a mut [u8],
    pub offset: usize,
}

impl<'a> TransBuf<'a> {
    pub fn New(buf: &'a mut [u8]) -> Self {
        return Self {
            buf: buf,
            offset: 0,
        }
    }

    pub fn Append<T: Sized + Copy>(&mut self, data: &T) -> Result<&'static T> {
        let size = mem::size_of::<T>();
        if self.offset + size > self.buf.len() {
            return Err(Error::NoEnoughMemory)
        }

        let ptr = unsafe {
            &mut *(&self.buf[self.offset] as *const _ as *mut T)
        };

        *ptr = *data;
        self.offset += size;
        return Ok(ptr)
    }

    pub fn AppendSlice<T: Sized + Copy>(&mut self, data: &[T]) -> Result<&'static [T]> {
        let size = mem::size_of::<T>() * data.len();
        if self.offset + size > self.buf.len() {
            return Err(Error::NoEnoughMemory)
        }

        let ptr = unsafe {
            &mut *(&self.buf[self.offset] as *const _ as *mut T)
        };

        let slice = unsafe { slice::from_raw_parts_mut(ptr, data.len()) };

        for i in 0..data.len() {
            slice[i] = data[i]
        }

        self.offset += size;
        return Ok(slice)
    }

    pub fn AppendStr(&mut self, str: &str) -> Result<&'static [u8]> {
        let arr = str.as_bytes();
        return self.AppendSlice(arr);
    }

    pub fn AppendStrArr(&mut self, strs: &[String]) -> Result<&'static [&'static [u8]]> {
        let mut arr = Vec::with_capacity(strs.len());

        for str in strs {
            arr.push(self.AppendStr(str)?);
        }

        return self.AppendSlice(&arr[..])
    }
}