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
use alloc::str;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::mem::*;
use core::sync::atomic::{AtomicU32, Ordering};

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::mem::block::*;
use super::super::memmgr::mm::*;
use super::super::task::*;
use super::super::util::cstring::*;

impl MemoryManager {
    // copy raw data from user to kernel
    pub fn CopyDataIn(&self, task: &Task, vaddr: u64, to: u64, len: usize, allowPartial: bool) -> Result<()> {
        let _ml = self.MappingWriteLock();

        return self.CopyDataInLocked(task, vaddr, to, len, allowPartial);
    }

    pub fn CopyDataInLocked(&self, task: &Task, vaddr: u64, to: u64, len: usize, allowPartial: bool) -> Result<()> {
        self.V2PLocked(task, vaddr, len as u64, &mut task.GetMut().iovs, false, allowPartial)?;
        defer!(task.GetMut().iovs.clear());
        let target: *mut u8 = to as *mut u8;
        let dst = unsafe { slice::from_raw_parts_mut(target, len) };

        let mut offset = 0;
        for iov in &task.GetMut().iovs {
            let src = iov.start as *const u8;
            let src = unsafe { slice::from_raw_parts(src, iov.len) };
            dst[offset..offset + iov.len].clone_from_slice(src);

            offset += iov.len;
        }

        return Ok(());
    }

    pub fn CopyDataOutLocked(&self, task: &Task, from: u64, vaddr: u64, len: usize, allowPartial: bool) -> Result<()> {
        self.V2PLocked(task, vaddr, len as u64, &mut task.GetMut().iovs, true, allowPartial)?;
        defer!(task.GetMut().iovs.clear());
        let from: *const u8 = from as *const u8;
        let src = unsafe { slice::from_raw_parts(from, len) };

        let mut offset = 0;
        for iov in &task.GetMut().iovs {
            let dst = iov.start as *mut u8;
            let dst = unsafe { slice::from_raw_parts_mut(dst, iov.len) };
            
            dst[0..iov.len].clone_from_slice(&src[offset..offset + iov.len]);
            offset += iov.len;
        }

        return Ok(());
    }

    pub fn CopyDataOut(&self, task: &Task, from: u64, vaddr: u64, len: usize, allowPartial: bool) -> Result<()> {
        let _ml = self.MappingWriteLock();

        return self.CopyDataOutLocked(task, from, vaddr, len, allowPartial);
    }

    pub fn CopyDataOutToIovsLocked(
        &self,
        task: &Task,
        buf: &[u8],
        dsts: &[IoVec],
        allowPartial: bool
    ) -> Result<usize> {
        if buf.len() == 0 {
            return Ok(0);
        }

        let mut offset = 0;
        for iov in dsts {
            if offset >= buf.len() {
                break;
            }

            let mut len = buf.len() - offset;
            if len > iov.len {
                len = iov.len
            }

            self.CopyDataOutLocked(task, &buf[offset] as *const _ as u64, iov.start, len, allowPartial)?;
            offset += len;
        }

        return Ok(offset);
    }

    pub fn CopyDataOutToIovs(&self, task: &Task, buf: &[u8], iovs: &[IoVec], allowPartial: bool) -> Result<usize> {
        let _ml = self.MappingWriteLock();

        return self.CopyDataOutToIovsLocked(task, buf, iovs, allowPartial);
    }

    pub fn CopyIovsOutToIovs(
        &self,
        task: &Task,
        srcIovs: &[IoVec],
        dstIovs: &[IoVec],
        allowPartial: bool
    ) -> Result<usize> {
        let _ml = self.MappingWriteLock();

        let mut dsts = dstIovs;
        let mut count = 0;
        let mut tmp;

        for iov in srcIovs {
            let buf = iov.ToSlice();
            let n = self.CopyDataOutToIovsLocked(task, buf, dsts, allowPartial)?;
            count += n;
            tmp = Iovs(dsts).DropFirst(n as usize);
            dsts = &tmp;
        }

        return Ok(count);
    }

    pub fn CopyDataInFromIovsLocked(
        &self,
        task: &Task,
        buf: &mut [u8],
        iovs: &[IoVec],
        allowPartial: bool
    ) -> Result<usize> {
        if buf.len() == 0 {
            return Ok(0);
        }

        let mut offset = 0;
        for iov in iovs {
            if offset >= buf.len() {
                break;
            }

            let mut len = buf.len() - offset;
            if len > iov.len {
                len = iov.len
            }

            self.CopyDataInLocked(task, iov.start, &buf[offset] as *const _ as u64, len, allowPartial)?;
            offset += len;
        }

        return Ok(offset);
    }

    pub fn CopyDataInFromIovs(&self, task: &Task, buf: &mut [u8], iovs: &[IoVec], allowPartial: bool) -> Result<usize> {
        let _ml = self.MappingWriteLock();

        return self.CopyDataInFromIovsLocked(task, buf, iovs, allowPartial);
    }

    pub fn CopyIovsInFromIovs(
        &self,
        task: &Task,
        srcIovs: &[IoVec],
        dstIovs: &[IoVec],
        allowPartial: bool
    ) -> Result<usize> {
        let _ml = self.MappingWriteLock();

        let mut srcs = srcIovs;
        let mut count = 0;
        let mut tmp;

        for iov in dstIovs {
            let buf = iov.ToSliceMut();
            let n = self.CopyDataInFromIovsLocked(task, buf, srcs, allowPartial)?;
            count += n;
            tmp = Iovs(srcs).DropFirst(n as usize);
            srcs = &tmp;
        }

        return Ok(count);
    }

    pub fn CopyIovsOutFromIovs(
        &self,
        task: &Task,
        srcIovs: &[IoVec],
        dstIovs: &[IoVec],
        allowPartial: bool
    ) -> Result<usize> {
        let _ml = self.MappingWriteLock();

        let mut dsts = dstIovs;
        let mut count = 0;
        let mut tmp;

        for iov in srcIovs {
            let buf = iov.ToSliceMut();
            let n = self.CopyDataOutToIovsLocked(task, buf, dsts, allowPartial)?;
            count += n;
            tmp = Iovs(dsts).DropFirst(n as usize);
            dsts = &tmp;
        }

        return Ok(count);
    }

    pub fn CopyInObjLocked<T: Sized + Copy>(&self, task: &Task, src: u64) -> Result<T> {
        let data: T = unsafe { MaybeUninit::uninit().assume_init() };
        let size = size_of::<T>();
        self.CopyDataInLocked(task, src, &data as *const _ as u64, size, false)?;
        return Ok(data);
    }

    pub fn CopyInObj<T: Sized + Copy>(&self, task: &Task, src: u64) -> Result<T> {
        let data: T = unsafe { MaybeUninit::uninit().assume_init() };
        let size = size_of::<T>();
        self.CopyDataIn(task, src, &data as *const _ as u64, size, false)?;
        return Ok(data);
    }

    pub fn CopyOutObjLocked<T: Sized + Copy>(&self, task: &Task, data: &T, dst: u64) -> Result<()> {
        let size = size_of::<T>();
        self.CopyDataOutLocked(task, data as *const _ as u64, dst, size, false)?;

        return Ok(());
    }

    pub fn CopyOutObj<T: Sized + Copy>(&self, task: &Task, data: &T, dst: u64) -> Result<()> {
        let size = size_of::<T>();
        self.CopyDataOut(task, data as *const _ as u64, dst, size, false)?;

        return Ok(());
    }

    pub fn CopyInVecLocked<T: Sized + Copy>(
        &self,
        task: &Task,
        src: u64,
        count: usize,
        allowPartial: bool
    ) -> Result<Vec<T>> {
        if src == 0 && count == 0 {
            return Ok(Vec::new());
        }

        let recordLen = core::mem::size_of::<T>();
        let mut vec: Vec<T> = Vec::with_capacity(count);
        unsafe {
            vec.set_len(count);
        }
        self.CopyDataInLocked(task, src, vec.as_ptr() as u64, recordLen * count, allowPartial)?;
        return Ok(vec);
    }

    pub fn CopyInVec<T: Sized + Copy>(
        &self,
        task: &Task,
        src: u64,
        count: usize,
    ) -> Result<Vec<T>> {
        if src == 0 && count == 0 {
            return Ok(Vec::new());
        }

        let recordLen = core::mem::size_of::<T>();
        let mut vec: Vec<T> = Vec::with_capacity(count);
        unsafe {
            vec.set_len(count);
        }
        self.CopyDataIn(task, src, vec.as_ptr() as u64, recordLen * count, false)?;
        return Ok(vec);
    }

    //Copy a slice to user memory
    pub fn CopyOutSlice<T: Sized + Copy>(
        &self,
        task: &Task,
        src: &[T],
        dst: u64,
        len: usize,
    ) -> Result<()> {
        if src.len() == 0 {
            return Ok(());
        }

        if len < src.len() {
            return Err(Error::SysError(SysErr::ERANGE));
        }

        let size = size_of::<T>() * src.len();
        return self.CopyDataOut(task, src.as_ptr() as u64, dst, size, false);
    }

    pub fn SwapU32(&self, task: &Task, vaddr: u64, new: u32) -> Result<u32> {
        let _ml = self.MappingWriteLock();

        assert!(vaddr % 4 == 0);

        self.V2PLocked(task, vaddr, 4, &mut task.GetMut().iovs, false, false)?;
        defer!(task.GetMut().iovs.clear());

        assert!(task.GetMut().iovs.len() == 1);
        let addr = task.GetMut().iovs[0].start;
        let val = unsafe { &*(addr as *const AtomicU32) };

        val.swap(new, Ordering::SeqCst);
        return Ok(new);
    }

    pub fn CompareAndSwapU32(&self, task: &Task, vaddr: u64, old: u32, new: u32) -> Result<u32> {
        let _ml = self.MappingWriteLock();

        assert!(vaddr % 4 == 0);

        self.V2PLocked(task, vaddr, 4, &mut task.GetMut().iovs, false, false)?;
        defer!(task.GetMut().iovs.clear());

        assert!(task.GetMut().iovs.len() == 1);
        let addr = task.GetMut().iovs[0].start;
        let val = unsafe { &*(addr as *const AtomicU32) };

        match val.compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(v) => return Ok(v),
            Err(v) => return Ok(v),
        }
    }

    // CopyInVector copies a NULL-terminated vector of strings from the task's
    // memory. The copy will fail with syscall.EFAULT if it traverses
    // user memory that is unmapped or not readable by the user.
    //
    // maxElemSize is the maximum size of each individual element.
    //
    // maxTotalSize is the maximum total length of all elements plus the total
    // number of elements. For example, the following strings correspond to
    // the following set of sizes:
    //
    //     { "a", "b", "c" } => 6 (3 for lengths, 3 for elements)
    //     { "abc" }         => 4 (3 for length, 1 for elements)
    pub fn CopyInVector(
        &self,
        task: &Task,
        addr: u64,
        maxElemSize: usize,
        maxTotalSize: i32,
    ) -> Result<Vec<String>> {
        if addr == 0 {
            return Ok(Vec::new());
        }

        let mut maxTotalSize = maxTotalSize;

        let maxlen = self.FixPermission(task, addr, maxElemSize as u64 * 8, false, true)? as usize;
        let addresses: Vec<u64> = self.CopyInVec(task, addr, maxlen / 8)?;

        let mut v = Vec::new();
        for i in 0..addresses.len() {
            let ptr = addresses[i];
            if ptr == 0 {
                return Ok(v);
            }

            // Each string has a zero terminating byte counted, so copying out a string
            // requires at least one byte of space. Also, see the calculation below.
            if maxTotalSize <= 0 {
                return Err(Error::SysError(SysErr::ENOMEM));
            }

            let mut thisMax = maxElemSize;
            if (maxTotalSize as usize) < thisMax {
                thisMax = maxTotalSize as usize;
            }

            //let maxlen = self.FixPermission(task, ptr, thisMax as u64, false, true)? as usize;
            //error!("CopyInVector 2 ptr is {:x?}, thisMax is {}", &ptr, maxlen);
            let (str, err) = self.CopyInString(task, ptr, thisMax);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }

            let strlen = str.len();
            v.push(str);
            maxTotalSize -= (strlen as i32) + 1;
        }

        return Ok(v);
    }

    // CopyStringIn copies a NUL-terminated string of unknown length from the
    // memory mapped at addr in uio and returns it as a string (not including the
    // trailing NUL). If the length of the string, including the terminating NUL,
    // would exceed maxlen, CopyStringIn returns the string truncated to maxlen and
    // ENAMETOOLONG.
    pub fn CopyInString(&self, task: &Task, addr: u64, maxlen: usize) -> (String, Result<()>) {
        let _ml = self.MappingWriteLock();

        let maxlen = match self.CheckPermissionLocked(task, addr, maxlen as u64, false, true) {
            Err(e) => return ("".to_string(), Err(e)),
            Ok(l) => l as usize,
        };

        let data: Vec<u8> = self
            .CopyInVecLocked(task, addr, maxlen, false)
            .expect("CopyInString fail ...");

        for i in 0..data.len() {
            if data[i] == 0 {
                match str::from_utf8(&data[0..i]) {
                    Ok(str) => {
                        return (str.to_string(), Ok(()));
                    }
                    _ => {
                        return (
                            "".to_string(),
                            Err(Error::Common("Invalid from_utf8".to_string())),
                        )
                    }
                }
            }
        }

        match str::from_utf8(&data[0..maxlen]) {
            Ok(str) => {
                return (str.to_string(), Err(Error::SysError(SysErr::ENAMETOOLONG)));
            }
            _ => {
                return (
                    "".to_string(),
                    Err(Error::Common("Invalid from_utf8".to_string())),
                )
            }
        }
    }

    // check whether the address range is legal.
    // 1. whether the range belong to user's space
    // 2. Whether the read/write permission meet requirement
    // 3. if need cow, fix the page.
    pub fn CheckPermissionLocked(
        &self,
        task: &Task,
        vAddr: u64,
        len: u64,
        writeReq: bool,
        allowPartial: bool,
    ) -> Result<u64> {
        if len == 0 {
            return Ok(0);
        }

        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        return self.FixPermissionLocked(task, vAddr, len, writeReq, allowPartial);
    }
}

impl Task {
    //Copy a vec from user memory
    pub fn CopyInVec<T: Sized + Copy>(&self, addr: u64, size: usize) -> Result<Vec<T>> {
        return self.mm.CopyInVec(self, addr, size);
    }

    //Copy a slice to user memory
    pub fn  CopyOutSlice<T: Sized + Copy>(&self, src: &[T], dst: u64, len: usize) -> Result<()> {
        return self.mm.CopyOutSlice(self, src, dst, len);
    }

    pub fn CopyDataOutToIovs(&self, src: &[u8], dsts: &[IoVec], allowPartial: bool) -> Result<usize> {
        return self.mm.CopyDataOutToIovs(self, src, dsts, allowPartial);
    }

    pub fn CopyIovsOutToIovs(&self, srcs: &[IoVec], dsts: &[IoVec], allowPartial: bool) -> Result<usize> {
        return self.mm.CopyIovsOutToIovs(self, srcs, dsts, allowPartial);
    }

    pub fn CopyDataInFromIovs(&self, buf: &mut [u8], iovs: &[IoVec], allowPartial: bool) -> Result<usize> {
        return self.mm.CopyDataInFromIovs(&self, buf, iovs, allowPartial);
    }

    //Copy an Object from user memory
    pub fn CopyInObj<T: Sized + Copy>(&self, src: u64) -> Result<T> {
        return self.mm.CopyInObj(self, src);
    }

    //Copy an Object to user memory
    pub fn CopyOutObj<T: Sized + Copy>(&self, src: &T, dst: u64) -> Result<()> {
        return self.mm.CopyOutObj(self, src, dst);
    }

    //Copy an str to user memory
    pub fn CopyOutString(&self, vAddr: u64, len: usize, s: &str) -> Result<()> {
        let str = CString::New(s);
        self.CopyOutSlice(str.Slice(), vAddr, len)
    }

    // CopyStringIn copies a NUL-terminated string of unknown length from the
    // memory mapped at addr in uio and returns it as a string (not including the
    // trailing NUL). If the length of the string, including the terminating NUL,
    // would exceed maxlen, CopyStringIn returns the string truncated to maxlen and
    // ENAMETOOLONG.
    pub fn CopyInString(&self, addr: u64, maxlen: usize) -> (String, Result<()>) {
        return self.mm.CopyInString(self, addr, maxlen);
    }

    pub fn CopyInVector(
        &self,
        addr: u64,
        maxElemSize: usize,
        maxTotalSize: i32,
    ) -> Result<Vec<String>> {
        return self.mm.CopyInVector(self, addr, maxElemSize, maxTotalSize);
    }

    pub fn CheckIOVecPermission(&self, iovs: &[IoVec], writeReq: bool) -> Result<()> {
        for iov in iovs {
            self.CheckPermission(iov.start, iov.len as u64, writeReq, false)?;
        }

        return Ok(());
    }

    pub fn GetIOVecPermission(&self, iovs: &[IoVec], writeReq: bool) -> Result<Vec<IoVec>> {
        let mut output = Vec::new();

        for iov in iovs {
            match self.CheckPermission(iov.start, iov.len as u64, writeReq, true) {
                Err(e) => {
                    if output.len() == 0 {
                        return Err(e)
                    }
                    return Ok(output)
                }
                Ok(len) => {
                    output.push(IoVec {
                        start: iov.start,
                        len: len as _
                    })
                }
            }
        }

        return Ok(output);
    }

    // check whether the address range is legal.
    // 1. whether the range belong to user's space
    // 2. Whether the read/write permission meet requirement
    // 3. if need cow, fix the page.
    pub fn CheckPermission(
        &self,
        vAddr: u64,
        len: u64,
        writeReq: bool,
        allowPartial: bool,
    ) -> Result<u64> {
        if len == 0 {
            return Ok(0);
        }

        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        return self
            .mm
            .FixPermission(self, vAddr, len, writeReq, allowPartial);
    }

    pub fn IovsFromAddr(&self, iovs: u64, iovsnum: usize) -> Result<Vec<IoVec>> {
        return self.mm.CopyInVec(self, iovs, iovsnum);
    }

    pub fn V2P(&self, start: u64, len: u64, output: &mut Vec<IoVec>, writable: bool, allowPartial: bool) -> Result<()> {
        return self.mm.V2P(self, start, len, output, writable, allowPartial);
    }

    pub fn V2PIov(&self, iov: &IoVec, output: &mut Vec<IoVec>, writable: bool, allowPartial: bool) -> Result<()> {
        return self.V2P(iov.start, iov.len as u64, output, writable, allowPartial);
    }

    pub fn V2PIovs(&self, iovs: &[IoVec], writable: bool, output: &mut Vec<IoVec>, allowPartial: bool) -> Result<()> {
        for iov in iovs {
            self.V2PIov(iov, output, writable, allowPartial)?;
        }

        return Ok(());
    }
}
