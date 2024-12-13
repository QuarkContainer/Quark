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

use alloc::boxed::Box;
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
use super::super::super::mutex::*;
use super::super::memmgr::mm::*;
use super::super::task::*;
use super::super::util::cstring::*;
use crate::kernel_def::IsKernel;
use crate::qlib::kernel::SHARESPACE;
use crate::GuestHostSharedAllocator;
use crate::GUEST_HOST_SHARED_ALLOCATOR;

impl MemoryManager {
    // copy raw data from user to kernel
    pub fn CopyDataIn(
        &self,
        task: &Task,
        vaddr: u64,
        to: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<()> {
        if SHARESPACE.config.read().CopyDataWithPf && !allowPartial {
            self.HandleTlbShootdown();
            return self.CopyDataWithPf(task, vaddr, to, len, allowPartial);
        }

        return self.CopyDataInManual(task, vaddr, to, len, allowPartial);
    }

    pub fn CopyDataInManual(
        &self,
        task: &Task,
        vaddr: u64,
        to: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<()> {
        let rl = self.MappingReadLock();

        return self.CopyDataInLocked(task, &rl, vaddr, to, len, allowPartial);
    }

    pub fn CopyDataWithPf(
        &self,
        _task: &Task,
        from: u64,
        to: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<()> {
        assert!(allowPartial == false);
        assert!(IsKernel());
        Self::Memcpy(to, from, len);

        return Ok(());
    }

    pub fn CopyDataInLocked(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        vaddr: u64,
        to: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<()> {
        if vaddr == 0 && len == 0 {
            return Ok(());
        }
        let iovs = self.V2PLocked(task, rl, vaddr, len as u64, false, allowPartial)?;

        let mut offset = 0;
        for iov in &iovs {
            unsafe {
                let dstPtr = (to + offset) as *mut u8;
                let srcPtr = iov.start as *const u8;
                core::ptr::copy_nonoverlapping(srcPtr, dstPtr, iov.len);
            }

            offset += iov.len as u64;
        }

        return Ok(());
    }

    pub fn CopyDataOutLocked(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        from: u64,
        vaddr: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<()> {
        if vaddr == 0 && len == 0 {
            return Ok(());
        }
        let iovs = self.V2PLocked(task, rl, vaddr, len as u64, true, allowPartial)?;

        let mut offset = 0;
        for iov in &iovs {
            unsafe {
                let dstPtr = iov.start as *mut u8;
                let srcPtr = (from + offset) as *const u8;
                core::ptr::copy_nonoverlapping(srcPtr, dstPtr, iov.len);
            }

            offset += iov.len as u64;
        }

        return Ok(());
    }

    pub fn ZeroDataOutLocked(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        vaddr: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<usize> {
        let iovs = self.V2PLocked(task, rl, vaddr, len as u64, true, allowPartial)?;

        let mut len = 0;
        for iov in &iovs {
            let dst = iov.start as *mut u8;
            let dst = unsafe { slice::from_raw_parts_mut(dst, iov.len) };
            for i in 0..iov.len {
                dst[i] = 0;
            }

            len += iov.len;
        }

        return Ok(len);
    }

    pub fn CopyDataOut(
        &self,
        task: &Task,
        from: u64,
        vaddr: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<()> {
        if SHARESPACE.config.read().CopyDataWithPf && !allowPartial {
            self.HandleTlbShootdown();
            return self.CopyDataWithPf(task, from, vaddr, len, allowPartial);
        }
        let rl = self.MappingReadLock();
        return self.CopyDataOutLocked(task, &rl, from, vaddr, len, allowPartial);
    }

    pub fn CopyDataOutManual(
        &self,
        task: &Task,
        from: u64,
        vaddr: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<()> {
        let rl = self.MappingReadLock();

        return self.CopyDataOutLocked(task, &rl, from, vaddr, len, allowPartial);
    }

    pub fn ZeroDataOut(
        &self,
        task: &Task,
        vaddr: u64,
        len: usize,
        allowPartial: bool,
    ) -> Result<usize> {
        let rl = self.MappingReadLock();

        return self.ZeroDataOutLocked(task, &rl, vaddr, len, allowPartial);
    }

    pub fn CopyDataOutToIovsLocked(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        buf: &[u8],
        dsts: &[IoVec],
        allowPartial: bool,
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

            self.CopyDataOutLocked(
                task,
                rl,
                &buf[offset] as *const _ as u64,
                iov.start,
                len,
                allowPartial,
            )?;
            offset += len;
        }

        return Ok(offset);
    }

    pub fn ZeroDataOutToIovsLocked(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        dsts: &[IoVec],
        size: usize,
        allowPartial: bool,
    ) -> Result<usize> {
        let mut offset = 0;
        for iov in dsts {
            if offset >= size {
                break;
            }

            let mut len = size - offset;
            if len > iov.len {
                len = iov.len
            }

            let cnt = self.ZeroDataOutLocked(task, rl, iov.start, len, allowPartial)?;
            offset += cnt;
            if cnt < len {
                break;
            }
        }

        return Ok(offset);
    }

    pub fn CopyDataOutToIovsWithPf(
        &self,
        _task: &Task,
        buf: &[u8],
        iovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        assert!(allowPartial == false);
        assert!(IsKernel());

        let src = buf.as_ptr() as u64; // &buf[0] as * const _ as u64;
        let mut offset = 0;

        for dst in iovs {
            let left = buf.len() as u64 - offset;
            let len = left.min(dst.Len() as u64);

            Self::Memcpy(dst.Start(), src + offset, len as usize);

            offset += len;
            if offset >= buf.len() as u64 {
                break;
            }
        }

        return Ok(offset as usize);
    }

    pub fn CopyDataOutToIovs(
        &self,
        task: &Task,
        buf: &[u8],
        iovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        if SHARESPACE.config.read().CopyDataWithPf && !allowPartial {
            self.HandleTlbShootdown();
            return self.CopyDataOutToIovsWithPf(task, buf, iovs, allowPartial);
        }

        return self.CopyDataOutToIovsManual(task, buf, iovs, allowPartial);
    }

    pub fn CopyDataOutToIovsManual(
        &self,
        task: &Task,
        buf: &[u8],
        iovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        let rl = self.MappingReadLock();

        return self.CopyDataOutToIovsLocked(task, &rl, buf, iovs, allowPartial);
    }

    pub fn ZeroDataOutToIovs(
        &self,
        task: &Task,
        iovs: &[IoVec],
        size: usize,
        allowPartial: bool,
    ) -> Result<usize> {
        let rl = self.MappingReadLock();

        return self.ZeroDataOutToIovsLocked(task, &rl, iovs, size, allowPartial);
    }

    pub fn CopyIovsOutToIovs(
        &self,
        task: &Task,
        srcIovs: &[IoVec],
        dstIovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        let rl = self.MappingReadLock();

        let mut dsts = dstIovs;
        let mut count = 0;
        let mut tmp;

        for iov in srcIovs {
            let buf = iov.ToSlice();
            let n = self.CopyDataOutToIovsLocked(task, &rl, buf, dsts, allowPartial)?;
            count += n;
            tmp = Iovs(dsts).DropFirst(n as usize);
            dsts = &tmp;
        }

        return Ok(count);
    }

    pub fn CopyDataInFromIovsLocked(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        buf: &mut [u8],
        iovs: &[IoVec],
        allowPartial: bool,
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

            self.CopyDataInLocked(
                task,
                rl,
                iov.start,
                &buf[offset] as *const _ as u64,
                len,
                allowPartial,
            )?;
            offset += len;
        }

        return Ok(offset);
    }

    pub fn CopyDataInFromIovsWithPf(
        &self,
        _task: &Task,
        buf: &mut [u8],
        iovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        assert!(allowPartial == false);
        assert!(IsKernel());

        let dst = buf.as_ptr() as u64; // &buf[0] as * const _ as u64;
        let mut offset = 0;

        for src in iovs {
            let left = buf.len() as u64 - offset;
            let len = left.min(src.Len() as u64);

            Self::Memcpy(dst + offset, src.Start(), len as usize);

            offset += len;
            if offset >= buf.len() as u64 {
                break;
            }
        }

        return Ok(offset as usize);
    }

    pub fn CopyDataInFromIovs(
        &self,
        task: &Task,
        buf: &mut [u8],
        iovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        if SHARESPACE.config.read().CopyDataWithPf && !allowPartial {
            self.HandleTlbShootdown();
            return self.CopyDataInFromIovsWithPf(task, buf, iovs, allowPartial);
        }

        let rl = self.MappingReadLock();

        return self.CopyDataInFromIovsLocked(task, &rl, buf, iovs, allowPartial);
    }

    pub fn CopyIovsInFromIovs(
        &self,
        task: &Task,
        srcIovs: &[IoVec],
        dstIovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        if SHARESPACE.config.read().CopyDataWithPf && !allowPartial {
            self.HandleTlbShootdown();
            return self.CopyBetweenIovsWithPf(task, srcIovs, dstIovs, allowPartial);
        }

        let rl = self.MappingReadLock();

        let mut srcs = srcIovs;
        let mut count = 0;
        let mut tmp;

        for iov in dstIovs {
            let buf = iov.ToSliceMut();
            let n = self.CopyDataInFromIovsLocked(task, &rl, buf, srcs, allowPartial)?;
            count += n;
            tmp = Iovs(srcs).DropFirst(n as usize);
            srcs = &tmp;
        }

        return Ok(count);
    }

    pub fn CopyBetweenIovsWithPf(
        &self,
        _task: &Task,
        srcIovs: &[IoVec],
        dstIovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        assert!(allowPartial == false);
        assert!(IsKernel());

        let mut srcIdx: usize = 0;
        let mut dstIdx: usize = 0;
        let mut srcOffset = 0;
        let mut dstOffset = 0;

        let mut total = 0;

        while srcIdx < srcIovs.len() && dstIdx < dstIovs.len() {
            let srcLen = srcIovs[srcIdx].Len() - srcOffset as usize;
            let dstLen = dstIovs[dstIdx].Len() - dstOffset as usize;
            if srcLen == dstLen {
                Self::Memcpy(
                    dstIovs[dstIdx].Start() + dstOffset,
                    srcIovs[srcIdx].Start() + srcOffset,
                    srcLen,
                );

                srcIdx += 1;
                srcOffset = 0;

                dstIdx += 1;
                dstOffset = 0;

                total += srcLen;
            } else if srcLen < dstLen {
                Self::Memcpy(
                    dstIovs[dstIdx].Start() + dstOffset,
                    srcIovs[srcIdx].Start() + srcOffset,
                    srcLen,
                );

                srcIdx += 1;
                srcOffset = 0;

                dstOffset += srcLen as u64;

                total += srcLen;
            } else {
                // srcLen > dstLen
                Self::Memcpy(
                    dstIovs[dstIdx].Start() + dstOffset,
                    srcIovs[srcIdx].Start() + srcOffset,
                    dstLen,
                );

                srcOffset += dstLen as u64;

                dstIdx += 1;
                dstOffset = 0;

                total += dstLen;
            }
        }

        return Ok(total);
    }

    pub fn Memcpy(dst: u64, src: u64, count: usize) {
        unsafe {
            let dstPtr = dst as *mut u8;
            let srcPtr = src as *const u8;
            core::ptr::copy_nonoverlapping(srcPtr, dstPtr, count);
        }
    }

    pub fn CopyIovsOutFromIovs(
        &self,
        task: &Task,
        srcIovs: &[IoVec],
        dstIovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        if SHARESPACE.config.read().CopyDataWithPf && !allowPartial {
            self.HandleTlbShootdown();
            return self.CopyBetweenIovsWithPf(task, srcIovs, dstIovs, allowPartial);
        }

        let rl = self.MappingReadLock();

        let mut dsts = dstIovs;
        let mut count = 0;
        let mut tmp;

        for iov in srcIovs {
            let buf = iov.ToSliceMut();
            let n = self.CopyDataOutToIovsLocked(task, &rl, buf, dsts, allowPartial)?;
            count += n;
            tmp = Iovs(dsts).DropFirst(n as usize);
            dsts = &tmp;
        }

        return Ok(count);
    }

    pub fn CopyInObjLocked<T: Sized + Copy>(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        src: u64,
    ) -> Result<T> {
        let data: T = unsafe { MaybeUninit::uninit().assume_init() };
        let size = size_of::<T>();
        self.CopyDataInLocked(task, rl, src, &data as *const _ as u64, size, false)?;
        return Ok(data);
    }

    pub fn CopyInObj<T: Sized + Copy>(&self, task: &Task, src: u64) -> Result<T> {
        let data: T = unsafe { MaybeUninit::uninit().assume_init() };
        let size = size_of::<T>();
        self.CopyDataIn(task, src, &data as *const _ as u64, size, false)?;
        return Ok(data);
    }

    pub fn CopyInObjShared<T: Sized + Copy>(&self, task: &Task, src: u64) -> Result<Box<T, GuestHostSharedAllocator>> {
        let data = Box::new_in(unsafe { MaybeUninit::<T>::uninit().assume_init() }, GUEST_HOST_SHARED_ALLOCATOR);
        let size = size_of::<T>();
        self.CopyDataIn(task, src, &*data as *const _ as u64, size, false)?;
        return Ok(data);
    }

    pub fn CopyOutObjLocked<T: Sized + Copy>(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        data: &T,
        dst: u64,
    ) -> Result<()> {
        let size = size_of::<T>();
        self.CopyDataOutLocked(task, rl, data as *const _ as u64, dst, size, false)?;

        return Ok(());
    }

    pub fn CopyOutObj<T: Sized + Copy>(&self, task: &Task, data: &T, dst: u64) -> Result<()> {
        let size = size_of::<T>();
        self.CopyDataOut(task, data as *const _ as u64, dst, size, false)?;

        return Ok(());
    }

    pub fn CopyOutObjManual<T: Sized + Copy>(&self, task: &Task, data: &T, dst: u64) -> Result<()> {
        let size = size_of::<T>();
        self.CopyDataOutManual(task, data as *const _ as u64, dst, size, false)?;

        return Ok(());
    }

    pub fn CopyInVecLocked<T: Sized + Copy>(
        &self,
        task: &Task,
        rl: &QUpgradableLockGuard,
        src: u64,
        count: usize,
        allowPartial: bool,
    ) -> Result<Vec<T>> {
        if src == 0 && count == 0 {
            return Ok(Vec::new());
        }

        let recordLen = core::mem::size_of::<T>();
        let mut vec: Vec<T> = Vec::with_capacity(count);
        unsafe {
            vec.set_len(count);
        }
        self.CopyDataInLocked(
            task,
            rl,
            src,
            vec.as_ptr() as u64,
            recordLen * count,
            allowPartial,
        )?;
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

        if src == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let recordLen = core::mem::size_of::<T>();
        let mut vec: Vec<T> = Vec::with_capacity(count);
        unsafe {
            vec.set_len(count);
        }
        self.CopyDataIn(task, src, vec.as_ptr() as u64, recordLen * count, false)?;
        return Ok(vec);
    }

    pub fn CopyInVecShared<T: Sized + Copy>(
        &self,
        task: &Task,
        src: u64,
        count: usize,
    ) -> Result<Vec<T, GuestHostSharedAllocator>> {
        if src == 0 && count == 0 {
            return Ok(Vec::new_in(GUEST_HOST_SHARED_ALLOCATOR));
        }

        if src == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let recordLen = core::mem::size_of::<T>();
        let mut vec = Vec::with_capacity_in(count, GUEST_HOST_SHARED_ALLOCATOR);
        unsafe {
            vec.set_len(count);
        }
        self.CopyDataIn(task, src, vec.as_ptr() as u64, recordLen * count, false)?;
        return Ok(vec);
    }

    pub fn CopyInVecManaul<T: Sized + Copy>(
        &self,
        task: &Task,
        src: u64,
        count: usize,
    ) -> Result<Vec<T>> {
        if src == 0 && count == 0 {
            return Ok(Vec::new());
        }

        if src == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let recordLen = core::mem::size_of::<T>();
        let mut vec: Vec<T> = Vec::with_capacity(count);
        unsafe {
            vec.set_len(count);
        }
        self.CopyDataInManual(task, src, vec.as_ptr() as u64, recordLen * count, false)?;
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
        let rl = self.MappingReadLock();

        assert!(vaddr % 4 == 0);

        let iovs = self.V2PLocked(task, &rl, vaddr, 4, false, false)?;

        assert!(iovs.len() == 1);
        let addr = iovs[0].start;
        let val = unsafe { &*(addr as *const AtomicU32) };

        val.swap(new, Ordering::SeqCst);
        return Ok(new);
    }

    pub fn CompareAndSwapU32(&self, task: &Task, vaddr: u64, old: u32, new: u32) -> Result<u32> {
        let rl = self.MappingReadLock();

        assert!(vaddr % 4 == 0);

        let iovs = self.V2PLocked(task, &rl, vaddr, 4, false, false)?;

        assert!(iovs.len() == 1);
        let addr = iovs[0].start;
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

        //// todo: fix this, lock for whole function
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
        let rl = self.MappingReadLock();

        let maxlen = match self.CheckPermissionLocked(task, &rl, addr, maxlen as u64, false, true) {
            Err(e) => return ("".to_string(), Err(e)),
            Ok(l) => l as usize,
        };

        let data: Vec<u8> = self
            .CopyInVecLocked(task, &rl, addr, maxlen, false)
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
        rl: &QUpgradableLockGuard,
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

        return self.FixPermissionLocked(task, rl, vAddr, len, writeReq, None, allowPartial);
    }
}

impl Task {
    //Copy a vec from user memory
    pub fn CopyInVec<T: Sized + Copy>(&self, addr: u64, size: usize) -> Result<Vec<T>> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyInVec(self, addr, size);
    }

    pub fn CopyInVecShared<T: Sized + Copy>(&self, addr: u64, size: usize) -> Result<Vec<T, GuestHostSharedAllocator>> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyInVecShared(self, addr, size);
    }

    pub fn CopyInVecManaul<T: Sized + Copy>(&self, addr: u64, size: usize) -> Result<Vec<T>> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyInVecManaul(self, addr, size);
    }

    pub fn FixPermissionForIovs(&self, iovs: &[IoVec], writable: bool) -> Result<()> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.FixPermissionForIovs(self, iovs, writable);
    }

    //Copy a slice to user memory
    pub fn CopyOutSlice<T: Sized + Copy>(&self, src: &[T], dst: u64, len: usize) -> Result<()> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyOutSlice(self, src, dst, len);
    }

    pub fn CopyDataOutToIovs(
        &self,
        src: &[u8],
        dsts: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyDataOutToIovs(self, src, dsts, allowPartial);
    }

    pub fn CopyDataOutToIovsManual(
        &self,
        src: &[u8],
        dsts: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        return self
            .mm
            .CopyDataOutToIovsManual(self, src, dsts, allowPartial);
    }

    pub fn ZeroDataOutToIovs(
        &self,
        src: &[u8],
        dsts: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyDataOutToIovs(self, src, dsts, allowPartial);
    }

    pub fn CopyIovsOutToIovs(
        &self,
        srcs: &[IoVec],
        dsts: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyIovsOutToIovs(self, srcs, dsts, allowPartial);
    }

    pub fn CopyDataInFromIovs(
        &self,
        buf: &mut [u8],
        iovs: &[IoVec],
        allowPartial: bool,
    ) -> Result<usize> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyDataInFromIovs(&self, buf, iovs, allowPartial);
    }

    //Copy an Object from user memory
    pub fn CopyInObj<T: Sized + Copy>(&self, src: u64) -> Result<T> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyInObj(self, src);
    }

    pub fn CopyInObjShared<T: Sized + Copy>(&self, src: u64) -> Result<Box<T, GuestHostSharedAllocator>> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyInObjShared(self, src);
    }

    //Copy an Object to user memory
    pub fn CopyOutObj<T: Sized + Copy>(&self, src: &T, dst: u64) -> Result<()> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyOutObj(self, src, dst);
    }

    pub fn CopyOutObjManual<T: Sized + Copy>(&self, src: &T, dst: u64) -> Result<()> {
        return self.mm.CopyOutObjManual(self, src, dst);
    }

    pub fn CopyDataOut(&self, from: u64, vaddr: u64, len: usize, allowPartial: bool) -> Result<()> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyDataOut(self, from, vaddr, len, allowPartial);
    }

    //Copy an str to user memory
    pub fn CopyOutString(&self, vAddr: u64, len: usize, s: &str) -> Result<()> {
        assert!(self.Addr() == Task::Current().Addr());
        let str = CString::New(s);
        self.CopyOutSlice(str.Slice(), vAddr, len)
    }

    // CopyStringIn copies a NUL-terminated string of unknown length from the
    // memory mapped at addr in uio and returns it as a string (not including the
    // trailing NUL). If the length of the string, including the terminating NUL,
    // would exceed maxlen, CopyStringIn returns the string truncated to maxlen and
    // ENAMETOOLONG.
    pub fn CopyInString(&self, addr: u64, maxlen: usize) -> (String, Result<()>) {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyInString(self, addr, maxlen);
    }

    pub fn CopyInVector(
        &self,
        addr: u64,
        maxElemSize: usize,
        maxTotalSize: i32,
    ) -> Result<Vec<String>> {
        assert!(self.Addr() == Task::Current().Addr());
        return self.mm.CopyInVector(self, addr, maxElemSize, maxTotalSize);
    }

    pub fn AdjustIOVecPermission(
        &self,
        iovs: &[IoVec],
        writeReq: bool,
        allowPartial: bool,
    ) -> Result<Vec<IoVec>> {
        let mut vec = Vec::new();
        for iov in iovs {
            let len = self.CheckPermission(iov.start, iov.len as u64, writeReq, allowPartial)?;
            vec.push(IoVec {
                start: iov.start,
                len: len as usize,
            })
        }

        return Ok(vec);
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
                        return Err(e);
                    }
                    return Ok(output);
                }
                Ok(len) => output.push(IoVec {
                    start: iov.start,
                    len: len as _,
                }),
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
        if iovsnum > UIO_MAXIOV {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        return self.mm.CopyInVec(self, iovs, iovsnum);
    }

    pub fn V2P(
        &self,
        start: u64,
        len: u64,
        writable: bool,
        allowPartial: bool,
    ) -> Result<Vec<IoVec>> {
        return self.mm.V2P(self, start, len, writable, allowPartial);
    }
}
