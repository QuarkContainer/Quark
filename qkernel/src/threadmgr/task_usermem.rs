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

use alloc::str;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::slice;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::util::cstring::*;
use super::super::qlib::addr::*;
use super::super::qlib::mem::seq::*;
use super::super::task::*;

impl Task {
    //Copy a vec from user memory
    pub fn CopyIn<T: Sized + Copy>(&self, addr: u64, size: usize) -> Result<Vec<T>> {
        if addr == 0 && size == 0 {
            return Ok(Vec::new())
        }

        let mut res = Vec::with_capacity(size);
        let slice = self.GetSlice::<T>(addr, size)?;

        for i in 0..size {
            res.push(slice[i]);
        }

        return Ok(res);
    }

    //Copy a slice to user memory
    pub fn CopyOutSlice<T: Sized + Copy>(&self, src: &[T], dst: u64, len: usize) -> Result<()> {
        if len < src.len() {
            return Err(Error::SysError(SysErr::ERANGE));
        }

        let dst = self.GetSliceMut::<T>(dst, src.len())?;
        for i in 0..dst.len() {
            dst[i] = src[i]
        }

        return Ok(())
    }

    //Copy an Object from user memory
    pub fn CopyInObj<T: Sized + Copy>(&self, src: u64, dst: &mut T) -> Result<()> {
        *dst = *self.GetType::<T>(src)?;
        return Ok(())
    }

    //Copy an Object to user memory
    pub fn CopyOutObj<T: Sized + Copy>(&self, src: &T, dst: u64) -> Result<()> {
        *self.GetTypeMut::<T>(dst)? = *src;
        return Ok(())
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
        let maxlen = match self.MaxMappedAddr(addr, maxlen as u64) {
            Err(e) => return ("".to_string(), Err(e)),
            Ok(l) => l
        };

        let slice = match self.GetSlice::<u8>(addr, maxlen) {
            Err(e) => return ("".to_string(), Err(e)),
            Ok(s) => s,
        };

        for i in 0..maxlen {
            if slice[i] == 0 {
                return (str::from_utf8(&slice[0..i]).unwrap().to_string(), Ok(()));
            }
        }

        return (str::from_utf8(&slice[0..maxlen]).unwrap().to_string(), Err(Error::SysError(SysErr::ENAMETOOLONG)));
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
    pub fn CopyInVector(&self, addr: u64, maxElemSize: usize, maxTotalSize: i32) -> Result<Vec<String>> {
        if addr == 0 {
            return Ok(Vec::new());
        }

        let mut maxTotalSize = maxTotalSize;

        let maxlen = self.MaxMappedAddr(addr, maxElemSize as u64 * 8)?;
        let addresses = self.GetSlice::<u64>(addr, maxlen/8)?;

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

            let maxlen = self.MaxMappedAddr(ptr, thisMax as u64)?;
            let (str, err) = self.CopyInString(ptr, maxlen);
            match err {
                Err(e) => return Err(e),
                _ => (),
            }

            let strlen = str.len();
            v.push(str);
            maxTotalSize -= (strlen as i32) + 1;
        }

        return Ok(v)
    }

    pub fn GetSlice<T: Sized>(&self, vAddr: u64, count: usize) -> Result<&[T]> {
        let recordLen = core::mem::size_of::<T>();
        let len = self.CheckPermission(vAddr, count as u64 * recordLen as u64, false, true)?;

        let t: *const T = vAddr as *const T;
        let slice = unsafe { slice::from_raw_parts(t, (len as usize) / recordLen) };
        return Ok(slice)
    }

    pub fn GetSliceMut<T: Sized>(&self, vAddr: u64, count: usize) -> Result<&mut [T]> {
        let recordLen = core::mem::size_of::<T>();
        // only check whether the address is valid, if readonly, will cow
        let len = self.CheckPermission(vAddr, count as u64 * recordLen as u64, true, true)?;

        let t: *mut T = vAddr as *mut T;
        let slice = unsafe { slice::from_raw_parts_mut(t, (len as usize) / recordLen) };
        return Ok(slice)
    }

    pub fn CheckIOVecPermission(&self, iovs: &[IoVec], writeReq: bool) -> Result<()> {
        for iov in iovs {
            self.CheckPermission(iov.start, iov.len as u64, writeReq, false)?;
        }

        return Ok(())
    }

    pub fn GetType<T: Sized>(&self, vAddr: u64) -> Result<&T> {
        let len = core::mem::size_of::<T>();
        self.CheckPermission(vAddr, len as u64, false, false)?;

        let t: *const T = vAddr as *const T;

        return Ok(unsafe { &(*t) })
    }

    pub fn GetTypeMut<T: Sized>(&self, vAddr: u64) -> Result<&mut T> {
        let len = core::mem::size_of::<T>();
        // only check whether the address is valid, if readonly, will cow
        self.CheckPermission(vAddr, len as u64, false, false)?;

        let t: *mut T = vAddr as *mut T;

        return Ok(unsafe { &mut (*t) })
    }

    pub fn MaxMappedAddr(&self, vAddr: u64, len: u64) -> Result<usize> {
        let mut addr = Addr(vAddr).RoundDown()?.0;
        while addr <= vAddr + len {
            let (_, _) = match self.mm.VirtualToPhy(addr) {
                Err(_) => {
                    if addr < vAddr {
                        return Err(Error::SysError(SysErr::EFAULT))
                    } else {
                        return Ok((addr - vAddr) as usize);
                    }
                }
                Ok((a, w)) => (a, w)
            };

            addr += MemoryDef::PAGE_SIZE;
        }

        return Ok(len as usize)
    }

    // check whether the address range is legal.
    // 1. whether the range belong to user's space
    // 2. Whether the read/write permission meet requirement
    // 3. if need cow, fix the page.
    pub fn CheckPermission(&self, vAddr: u64, len: u64, writeReq: bool, allowPartial: bool) -> Result<u64> {
        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT))
        }

        return self.mm.FixPermission(self, vAddr, len, writeReq, allowPartial)
    }

    #[cfg(not(test))]
    pub fn VirtualToPhy(&self, vAddr: u64) -> Result<u64> {
        let (addr, _) = self.mm.VirtualToPhy(vAddr)?;
        return Ok(addr);
    }

    #[cfg(test)]
    pub fn VirtualToPhy(&self, vAddr: u64) -> Result<u64> {
        return Ok(vAddr)
    }

    pub fn IovsFromAddr(&self, iovs: u64, iovsnum: usize) -> Result<&mut [IoVec]> {
        return self.GetSliceMut::<IoVec>(iovs, iovsnum);
    }

    pub fn V2P(&self, start: u64, len: u64, output: &mut Vec<IoVec>, writable: bool) -> Result<()> {
        if len == 0 {
            return Ok(())
        }

        self.CheckPermission(start, len, writable, false)?;

        //etcd has such weird call, handle that with special case
        if len == 0 {
            match self.VirtualToPhy(start) {
                Err(e) => {
                    info!("convert to phyaddress fail, addr = {:x} e={:?}", start, e);
                    return Err(Error::SysError(SysErr::EFAULT))
                }
                Ok(pAddr) => {
                    output.push(IoVec {
                        start: pAddr,
                        len: 0, //iov.len,
                    });

                }
            }

            return Ok(())
        }

        let mut start = start;
        let end = start + len;

        while start < end {
            let next = if Addr(start).IsPageAligned() {
                start + MemoryDef::PAGE_SIZE
            } else {
                Addr(start).RoundUp().unwrap().0
            };

            match self.VirtualToPhy(start) {
                Err(e) => {
                    info!("convert to phyaddress fail, addr = {:x} e={:?}", start, e);
                    return Err(Error::SysError(SysErr::EFAULT))
                }
                Ok(pAddr) => {
                    let iov = IoVec {
                        start: pAddr,
                        len: if end < next {
                            (end - start) as usize
                        } else {
                            (next - start) as usize
                        },
                    };

                    let cnt = output.len();
                    if cnt > 0 && output[cnt-1].End() == iov.start {
                        // use the last entry
                        output[cnt-1].len += iov.len;
                    } else {
                        output.push(iov);
                    }
                }
            }

            start = next;
        }

        return Ok(())
    }

    pub fn V2PIov(&self, iov: &IoVec, output: &mut Vec<IoVec>, writable: bool) -> Result<()> {
        return self.V2P(iov.start, iov.len as u64, output, writable)
    }

    pub fn V2PIovs(&self, iovs: &[IoVec], writable: bool, output: &mut Vec<IoVec>) -> Result<()> {
        for iov in iovs {
            self.V2PIov(iov, output, writable)?;
        }

        return Ok(())
    }

    pub fn V2PBlockSeq(&self, bs: BlockSeq, output: &mut Vec<IoVec>, writable: bool) -> Result<()> {
        let mut bs = bs;
        while !bs.IsEmpty() {
            let iov = bs.Head();
            self.V2PIov(&iov, output, writable)?;

            bs = bs.Tail();
        }

        return Ok(())
    }
}