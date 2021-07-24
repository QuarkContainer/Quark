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


use super::Kernel::HostSpace;
use super::qlib::common::*;
use super::task::*;
use super::qlib::linux_def::*;
use super::qlib::mem::io::*;

pub struct MemBuf<'a> {
    pub data: &'a mut [u8],
    pub offset: usize,
}

impl<'a> MemBuf<'a> {
    pub fn NewFromAddr(task: &'a Task, addr: u64, size: usize) -> Result<Self> {
        let slice = task.GetSliceMut::<u8>(addr, size)?;

        return Ok(Self {
            data: slice,
            offset: 0,
        })
    }

    pub fn New(data: &'a mut [u8]) -> Self {
        return Self {
            data: data,
            offset: 0,
        }
    }

    pub fn Len(&self) -> usize {
        return self.data.len();
    }
}

impl<'a> IOReader for MemBuf<'a> {
    fn Read(&mut self, buf: &mut [u8]) -> Result<i64> {
        let tmp = &self.data[self.offset..];

        let mut len = buf.len();
        if len > tmp.len() {
            len = tmp.len();
        }

        for i in 0..len {
            buf[i] = tmp[i]
        }

        self.offset += len;

        return Ok(len as i64)
    }
}

impl<'a> IOWriter for MemBuf<'a> {
    fn Write(&mut self, buf: &[u8]) -> Result<i64> {
        let tmp = &mut self.data[self.offset..];

        let mut len = buf.len();
        if len > tmp.len() {
            len = tmp.len();
        }

        for i in 0..len {
            tmp[i] = buf[i]
        }

        self.offset += len;

        return Ok(len as i64)
    }
}

pub fn IORead(fd: i32, buf: &[IoVec]) -> Result<i64> {
    if buf.len() == 0 {
        return Ok(0)
    }

    let iovsAddr = &buf[0] as *const _ as u64;
    let iovcnt = buf.len() as i32;

    let ret = HostSpace::IORead(fd, iovsAddr, iovcnt);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret)
}

pub fn IOReadAt(fd: i32, buf: &[IoVec], offset: u64) -> Result<i64> {
    if buf.len() == 0 {
        return Ok(0)
    }

    let iovsAddr = &buf[0] as *const _ as u64;
    let mut iovcnt = buf.len() as i32;
    if iovcnt > 1024 {
        iovcnt = 1024;
    }

    let ret = HostSpace::IOReadAt(fd, iovsAddr, iovcnt, offset);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret)
}

pub fn IOTTYRead(fd: i32, buf: &[IoVec]) -> Result<i64> {
    if buf.len() == 0 {
        return Ok(0)
    }

    let iovsAddr = &buf[0] as *const _ as u64;
    let mut iovcnt = buf.len() as i32;
    if iovcnt > 1024 {
        iovcnt = 1024;
    }

    let ret = HostSpace::IOTTYRead(fd, iovsAddr, iovcnt);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret)
}

pub fn IOWrite(fd: i32, buf: &[IoVec]) -> Result<i64> {
    if buf.len() == 0 {
        return Ok(0)
    }

    let iovsAddr = &buf[0] as *const _ as u64;
    let iovcnt = buf.len() as i32;
    let ret = HostSpace::IOWrite(fd, iovsAddr, iovcnt);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret)
}

pub fn IOWriteAt(fd: i32, buf: &[IoVec], offset: u64) -> Result<i64> {
    if buf.len() == 0 {
        return Ok(0)
    }

    let iovsAddr = &buf[0] as *const _ as u64;
    let mut iovcnt = buf.len() as i32;
    if iovcnt > 1024 {
        iovcnt = 1024;
    }
    let ret = HostSpace::IOWriteAt(fd, iovsAddr, iovcnt, offset);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    return Ok(ret)
}

pub struct RangeReader<'a> {
    r: &'a mut IOReaderAt,
    off: i64,
    limit: i64,
}

impl<'a> RangeReader<'a> {
    pub fn NewOffsetReader(r: &'a mut IOReaderAt, off: i64) -> Self {
        return Self {
            r,
            off,
            limit: -1,
        }
    }

    pub fn New(r: &'a mut IOReaderAt, off: i64, limit: i64) -> Self {
        assert!(off <= limit, "RangeReader");
        return Self {
            r,
            off,
            limit,
        }
    }
}

impl<'a> IOReader for RangeReader<'a> {
    fn Read(&mut self, buf: &mut [u8]) -> Result<i64> {
        let mut buf = buf;
        if self.limit >= 0 {
            let max = self.limit - self.off;
            if max < buf.len() as i64 {
                buf = &mut buf[..max as usize]
            }
        }

        let cnt = self.r.ReadAt(buf, self.off)?;
        self.off += cnt;

        return Ok(cnt)
    }
}

pub struct RangeWriter<'a> {
    w: &'a mut IOWriterAt,
    off: i64,
    limit: i64,
}

impl<'a> RangeWriter<'a> {
    pub fn NewOffsetWriter(w: &'a mut IOWriterAt, off: i64) -> Self {
        return Self {
            w,
            off,
            limit: -1,
        }
    }

    pub fn New(w: &'a mut IOWriterAt, off: i64, limit: i64) -> Self {
        assert!(off <= limit, "RangeWriter");
        return Self {
            w,
            off,
            limit,
        }
    }
}

impl<'a> IOWriter for RangeWriter<'a> {
    fn Write(&mut self, buf: &[u8]) -> Result<i64> {
        let mut buf = buf;
        if self.limit >= 0 {
            let max = self.limit - self.off;
            if max < buf.len() as i64 {
                buf = &buf[..max as usize]
            }
        }

        let cnt = self.w.WriteAt(buf, self.off)?;
        self.off += cnt;

        return Ok(cnt)
    }
}
