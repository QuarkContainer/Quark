use alloc::vec::Vec;
use  core::slice;

use super::qlib::mem::seq::BlockSeq;
use super::qlib::linux_def::*;
use super::BUF_MGR;

pub struct DataBuff {
    pub buf: Vec<u8>
}

impl DataBuff {
    pub fn New(size: usize) -> Self {
        let mut buf = Vec::with_capacity(size);
        unsafe {
            buf.set_len(size);
        }

        return Self {
            buf: buf
        }
    }

    pub fn Zero(&mut self) {
        for i in 0..self.buf.len() {
            self.buf[i] = 0;
        }
    }

    pub fn Buf(&mut self) -> &'static mut[u8] {
        let ptr = &mut self.buf[0] as * mut u8;
        let toSlice = unsafe {
            slice::from_raw_parts_mut (ptr, self.buf.len())
        };

        return toSlice;
    }

    pub fn Ptr(&self) -> u64 {
        return self.buf.as_ptr() as u64;
    }

    pub fn Len(&self) -> usize {
        return self.buf.len()
    }

    pub fn IoVec(&self) -> IoVec {
        if self.Len() == 0 {
            return IoVec::NewFromAddr(0, 0)
        }

        return IoVec {
            start: self.Ptr(),
            len: self.Len(),
        }
    }

    pub fn Iovs(&self) -> [IoVec; 1] {
        return [self.IoVec()]
    }


    pub fn BlockSeq(&self) -> BlockSeq {
        return BlockSeq::New(&self.buf);
    }
}

pub struct IOBuff {
    pub addr: u64,
    pub size: usize,
}

impl Drop for IOBuff {
    fn drop(&mut self) {
        BUF_MGR.Free(self.addr, self.size as u64)
    }
}

impl IOBuff {
    pub fn New(size: usize) -> Self {
        let addr = BUF_MGR.Alloc(size as u64).expect("IOBuff allocate fail");

        return Self {
            addr: addr,
            size: size
        }
    }

    pub fn Buf(&self) -> &'static mut[u8] {
        let ptr = self.addr as * mut u8;
        let toSlice = unsafe {
            slice::from_raw_parts_mut (ptr, self.size)
        };

        return toSlice;
    }

    pub fn Ptr(&self) -> u64 {
        return self.addr;
    }

    pub fn Len(&self) -> usize {
        return self.size
    }

    pub fn IoVec(&self) -> IoVec {
        if self.Len() == 0 {
            return IoVec::NewFromAddr(0, 0)
        }

        return IoVec {
            start: self.Ptr(),
            len: self.Len(),
        }
    }

    pub fn Iovs(&self) -> [IoVec; 1] {
        return [self.IoVec()]
    }


    pub fn BlockSeq(&self) -> BlockSeq {
        return BlockSeq::NewFromBlock(self.IoVec());
    }
}