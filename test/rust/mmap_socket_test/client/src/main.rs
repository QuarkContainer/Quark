#![allow(non_snake_case)]
#![feature(asm)]

use libc::*;
use std::ptr;
use std::cmp;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SockAddrInet {
    pub Family: u16,
    pub Port: u16,
    pub Addr: [u8; 4],
    pub Zero: [u8; 8], // pad to sizeof(struct sockaddr).
}

fn main() {
    println!("Hello, world!");
    Client();
}

pub const MMAP : bool = true;
pub const MAP_SOCKT_READ : i32 = 1 << 31;

pub fn Client() {
    let fd = unsafe {
        socket(libc::AF_INET, libc::SOCK_STREAM, 0)
    };

    assert!(fd > 0);

    let addr = SockAddrInet {
        Family: libc::AF_INET as _,
        Port: 1234,
        Addr: [127, 0, 0, 1],
        Zero: [0, 0, 0, 0, 0, 0, 0, 0],
    };

    let ret = unsafe {
        libc::connect(fd, &addr as * const _ as u64 as _, 16)
    };
    println!("connect result {}", ret);

    let str = "hello world!";
    let bytes = str.as_bytes();
    let mut buf = Vec::with_capacity(1024);

    unsafe {
        buf.set_len(1024);
    }

    let mut readIovs = RingBufIovs::default();
    let mut writeIovs = RingBufIovs::default();

    let mut _readAddr = 0;
    let mut writeAddr = 0;
    
   if MMAP {
        _readAddr = unsafe {
            libc::mmap(
                0 as _, 
                65536, 
                libc::PROT_READ, 
                libc::MAP_SHARED | MAP_SOCKT_READ,
                fd,
                0
            )
        } as u64;

        writeAddr = unsafe {
            libc::mmap(
                0 as _, 
                65536, 
                libc::PROT_READ | libc::PROT_WRITE, 
                libc::MAP_SHARED,
                fd,
                0
            )
        } as u64;

        println!("MMAP readaddr is {:x}/write addr {:x}", _readAddr, writeAddr);
        
        let ret1 = SocketProduce(fd, 0, &mut writeIovs, libc::SOCK_NONBLOCK);
        println!("writeIovs {}/{:x?}", ret1, &writeIovs);

        let ret2 = SocketConsume(fd, 0, &mut readIovs, libc::SOCK_NONBLOCK);
        println!("readIovs {}/{:x?}", ret2, &readIovs);
    }

    let mut readCount = 0;

    for _i in 0..100000 {
        let wcnt = if MMAP {
            let count = writeIovs.CopyIn(writeAddr, &bytes);
            SocketProduce(fd, count as i32, &mut writeIovs, 0)
        } else {
            unsafe {
                write(fd, &bytes[0] as * const _ as u64 as * const c_void, bytes.len() as _)
            }
        };
            
        let rcnt = if MMAP {
            readCount = SocketConsume(fd, readCount as i32, &mut readIovs, 0);
            readCount
        } else {
            unsafe {
                read(fd, &mut buf[0] as * mut c_void, buf.len() as _)
            }
        };
   
        //println!("process result {}/{}/{}", i, wcnt, rcnt);
        
        if wcnt < 0 || rcnt < 0 {
            return 
        } 
    }
    
    let ret = unsafe {
        close(fd)
    };
    println!("close result {}", ret);
}

#[repr(C)]
#[derive(Clone, Default, Debug, Copy, Eq, PartialEq)]
pub struct IoVec {
    pub start: u64,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RingBufIovs {
    pub iovs: [IoVec; 2],
    pub iovcnt: i32,
}

impl RingBufIovs {
    pub fn CopyIn(&self, addr: u64, data: &[u8]) -> usize {
        if self.iovcnt == 0 {
            return 0;
        }

        let mut count = cmp::min(data.len(), self.iovs[0].len);
        unsafe {
            ptr::copy_nonoverlapping(&data[0], (addr + self.iovs[0].start) as _, count);
        }

        if self.iovcnt > 1 && data.len() > count {
            let more = cmp::min(data.len()-count, self.iovs[1].len);
            unsafe {
                ptr::copy_nonoverlapping(&data[count], (addr + self.iovs[1].start) as _, more);
            }
            count += more;
        }
        return count;
    }
}

#[inline(always)]
pub unsafe fn syscall4(n: usize, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
    let ret: usize;
    asm!(
        "syscall",
        inlateout("rax") n as usize => ret,
        in("rdi") a1,
        in("rsi") a2,
        in("rdx") a3,
        in("r10") a4,
    );
    ret
}


pub fn SocketProduce(fd: i32, count: i32, iovs: &mut RingBufIovs, flags: i32) -> isize {
    return unsafe {
        syscall4(10001, fd as usize, count as _, iovs as * const _ as usize, flags as usize) as isize
    }
}

pub fn SocketConsume(fd: i32, count: i32, iovs: &mut RingBufIovs, flags: i32) -> isize {
    return unsafe {
        syscall4(10002, fd as usize, count as _, iovs as * const _ as usize, flags as usize) as isize
    }
}