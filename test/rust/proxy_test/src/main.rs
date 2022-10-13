#![allow(non_snake_case)]

use core::arch::asm;

fn main() {
    println!("Hello, world!");
    Test();
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Cmd1In {
    pub val: u64,
}

#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct Cmd1Out {
    pub val1: u64,
    pub val2: u64,
}

pub fn Test() {
    let dataIn = Cmd1In {
        val: 123
    };
    let mut dataOut = Cmd1Out::default();

    let ret = Proxy(1, &dataIn as * const _ as u64, &mut dataOut as * mut _ as u64);
    println!("output {:?}/{}", dataOut, ret);
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

pub fn Proxy(cmd: u64, addrIn: u64, addrOut:u64) -> i64 {
    return unsafe {
        syscall4(10003, cmd as usize, addrIn as _, addrOut as usize, 0 as usize) as i64
    }
}