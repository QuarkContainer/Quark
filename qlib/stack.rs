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
use alloc::string::String;
use alloc::vec::Vec;
use core::mem;

// StackLayout describes the location of the arguments and environment on the
// stack.
#[derive(Default)]
pub struct StackLayout {
    // ArgvStart is the beginning of the argument vector.
    pub ArgvStart: u64,

    // ArgvEnd is the end of the argument vector.
    pub ArgvEnd: u64,

    // EnvvStart is the beginning of the environment vector.
    pub EnvvStart: u64,

    // EnvvEnd is the end of the environment vector.
    pub EvvvEnd: u64,
}

pub struct Stack {
    pub sp: u64
}

impl Stack {
    pub fn New(addr: u64) -> Self {
        return Stack {
            sp: addr
        }
    }

    pub fn PushType<T>(&mut self, data: &T) -> u64 {
        let size = mem::size_of::<T>();
        self.sp -= size as u64;
        let to = unsafe {
            slice::from_raw_parts_mut(self.sp as *mut u8, size)
        };
        let from = unsafe {
            slice::from_raw_parts(data as *const _ as *const u8, size)
        };
        for i in 0..size {
            to[i] = from[i]
        }

        return self.sp;
    }

    pub fn PopType<T>(&mut self, data: &mut T) -> u64 {
        let size = mem::size_of::<T>();
        let to = unsafe {
            slice::from_raw_parts_mut(data as *const _ as *mut u8, size)
        };
        let from = unsafe {
            slice::from_raw_parts(self.sp as *const u8, size)
        };

        for i in 0..size {
            to[i] = from[i]
        }
        self.sp += size as u64;
        return self.sp;
    }

    pub fn PushStr(&mut self, str: &str) -> u64 {
        let len = str.len();
        self.sp = self.sp - len as u64 - 1;
        unsafe {
            //let to = slice::from_raw_parts_mut(self.sp as *mut u8, len + 1);
            let to = slice::from_raw_parts_mut(self.sp as *mut u8, len);
            let from = slice::from_raw_parts(str.as_ptr(), len);
            to[0..len].clone_from_slice(from);

            //to[len] = 0;
        }
        return self.sp
    }

    pub fn PushU64(&mut self, val: u64) -> u64 {
        self.sp = self.sp - 8;
        unsafe {
            *(self.sp as *mut u64) = val
        }
        return self.sp
    }

    pub fn PushU32(&mut self, val: u32) -> u64 {
        self.sp = self.sp - 4;
        unsafe {
            *(self.sp as *mut u32) = val
        }
        return self.sp
    }

    pub fn PushU16(&mut self, val: u16) -> u64 {
        self.sp = self.sp - 2;
        unsafe {
            *(self.sp as *mut u16) = val
        }
        return self.sp
    }

    pub fn PushU8(&mut self, val: u8) -> u64 {
        self.sp = self.sp - 1;
        unsafe {
            *(self.sp as *mut u8) = val
        }
        return self.sp
    }

    pub fn Pad16(&mut self) -> u64 {
        let offset = self.sp & 0xf;
        self.sp -= offset;
        return self.sp
    }

    // LoadEnv pushes the given args, env and aux vector to the stack using the
    // well-known format for a new executable. It returns the start and end
    // of the argument and environment vectors.
    pub fn LoadEnv(&mut self, envs: &[String], args: &[String], auxv: &[AuxEntry]) -> StackLayout {
        let mut l = StackLayout::default();

        // Make sure we start with a 16-byte alignment.
        self.Pad16();

        // Push the environment vector so the end of the argument vector is adjacent to
        // the beginning of the environment vector.
        // While the System V abi for x86_64 does not specify an ordering to the
        // Information Block (the block holding the arg, env, and aux vectors),
        // support features like setproctitle(3) naturally expect these segments
        // to be in this order. See: https://www.uclibc.org/docs/psABI-x86_64.pdf
        // page 29.
        l.EvvvEnd = self.sp;
        let mut envAddrs = Vec::new();
        for i in 0..envs.len() {
            let idx = envs.len() - i - 1;
            let addr = self.PushStr(envs[idx].as_str());
            envAddrs.push(addr);
        }
        l.EnvvStart = self.sp;

        // Push our args.
        l.ArgvEnd = self.sp;
        let mut argAddrs: Vec<u64> = Vec::new();
        for i in 0..args.len() {
            let idx = args.len() - i - 1;
            let addr = self.PushStr(args[idx].as_str());
            argAddrs.push(addr);
        }
        l.ArgvStart = self.sp;

        // We need to align the arguments appropriately.
        //
        // We must finish on a 16-byte alignment, but we'll play it
        // conservatively and finish at 32-bytes. It would be nice to be able
        // to call Align here, but unfortunately we need to align the stack
        // with all the variable sized arrays pushed. So we just need to do
        // some calculations.
        let argvSize = 8 * (args.len() + 1);
        let envvSize = 8 * (envs.len() + 1);
        let auxvSize = 8 * 2 * (auxv.len() + 1);
        let total = argvSize + envvSize + auxvSize + 8;
        let expectedBottom = self.sp - total as u64;
        if expectedBottom % 32 != 0 {
            self.sp -= expectedBottom % 32;
        }

        for i in 0..auxv.len() {
            self.PushU64(auxv[i].Val);
            self.PushU64(auxv[i].Key as u64);
        }
        self.PushU64(0);

        /*env*/
        for i in 0..envAddrs.len() {
            self.PushU64(envAddrs[i]);
        }
        self.PushU64(0);

        /*argv*/
        for i in 0..argAddrs.len() {
            self.PushU64(argAddrs[i]);
        }

        /*argc*/
        self.PushU64(argAddrs.len() as u64);
        return l
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub enum AuxVec {
    AT_NULL = 0,

    // AT_IGNORE should be ignored.
    AT_IGNORE = 1,

    // AT_EXECFD is the file descriptor of the program.
    AT_EXECFD = 2,

    // AT_PHDR points to the program headers.
    AT_PHDR = 3,

    // AT_PHENT is the size of a program header entry.
    AT_PHENT = 4,

    // AT_PHNUM is the number of program headers.
    AT_PHNUM = 5,

    // AT_PAGESZ is the system page size.
    AT_PAGESZ = 6,

    // AT_BASE is the base address of the interpreter.
    AT_BASE = 7,

    // AT_FLAGS are flags.
    AT_FLAGS = 8,

    // AT_ENTRY is the program entry point.
    AT_ENTRY = 9,

    // AT_NOTELF indicates that the program is not an ELF binary.
    AT_NOTELF = 10,

    // AT_UID is the real UID.
    AT_UID = 11,

    // AT_EUID is the effective UID.
    AT_EUID = 12,

    // AT_GID is the real GID.
    AT_GID = 13,

    // AT_EGID is the effective GID.
    AT_EGID = 14,

    // AT_PLATFORM is a string identifying the CPU.
    AT_PLATFORM = 15,

    // AT_HWCAP are arch-dependent CPU capabilities.
    AT_HWCAP = 16,

    // AT_CLKTCK is the frequency used by times(2).
    AT_CLKTCK = 17,

    // AT_SECURE indicate secure mode.
    AT_SECURE = 23,

    // AT_BASE_PLATFORM is a string identifying the "real" platform. It may
    // differ from AT_PLATFORM.
    AT_BASE_PLATFORM = 24,

    // AT_RANDOM points to 16-bytes of random data.
    AT_RANDOM = 25,

    // AT_HWCAP2 is an extension of AT_HWCAP.
    AT_HWCAP2 = 26,

    // AT_EXECFN is the path used to execute the program.
    AT_EXECFN = 31,

    // AT_SYSINFO_EHDR is the address of the VDSO.
    AT_SYSINFO_EHDR = 33,
}

#[derive(Debug, Copy, Clone)]
pub struct AuxEntry {
    pub Key: AuxVec,
    pub Val: u64,
}

