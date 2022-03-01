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

use alloc::string::String;
use alloc::vec::Vec;
use core::mem;

use super::super::auxv::*;
use super::super::common::*;
use super::task::*;

// StackLayout describes the location of the arguments and environment on the
// stack.
#[derive(Default, Debug)]
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

    pub fn PushType<T: Copy>(&mut self, task: &Task, data: &T) -> Result<u64> {
        let size = mem::size_of::<T>();
        self.sp -= size as u64;
        task.CopyOutObj(data, self.sp).unwrap();
        return Ok(self.sp);
    }

    pub fn PopType<T: Copy>(&mut self, task: &Task, data: &mut T) -> Result<u64> {
        let size = mem::size_of::<T>();
        *data = task.CopyInObj(self.sp)?;

        self.sp += size as u64;
        return Ok(self.sp);
    }

    pub fn PushStr(&mut self, task: &Task, str: &str) -> Result<u64> {
        let len = str.len();
        self.sp = self.sp - len as u64 - 1;
        task.CopyOutString(self.sp, len + 1, str)?;
        return Ok(self.sp)
    }

    pub fn PushU64(&mut self, task: &Task, val: u64) -> Result<u64> {
        self.sp = self.sp - 8;
        task.CopyOutObj(&val, self.sp)?;
        return Ok(self.sp)
    }

    pub fn PushU32(&mut self, task: &Task, val: u32) -> Result<u64> {
        self.sp = self.sp - 4;
        task.CopyOutObj(&val, self.sp)?;
        return Ok(self.sp)
    }

    pub fn PushU16(&mut self, task: &Task, val: u16) -> Result<u64> {
        self.sp = self.sp - 2;
        task.CopyOutObj(&val, self.sp)?;
        return Ok(self.sp)
    }

    pub fn PushU8(&mut self, task: &Task, val: u8) -> Result<u64> {
        self.sp = self.sp - 1;
        task.CopyOutObj(&val, self.sp)?;
        return Ok(self.sp)
    }

    pub fn Pad16(&mut self, _task: &Task) -> Result<u64> {
        let offset = self.sp & 0xf;
        self.sp -= offset;
        return Ok(self.sp)
    }

    // LoadEnv pushes the given args, env and aux vector to the stack using the
    // well-known format for a new executable. It returns the start and end
    // of the argument and environment vectors.
    pub fn LoadEnv(&mut self, task: &Task, envs: &[String], args: &[String], auxv: &[AuxEntry]) -> Result<StackLayout> {
        let mut l = StackLayout::default();

        // Make sure we start with a 16-byte alignment.
        self.Pad16(task)?;

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
            let addr = self.PushStr(task, envs[idx].as_str())?;
            envAddrs.push(addr);
        }
        l.EnvvStart = self.sp;

        // Push our args.
        l.ArgvEnd = self.sp;
        let mut argAddrs: Vec<u64> = Vec::new();
        for i in 0..args.len() {
            let idx = args.len() - i - 1;
            let addr = self.PushStr(task, args[idx].as_str())?;
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
            self.PushU64(task, auxv[i].Val)?;
            self.PushU64(task, auxv[i].Key as u64)?;
        }
        self.PushU64(task, 0)?;

        /*env*/
        for i in 0..envAddrs.len() {
            self.PushU64(task, envAddrs[i])?;
        }
        self.PushU64(task, 0)?;

        /*argv*/
        for i in 0..argAddrs.len() {
            self.PushU64(task, argAddrs[i])?;
        }

        /*argc*/
        self.PushU64(task, argAddrs.len() as u64)?;
        return Ok(l)
    }
}


pub struct KernelStack {
    pub sp: u64
}

impl KernelStack {
    pub fn New(addr: u64) -> Self {
        return Self {
            sp: addr
        }
    }

    pub fn PushU64(&mut self, val: u64) -> u64 {
        self.sp = self.sp - 8;
        unsafe {
            *(self.sp as * mut u64) = val;
        }
        return self.sp
    }

    pub fn Pad16(&mut self, _task: &Task) -> Result<u64> {
        let offset = self.sp & 0xf;
        self.sp -= offset;
        return Ok(self.sp)
    }
}