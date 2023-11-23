// Copyright (c) 2021 Quark Container Authors / https://github.com/gz/backtracer
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

#[derive(Debug, Clone)]
pub struct Frame {
    pub rbp: u64,
    pub rsp: u64,
    pub rip: u64,
}

impl Frame {
    pub fn new(rbp: u64, rsp: u64, rip: u64) -> Frame {
        Frame {
            rbp: rbp,
            rsp: rsp,
            rip: rip,
        }
    }

    pub fn ip(&self) -> *mut u8 {
        (self.rip - 1) as *mut u8
    }

    pub fn symbol_address(&self) -> *mut u8 {
        0 as *mut u8
    }
}

#[inline(always)]
pub fn trace_from(mut curframe: Frame, cb: &mut dyn FnMut(&Frame) -> bool) {
    for _ in 0..20 {
        let ctxt = curframe.clone();

        let keep_going = cb(&ctxt);

        if keep_going {
            unsafe {
                curframe.rip = *((curframe.rbp + 8) as *mut u64);
                curframe.rsp = curframe.rbp;
                curframe.rbp = *(curframe.rbp as *mut u64);

                if curframe.rip == 0 || curframe.rbp <= 0xfff {
                    break;
                }
            }
        } else {
            break;
        }
    }
}

#[inline(always)]
pub fn trace(rip: u64, rsp: u64, rbp: u64, cb: &mut dyn FnMut(&Frame) -> bool) {
    let curframe = Frame::new(rbp, rsp, rip);
    trace_from(curframe, cb);
}

/*
#[inline(always)]
pub fn trace(cb: &mut dyn FnMut(&Frame) -> bool) {
    use x86::current::registers;
    let curframe = Frame::new(registers::rbp(), registers::rsp(), registers::rip());
    trace_from(curframe.clone(), cb);
}
*/
