// Copyright (c) 2021 Quark Container Authors
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

use core::alloc::{GlobalAlloc, Layout};
use core::ops::Deref;
use buddy_system_allocator::Heap;
use spin::Mutex;
use core::cmp::max;
use core::mem::size_of;
use core::ptr::NonNull;

use super::super::asm::*;
use super::super::qlib::common::*;

pub const BUF_CNT: usize = 16;
pub const CLASS_CNT : usize = 32;
pub const ENABLE_BUF : bool = true;

pub struct SliceStack128  {
    pub arr: [u64; 128],
    pub top: usize,
    pub count: u64,
    pub hit: u64,
}

impl SliceStack128 {
    pub const fn New() -> Self {
        return Self {
            arr: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            top: 0,
            count: 0,
            hit: 0,
        }
    }

    fn Push(&mut self, data: u64) -> Result<()> {
        self.count += 1;
        if let Some(slot) = self.arr.get_mut(self.top) {
            *slot = data;
            self.top += 1;
            self.hit += 1;
            return Ok(())
        }

        return Err(Error::QueueFull)
    }

    fn Pop(&mut self) -> Result<u64> {
        self.count += 1;
        if self.top == 0 {
            return Err(Error::NoData)
        }

        self.top -= 1;
        if let Some(slot) = self.arr.get(self.top) {
            self.hit += 1;
            return Ok(*slot)
        }

        panic!("SliceStack nodata");
    }

    fn PerfData(&self) -> (u64, u64) {
        return (self.count, self.hit)
    }
}

pub struct SliceStack64  {
    pub arr: [u64; 64],
    pub top: usize,
    pub count: u64,
    pub hit: u64,
}

impl SliceStack64  {
    pub const fn New() -> Self {
        return Self {
            arr: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            top: 0,
            count: 0,
            hit: 0,
        }
    }

    fn Push(&mut self, data: u64) -> Result<()> {
        self.count += 1;

        if let Some(slot) = self.arr.get_mut(self.top) {
            *slot = data;
            self.top += 1;
            self.hit += 1;
            return Ok(())
        }

        return Err(Error::QueueFull)
    }

    fn Pop(&mut self) -> Result<u64> {
        self.count += 1;
        if self.top == 0 {
            return Err(Error::NoData)
        }

        self.top -= 1;
        if let Some(slot) = self.arr.get(self.top) {
            self.hit += 1;
            return Ok(*slot)
        }

        panic!("SliceStack nodata");
    }

    fn PerfData(&self) -> (u64, u64) {
        return (self.count, self.hit)
    }
}

pub struct SliceStack32  {
    pub arr: [u64; 32],
    pub top: usize,
    pub count: u64,
    pub hit: u64,
}

impl SliceStack32  {
    pub const fn New() -> Self {
        return Self {
            arr: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            top: 0,
            count: 0,
            hit: 0,
        }
    }

    fn Push(&mut self, data: u64) -> Result<()> {
        self.count += 1;
        if let Some(slot) = self.arr.get_mut(self.top) {
            *slot = data;
            self.hit += 1;
            self.top += 1;
            return Ok(())
        }

        return Err(Error::QueueFull)
    }

    fn Pop(&mut self) -> Result<u64> {
        self.count += 1;
        if self.top == 0 {
            return Err(Error::NoData)
        }

        self.top -= 1;
        if let Some(slot) = self.arr.get(self.top) {
            self.hit += 1;
            return Ok(*slot)
        }

        panic!("SliceStack nodata");
    }

    fn PerfData(&self) -> (u64, u64) {
        return (self.count, self.hit)
    }
}

#[derive(Default)]
pub struct SliceStack16  {
    pub arr: [u64; 16],
    pub top: usize,
    pub count: u64,
    pub hit: u64,
}

impl SliceStack16  {
    pub const fn New() -> Self {
        return Self {
            arr: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            top: 0,
            count: 0,
            hit: 0,
        }
    }

    fn Push(&mut self, data: u64) -> Result<()> {
        self.count += 1;
        if let Some(slot) = self.arr.get_mut(self.top) {
            *slot = data;
            self.hit += 1;
            self.top += 1;
            return Ok(())
        }

        return Err(Error::QueueFull)
    }

    fn Pop(&mut self) -> Result<u64> {
        self.count += 1;
        if self.top == 0 {
            return Err(Error::NoData)
        }

        self.top -= 1;
        if let Some(slot) = self.arr.get(self.top) {
            self.hit += 1;
            return Ok(*slot)
        }

        panic!("SliceStack nodata");
    }

    fn PerfData(&self) -> (u64, u64) {
        return (self.count, self.hit)
    }
}

pub struct StackHeapInternal {
    pub buf3    : SliceStack128,
    pub buf4    : SliceStack128,
    pub buf5    : SliceStack128,
    pub buf6    : SliceStack64,
    pub buf7    : SliceStack64,
    pub buf8    : SliceStack64,
    pub buf9    : SliceStack32,
    pub buf10   : SliceStack32,
    pub buf11   : SliceStack16,
    pub buf12   : SliceStack128,
    pub buf13   : SliceStack16,

    pub heap    : Heap,
    pub count   : i64,
    pub hit     : i64,
    pub time    : i64,
}

pub struct StackHeap(Mutex<StackHeapInternal>);

impl Deref for StackHeap {
    type Target = Mutex<StackHeapInternal>;

    fn deref(&self) -> &Mutex<StackHeapInternal> {
        &self.0
    }
}

impl StackHeap {
    pub const fn Empty() -> Self {
        let internal = StackHeapInternal {
            buf3: SliceStack128::New(),
            buf4: SliceStack128::New(),
            buf5: SliceStack128::New(),
            buf6: SliceStack64::New(),
            buf7: SliceStack64::New(),
            buf8: SliceStack64::New(),
            buf9: SliceStack32::New(),
            buf10: SliceStack32::New(),
            buf11: SliceStack16::New(),
            buf12: SliceStack128::New(),
            buf13: SliceStack16::New(),

            heap: Heap::empty(),
            count: 0,
            hit: 0,
            time: 0,
        };

        return Self(Mutex::new(internal))
    }

    pub fn AddToHead(&self, start: usize, end: usize) {
        let mut intern = self.lock();
        unsafe {
            intern.heap.add_to_heap(start, end);
        }
    }

    pub fn Add(&self, start: usize, size: usize) {
        self.AddToHead(start, start + size)
    }

    pub fn Print(&self) {
        let time;
        let count;
        let hit;
        {
            let intern =self.lock();
            time = intern.time;
            count = intern.count;
            hit = intern.hit;
        }

        let mut str = format!("");
        let data = self.lock().buf3.PerfData();
        str += &format!("{}/{:?}", 8, data);
        let data = self.lock().buf4.PerfData();
        str += &format!("{}/{:?}", 16, data);
        let data = self.lock().buf5.PerfData();
        str += &format!("{}/{:?}", 32, data);
        let data = self.lock().buf6.PerfData();
        str += &format!("{}/{:?}", 64, data);
        let data = self.lock().buf7.PerfData();
        str += &format!("{}/{:?}", 128, data);
        let data = self.lock().buf8.PerfData();
        str += &format!("{}/{:?}", 256, data);
        let data = self.lock().buf9.PerfData();
        str += &format!("{}/{:?}", 512, data);
        let data = self.lock().buf10.PerfData();
        str += &format!("{}/{:?}", 1024, data);
        let data = self.lock().buf11.PerfData();
        str += &format!("{}/{:?}", 2048, data);
        let data = self.lock().buf12.PerfData();
        str += &format!("{}/{:?}", 4096, data);
        let data = self.lock().buf13.PerfData();
        str += &format!("{}/{:?}", 8192, data);

        error!("GlobalAlloc::Print All/{}/{}/{}, {}", time / FACTOR as i64, count, hit, str);
    }
}

unsafe impl GlobalAlloc for StackHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let start = Rdtsc();
        let mut intern = self.lock();

        let mut hit = 0;

        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );

        let class = size.trailing_zeros() as usize;

        let ret = match class {
            3 => intern.buf3.Pop(),
            4 => intern.buf4.Pop(),
            5 => intern.buf5.Pop(),
            6 => intern.buf6.Pop(),
            7 => intern.buf7.Pop(),
            8 => intern.buf8.Pop(),
            9 => intern.buf9.Pop(),
            10 => intern.buf10.Pop(),
            11 => intern.buf11.Pop(),
            12 => intern.buf12.Pop(),
            13 => intern.buf13.Pop(),
            _ => Err(Error::NoData)
        };

        let addr = match ret {
            Ok(addr) => {
                hit = 1;
                addr
            },
            _ => {
                let ret = intern
                    .heap
                    .alloc(layout)
                    .ok()
                    .map_or(0 as *mut u8, |allocation| allocation.as_ptr()) as u64;

                if ret == 0 {
                    super::super::Kernel::HostSpace::KernelMsg(ret);
                    super::super::Kernel::HostSpace::KernelOOM();
                    loop {}
                }
                ret
            }
        };

        let current = Rdtsc();
        intern.time += current - start;
        intern.count += 1;
        intern.hit += hit;

        return addr as *mut u8;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let start = Rdtsc();
        let mut intern = self.lock();

        let mut hit = 0;

        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        let addr = ptr as u64;

        let ret = match class {
            3 => intern.buf3.Push(addr),
            4 => intern.buf4.Push(addr),
            5 => intern.buf5.Push(addr),
            6 => intern.buf6.Push(addr),
            7 => intern.buf7.Push(addr),
            8 => intern.buf8.Push(addr),
            9 => intern.buf9.Push(addr),
            10 => intern.buf10.Push(addr),
            11 => intern.buf11.Push(addr),
            12 => intern.buf12.Push(addr),
            13 => intern.buf13.Push(addr),
            _ => Err(Error::QueueFull)
        };

        match ret {
            Ok(()) => {
                hit = 1;
            },
            _ => {
                intern.heap.dealloc(NonNull::new_unchecked(ptr), layout)
            }
        }

        let current = Rdtsc();
        intern.time += current - start;
        intern.count += 1;
        intern.hit += hit;
    }
}

#[repr(align(128))]
pub struct BufHeapInternal {
    pub bufs: [[u64; BUF_CNT]; CLASS_CNT],
    pub tops: [usize; CLASS_CNT],
    pub times: [(u64, u64, u64); CLASS_CNT], // (consume cycles, access count, cache hit)
    pub heap: Heap,
}

pub struct BufHeap(Mutex<BufHeapInternal>);

impl Deref for BufHeap {
    type Target = Mutex<BufHeapInternal>;

    fn deref(&self) -> &Mutex<BufHeapInternal> {
        &self.0
    }
}

pub const FACTOR : u64 = 100_000;

impl BufHeap {
    pub const fn Empty() -> Self {
        let internal = BufHeapInternal {
            bufs: [[0; BUF_CNT]; CLASS_CNT],
            tops: [0; CLASS_CNT],
            times: [(0, 0, 0); CLASS_CNT],
            heap: Heap::empty(),
        };

        return Self(Mutex::new(internal))
    }

    pub fn AddToHead(&self, start: usize, end: usize) {
        let mut intern = self.lock();
        unsafe {
            intern.heap.add_to_heap(start, end);
        }
    }

    pub fn Init(&self, start: usize, size: usize) {
        self.AddToHead(start, start + size)
    }

    pub fn Print(&self) {
        let times = self.lock().times;
        let mut str = format!("");
        let mut cycles = 0;
        let mut counts = 0;
        let mut cache = 0;
        for i in 0..times.len() {
            if times[i].0 > 0 {
                cycles += times[i].0;
                counts += times[i].1;
                cache += times[i].2;
                str += &format!("\t{}/{}/{}/{}", 1<<i, times[i].0 / FACTOR, times[i].1, times[i].2);
            }
        }

        error!("GlobalAlloc::Print All/{}/{}/{} {}", cycles/FACTOR, counts, cache, str);
    }
}

unsafe impl GlobalAlloc for BufHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let start = Rdtsc();
        let mut intern = self.lock();
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        let top = intern.tops[class];

        let ret = if ENABLE_BUF && top > 0 {
            intern.tops[class] = top - 1; // pop the top
            let ret = intern.bufs[class][top-1];
            intern.times[class].2 += 1;
            ret
        } else {
            let ret = intern
                .heap
                .alloc(layout)
                .ok()
                .map_or(0 as *mut u8, |allocation| allocation.as_ptr()) as u64;

            if ret == 0 {
                panic!("GlobalAlloc::alloc OOM")
            }
            ret
        };

        let now = Rdtsc();
        intern.times[class].0 += (now - start) as u64;
        intern.times[class].1 += 1;

        return ret as *mut u8;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let start = Rdtsc();
        let mut intern = self.lock();

        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        let top = intern.tops[class];
        if ENABLE_BUF && top < BUF_CNT {
            intern.tops[class] = top + 1;
            intern.bufs[class][top] = ptr as u64;
            intern.times[class].2 += 1;
        } else {
            intern.heap.dealloc(NonNull::new_unchecked(ptr), layout)
        }

        let now = Rdtsc();
        intern.times[class].0 += (now - start) as u64;
        intern.times[class].1 += 1;
    }
}