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
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use core::cmp::max;
use core::mem::size_of;
use core::ptr::NonNull;
//use alloc::slice;
use spin::Mutex;
use buddy_system_allocator::Heap;

pub const CLASS_CNT : usize = 14;
pub const FREE_THRESHOLD: usize = 30; // when free size less than 30%, need to free buffer
pub const BUFF_THRESHOLD: usize = 50; // when buff size takes more than 50% of free size, needs to free
pub const FREE_BATCH: usize = 10; // free 10 blocks each time.
pub const ORDER : usize = 30;

pub struct ListAllocator {
    pub bufs: [Mutex<FreeMemBlockMgr>; CLASS_CNT],
    pub heap: Mutex<Heap<ORDER>>,
    pub total: AtomicUsize,
    pub free: AtomicUsize,
    pub bufSize: AtomicUsize,
}

impl ListAllocator {
    pub const fn Empty() -> Self {
        let bufs : [Mutex<FreeMemBlockMgr>; CLASS_CNT] = [
            Mutex::new(FreeMemBlockMgr::New(0, 0)),
            Mutex::new(FreeMemBlockMgr::New(0, 1)),
            Mutex::new(FreeMemBlockMgr::New(0, 2)),
            Mutex::new(FreeMemBlockMgr::New(128, 3)),
            Mutex::new(FreeMemBlockMgr::New(128, 4)),
            Mutex::new(FreeMemBlockMgr::New(128, 5)),
            Mutex::new(FreeMemBlockMgr::New(64, 6)),
            Mutex::new(FreeMemBlockMgr::New(64, 7)),
            Mutex::new(FreeMemBlockMgr::New(64, 8)),
            Mutex::new(FreeMemBlockMgr::New(32, 9)),
            Mutex::new(FreeMemBlockMgr::New(32, 10)),
            Mutex::new(FreeMemBlockMgr::New(16, 11)),
            Mutex::new(FreeMemBlockMgr::New(1024, 12)),
            Mutex::new(FreeMemBlockMgr::New(16, 13))
        ];

        return Self {
            bufs: bufs,
            heap: Mutex::new(Heap::empty()),
            total: AtomicUsize::new(0),
            free: AtomicUsize::new(0),
            bufSize: AtomicUsize::new(0),
        }
    }

    pub fn AddToHead(&self, start: usize, end: usize) {
        unsafe {
            self.heap.lock().add_to_heap(start, end);
        }

        let size = end - start;
        self.total.fetch_add(size, Ordering::Release);
        self.free.fetch_add(size, Ordering::Release);
    }

    pub fn Add(&self, start: usize, size: usize) {
        let mut start = start;
        let end = start + size;
        let size = 1 << 30; // 1GB
        // note: we can't add full range (>4GB) to the buddyallocator
        while start + size < end {
            self.AddToHead(start, start + size);
            start  += size;
        }

        if start < end {
            self.AddToHead(start, end)
        }
    }

    pub fn NeedFree(&self) -> bool {
        let total = self.total.load(Ordering::Acquire);
        let free = self.free.load(Ordering::Acquire);
        let bufSize = self.bufSize.load(Ordering::Acquire);

        if free > core::usize::MAX / 100 || total > core::usize::MAX / 100 {
            error!("total is {:x}, free is {:x}, buffsize is {:x}", total, free, bufSize);
        }

        if total * FREE_THRESHOLD / 100 > free && // there are too little free memory
            free * BUFF_THRESHOLD /100 < bufSize { // there are too much bufferred memory
            return true
        }

        return false
    }

    // ret: true: free some memory, false: no memory freed
    pub fn Free(&self) -> bool {
        let mut count = 0;
        for i in 0..self.bufs.len() {
            if !self.NeedFree() || count == FREE_BATCH {
                return count > 0
            }

            let idx = self.bufs.len() - i - 1; // free from larger size
            let cnt = self.bufs[idx].lock().FreeMultiple(&self.heap, FREE_BATCH - count);
            self.bufSize.fetch_sub(cnt * self.bufs[idx].lock().size, Ordering::Release);
            count += cnt;
        }

        return count > 0;
    }
}

unsafe impl GlobalAlloc for ListAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );

        let class = size.trailing_zeros() as usize;

        self.free.fetch_sub(size, Ordering::Release);

        if 3 <= class && class < self.bufs.len() {
            let (ret, fromBuf) = self.bufs[class].lock().Alloc(&self.heap);
            if fromBuf {
                self.bufSize.fetch_sub(size, Ordering::Release);
            }

            return ret;
        }

        let ret = self
            .heap
            .lock()
            .alloc(layout)
            .ok()
            .map_or(0 as *mut u8, |allocation| allocation.as_ptr()) as u64;

        if ret == 0 {
            super::super::Kernel::HostSpace::KernelMsg(ret, 0);
            super::super::Kernel::HostSpace::KernelOOM(size as u64, layout.align() as u64);
            loop {}
        }

        return ret as *mut u8;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        self.free.fetch_add(size, Ordering::Release);
        self.bufSize.fetch_add(size, Ordering::Release);
        if class < self.bufs.len() {
            return self.bufs[class].lock().Dealloc(ptr, &self.heap);
        }

        self.heap.lock().dealloc(NonNull::new_unchecked(ptr), layout)
    }
}

pub struct FreeMemBlockMgr {
    pub size: usize,
    pub count: usize,
    pub capacity: usize,
    pub list: MemList,
}

impl FreeMemBlockMgr {
    pub const fn New(capacity: usize, class: usize) -> Self {
        return Self {
            size: 1<<class,
            capacity: capacity,
            count: 0,
            list: MemList::New(),
        }
    }

    pub fn Layout(&self) -> Layout {
        return Layout::from_size_align(self.size, self.size).unwrap();
    }


    // ret: (data, whether it is from list)
    pub fn Alloc(&mut self, heap: &Mutex<Heap<ORDER>>) -> (*mut u8, bool) {
        if self.count > 0 {
            self.count -= 1;
            let ret = self.list.Pop();

            let ptr = ret as * mut MemBlock;
            unsafe {
                ptr.write(0)
            }

            /*let size = self.size / 8;
            unsafe {
                let toArr = slice::from_raw_parts_mut(ret as *mut u64, size);
                for i in 0..size {
                    toArr[i] = 0;
                }
            }*/

            return (ret as * mut u8, true)
        }

        match heap.lock().alloc(self.Layout()) {
            Err(_) => {
                super::super::Kernel::HostSpace::KernelMsg(0, 0);
                super::super::Kernel::HostSpace::KernelOOM(self.size as u64, 1);
                loop {}
            }
            Ok(ret) => {
                return (ret.as_ptr(), false)
            }
        }
    }

    pub fn Dealloc(&mut self, ptr: *mut u8, _heap: &Mutex<Heap<ORDER>>) {
        /*let size = self.size / 8;
        unsafe {
            let toArr = slice::from_raw_parts(ptr as *mut u64, size);
            for i in 0..size {
                assert!(toArr[i] == 0);
            }
        }*/

        self.count += 1;
        self.list.Push(ptr as u64);
    }

    fn Free(&mut self, heap: &Mutex<Heap<ORDER>>) {
        assert!(self.count > 0);
        self.count -= 1;
        let addr = self.list.Pop();

        unsafe {
            heap.lock().dealloc(NonNull::new_unchecked(addr as * mut u8), self.Layout());
        }
    }

    pub fn FreeMultiple(&mut self, heap: &Mutex<Heap<ORDER>>, count: usize) -> usize {
        for i in 0..count {
            if self.count <= self.capacity {
                return i;
            }

            self.Free(heap)
        }

        return count;
    }
}


type MemBlock = u64;


pub struct MemList {
    head: MemBlock,
    tail: MemBlock,
}

impl MemList {
    pub const fn New() -> Self {
        return Self {
            head: 0,
            tail: 0,
        }
    }

    pub fn Push(&mut self, addr: u64) {
        if addr % 8 != 0 {
            super::super::Kernel::HostSpace::KernelMsg(101, addr);
        }
        assert!(addr % 8 == 0);

        let newB = addr as * mut MemBlock;
        unsafe {
            *newB = 0;
        }

       if self.head == 0 {
            self.head = addr;
            self.tail = addr;
            return
        }

        let tail = self.tail;

        let ptr = tail as * mut MemBlock;
        unsafe {
            *ptr = addr;
        }
        self.tail = addr;
    }

    pub fn Pop(&mut self) -> u64 {
        if self.head == 0 {
            return 0
        }

        let next = self.head;

        if self.head == self.tail {
            self.head = 0;
            self.tail = 0;
            return next;
        }

        let ptr = unsafe {
           &mut *(next as * mut MemBlock)
        };

        self.head = *ptr;

        if next % 8 != 0 {
            super::super::Kernel::HostSpace::KernelMsg(100, next);
        }
        assert!(next % 8 == 0);
        return next;
    }
}