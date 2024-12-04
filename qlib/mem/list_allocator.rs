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

use alloc::collections::vec_deque::VecDeque;
use alloc::string::String;
use cache_padded::CachePadded;
use core::alloc::{GlobalAlloc, Layout};
use core::cmp::max;
use core::mem::size_of;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

use crate::qlib::kernel::arch::tee::is_cc_active;
use super::super::super::kernel_def::VcpuId;
use super::super::kernel::vcpu::CPU_LOCAL;
use crate::PRIVATE_VCPU_ALLOCATOR;
use super::super::linux_def::*;
use super::super::mutex::*;
use super::super::pagetable::AlignedAllocator;
use super::buddy_allocator::Heap;
use crate::qlib::vcpu_mgr::CPULocal;
use crate::GLOBAL_ALLOCATOR;

pub const CLASS_CNT: usize = 16;
pub const FREE_THRESHOLD: usize = 30; // when free size less than 30%, need to free buffer
pub const BUFF_THRESHOLD: usize = 50; // when buff size takes more than 50% of free size, needs to free
pub const FREE_BATCH: usize = 1024; // free 10 blocks each time.
pub const ORDER: usize = 33; //1GB

//pub static GLOBAL_ALLOCATOR: HostAllocator = HostAllocator::New();

pub fn CheckZeroPage(pageStart: u64) {
    use alloc::slice;
    unsafe {
        let arr = slice::from_raw_parts_mut(pageStart as *mut u64, 512);
        for i in 0..512 {
            if arr[i] != 0 {
                panic!("alloc non zero page {:x}", pageStart);
            }
        }
    }
}

pub fn SetZeroPage(pageStart: u64) {
    use alloc::slice;
    unsafe {
        let arr = slice::from_raw_parts_mut(pageStart as *mut u64, 512);
        for i in 0..512 {
            arr[i] = 0;
        }
    }
}

#[derive(Default)]
pub struct GlobalVcpuAllocator {
    pub init: AtomicBool,
}

impl GlobalVcpuAllocator {
    pub const fn New() -> Self {
        return Self {
            init: AtomicBool::new(false),
        };
    }

    pub fn Print(&self) {
        if is_cc_active() {
            error!("GlobalVcpuAllocator {}/{}", VcpuId(), unsafe {
                (*PRIVATE_VCPU_ALLOCATOR.allocators[VcpuId()].get()).bufs.len()
            })
        } else {
            error!("GlobalVcpuAllocator {}/{}", VcpuId(), unsafe {
                (*CPU_LOCAL[VcpuId()].allocator.get()).bufs.len()
            })
        }
    }

    pub fn Initializated(&self) {
        self.init.store(true, Ordering::Release)
    }
}

/*
unsafe impl GlobalAlloc for GlobalVcpuAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !self.init.load(Ordering::Relaxed) {
            return GLOBAL_ALLOCATOR
                .alloc(layout);
        }
        return CPU_LOCAL[VcpuId()].AllocatorMut().alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !self.init.load(Ordering::Relaxed) {
            return GLOBAL_ALLOCATOR
                .dealloc(ptr, layout);
        }
        return CPU_LOCAL[VcpuId()].AllocatorMut().dealloc(ptr, layout)
    }
}*/

#[derive(Debug, Default)]
pub struct PageAllocator {
    pub pages: VecDeque<u64>,
}

impl PageAllocator {
    pub const PAGE_CACHE_COUNT: usize = 16;
    pub const PAGE_CACHE_MAX_COUNT: usize = 128;

    pub fn AllocPage(&mut self) -> Option<u64> {
        match self.pages.pop_front() {
            None => return None,
            Some(addr) => return Some(addr),
        }
    }

    pub fn FreePage(&mut self, page: u64) {
        if self.pages.len() >= Self::PAGE_CACHE_MAX_COUNT {
            AlignedAllocator::New(MemoryDef::PAGE_SIZE as usize, MemoryDef::PAGE_SIZE as usize)
                .Free(page)
                .unwrap();
        } else {
            self.pages.push_front(page);
        }
    }

    pub fn Clean(&mut self) {
        while self.pages.len() > Self::PAGE_CACHE_COUNT {
            match self.pages.pop_back() {
                None => break,
                Some(addr) => {
                    AlignedAllocator::New(
                        MemoryDef::PAGE_SIZE as usize,
                        MemoryDef::PAGE_SIZE as usize,
                    )
                    .Free(addr)
                    .unwrap();
                }
            }
        }
    }
}

pub const STACK_CNT: usize = 64;

#[derive(Debug)]
pub struct StackAllocator {
    pub stack: [u64; STACK_CNT - 1],
    pub next: usize,
}

impl Default for StackAllocator {
    fn default() -> Self {
        return Self {
            stack: [0; STACK_CNT - 1],
            next: 0,
        };
    }
}

impl StackAllocator {
    pub fn Push(&mut self, addr: u64) {
        assert!(self.next < self.stack.len());
        self.stack[self.next] = addr;
        self.next += 1;
    }

    pub fn Pop(&mut self) -> u64 {
        assert!(self.next != 0);
        self.next -= 1;
        return self.stack[self.next];
    }

    pub fn IsEmpty(&self) -> bool {
        return self.next == 0;
    }

    pub fn IsFull(&self) -> bool {
        return self.next == self.stack.len();
    }
}

#[derive(Debug, Default)]
pub struct VcpuAllocator {
    pub bufs: [StackAllocator; 8],
}

impl VcpuAllocator {
    pub fn Clear(&mut self) {
        for idx in 3..self.bufs.len() {
            let size = 1 << idx;
            let layout =
                Layout::from_size_align(size, size).expect("VcpuAllocator layout alloc fail");
            while !self.bufs[idx].IsEmpty() {
                let addr = self.bufs[idx].Pop();
                unsafe {
                    GLOBAL_ALLOCATOR.dealloc(addr as _, layout);
                }
            }
        }
    }
}

impl VcpuAllocator {
    #[inline(never)]
    pub fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;
        assert!(class >= 3);
        let idx = class - 3;

        let ret;
        if idx < self.bufs.len() && !self.bufs[idx].IsEmpty() {
            ret = self.bufs[idx].Pop();
        } else {
            unsafe { ret = GLOBAL_ALLOCATOR.alloc(layout) as u64 };
        }
        return ret as *mut u8;
    }

    pub fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;
        assert!(class >= 3);
        let idx = class - 3;

        if idx < self.bufs.len() && !self.bufs[idx].IsFull() {
            self.bufs[idx].Push(ptr as u64);
        } else {
            unsafe { GLOBAL_ALLOCATOR.dealloc(ptr, layout) }
        }
    }
}

#[derive(Debug, Default)]
pub struct HostAllocator {
    pub ioHeapAddr: AtomicU64,
    pub hostInitHeapAddr: AtomicU64,
    pub guestPrivHeapAddr: AtomicU64,
    pub sharedHeapAddr: AtomicU64,
    pub vmLaunched: AtomicBool,
    pub initialized: AtomicBool,
}

impl HostAllocator {
    pub fn IOAllocator(&self) -> &mut ListAllocator {
        return unsafe { &mut *(self.ioHeapAddr.load(Ordering::SeqCst) as *mut ListAllocator) };
    }

    pub fn IsHeapAddr(addr: u64) -> bool {
        return addr < MemoryDef::HEAP_END;
    }

    pub fn IsIOBuf(addr: u64) -> bool {
        return MemoryDef::HEAP_END <= addr && addr < MemoryDef::HEAP_END + MemoryDef::IO_HEAP_SIZE;
    }

    pub unsafe fn AllocIOBuf(&self, size: usize) -> *mut u8 {
        let layout = Layout::from_size_align(size, size)
            .expect("RingeBufAllocator::AllocHeadTail can't allocate memory");
        return self.IOAllocator().alloc(layout);
    }

    unsafe fn DeallocIOBuf(&self, ptr: *mut u8, layout: Layout) {
        self.IOAllocator().dealloc(ptr, layout);
    }

    // should be called by host
    pub unsafe fn AllocGuestPrivatMem(&self, size: usize, align: usize) -> *mut u8 {
        let layout = Layout::from_size_align(size, align)
            .expect("AllocGuestPrivatMem can't allocate memory");
        let ptr = self.GuestPrivateAllocator().alloc(layout);
        return ptr;
    }

    pub unsafe fn AllocSharedBuf(&self, size: usize, align: usize) -> *mut u8 {
        let layout =
            Layout::from_size_align(size, align).expect("AllocSharedBuf can't allocate memory");

        return self.GuestHostSharedAllocator().alloc(layout);
    }

    pub unsafe fn DeallocShareBuf(&self, ptr: *mut u8, size: usize, align: usize) {
        let layout =
            Layout::from_size_align(size, align).expect("DeallocShareBuf can't dealloc memory");

        return self.GuestHostSharedAllocator().dealloc(ptr, layout);
    }
}

#[derive(Debug)]
pub struct ListAllocator {
    pub bufs: [CachePadded<QMutex<FreeMemBlockMgr>>; CLASS_CNT],
    pub heap: QMutex<Heap<ORDER>>,
    pub total: AtomicUsize,
    pub free: AtomicUsize,
    pub allocated: AtomicUsize,
    pub bufSize: AtomicUsize,
    pub heapStart: u64,
    pub heapEnd: u64,
    pub counts: [AtomicUsize; 36],
    pub maxnum: [AtomicUsize; 36],
    //pub errorHandler: Arc<OOMHandler>
    pub initialized: AtomicBool,
}

pub trait OOMHandler {
    fn handleError(&self, a: u64, b: u64) -> ();
}

impl Default for ListAllocator {
    fn default() -> Self {
        return Self::New(0, 0);
    }
}

impl ListAllocator {
    pub const fn New(heapStart: u64, heapEnd: u64) -> Self {
        let bufs: [CachePadded<QMutex<FreeMemBlockMgr>>; CLASS_CNT] = [
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(0, 0))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(0, 1))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(0, 2))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(128, 3))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(128, 4))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(128, 5))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(64, 6))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(64, 7))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(64, 8))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(32, 9))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(32, 10))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(16, 11))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(1024, 12))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(16, 13))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(8, 14))),
            CachePadded::new(QMutex::new(FreeMemBlockMgr::New(8, 15))),
        ];

        return Self {
            bufs: bufs,
            heap: QMutex::new(Heap::empty()),
            total: AtomicUsize::new(0),
            free: AtomicUsize::new(0),
            allocated: AtomicUsize::new(0),
            bufSize: AtomicUsize::new(0),
            heapStart,
            heapEnd,
            initialized: AtomicBool::new(false),
            counts: [
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
            ],
            maxnum: [
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
            ],
        };
    }

    pub fn Print(&self, _class: usize) -> String {
        /*print!("heap addr is {:x}", self.heap.MutexId());
        for i in 0..self.bufs.len() {
            print!("ListAllocator[{}] {:x}", i, self.bufs[i].MutexId());
        }*/

        //return "".to_string();
        let mut total = 0;
        for i in 3..24 {
            total += (1 << i) * self.maxnum[i].load(Ordering::Relaxed);
        }
        return format!(
            "total {:?}/{}, {:?}",
            self.allocated,
            total,
            &self.maxnum[3..24]
        );
    }

    pub fn AddToHead(&self, start: usize, end: usize) {
        unsafe {
            self.heap.lock().add_to_heap(start, end);
        }

        let size = end - start;
        self.total.fetch_add(size, Ordering::Release);
        self.free.fetch_add(size, Ordering::Release);
    }

    pub fn Init(&self, start: usize, size: usize) {
        self.Add(start, size);
    }

    /// add the chunk of memory (start, start+size) to heap for allocating dynamic memory
    pub fn Add(&self, start: usize, size: usize) {
        let mut start = start;
        let end = start + size;
        //let order = 22;
        let size = 1 << ORDER; // 2MB
                               // note: we can't add full range (>4GB) to the buddyallocator
                               /*let alignStart = start & !(size - 1);
                               if start != alignStart {
                                   self.AddToHead(start, alignStart + size);
                                   start = alignStart + size;
                               }*/

        while start + size < end {
            self.AddToHead(start, start + size);
            start += size;
        }

        if start < end {
            self.AddToHead(start, end)
        }

        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn NeedFree(&self) -> bool {
        let total = self.total.load(Ordering::Acquire);
        let free = self.free.load(Ordering::Acquire);
        let bufSize = self.bufSize.load(Ordering::Acquire);

        /*if free > core::usize::MAX / 100 || total > core::usize::MAX / 100 {
            error!("total is {:x}, free is {:x}, buffsize is {:x}", total, free, bufSize);
        }*/

        if total * FREE_THRESHOLD / 100 > free && // there are too little free memory
            free * BUFF_THRESHOLD /100 < bufSize
        {
            // there are too much bufferred memory
            return true;
        }

        return false;
    }

    // ret: true: free some memory, false: no memory freed
    pub fn Free(&self) -> bool {
        let mut count = 0;
        for i in 0..self.bufs.len() {
            if !self.NeedFree() || count == FREE_BATCH {
                return count > 0;
            }

            let idx = self.bufs.len() - i - 1; // free from larger size
            let cnt = self.bufs[idx]
                .lock()
                .FreeMultiple(&self.heap, FREE_BATCH - count);
            self.bufSize
                .fetch_sub(cnt * self.bufs[idx].lock().size, Ordering::Release);
            count += cnt;
        }

        return count > 0;
    }

    pub fn FreeAll(&self) -> bool {
        let mut count = 0;
        //let mut free = 0;
        for i in 0..self.bufs.len() {
            let idx = self.bufs.len() - i - 1; // free from larger size
            let cnt = self.bufs[idx]
                .lock()
                .FreeMultiple(&self.heap, FREE_BATCH - count);
            //.FreeAll(&self.heap);
            /*free += self.bufs[idx]
            .lock()
            .TotalFree();*/
            self.bufSize
                .fetch_sub(cnt * self.bufs[idx].lock().size, Ordering::Release);
            count += cnt;
        }

        return count > 0;
    }

    pub fn enlarge(&mut self, newHeapStart: u64, newHeapEnd: u64) {
        assert!(self.heapStart >= newHeapStart);
        assert!(self.heapEnd <= newHeapEnd);
        self.heapStart = newHeapStart;
        self.heapEnd = newHeapEnd;
    }
}

pub const PRINT_CLASS: usize = 0;
pub const MEMORY_CHECKING: bool = false;

unsafe impl GlobalAlloc for ListAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.Check();

        let initialized = self.initialized.load(Ordering::Relaxed);
        if !initialized {
            self.initialize();
        }

        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );

        let class = size.trailing_zeros() as usize;

        if MEMORY_CHECKING {
            self.counts[class].fetch_add(1, Ordering::Release);
            self.maxnum[class].fetch_add(1, Ordering::Release);
            self.allocated.fetch_add(size, Ordering::Release);
        }

        if 3 <= class && class < self.bufs.len() {
            let ret = self.bufs[class].lock().Alloc();
            if let Some(addr) = ret {
                self.bufSize.fetch_sub(size, Ordering::Release);
                if MEMORY_CHECKING && class == PRINT_CLASS {
                    //error!("L#{} alloc {:x?}", class, addr as u64);
                    let idx = self.maxnum[class].load(Ordering::Acquire);
                    let cpuid = (idx << 8) | CPULocal::CpuId();
                    raw!(1, class as u64, addr as u64, cpuid as u64);
                }
                return addr;
            }
        }

        /*if size > 1 << 21 {
            panic!("alloc size is {}", layout.size());
        }*/

        let ret = self
            .heap
            .lock()
            .alloc(layout)
            .ok()
            .map_or(0 as *mut u8, |allocation| allocation.as_ptr()) as u64;

        if ret == 0 {
            self.handleError(size as u64, layout.align() as u64);
            loop {}
        }

        if MEMORY_CHECKING && class == PRINT_CLASS {
            //error!("L#{} alloc {:x}", class, ret as u64);
            let idx = self.maxnum[class].load(Ordering::Acquire);
            let cpuid = (idx << 8) | CPULocal::CpuId();
            raw!(1, class as u64, ret as u64, cpuid as u64);
        }

        if ret % size as u64 != 0 {
            raw!(0x236, ret, size as u64, 0);
            panic!("alloc next fail");
        }

        // Subtract when ret != 0 to avoid overflow
        self.free.fetch_sub(size, Ordering::Release);

        return ret as *mut u8;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.Check();

        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        if MEMORY_CHECKING && class == PRINT_CLASS {
            //error!("L#{} free {:x}", class, ptr as u64);
            let idx = self.maxnum[class].load(Ordering::Acquire);
            let cpuid = (idx << 8) | CPULocal::CpuId();
            raw!(2, class as u64, ptr as u64, cpuid as u64);
        }

        if MEMORY_CHECKING {
            self.maxnum[class].fetch_sub(1, Ordering::Release);
            self.allocated.fetch_sub(size, Ordering::Release);
        }
        self.free.fetch_add(size, Ordering::Release);
        self.bufSize.fetch_add(size, Ordering::Release);
        if class < self.bufs.len() {
            return self.bufs[class].lock().Dealloc(ptr, &self.heap);
        }

        self.heap
            .lock()
            .dealloc(NonNull::new_unchecked(ptr), layout)
    }
}

/// FreeMemoryBlockMgr is used to manage heap memory block allocated by allocator
#[repr(align(128))]
#[derive(Debug, Default)]
pub struct FreeMemBlockMgr {
    pub size: usize,
    pub count: usize,
    pub reserve: usize,
    pub list: MemList,
    //pub queue: [bool; 16],
    //pub idx: usize,
}

impl FreeMemBlockMgr {
    /// Return a newly created FreeMemBlockMgr
    /// # Arguments
    ///
    /// * `reserve` - number of clocks the Block Manager keeps for itself when free multiple is called.
    /// * `class` - denotes the block size this manager is in charge of. class i means the block is of size 2^i bytes
    pub const fn New(reserve: usize, class: usize) -> Self {
        return Self {
            size: 1 << class,
            reserve: reserve,
            count: 0,
            list: MemList::New(1 << class),
            //queue: [false; 16],
            //idx: 0,
        };
    }

    pub fn Layout(&self) -> Layout {
        return Layout::from_size_align(self.size, self.size).unwrap();
    }

    pub fn Alloc(&mut self) -> Option<*mut u8> {
        if self.count != self.list.count as usize {
            error!(
                "FreeMemBlockMgr::Alloc {}/{}/{}",
                self.size, self.count, self.list.count
            );
        }

        if self.count > 0 {
            self.count -= 1;
            let ret = self.list.Pop();

            assert!(
                ret != 0,
                "self.count is {}, size is {}",
                self.count,
                self.size
            );
            let ptr = ret as *mut MemBlock;
            unsafe { ptr.write(0) }

            //self.queue[self.idx%16] = true;
            //self.idx += 1;
            return Some(ret as *mut u8);
        } else {
            return None;
        }
    }

    pub fn Dealloc(&mut self, ptr: *mut u8, _heap: &QMutex<Heap<ORDER>>) {
        if self.count != self.list.count as usize {
            error!(
                "FreeMemBlockMgr::Dealloc {}/{}/{}",
                self.size, self.count, self.list.count
            );
        }
        self.count += 1;
        self.list.Push(ptr as u64);
        //self.queue[self.idx%16] = false;
        //self.idx += 1;
    }

    fn Free(&mut self, heap: &QMutex<Heap<ORDER>>) -> bool {
        if self.count == 0 {
            return false;
        }

        if self.count != self.list.count as usize {
            error!(
                "FreeMemBlockMgr::Dealloc {}/{}/{}",
                self.size, self.count, self.list.count
            );
        }

        self.count -= 1;
        let addr = self.list.Pop();

        unsafe {
            heap.lock()
                .dealloc(NonNull::new_unchecked(addr as *mut u8), self.Layout());
        }

        return true;
    }

    pub fn TotalFree(&self) -> usize {
        return self.size * self.count;
    }

    pub fn FreeAll(&mut self, heap: &QMutex<Heap<ORDER>>) -> usize {
        let mut i = 0;
        loop {
            if self.count <= self.reserve {
                return i;
            }

            i += 1;
            self.Free(heap);
        }
    }

    pub fn FreeMultiple(&mut self, heap: &QMutex<Heap<ORDER>>, count: usize) -> usize {
        for i in 0..count {
            if self.count <= self.reserve {
                return i;
            }

            self.Free(heap);
        }

        return count;
    }
}

type MemBlock = u64;

#[derive(Debug, Default)]
pub struct MemList {
    size: u64,
    count: u64,
    head: MemBlock,
}

impl MemList {
    pub const fn New(size: usize) -> Self {
        return Self {
            size: size as u64,
            count: 0,
            head: 0,
        };
    }

    pub fn Push(&mut self, addr: u64) {
        if addr % self.size != 0 {
            raw!(235, addr, self.size, 0);
            panic!("Push next fail");
        }

        self.count += 1;
        let newB = addr as *mut MemBlock;
        unsafe {
            *newB = 0;
        }

        let ptr = addr as *mut MemBlock;
        unsafe {
            *ptr = self.head;
        }
        self.head = addr;
    }

    pub fn Pop(&mut self) -> u64 {
        if self.head == 0 {
            if self.count != 0 {
                error!("MemList::pop self.size is {}/{}", self.size, self.count);
            }

            return 0;
        }

        self.count -= 1;

        let next = self.head;

        let ptr = unsafe { &mut *(next as *mut MemBlock) };

        self.head = *ptr;

        assert!(
            !(self.head == 0 && self.count != 0),
            "MemList::pop2 self.size is {}/{}/{:x}",
            self.size,
            self.count,
            next
        );
        if next % self.size != 0 {
            raw!(0x234, next, self.size as u64, 0);
            panic!("Pop next fail");
        }
        //assert!(next % self.size == 0, "Pop next is {:x}/size is {:x}", next, self.size);
        return next;
    }
}
