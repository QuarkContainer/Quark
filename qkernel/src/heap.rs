use core::alloc::{GlobalAlloc, Layout};
//use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
//use core::ptr::NonNull;
//use core::mem::size_of;
//use core::cmp::max;
use alloc::string::String;

//use super::qlib::vcpu_mgr::*;
use super::qlib::mem::list_allocator::*;

pub const HEAP_ADDR: u64 = 0x4040000000;
pub struct GuestAllocator {
    pub heapAddr: AtomicU64,
}

impl GuestAllocator {
    pub const fn New() -> Self {
        return Self {
            heapAddr: AtomicU64::new(0)
        }
    }

    pub fn Init(&self, heapAddr: u64) {
        self.heapAddr.store(heapAddr, Ordering::SeqCst)
    }

    #[inline(always)]
    pub fn Allocator(&self) -> &'static ListAllocator {
        return unsafe {
            &*(self.heapAddr.load(Ordering::Relaxed) as * const ListAllocator)
        }
    }

    pub fn Print(&self, class: usize) -> String {
        return format!("GuestAllocator {} {:x}", class, HEAP_ADDR);
    }
}

unsafe impl GlobalAlloc for GuestAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        return self.Allocator().alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        return self.Allocator().dealloc(ptr, layout)
    }
}

/*
impl VcpuAllocator {
    pub fn handleError(&self, size:u64, alignment:u64) {
        super::Kernel::HostSpace::KernelOOM(size, alignment);
    }
}

#[derive(Default, Debug)]
pub struct Count {
    pub curr: AtomicU64,
    pub max: AtomicU64,
}

impl Count {
    pub const fn New() -> Self {
        return Self {
            curr: AtomicU64::new(0),
            max: AtomicU64::new(0)
        }
    }

    pub fn Incr(&self) {
        let val = self.curr.fetch_add(1, Ordering::SeqCst);
        if val + 1 > self.max.load(Ordering::Relaxed) {
            self.max.store(val + 1, Ordering::SeqCst)
        }
    }
    pub fn Decr(&self) {
        self.curr.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Default, Debug)]
pub struct QAllocator {
    pub localReady: AtomicBool,
    pub counts: [Count; 16],
}

unsafe impl GlobalAlloc for QAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        /*let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );

        let class = size.trailing_zeros() as usize;
        if class < self.counts.len() {
            self.counts[class].Incr();
        }*/

        if self.Ready() {
            return CPULocal::Myself().allocator.alloc(layout);
        }

        return GLOBAL_ALLOCATOR.lock().alloc(layout)
            .ok()
            .map_or(0 as *mut u8, |allocation| allocation.as_ptr());
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        /*let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );

        let class = size.trailing_zeros() as usize;
        if class < self.counts.len() {
            self.counts[class].Decr();
        }*/

        if self.Ready() {
            return CPULocal::Myself().allocator.dealloc(ptr, layout);
        }

        return GLOBAL_ALLOCATOR.lock().dealloc(NonNull::new_unchecked(ptr), layout);
    }
}

impl QAllocator {
    pub const fn New() -> Self {
        return Self {
            localReady: AtomicBool::new(false),
            counts: [
                Count::New(), Count::New(), Count::New(), Count::New(),
                Count::New(), Count::New(), Count::New(), Count::New(),
                Count::New(), Count::New(), Count::New(), Count::New(),
                Count::New(), Count::New(), Count::New(), Count::New()
            ],
        }
    }

    pub fn Print(&self, class: usize) -> String {
        return format!("alloc[{}] xxx {:?}", class, &self.counts[class]);
        //return format!("alloc[{}] xxx {:#?}", class, &self.counts);
    }

    pub fn AddToHead(&self, start: usize, end: usize) {
        //Kernel::HostSpace::KernelMsg(2, start as u64, end as u64);
        unsafe {
            GLOBAL_ALLOCATOR.lock().add_to_heap(start, end);
        }
        //Kernel::HostSpace::KernelMsg(2, 0, 2);

    }

    pub fn Init(&self, start: usize, size: usize) {
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

    pub fn SetReady(&self, val: bool) {
        self.localReady.store(val, Ordering::SeqCst);
    }

    pub fn Ready(&self) -> bool {
        return self.localReady.load(Ordering::Relaxed);
    }
}


unsafe impl GlobalAlloc for VcpuAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );

        let class = size.trailing_zeros() as usize;

        //super::Kernel::HostSpace::KernelMsg(1, 1, size as u64);
        //defer!(super::Kernel::HostSpace::KernelMsg(1, 2, size as u64));

        if 3 <= class && class < self.bufs.len() {
            let ret = self.bufs[class].lock().Alloc();
            if ret.is_some() {
                return ret.unwrap();
            }
        }

        let ret = GLOBAL_ALLOCATOR
            .lock()
            .alloc(layout)
            .ok()
            .map_or(0 as *mut u8, |allocation| allocation.as_ptr()) as u64;

        if ret == 0 {
            self.handleError(size as u64, layout.align() as u64);
            loop {}
        }

        if ret % size as u64 != 0 {
            raw!(0x236, ret, size as u64);
            panic!("alloc next fail");
        }

        return ret as *mut u8;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        //super::Kernel::HostSpace::KernelMsg(1, 3, size as u64);
        //defer!(super::Kernel::HostSpace::KernelMsg(1, 4, size as u64));

        if class < self.bufs.len() {
            return self.bufs[class].lock().Dealloc(ptr, &GLOBAL_ALLOCATOR);
        }

        GLOBAL_ALLOCATOR.lock().dealloc(NonNull::new_unchecked(ptr), layout)
    }
}
*/