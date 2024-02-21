use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc;

use super::qlib::linux_def::MemoryDef;
use super::qlib::mem::bitmap_allocator::*;
use super::qlib::mem::list_allocator::*;

pub const ENABLE_HUGEPAGE: bool = false;

impl BitmapAllocatorWrapper {
    pub const fn New() -> Self {
        return Self {
            addr: AtomicU64::new(0),
        };
    }

    pub fn Init(&self) {
        let heapSize = MemoryDef::HEAP_SIZE as usize;
        let heapAddr = MemoryDef::HEAP_OFFSET;
        let addr = unsafe {
            let mut flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                heapAddr as _,
                heapSize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            ) as u64
        };

        if addr == libc::MAP_FAILED as u64 {
            panic!("mmap: failed to get mapped memory area for heap");
        }

        assert!(
            heapAddr == addr,
            "expect is {:x}, actual is {:x}",
            heapAddr,
            addr
        );

        self.addr.store(heapAddr, Ordering::SeqCst);
    }
}

impl HostAllocator {
    pub const fn New() -> Self {
        return Self {
            listHeapAddr: AtomicU64::new(MemoryDef::HEAP_OFFSET),
            initialized: AtomicBool::new(false),
        };
    }

    pub fn Init(&self) {
        let heapSize = MemoryDef::HEAP_SIZE as usize;
        let addr = unsafe {
            let mut flags = libc::MAP_SHARED | libc::MAP_ANON | libc::MAP_FIXED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                self.listHeapAddr.load(Ordering::Relaxed) as _,
                heapSize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            ) as u64
        };

        if addr == libc::MAP_FAILED as u64 {
            panic!("mmap: failed to get mapped memory area for heap");
        }

        assert!(
            self.listHeapAddr.load(Ordering::Relaxed) == addr,
            "listHeapAddr is {:x}, addr is {:x}",
            self.listHeapAddr.load(Ordering::Relaxed),
            addr
        );

        let heapStart = self.listHeapAddr.load(Ordering::Relaxed);
        let heapEnd = heapStart + MemoryDef::HEAP_SIZE as u64;
        *self.Allocator() = ListAllocator::New(heapStart as _, heapEnd);

        // reserve first 4KB gor the listAllocator
        let size = core::mem::size_of::<ListAllocator>();
        self.Allocator().Add(MemoryDef::HEAP_OFFSET as usize + size, MemoryDef::HEAP_SIZE as usize - size);
        self.initialized.store(true, Ordering::SeqCst);
    }

    pub fn Clear(&self) -> bool {
        //return self.Allocator().Free();
        return false;
    }
}

unsafe impl GlobalAlloc for HostAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let initialized = self.initialized.load(Ordering::Relaxed);
        if !initialized {
            self.Init();
        }

        return self.Allocator().alloc(layout);
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.Allocator().dealloc(ptr, layout);
    }
}

impl OOMHandler for ListAllocator {
    fn handleError(&self, _a: u64, _b: u64) {
        panic!("qvisor OOM: Heap allocator fails to allocate memory block");
    }
}

impl ListAllocator {
    pub fn initialize(&self) {
        /*let listHeapAddr = MemoryDef::PHY_LOWER_ADDR + HEAP_OFFSET;
        let heapSize = 1 << KERNEL_HEAP_ORD as usize;
        let address: usize;
        unsafe {
            address = libc::mmap(listHeapAddr as _, heapSize, libc::PROT_READ | libc::PROT_WRITE,
                                 libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0) as usize;
            if address == libc::MAP_FAILED as usize {
                panic!("mmap: failed to get mapped memory area for heap");
            }
            self.heap.lock().init(address + 0x1000 as usize, heapSize - 0x1000);
        }*/
        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn Check(&self) {}
}

impl VcpuAllocator {
    pub fn handleError(&self, _size: u64, _alignment: u64) {}
}
