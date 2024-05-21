use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc;
use std::sync::atomic::AtomicU8;

use super::qlib::linux_def::MemoryDef;
use super::qlib::mem::bitmap_allocator::*;
use super::qlib::mem::list_allocator::*;
use crate::qlib::common::*;
use crate::qlib::range::Range;

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
        const ARRAY_REPEAT_VALUE_U8: AtomicU8 = AtomicU8::new(0);
        const ARRAY_REPEAT_VALUE_U64: AtomicU64 = AtomicU64::new(0);
        let a = Self {
            allocators: [ARRAY_REPEAT_VALUE_U64; ListAllocatorType::Last as usize],
            initialized: AtomicBool::new(false),
            addrMap: [ARRAY_REPEAT_VALUE_U8; 1024],
        };

        return a;
    }

    pub fn IncreaseHeapSize(&self, type_: ListAllocatorType) -> Result<()> {
        let allocator = self.GetAllocator(type_);
        let range = allocator.IncreaseHeapSize()?;
        self.AddRange(&range, type_);
        return Ok(());
    }

    pub fn Init(&self) {
        let heapStart = MemoryDef::HEAP_OFFSET; //self.listHeapAddr.load(Ordering::Relaxed);
        let heapSize = MemoryDef::HEAP_SIZE as usize + MemoryDef::IO_HEAP_SIZE as usize;
        let addr = unsafe {
            let mut flags = libc::MAP_SHARED | libc::MAP_ANON | libc::MAP_FIXED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                heapStart as _,
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
            heapStart == addr,
            "heapstart is {:x}, addr is {:x}",
            heapStart,
            addr
        );

        self.SetAllocator(ListAllocatorType::Global, MemoryDef::HEAP_OFFSET);
        self.SetAllocator(
            ListAllocatorType::IO,
            MemoryDef::HEAP_OFFSET + MemoryDef::HEAP_SIZE,
        );

        let heapEnd = heapStart + MemoryDef::HEAP_SIZE as u64;
        *self.Allocator() = ListAllocator::New(ListAllocatorType::Global, heapStart as _, heapEnd);

        let ioHeapEnd = heapStart + MemoryDef::HEAP_SIZE as u64 + MemoryDef::IO_HEAP_SIZE;
        *self.IOAllocator() = ListAllocator::New(ListAllocatorType::IO, heapEnd as _, ioHeapEnd);

        // reserve first 4KB gor the listAllocator
        let size = core::mem::size_of::<ListAllocator>();
        self.Allocator().Add(
            MemoryDef::HEAP_OFFSET as usize + size,
            MemoryDef::HEAP_SIZE as usize - size,
        );
        self.AddRange(
            &Range::New(MemoryDef::HEAP_OFFSET, MemoryDef::HEAP_SIZE),
            ListAllocatorType::Global,
        );
        self.IOAllocator().Add(
            MemoryDef::HEAP_END as usize + size,
            MemoryDef::IO_HEAP_SIZE as usize - size,
        );
        self.AddRange(
            &Range::New(MemoryDef::HEAP_END, MemoryDef::IO_HEAP_SIZE),
            ListAllocatorType::IO,
        );
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
        let addr = ptr as u64;
        if !self.IsIOBuf(addr) {
            self.Allocator().dealloc(ptr, layout);
        } else {
            //self.Allocator().dealloc(ptr, layout);
            self.IOAllocator().dealloc(ptr, layout);
        }
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
