use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc;

use super::qlib::linux_def::MemoryDef;
use super::qlib::mem::list_allocator::*;

pub const ENABLE_HUGEPAGE: bool = false;

impl HostAllocator {
    pub const fn New() -> Self {
        return Self {
            host_initialization_heap: AtomicU64::new(MemoryDef::HOST_INIT_HEAP_OFFSET),
            host_guest_shared_heap: AtomicU64::new(MemoryDef::GUEST_HOST_SHARED_HEAP_OFFEST),
            guest_private_heap: AtomicU64::new(MemoryDef::GUEST_PRIVATE_HEAP_OFFSET),
            initialized: AtomicBool::new(false),
            is_host_allocator: AtomicBool::new(true),
            is_vm_lauched: AtomicBool::new(false),
        };
    }

    pub fn Init(&self) {
        // guest && host shared heap + guest private heap
        let guest_private_heap_addr = unsafe {
            let mut flags = libc::MAP_SHARED | libc::MAP_ANON | libc::MAP_FIXED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                self.guest_private_heap.load(Ordering::Relaxed) as _,
                MemoryDef::GUEST_PRIVATE_HEAP_PLUS_SHARED_HEAP_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            ) as u64
        };

        if guest_private_heap_addr == libc::MAP_FAILED as u64 {
            panic!("mmap: failed to get mapped memory area for heap");
        }



        // guest && host shared heap + guest private heap
        let host_init_heap_addr = unsafe {
            let mut flags = libc::MAP_SHARED | libc::MAP_ANON | libc::MAP_FIXED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                self.host_initialization_heap.load(Ordering::Relaxed) as _,
                MemoryDef::HOST_INIT_HEAP_OFFSET as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            ) as u64
        };

        if host_init_heap_addr == libc::MAP_FAILED as u64 {
            panic!("mmap: failed to get mapped memory area for heap");
        }


        assert!(
            self.guest_private_heap.load(Ordering::Relaxed) == guest_private_heap_addr,
            "guest_private_heap is {:x}, mmap addr is {:x}",
            self.guest_private_heap.load(Ordering::Relaxed),
            guest_private_heap_addr
        );

        assert!(
            self.host_initialization_heap.load(Ordering::Relaxed) == host_init_heap_addr,
            "host_init_heap_addr is {:x}, mmap addr is {:x}",
            self.host_initialization_heap.load(Ordering::Relaxed),
            host_init_heap_addr
        );

        // init private guest heap
        let guestPrivateHeapStart = self.guest_private_heap.load(Ordering::Relaxed);
        let guestPrivateHeapEnd = guestPrivateHeapStart + MemoryDef::GUEST_PRIVATE_HEAP_SIZE as u64;
        *self.GuestPrivateAllocator() = ListAllocator::New(guestPrivateHeapStart as _, guestPrivateHeapEnd);

        let hostInitHeapStart = self.host_initialization_heap.load(Ordering::Relaxed);
        let hostInitHeapEnd = hostInitHeapStart + MemoryDef::HOST_INIT_HEAP_SIZE as u64;
        *self.HostInitAllocator() = ListAllocator::New(hostInitHeapStart as _, hostInitHeapEnd);

        // reserve first 4KB gor the listAllocator
        let size = core::mem::size_of::<ListAllocator>();

        self.GuestPrivateAllocator().Add(MemoryDef::GUEST_PRIVATE_HEAP_OFFSET as usize + size, 
                        MemoryDef::GUEST_PRIVATE_HEAP_SIZE as usize - size);
        self.HostInitAllocator().Add(MemoryDef::HOST_INIT_HEAP_OFFSET as usize + size, 
                                            MemoryDef::HOST_INIT_HEAP_SIZE as usize - size);
        self.initialized.store(true, Ordering::SeqCst);

        self.is_vm_lauched.store(false, Ordering::SeqCst);

    }

    pub fn Clear(&self) -> bool {
        //return self.Allocator().Free();
        return false;
    }
}


#[cfg(feature = "cc")]
unsafe impl GlobalAlloc for HostAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let initialized = self.initialized.load(Ordering::SeqCst);
        if !initialized {
            self.Init();
        }

        let is_vm_init = self.is_vm_lauched.load(Ordering::SeqCst);
        if !is_vm_init {
            self.HostInitAllocator().alloc(layout)
        } else {
            self.GuestHostSharedAllocator().alloc(layout)
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let addr = ptr as u64;
        // prevent memory lead
        let is_vm_init = self.is_vm_lauched.load(Ordering::SeqCst);
        if !is_vm_init && Self::IsGuestPrivateHeapAddr(addr) {
            self.GuestPrivateAllocator().dealloc(ptr, layout);
            return
        }

        if Self::IsHostGuestSharedHeapAddr(addr) {
            self.GuestHostSharedAllocator().dealloc(ptr, layout);
        } else {
            self.HostInitAllocator().dealloc(ptr, layout);
        }
    }
}

#[cfg(not(feature = "cc"))]
unsafe impl GlobalAlloc for HostAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let initialized = self.initialized.load(Ordering::Relaxed);
        if !initialized {
            self.Init();
        }

        return self.GuestPrivateAllocator().alloc(layout);
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.GuestPrivateAllocator().dealloc(ptr, layout);
        
    }
}



impl OOMHandler for ListAllocator {
    fn handleError(&self, _a: u64, _b: u64) {
        panic!("qvisor OOM: Heap allocator fails to allocate memory block");
    }
}

impl ListAllocator {
    pub fn initialize(&self) {
        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn Check(&self) {}
}

impl VcpuAllocator {
    pub fn handleError(&self, _size: u64, _alignment: u64) {}
}

