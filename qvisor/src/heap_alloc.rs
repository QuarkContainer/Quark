use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc;

use super::qlib::linux_def::MemoryDef;
use super::qlib::mem::bitmap_allocator::*;
use super::qlib::mem::list_allocator::*;

pub const ENABLE_HUGEPAGE: bool = false;

use crate::qlib::kernel::arch::tee::is_cc_active;
use crate::qlib::kernel::Kernel::IDENTICAL_MAPPING;

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
            ioHeapAddr: AtomicU64::new(MemoryDef::HEAP_OFFSET + MemoryDef::HEAP_SIZE),
            hostInitHeapAddr: AtomicU64::new(MemoryDef::HOST_INIT_HEAP_OFFSET),
            guestPrivHeapAddr: AtomicU64::new(MemoryDef::GUEST_PRIVATE_HEAP_OFFSET),
            sharedHeapAddr: AtomicU64::new(MemoryDef::GUEST_HOST_SHARED_HEAP_OFFSET),
            vmLaunched: AtomicBool::new(false),
            initialized: AtomicBool::new(false),
        };
    }

    //Map shared heap here
    pub fn Init(&self) {
        let sharedHeapAddr = unsafe {
            let mut flags = libc::MAP_SHARED | libc::MAP_ANON | libc::MAP_FIXED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                self.sharedHeapAddr.load(Ordering::Relaxed) as _,
                (MemoryDef::GUEST_HOST_SHARED_HEAP_SIZE + MemoryDef::IO_HEAP_SIZE) as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            ) as u64
        };

        if sharedHeapAddr == libc::MAP_FAILED as u64 {
            panic!("mmap: failed to get mapped memory area for shared heap");
}

        assert!(
            self.sharedHeapAddr.load(Ordering::Relaxed) == sharedHeapAddr,
            "sharedHeapAddr expected address is {:x}, mmap address is {:x}",
            self.sharedHeapAddr.load(Ordering::Relaxed),
            sharedHeapAddr
        );

        let hostInitHeapAddr = unsafe {
            let mut flags = libc::MAP_SHARED | libc::MAP_ANON | libc::MAP_FIXED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                self.hostInitHeapAddr.load(Ordering::Relaxed) as _,
                MemoryDef::HOST_INIT_HEAP_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            ) as u64
        };

        if hostInitHeapAddr == libc::MAP_FAILED as u64 {
            panic!("mmap: failed to get mapped memory area for shared heap");
        }

        assert!(
            self.hostInitHeapAddr.load(Ordering::Relaxed) == hostInitHeapAddr,
            "hostInitHeapAddr expected address is {:x}, mmap address is {:x}",
            self.hostInitHeapAddr.load(Ordering::Relaxed),
            hostInitHeapAddr
        );

        let hostInitHeapStart = self.hostInitHeapAddr.load(Ordering::Relaxed);
        let hostInitHeapEnd = hostInitHeapStart + MemoryDef::HOST_INIT_HEAP_SIZE as u64;
        *self.HostInitAllocator() = ListAllocator::New(hostInitHeapStart as _, hostInitHeapEnd);
        //reserve first 4KB gor the listAllocator
        let size = core::mem::size_of::<ListAllocator>();
        self.HostInitAllocator().Add(
            MemoryDef::HOST_INIT_HEAP_OFFSET as usize + size,
            MemoryDef::HOST_INIT_HEAP_SIZE as usize - size,
        );

        self.initialized.store(true, Ordering::SeqCst);
    }

    pub fn Clear(&self) -> bool {
        //return self.Allocator().Free();
        return false;
    }

    pub fn InitAllocator(&self) {
        use std::convert::TryInto;

        let mut guestPrivHeapStart = self.guestPrivHeapAddr.load(Ordering::Acquire);
        let identical = IDENTICAL_MAPPING.load(Ordering::Acquire);
        if !identical {
            guestPrivHeapStart += MemoryDef::UNIDENTICAL_MAPPING_OFFSET;
            self.guestPrivHeapAddr
                .store(guestPrivHeapStart, Ordering::Release);
        }

        let guestPrivHeapAddr = unsafe {
            let mut flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED | libc::MAP_LOCKED;
            if ENABLE_HUGEPAGE {
                flags |= libc::MAP_HUGE_2MB;
            }
            libc::mmap(
                self.guestPrivHeapAddr.load(Ordering::Relaxed) as _,
                MemoryDef::GUEST_PRIVATE_HEAP_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            ) as u64
        };

        if guestPrivHeapAddr == libc::MAP_FAILED as u64 {
            panic!("mmap: failed to get mapped memory area for guest private heap");
        }
        assert!(
            self.guestPrivHeapAddr.load(Ordering::Relaxed) == guestPrivHeapAddr,
            "guestPrivHeapAddr expected address is {:x}, mmap address is {:x}",
            self.guestPrivHeapAddr.load(Ordering::Relaxed),
            guestPrivHeapAddr
        );

        unsafe {
            let m_lock = libc::mlock2(self.guestPrivHeapAddr.load(Ordering::Relaxed)
                as *const libc::c_void, MemoryDef::GUEST_PRIVATE_HEAP_SIZE.try_into().unwrap(),
                libc::MLOCK_ONFAULT.try_into().unwrap());
                if m_lock < 0 {
                    panic!("VM: Failed to lock heap memory - error:{:?}", std::io::Error::last_os_error());
                }
        }
        let heap_size = if identical {
            MemoryDef::GUEST_PRIVATE_HEAP_SIZE
        } else {
            MemoryDef::GUEST_PRIVATE_INIT_HEAP_SIZE
        };
        let guestPrivHeapEnd = guestPrivHeapStart + heap_size;
        *self.GuestPrivateAllocator() = ListAllocator::New(guestPrivHeapStart, guestPrivHeapEnd);

        let size = core::mem::size_of::<ListAllocator>();
        self.GuestPrivateAllocator().Add(
            guestPrivHeapAddr as usize + size,
            heap_size as usize - size,
        );

        if !is_cc_active() {
            self.sharedHeapAddr.store(MemoryDef::HEAP_OFFSET, Ordering::SeqCst);
            self.GuestHostSharedAllocator().enlarge(MemoryDef::HEAP_OFFSET, MemoryDef::HEAP_END);
            self.GuestHostSharedAllocator().Add(MemoryDef::GUEST_HOST_SHARED_HEAP_OFFSET as usize,
                MemoryDef::GUEST_HOST_SHARED_HEAP_SIZE as usize);

            let ioHeapEnd = MemoryDef::HEAP_END + MemoryDef::IO_HEAP_SIZE;
            *self.IOAllocator() = ListAllocator::New(MemoryDef::HEAP_END as _, ioHeapEnd);
            self.IOAllocator().Add(
                MemoryDef::HEAP_END as usize,
                MemoryDef::IO_HEAP_SIZE as usize,
            );
        }
    }
}

unsafe impl GlobalAlloc for HostAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let initialized = self.initialized.load(Ordering::Relaxed);
        if !initialized {
            self.Init();
        }

        let is_vm_init = self.vmLaunched.load(Ordering::Relaxed);
        if !is_vm_init {
            self.HostInitAllocator().alloc(layout)
        } else {
            self.GuestHostSharedAllocator().alloc(layout)
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let addr = ptr as u64;

        let is_vm_init = self.vmLaunched.load(Ordering::Relaxed);
        if !is_vm_init && self.IsGuestPrivateHeapAddr(addr) {
            self.GuestPrivateAllocator().dealloc(ptr, layout);
            return
        }

        if Self::IsSharedHeapAddr(addr) {
            self.GuestHostSharedAllocator().dealloc(ptr, layout);
        } else if Self::IsInitHeapAddr(addr) {
            self.HostInitAllocator().dealloc(ptr, layout);
        } else if Self::IsIOBuf(addr) {
            self.IOAllocator().dealloc(ptr, layout);
        } else if self.IsGuestPrivateHeapAddr(addr) && !is_cc_active(){
            self.GuestPrivateAllocator().dealloc(ptr, layout);
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
