use crate::kernel_def::VcpuId;
use crate::qlib::kernel::vcpu::VCPU_COUNT;
use crate::qlib::linux_def::*;
use crate::qlib::mem::list_allocator::*;
use crate::qlib::Vec;
use crate::GLOBAL_ALLOCATOR;
use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;

#[derive(Debug, Default)]
#[repr(C)]
#[repr(align(16))]
pub struct PrivateVcpuAllocators {
    pub allocators: Vec<UnsafeCell<VcpuAllocator>>,
}

unsafe impl Send for PrivateVcpuAllocators {}
unsafe impl Sync for PrivateVcpuAllocators {}

impl PrivateVcpuAllocators {
    pub fn New() -> Self {
        let mut v = Vec::new();
        let vcpu_number = VCPU_COUNT.load(Ordering::Acquire);

        info!("PrivateCPULocal new, cpu number {}", vcpu_number);

        for _ in 0..vcpu_number {
            let a = VcpuAllocator::default().into();
            v.push(a);
        }
        return Self { allocators: v };
    }

    pub fn AllocatorMut(&self) -> &mut VcpuAllocator {
        //return unsafe { &mut *(&self.allocator as *const _ as u64 as *mut VcpuAllocator) };
        return unsafe { &mut *self.allocators[VcpuId()].get() };
    }
}

#[derive(Debug, Default)]
#[repr(C)]
#[repr(align(16))]
pub struct PrivateVcpuSharedAllocators {
    pub allocators: Vec<UnsafeCell<VcpuSharedAllocator>>,
}

unsafe impl Send for PrivateVcpuSharedAllocators {}
unsafe impl Sync for PrivateVcpuSharedAllocators {}

impl PrivateVcpuSharedAllocators {
    pub fn New() -> Self {
        let mut v = Vec::new();
        let vcpu_number = VCPU_COUNT.load(Ordering::Acquire);

        info!("PrivateCPULocal new, cpu number {}", vcpu_number);

        for _ in 0..vcpu_number {
            let a = VcpuSharedAllocator::default().into();
            v.push(a);
        }
        return Self { allocators: v };
    }

    pub fn AllocatorMut(&self) -> &mut VcpuSharedAllocator {
        //return unsafe { &mut *(&self.allocator as *const _ as u64 as *mut VcpuAllocator) };
        return unsafe { &mut *self.allocators[VcpuId()].get() };
    }
}

impl HostAllocator {
    pub fn HostInitAllocator(&self) -> &mut ListAllocator {
        return unsafe {
            &mut *(self.hostInitHeapAddr.load(Ordering::Relaxed) as *mut ListAllocator)
        };
    }

    pub fn GuestPrivateAllocator(&self) -> &mut ListAllocator {
        return unsafe {
            &mut *(self.guestPrivHeapAddr.load(Ordering::Relaxed) as *mut ListAllocator)
        };
    }

    pub fn GuestHostSharedAllocator(&self) -> &mut ListAllocator {
        return unsafe { &mut *(self.sharedHeapAddr.load(Ordering::Relaxed) as *mut ListAllocator) };
    }

    pub fn IsGuestPrivateHeapAddr(&self, addr: u64) -> bool {
        let heapStart = self.guestPrivHeapAddr.load(Ordering::Relaxed);
        let heapSize = if crate::qlib::kernel::Kernel::IDENTICAL_MAPPING.load(Ordering::Acquire) {
            MemoryDef::GUEST_PRIVATE_HEAP_SIZE
        } else {
            MemoryDef::GUEST_PRIVATE_RUNNING_HEAP_SIZE
        };
        let heapEnd = heapStart + heapSize;
        return addr < heapEnd && addr >= heapStart;
    }

    pub fn IsInitHeapAddr(addr: u64) -> bool {
        return addr < MemoryDef::HOST_INIT_HEAP_END && addr >= MemoryDef::HOST_INIT_HEAP_OFFSET;
    }

    pub fn IsSharedHeapAddr(addr: u64) -> bool {
        return addr < MemoryDef::GUEST_HOST_SHARED_HEAP_END
            && addr >= MemoryDef::GUEST_HOST_SHARED_HEAP_OFFSET;
    }

    #[inline]
    pub fn HeapRange(&self) -> (u64, u64) {
        let allocator = self.GuestPrivateAllocator();
        return (allocator.heapStart, allocator.heapEnd);
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct GlobalVcpuSharedAllocator {}

impl GlobalVcpuSharedAllocator {
    pub const fn New() -> Self {
        return Self {};
    }
}

unsafe impl GlobalAlloc for GlobalVcpuSharedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let addr = GLOBAL_ALLOCATOR.AllocSharedBuf(layout.size(), layout.align());
        return addr;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        GLOBAL_ALLOCATOR.dealloc(ptr, layout);
    }
}

unsafe impl Allocator for GlobalVcpuSharedAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        //info!("GuestHostSharedAllocator allocate");

        unsafe {
            let ptr = GLOBAL_ALLOCATOR.AllocSharedBuf(layout.size(), layout.align());
            let slice = core::slice::from_raw_parts_mut(ptr, layout.size());

            Ok(NonNull::new_unchecked(slice))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        //info!("GuestHostSharedAllocator deallocate");
        let ptr = ptr.as_ptr();
        GLOBAL_ALLOCATOR.dealloc(ptr, layout);
    }
}