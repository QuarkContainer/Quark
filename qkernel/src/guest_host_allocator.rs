use alloc::alloc::{Layout, GlobalAlloc, Allocator, AllocError};
use GLOBAL_ALLOCATOR;
use ptr::NonNull;


#[derive(Debug, Default, Copy, Clone)]
pub struct GuestHostSharedAllocator {
}


impl GuestHostSharedAllocator {
    pub const fn New() -> Self {
        return Self {};
    }

}

unsafe impl GlobalAlloc for GuestHostSharedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {

        let addr = GLOBAL_ALLOCATOR.AllocSharedBuf(layout.size(), layout.align());
        return addr;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        GLOBAL_ALLOCATOR.dealloc(ptr, layout);
    }
}

unsafe impl Allocator for GuestHostSharedAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {

        info!("GuestHostSharedAllocator allocate");

        unsafe {
            let ptr = GLOBAL_ALLOCATOR.AllocSharedBuf(layout.size(), layout.align());
            let slice = core::slice::from_raw_parts_mut(ptr, layout.size());
            
            Ok(NonNull::new_unchecked(slice))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        info!("GuestHostSharedAllocator deallocate");
        let ptr = ptr.as_ptr();
        GLOBAL_ALLOCATOR.dealloc(ptr, layout);
    }
}
