use super::qlib::mem::list_allocator::*;
use libc;
use core::sync::atomic::Ordering;
use core::ptr;


impl OOMHandler for ListAllocator {
    fn handleError(&self, _a:u64, _b:u64) {
        panic!("qvisor OOM: Heap allocator fails to allocate memory block");
    }
}

impl ListAllocator {
    pub fn initialize(&self) {
        let address: *mut libc::c_void;
        unsafe {
            address = libc::mmap(ptr::null_mut(), 1<<29 as usize, libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0);
            if address == libc::MAP_FAILED {
                panic!("mmap: failed to get mapped memory area for heap");
            }
            self.heap.lock().init(address as usize, 1<<29 as usize);
        }
        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn Check(&self) {

    }
}