use super::super::super::mem::list_allocator::*;
use core::sync::atomic::Ordering;

impl OOMHandler for ListAllocator {
    fn handleError(&self, size:u64, alignment:u64) {
        super::super::Kernel::HostSpace::KernelOOM(size, alignment);
    }
}

impl ListAllocator {
    pub fn initialize(&self)-> () {
        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn Check(&self) {
        super::super::task::Task::StackOverflowCheck();
    }
}
