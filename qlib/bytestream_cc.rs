use super::linux_def::*;
use crate::qlib::bytestream::*;
use crate::GLOBAL_ALLOCATOR;
use alloc::alloc::{alloc, dealloc, Layout};
use alloc::slice;
use core::sync::atomic::Ordering;
use core::sync::atomic::{AtomicBool, AtomicU32};

impl RingeBufAllocator {
    pub fn AllocHeadTail(&self) -> &'static [AtomicU32] {
        match self {
            Self::GuestPrivateHeapAllocator => return GuestPrivateHeapAllocator::AllocHeadTail(),
            Self::GuestSharedHeapAllocator => return GuestSharedHeapAllocator::AllocHeadTail(),
            Self::ShareAllocator(headTailAddr, _, _, init) => {
                return ShareAllocator::AllocHeadTail(*headTailAddr, *init)
            }
        }
    }

    pub fn AllocWaitingRW(&self) -> &'static [AtomicBool] {
        match self {
            Self::GuestPrivateHeapAllocator => return GuestPrivateHeapAllocator::AllocWaitingRW(),
            Self::GuestSharedHeapAllocator => return GuestSharedHeapAllocator::AllocWaitingRW(),
            Self::ShareAllocator(_, waitingRWAddr, _, init) => {
                return ShareAllocator::AllocWaitingRW(*waitingRWAddr, *init)
            }
        }
    }

    pub fn FreeHeadTail(&self, data: &'static [AtomicU32]) {
        match self {
            Self::GuestPrivateHeapAllocator => {
                return GuestPrivateHeapAllocator::FreeHeadTail(data)
            }
            Self::GuestSharedHeapAllocator => return GuestSharedHeapAllocator::FreeHeadTail(data),
            Self::ShareAllocator(_, _, _, _) => return ShareAllocator::FreeHeadTail(data),
        }
    }

    pub fn FreeWaitingRW(&self, data: &'static [AtomicBool]) {
        match self {
            Self::GuestPrivateHeapAllocator => {
                return GuestPrivateHeapAllocator::FreeWaitingRW(data)
            }
            Self::GuestSharedHeapAllocator => return GuestSharedHeapAllocator::FreeWaitingRW(data),
            Self::ShareAllocator(_, _, _, _) => return ShareAllocator::FreeWaitingRW(data),
        }
    }

    pub fn AlllocBuf(&self, pageCount: usize) -> u64 {
        match self {
            Self::GuestPrivateHeapAllocator => {
                return GuestPrivateHeapAllocator::AlllocBuf(pageCount)
            }
            Self::GuestSharedHeapAllocator => {
                return GuestSharedHeapAllocator::AlllocBuf(pageCount)
            }
            Self::ShareAllocator(_, _, buffAddr, _) => return ShareAllocator::AlllocBuf(*buffAddr),
        }
    }

    pub fn FreeBuf(&self, addr: u64, size: usize) {
        match self {
            Self::GuestPrivateHeapAllocator => {
                return GuestPrivateHeapAllocator::FreeBuf(addr, size)
            }
            Self::GuestSharedHeapAllocator => return GuestSharedHeapAllocator::FreeBuf(addr, size),
            Self::ShareAllocator(_, _, _, _) => return ShareAllocator::FreeBuf(addr, size),
        }
    }
}

pub struct GuestSharedHeapAllocator {}

unsafe impl Send for GuestSharedHeapAllocator {}
unsafe impl Sync for GuestSharedHeapAllocator {}

impl GuestSharedHeapAllocator {
    pub fn AllocHeadTail() -> &'static [AtomicU32] {
        let addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(8, 8) };

        let ptr = addr as *mut AtomicU32;
        let slice = unsafe { slice::from_raw_parts(ptr, 2 as usize) };
        slice[0].store(0, Ordering::Release);
        slice[1].store(0, Ordering::Release);
        return slice;
    }

    pub fn AllocWaitingRW() -> &'static [AtomicBool] {
        let addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(2, 8) };

        let ptr = addr as *mut AtomicBool;
        let slice = unsafe { slice::from_raw_parts(ptr, 2 as usize) };
        slice[0].store(true, Ordering::Release);
        slice[1].store(false, Ordering::Release);
        return slice;
    }

    pub fn FreeHeadTail(data: &'static [AtomicU32]) {
        assert!(data.len() == 2);
        let addr = &data[0] as *const _ as u64;
        let layout = Layout::from_size_align(8, 8)
            .expect("GuestSharedHeapAllocator::FreeHeadTail can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }

    pub fn FreeWaitingRW(data: &'static [AtomicBool]) {
        assert!(data.len() == 2);
        let addr = &data[0] as *const _ as u64;
        let layout = Layout::from_size_align(2, 8)
            .expect("GuestSharedHeapAllocator::FreeWaitingRW can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }

    pub fn AlllocBuf(pageCount: usize) -> u64 {
        assert!(IsPowerOfTwo(pageCount));

        let size = pageCount * MemoryDef::PAGE_SIZE as usize;
        let align = MemoryDef::PAGE_SIZE as usize;
        let addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size, align) };

        return addr as u64;
    }

    pub fn FreeBuf(addr: u64, size: usize) {
        assert!(IsPowerOfTwo(size) && addr % MemoryDef::PAGE_SIZE == 0);
        let layout = Layout::from_size_align(size, MemoryDef::PAGE_SIZE as usize)
            .expect("HeapAllocator::FreeBuf can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }
}

pub struct GuestPrivateHeapAllocator {}

unsafe impl Send for GuestPrivateHeapAllocator {}
unsafe impl Sync for GuestPrivateHeapAllocator {}

impl GuestPrivateHeapAllocator {
    pub fn AllocHeadTail() -> &'static [AtomicU32] {
        let layout = Layout::from_size_align(8, 8)
            .expect("RingeBufAllocator::AllocHeadTail can't allocate memory");
        let addr = unsafe { alloc(layout) };

        let ptr = addr as *mut AtomicU32;
        let slice = unsafe { slice::from_raw_parts(ptr, 2 as usize) };
        slice[0].store(0, Ordering::Release);
        slice[1].store(0, Ordering::Release);
        return slice;
    }

    pub fn AllocWaitingRW() -> &'static [AtomicBool] {
        let layout = Layout::from_size_align(2, 8)
            .expect("HeapAllocator::AllocWaitingRW can't allocate memory");
        let addr = unsafe { alloc(layout) };

        let ptr = addr as *mut AtomicBool;
        let slice = unsafe { slice::from_raw_parts(ptr, 2 as usize) };
        slice[0].store(true, Ordering::Release);
        slice[1].store(false, Ordering::Release);
        return slice;
    }

    pub fn FreeHeadTail(data: &'static [AtomicU32]) {
        assert!(data.len() == 2);
        let addr = &data[0] as *const _ as u64;
        let layout =
            Layout::from_size_align(8, 8).expect("HeapAllocator::FreeHeadTail can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }

    pub fn FreeWaitingRW(data: &'static [AtomicBool]) {
        assert!(data.len() == 2);
        let addr = &data[0] as *const _ as u64;
        let layout =
            Layout::from_size_align(2, 8).expect("HeapAllocator::FreeWaitingRW can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }

    pub fn AlllocBuf(pageCount: usize) -> u64 {
        assert!(IsPowerOfTwo(pageCount));
        let layout = Layout::from_size_align(
            pageCount * MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize,
        )
        .expect("HeapAllocator::AlllocBuf can't allocate memory");
        let addr = unsafe { alloc(layout) };

        return addr as u64;
    }

    pub fn FreeBuf(addr: u64, size: usize) {
        assert!(IsPowerOfTwo(size) && addr % MemoryDef::PAGE_SIZE == 0);
        let layout = Layout::from_size_align(size, MemoryDef::PAGE_SIZE as usize)
            .expect("HeapAllocator::FreeBuf can't free memory");
        unsafe { dealloc(addr as *mut u8, layout) };
    }
}
