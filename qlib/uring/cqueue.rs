//! Completion Queue

use core::sync::atomic;

use super::sys;
use super::util::{unsync_load, Mmap};

pub struct CompletionQueue {
    pub(crate) head: *const atomic::AtomicU32,
    pub(crate) tail: *const atomic::AtomicU32,
    pub(crate) ring_mask: *const u32,
    pub(crate) ring_entries: *const u32,

    overflow: *const atomic::AtomicU32,

    pub(crate) cqes: *const sys::io_uring_cqe,

    #[allow(dead_code)]
    flags: *const atomic::AtomicU32,
}

impl Default for CompletionQueue {
    fn default() -> Self {
        return Self {
            head: 0 as *const atomic::AtomicU32,
            tail: 0 as *const atomic::AtomicU32,
            ring_mask: 0 as *const u32,
            ring_entries: 0 as *const u32,
            overflow: 0 as *const atomic::AtomicU32,
            cqes: 0 as *const sys::io_uring_cqe,
            flags: 0 as *const atomic::AtomicU32,
        };
    }
}

/// Completion Entry
#[repr(transparent)]
#[derive(Clone, Default, Debug)]
pub struct Entry(pub(crate) sys::io_uring_cqe);

impl CompletionQueue {
    pub(crate) unsafe fn new(cq_mmap: &Mmap, p: &sys::io_uring_params) -> CompletionQueue {
        mmap_offset! {
            let head            = cq_mmap + p.cq_off.head           => *const atomic::AtomicU32;
            let tail            = cq_mmap + p.cq_off.tail           => *const atomic::AtomicU32;
            let ring_mask       = cq_mmap + p.cq_off.ring_mask      => *const u32;
            let ring_entries    = cq_mmap + p.cq_off.ring_entries   => *const u32;
            let overflow        = cq_mmap + p.cq_off.overflow       => *const atomic::AtomicU32;
            let cqes            = cq_mmap + p.cq_off.cqes           => *const sys::io_uring_cqe;
            let flags           = cq_mmap + p.cq_off.flags          => *const atomic::AtomicU32;
        }

        CompletionQueue {
            head,
            tail,
            ring_mask,
            ring_entries,
            overflow,
            cqes,
            flags,
        }
    }

    /// If queue is full, the new event maybe dropped.
    /// This value records number of dropped events.
    pub fn overflow(&self) -> u32 {
        unsafe { (*self.overflow).load(atomic::Ordering::SeqCst) }
    }

    #[cfg(feature = "unstable")]
    pub fn eventfd_disabled(&self) -> bool {
        unsafe {
            (*self.flags).load(atomic::Ordering::Acquire) & sys::IORING_CQ_EVENTFD_DISABLED != 0
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        unsafe { self.ring_entries.read() as usize }
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe {
            let head = unsync_load(self.head);
            let tail = (*self.tail).load(atomic::Ordering::Acquire);

            tail.wrapping_sub(head) as usize
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    pub fn next(&mut self) -> Option<Entry> {
        unsafe {
            let head = unsync_load(self.head);
            let tail = (*self.tail).load(atomic::Ordering::Acquire);
            let ring_mask = self.ring_mask.read();

            if head != tail {
                let entry = self.cqes.add((head & ring_mask) as usize);
                (*self.head).store(head.wrapping_add(1), atomic::Ordering::Release);
                Some(Entry(*entry))
            } else {
                None
            }
        }
    }
}

impl Entry {
    /// Result value
    pub fn result(&self) -> i32 {
        self.0.res
    }

    /// User Data
    ///
    /// See [Entry::user_data](super::squeue::Entry::user_data).
    pub fn user_data(&self) -> u64 {
        self.0.user_data
    }

    /// Flags
    pub fn flags(&self) -> u32 {
        self.0.flags
    }
}
