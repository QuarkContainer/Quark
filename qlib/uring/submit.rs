use core::sync::atomic;
use core::{ptr};

use super::super::linux_def::IoVec;
use super::super::common::*;
use super::register::execute;
use super::register::Probe;
use super::squeue::SubmissionQueue;
use super::sys;
use super::util::{cast_ptr, unsync_load, Fd};
use super::porting::*;

#[cfg(feature = "unstable")]
use super::register::Restriction;

/// Submitter
pub struct Submitter<'a> {
    pub fd: &'a Fd,
    pub flags: u32,

    pub sq_head: *const atomic::AtomicU32,
    pub sq_tail: *const atomic::AtomicU32,
    pub sq_flags: *const atomic::AtomicU32,
}

impl<'a> Submitter<'a> {
    #[inline]
    pub(crate) const fn new(fd: &'a Fd, flags: u32, sq: &SubmissionQueue) -> Submitter<'a> {
        Submitter {
            fd,
            flags,
            sq_head: sq.head,
            sq_tail: sq.tail,
            sq_flags: sq.flags,
        }
    }

    pub fn sq_len(&self) -> usize {
        unsafe {
            let head = (*self.sq_head).load(atomic::Ordering::Acquire);
            let tail = unsync_load(self.sq_tail);

            tail.wrapping_sub(head) as usize
        }
    }

    pub fn sq_need_wakeup(&self) -> bool {
        unsafe {
            (*self.sq_flags).load(atomic::Ordering::Acquire) & sys::IORING_SQ_NEED_WAKEUP != 0
        }
    }

     #[cfg(feature = "unstable")]
    pub fn squeue_wait(&self) -> Result<usize> {
        let result = unsafe {
            sys::io_uring_enter(
                self.fd.as_raw_fd(),
                0,
                0,
                sys::IORING_ENTER_SQ_WAIT,
                ptr::null(),
            )
        };

        if result >= 0 {
            Ok(result as _)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Register buffers.
    pub fn register_buffers(&self, bufs: &[IoVec]) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_BUFFERS,
            bufs.as_ptr() as *const _,
            bufs.len() as _,
        )
        .map(drop)
    }

    /// Register files for I/O.
    pub fn register_files(&self, fds: &[RawFd]) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_FILES,
            fds.as_ptr() as *const _,
            fds.len() as _,
        )
        .map(drop)
    }

    /// It’s possible to use `eventfd(2)` to get notified of completion events on an io_uring instance.
    pub fn register_eventfd(&self, eventfd: RawFd) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_EVENTFD,
            cast_ptr::<RawFd>(&eventfd) as *const _,
            1,
        )
        .map(drop)
    }

    /// This operation replaces existing files in the registered file set with new ones,
    /// either turning a sparse entry (one where fd is equal to -1) into a real one, removing an existing entry (new one is set to -1),
    /// or replacing an existing entry with a new existing entry.
    pub fn register_files_update(&self, offset: u32, fds: &[RawFd]) -> Result<usize> {
        let fu = sys::io_uring_files_update {
            offset,
            resv: 0,
            fds: fds.as_ptr() as _,
        };
        let fu = cast_ptr::<sys::io_uring_files_update>(&fu);
        let ret = execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_FILES_UPDATE,
            fu as *const _,
            fds.len() as _,
        )?;
        Ok(ret as _)
    }

    /// This works just like [Submitter::register_eventfd],
    /// except notifications are only posted for events that complete in an async manner.
    pub fn register_eventfd_async(&self, eventfd: RawFd) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_EVENTFD_ASYNC,
            cast_ptr::<RawFd>(&eventfd) as *const _,
            1,
        )
        .map(drop)
    }

    /// This operation returns a structure `Probe`,
    /// which contains information about the opcodes supported by io_uring on the running kernel.
    pub fn register_probe(&self, probe: &mut Probe) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_PROBE,
            probe.as_mut_ptr() as *const _,
            Probe::COUNT as _,
        )
        .map(drop)
    }

    /// This operation registers credentials of the running application with io_uring,
    /// and returns an id associated with these credentials.
    pub fn register_personality(&self) -> Result<i32> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_PERSONALITY,
            ptr::null(),
            0,
        )
    }

    /// Unregister buffers.
    pub fn unregister_buffers(&self) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_UNREGISTER_BUFFERS,
            ptr::null(),
            0,
        )
        .map(drop)
    }

    /// Unregister files.
    pub fn unregister_files(&self) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_UNREGISTER_FILES,
            ptr::null(),
            0,
        )
        .map(drop)
    }

    /// Unregister an eventfd file descriptor to stop notifications.
    pub fn unregister_eventfd(&self) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_UNREGISTER_EVENTFD,
            ptr::null(),
            0,
        )
        .map(drop)
    }

    /// This operation unregisters a previously registered personality with io_uring.
    pub fn unregister_personality(&self, id: i32) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_UNREGISTER_PERSONALITY,
            ptr::null(),
            id as _,
        )
        .map(drop)
    }

    #[cfg(feature = "unstable")]
    pub fn register_restrictions(&self, res: &mut [Restriction]) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_RESTRICTIONS,
            res.as_mut_ptr().cast(),
            res.len() as _,
        )
        .map(drop)
    }

    #[cfg(feature = "unstable")]
    pub fn register_enable_rings(&self) -> Result<()> {
        execute(
            self.fd.as_raw_fd(),
            sys::IORING_REGISTER_ENABLE_RINGS,
            ptr::null(),
            0,
        )
        .map(drop)
    }
}
