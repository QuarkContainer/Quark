use core::mem;
use core::sync::atomic;

use super::porting::*;

macro_rules! mmap_offset {
    ( $mmap:ident + $offset:expr => $ty:ty ) => {
        $mmap.as_mut_ptr().add($offset as _) as $ty
    };
    ( $( let $val:ident = $mmap:ident + $offset:expr => $ty:ty );+ ; ) => {
        $(
            let $val = mmap_offset!($mmap + $offset => $ty);
        )*
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Mmap {
    pub addr: u64,
    pub len: usize,
}

impl Mmap {
    /*pub fn dontfork(&self) -> Result<()> {
        match unsafe { libc::madvise(self.addr.as_ptr(), self.len, libc::MADV_DONTFORK) } {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        }
    }*/

    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        //unsafe {
        self.addr as *mut u8
        //}
    }
}

#[derive(Default)]
pub struct Fd(pub RawFd);

impl AsRawFd for Fd {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

impl IntoRawFd for Fd {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        let fd = self.0;
        mem::forget(self);
        fd
    }
}

impl FromRawFd for Fd {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> Fd {
        Fd(fd)
    }
}

impl Drop for Fd {
    fn drop(&mut self) {
        //unsafe {
        //libc::close(self.0);
        panic!("todo....")
        //}
    }
}

#[inline(always)]
pub unsafe fn unsync_load(u: *const atomic::AtomicU32) -> u32 {
    u.cast::<u32>().read()
}

#[inline]
pub fn cast_ptr<T>(n: &T) -> *const T {
    n as *const T
}
