// copy and modify from https://github.com/polachok/tokio-eventfd/blob/master/src/lib.rs

use std::io::{self, Read, Result, Write};
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_lite::ready;
use tokio::io::unix::AsyncFd;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

struct Inner(RawFd);

impl Inner {
    fn new(fd: RawFd) -> Self {
        Inner(fd)
    }
}

impl AsRawFd for Inner {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

impl<'a> io::Read for &'a Inner {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let rv =
            unsafe { libc::read(self.0, buf.as_mut_ptr() as *mut std::ffi::c_void, buf.len()) };
        if rv < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(rv as usize)
    }
}

impl<'a> io::Write for &'a Inner {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let rv = unsafe { libc::write(self.0, buf.as_ptr() as *const std::ffi::c_void, buf.len()) };
        if rv < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(rv as usize)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct EventFd(AsyncFd<Inner>);

impl EventFd {
    pub fn new(fd: RawFd) -> Self {
        let inner = Inner::new(fd);
        return EventFd(AsyncFd::new(inner).unwrap())
    }
}

impl AsRawFd for EventFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0.get_ref().0
    }
}

impl FromRawFd for EventFd {
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        EventFd(AsyncFd::new(Inner(fd)).unwrap())
    }
}

impl AsyncRead for EventFd {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<Result<()>> {
        let mut guard = ready!(self.0.poll_read_ready(cx))?;
        
        let count = match guard.try_io(|inner| {
            let buf = unsafe {
                &mut *(buf.unfilled_mut() as *mut [std::mem::MaybeUninit<u8>] as *mut [u8])
            };
            inner.get_ref().read(buf)
        }) {
            Ok(result) => result?,
            Err(_) => return Poll::Pending,
        };
        unsafe { buf.assume_init(count) };
        buf.advance(count);
        Poll::Ready(Ok(()))
    }
}

impl AsyncWrite for EventFd {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        loop {
            let mut guard = ready!(self.0.poll_write_ready(cx))?;

            match guard.try_io(|inner| inner.get_ref().write(buf)) {
                Ok(result) => return Poll::Ready(result),
                Err(_would_block) => continue,
            }
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

