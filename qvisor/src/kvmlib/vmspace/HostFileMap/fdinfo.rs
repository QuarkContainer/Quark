// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use alloc::sync::Arc;
use core::ops::Deref;
use libc::*;

use super::super::*;
use super::super::qlib::common::*;
use super::super::super::util::*;

#[derive(Clone, Debug)]
pub struct FdInfo(Arc<FdInfoIntern>);

impl FdInfo {
    pub fn New(osfd: i32) -> Self {
        return Self(Arc::new(FdInfoIntern::New(osfd)))
    }
}

impl Deref for FdInfo {
    type Target = Arc<FdInfoIntern>;

    fn deref(&self) -> &Arc<FdInfoIntern> {
        &self.0
    }
}

#[derive(Debug)]
pub struct FdInfoIntern {
    pub osfd: i32,
}

impl FdInfoIntern {
    pub fn New(osfd: i32) -> Self {
        //info!("New osfd {}, hostfd{}: epollable is {}", osfd, hostfd, epollable);
        let res = Self {
            osfd: osfd,
        };

        return res;
    }

    pub fn IOBufWrite(&self, addr: u64, len: usize, offset: isize) -> i64 {
        let osfd = self.osfd;
        let ret = unsafe{
            if offset < 0 {
                write(osfd as c_int, addr as *const c_void, len as size_t)
            } else {
                pwrite(osfd as c_int, addr as *const c_void, len as size_t, offset as off_t)
            }
        };

        return SysRet(ret as i64)
    }

    pub fn IOWrite(&self, _taskId: u64, iovs: u64, iovcnt: i32) -> i64 {
        let osfd = self.osfd;
        let ret = unsafe {
            writev(osfd as c_int, iovs as *const iovec, iovcnt) as i64
        };

        return SysRet(ret as i64)
    }

    pub fn IOAppend(&self, _taskId: u64, iovs: u64, iovcnt: i32, fileLenAddr: u64) -> i64 {
        let osfd = self.osfd;

        //let nr = SysCallID::pwritev2 as usize;

        let end = unsafe {
            lseek(osfd as c_int, 0, libc::SEEK_END)
        };

        if end < 0 {
            panic!("IOAppend lseek1 fail")
        }

        let size = unsafe{
            //todo: don't know why RWF_APPEND doesn't work. need to fix.
            //syscall5(nr, osfd as usize, iovs as usize, iovcnt as usize, -1 as i32 as usize, Flags::RWF_APPEND as usize) as i64
            pwritev(osfd as c_int, iovs as *const iovec, iovcnt, end as i64) as i64
        };

        //error!("IOAppend: end is {:x}, size is {:x}, new end is {:x}", end, size, end + size);
        if size < 0 {
            return SysRet(size as i64)
        }

        unsafe {
            *(fileLenAddr as * mut i64) = (end + size) as i64
        }

        return size;

        // the pwritev2 doesn't work. It will break the bazel build.
        // Todo: root cause this.
        /*let osfd = self.osfd;

        let size = unsafe{
            pwritev2(osfd as c_int, iovs as *const iovec, iovcnt, -1, Flags::RWF_APPEND) as i64
        };

        if size < 0 {
            return SysRet(size as i64)
        }

        let end = unsafe {
            lseek(osfd as c_int, 0, libc::SEEK_END)
        };

        unsafe {
            *(fileLenAddr as * mut i64) = end as i64
        }

        return size as i64*/
    }

    pub fn IOReadAt(&self, _taskId: u64, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let osfd = self.osfd;

        let ret = unsafe {
            if offset as i64 == -1 {
                readv(osfd as c_int, iovs as *const iovec, iovcnt) as i64
            } else {
                preadv(osfd as c_int, iovs as *const iovec, iovcnt, offset as i64) as i64
            }
        };

        return SysRet(ret as i64)
    }

    pub fn IOWriteAt(&self, _taskId: u64, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let osfd = self.osfd;

        let ret = unsafe{
            if offset as i64 == -1 {
                writev(osfd as c_int, iovs as *const iovec, iovcnt) as i64
            } else {
                pwritev(osfd as c_int, iovs as *const iovec, iovcnt, offset as i64) as i64
            }
        };

        return SysRet(ret as i64)
    }

    pub fn IOAccept(&self, _taskId: u64, addr: u64, addrlen: u64, _flags: i32) -> i64 {
        let osfd = self.osfd;

        let newOsfd = unsafe{
            accept4(osfd, addr as  *mut sockaddr, addrlen as  *mut socklen_t, SocketFlags::SOCK_NONBLOCK | SocketFlags::SOCK_CLOEXEC)
        };

        if newOsfd < 0 {
            return SysRet(newOsfd as i64);
        }

        //let hostfd = IO_MGR.lock().AddFd(newOsfd);
        URING_MGR.lock().Addfd(newOsfd).unwrap();
        return SysRet(newOsfd as i64);
    }

    pub fn IOConnect(&self, _taskId: u64, addr: u64, addrlen: u32) -> i64 {
        let osfd = self.osfd;

        let ret = unsafe{
            connect(osfd, addr as *const sockaddr, addrlen as socklen_t)
        };

        return SysRet(ret as i64)
    }

    pub fn IORecvMsg(&self, _taskId: u64, msghdr: u64, flags: i32) -> i64 {
        let osfd = self.osfd;

        let ret = unsafe{
            recvmsg(osfd, msghdr as *mut msghdr, flags as c_int)
        };

        return SysRet(ret as i64);
    }

    pub fn IOSendMsg(&self, _taskId: u64, msghdr: u64, flags: i32) -> i64 {
        let osfd = self.osfd;

        let ret = unsafe{
            sendmsg(osfd, msghdr as *mut msghdr, flags as c_int)
        };

        return SysRet(ret as i64);
    }

    pub fn Fcntl(&self, _taskId: u64, cmd: i32, arg: u64) -> i64 {
        let osfd = self.osfd;

        /*if cmd == Cmd::F_GETFL {
            return self.GetFlags() as i64;
        }*/

        let ret = unsafe{
            fcntl(osfd as c_int, cmd, arg)
        };

        /*if cmd == Cmd::F_SETFL  {
            if ret == 0 {
                self.SetFlags(arg as i32);
            }
        }*/

        return SysRet(ret as i64);
    }

    pub fn IoCtl(&self, _taskId: u64, cmd: u64, argp: u64) -> i64 {
        //todo: fix this.
        /* when run /bin/bash, the second command as below return ENOTTY. Doesn't know why
        ioctl(0, TCGETS, {B38400 opost isig icanon echo ...}) = 0
        ioctl(2, TCGETS, 0x7ffdf82a09a0)        = -1 ENOTTY (Inappropriate ioctl for device)
        ioctl(-1, TIOCGPGRP, 0x7ffdf82a0a14)    = -1 EBADF (Bad file descriptor)
        */
        let osfd = self.osfd;

        if osfd == 2 {
            return -SysErr::ENOTTY as i64
        }

        //error!("IoCtl osfd is {}, cmd is {:x}, argp is {:x}", osfd, cmd, argp);

        let ret = unsafe{
            ioctl(osfd as c_int, cmd, argp)
        };

        return SysRet(ret as i64);
    }

    pub fn FSync(&self, _taskId: u64, dataSync: bool) -> i64 {
        let osfd = self.osfd;

        let ret = if dataSync {
            unsafe{
                fsync(osfd)
            }
        } else {
            unsafe{
                fdatasync(osfd)
            }
        };

        return SysRet(ret as i64);
    }

    pub fn Seek(&self, _taskId: u64, offset: i64, whence: i32) -> i64 {
        let osfd = self.osfd;

        let ret = unsafe {
            libc::lseek(osfd, offset, whence)
        };

        return SysRet(ret as i64)
    }


    pub fn GetSockErr(&self) -> Result<u64> {
        let mut err = 0;
        let mut len: u32 = 8;

        let ret = unsafe {
            getsockopt(self.osfd, SOL_SOCKET, SO_ERROR, &mut err as *mut _ as *mut c_void, &mut len as *mut socklen_t)
        };

        if ret == -1 {
            return Err(Error::SysError(errno::errno().0))
        }

        return Ok(err);
    }

    pub fn Close(&self) -> i32 {
        //let _ioMgr = IO_MGR.lock(); //global lock
        if self.osfd >= 0 {
            unsafe {
                // shutdown for socket, without shutdown, it the uring read won't be wake up
                // todo: handle this elegant
                shutdown(self.osfd, 2);
                return close(self.osfd)
            }
        }

        return 0;
    }
}

impl Drop for FdInfoIntern {
    fn drop(&mut self) {
        //error!("in fdInfo drop: guest fd is {}, osfd is {}", self.hostfd, self.osfd);
        //self.Close();
    }
}

