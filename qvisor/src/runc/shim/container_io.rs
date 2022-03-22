/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

use std::fs::OpenOptions;
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::FromRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::process::Command;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::process::Stdio as ProcessStdio;
use std::os::unix::io::AsRawFd;
use std::os::unix::io::IntoRawFd;

use libc::*;

use containerd_shim::util::IntoOption;

use super::super::super::util::*;

use nix::{ioctl_write_ptr_bad};
ioctl_write_ptr_bad!(ioctl_set_winsz, libc::TIOCSWINSZ, libc::winsize);

use super::super::super::console::pty::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;

#[derive(Clone, Debug, Default)]
pub struct ContainerStdio {
    pub stdin: String,
    pub stdout: String,
    pub stderr: String,
    pub terminal: bool,
}

impl ContainerStdio {
    pub fn is_null(&self) -> bool {
        self.stdin.is_empty() && self.stdout.is_empty() && self.stderr.is_empty()
    }

    pub fn CreateIO(&self) -> Result<ContainerIO> {
        if self.terminal {
            return Ok(ContainerIO::PtyIO(PtyIO::New()?))
        }

        if self.is_null() {
            let nio = NullIO::new()?;
            return Ok(ContainerIO::NullIO(nio))
        }

        let io = FifoIO {
            stdin: self.stdin.to_string().none_if(|x| x.is_empty()),
            stdout: self.stdout.to_string().none_if(|x| x.is_empty()),
            stderr: self.stderr.to_string().none_if(|x| x.is_empty()),
        };

        return Ok(ContainerIO::FifoIO(io))
    }
}

#[derive(Debug)]
pub enum ContainerIO {
    PtyIO(PtyIO),
    FifoIO(FifoIO),
    NullIO(NullIO),
    None,
}

impl Default for ContainerIO {
    fn default() -> Self {
        return Self::None
    }
}

impl ContainerIO {
    pub fn Set(&self, cmd: &mut Command) -> Result<()> {
        match self {
            Self::PtyIO(c) => return c.Set(cmd),
            Self::FifoIO(c) => return c.Set(cmd),
            Self::NullIO(c) => return c.Set(cmd),
            Self::None => panic!("ContainerIO::None"),
        }
    }

    pub fn CloseAfterStart(&self) {
        match self {
            Self::PtyIO(c) => return c.CloseAfterStart(),
            Self::FifoIO(c) => return c.CloseAfterStart(),
            Self::NullIO(c) => return c.CloseAfterStart(),
            Self::None => panic!("ContainerIO::None"),
        }
    }

    pub fn ResizePty(&self, height: u32, width: u32) -> Result<()> {
        match self {
            Self::PtyIO(c) => return c.ResizePty(height, width),
            Self::None => panic!("ContainerIO::None"),
            _ => return Err(Error::Common(format!("there is no console")))
        }
    }

    pub fn CopyIO(&self, stdio: &ContainerStdio) -> Result<()> {
        match self {
            Self::PtyIO(c) => return c.CopyIO(stdio),
            Self::None => panic!("ContainerIO::None"),
            _ => return Ok(())
        }
    }

    pub fn StdioFds(&self) -> Result<Vec<i32>> {
        match self {
            Self::PtyIO(c) => return c.StdioFds(),
            Self::FifoIO(c) => return c.StdioFds(),
            Self::NullIO(c) => return c.StdioFds(),
            Self::None => panic!("ContainerIO::StdioFds"),
        }
    }
}

#[derive(Debug)]
pub struct PtyIO {
    pub master: Master,
    pub slave: Slave,
    pub pipeRead: i32,
    pub pipeWrite: i32,
}

impl Drop for PtyIO {
    fn drop(&mut self) {
        unsafe {
            //error!("start to close pipe ....");
            libc::close(self.pipeWrite);
        }
    }
}

impl PtyIO {
    pub fn New() -> Result<Self> {
        let (master, slave) = NewPty()?;

        let mut fds : [i32; 2] = [0, 0];
        let ret = unsafe {
            pipe(&mut fds[0] as * mut i32)
        };

        if ret < 0 {
            return Err(Error::SysError(errno::errno().0))
        }

        return Ok(Self {
            master: master,
            slave: slave,
            pipeRead: fds[0],
            pipeWrite: fds[1],
        })
    }

    pub fn Set(&self, cmd: &mut Command) -> Result<()> {
        unsafe {
            //let tty = self.slave.dup()?;
            let tty = self.slave.pty;
            cmd.stdin(ProcessStdio::from_raw_fd(tty));
            cmd.stdout(ProcessStdio::from_raw_fd(tty));
            cmd.stderr(ProcessStdio::from_raw_fd(tty));
        }

        Ok(())
    }

    pub fn CloseAfterStart(&self) {
        unsafe {
            libc::close(self.slave.pty);
        }
    }

    pub fn ResizePty(&self, height: u32, width: u32) -> Result<()> {
        let tty = self.master.pty;
        unsafe {
            let w = libc::winsize {
                ws_row: height as u16,
                ws_col: width as u16,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            ioctl_set_winsz(tty, &w)
                .map(|_x| ())
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
        }

        return Ok(())
    }

    pub fn CopyIO(&self, stdio: &ContainerStdio) -> Result<()> {
        if !stdio.stdin.is_empty() {
            let tty = self.master.pty;
            let f = unsafe { File::from_raw_fd(tty) };
            //debug!("copy_console: pipe stdin to console {}/{}", f.as_raw_fd(), tty);
            let stdin = OpenOptions::new()
                .read(true)
                //.write(true)
                .open(stdio.stdin.as_str())
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            //spawn_copy(stdin, f, None);
            Redirect(stdin, f, None, self.pipeRead);
        }

        if !stdio.stdout.is_empty() {
            let tty = self.master.pty;
            let f = unsafe { File::from_raw_fd(tty) };
            //debug!("copy_console: pipe stdout from console {}/{}", f.as_raw_fd(), tty);
            let stdout = OpenOptions::new()
                .write(true)
                .open(stdio.stdout.as_str())
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            // open a read to make sure even if the read end of containerd shutdown,
            // copy still continue until the restart of containerd succeed
            let stdout_r = OpenOptions::new()
                .read(true)
                .open(stdio.stdout.as_str())
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            /*spawn_copy(
                f,
                stdout,
                Some(Box::new(move || {
                    drop(stdout_r);
                })),
            );*/

            Redirect(f,
                     stdout,
                     Some(Box::new(move || {
                         drop(stdout_r);
                     })),
                     self.pipeRead);
        }

        return Ok(())
    }

    pub fn StdioFds(&self) -> Result<Vec<i32>> {
        let tty = self.slave.pty;
        return Ok(vec![tty])
    }
}

#[derive(Debug)]
pub struct FifoIO {
    pub stdin: Option<String>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
}

impl FifoIO {
    pub fn Set(&self, cmd: &mut Command) -> Result<()> {
        if let Some(path) = self.stdin.as_ref() {
            let stdin = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(path)
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            cmd.stdin(stdin);
        }

        if let Some(path) = self.stdout.as_ref() {
            let stdout = OpenOptions::new().write(true).open(path)
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            cmd.stdout(stdout);
        }

        if let Some(path) = self.stderr.as_ref() {
            let stderr = OpenOptions::new().write(true).open(path)
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            cmd.stderr(stderr);
        }

        Ok(())
    }

    pub fn StdioFds(&self) -> Result<Vec<i32>> {
        let fd0;
        let fd1;
        let fd2;
        if let Some(path) = self.stdin.as_ref() {
            let stdin = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(path)
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            fd0 = stdin.into_raw_fd();
        } else {
            fd0 = nix::fcntl::open(
                "/dev/null",
                nix::fcntl::OFlag::O_RDONLY,
                nix::sys::stat::Mode::empty(),
            ).map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
        }

        if let Some(path) = self.stdout.as_ref() {
            let stdout = OpenOptions::new().write(true).open(path)
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            fd1 = stdout.into_raw_fd();
        } else {
            fd1 = nix::fcntl::open(
                "/dev/null",
                nix::fcntl::OFlag::O_RDONLY,
                nix::sys::stat::Mode::empty(),
            ).map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
        }

        if let Some(path) = self.stderr.as_ref() {
            let stderr = OpenOptions::new().write(true).open(path)
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            fd2 = stderr.into_raw_fd();
        } else {
            fd2 = nix::fcntl::open(
                "/dev/null",
                nix::fcntl::OFlag::O_RDONLY,
                nix::sys::stat::Mode::empty(),
            ).map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
        }

        return Ok(vec![fd0, fd1, fd2])
    }

    pub fn CloseAfterStart(&self) {}
}

/// IO driver to direct output/error messages to /dev/null.
///
/// With this Io driver, all methods of [crate::Runc] can't capture the output/error messages.
#[derive(Debug)]
pub struct NullIO {
    dev_null: Mutex<Option<File>>,
}

impl NullIO {
    pub fn new() -> Result<Self> {
        let fd = nix::fcntl::open(
            "/dev/null",
            nix::fcntl::OFlag::O_RDONLY,
            nix::sys::stat::Mode::empty(),
        ).map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
        let dev_null = unsafe { Mutex::new(Some(std::fs::File::from_raw_fd(fd))) };
        Ok(Self { dev_null })
    }

    pub fn Set(&self, cmd: &mut Command) -> Result<()> {
        if let Some(null) = self.dev_null.lock().unwrap().as_ref() {
            cmd.stdout(null.try_clone()
                           .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?);
            cmd.stderr(null.try_clone()
                           .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?);
        }
        Ok(())
    }

    pub fn CloseAfterStart(&self) {
        let mut m = self.dev_null.lock().unwrap();
        let _ = m.take();
    }

    pub fn StdioFds(&self) -> Result<Vec<i32>> {
        let fd = if let Some(null) = self.dev_null.lock().unwrap().as_ref() {
            null.as_raw_fd()
        } else {
            panic!("NullIO StdioFds fail")
        };
        return Ok(vec![fd, fd, fd])
    }
}

pub fn spawn_copy(
    mut from: File,
    mut to: File,
    on_close_opt: Option<Box<dyn FnOnce() + Send + Sync>>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        /*if let Err(e) = std::io::copy(&mut from, &mut to) {
            debug!("copy io error: {}", e);
        }*/

        let mut buf = [0; 1024];
        error!("spawn_copy start ....");
        loop {
            error!("spawn_copy 1");
            match from.read(&mut buf) {
                Err(e) => {
                    error!("spawn_copy e {:?}/{}", e, from.as_raw_fd());
                    break;
                },
                Ok(cnt) => {
                    error!("spawn_copy 2 .... {}/{}", cnt, std::str::from_utf8(&buf[0..cnt]).unwrap());
                    if cnt == 0 {
                        break;
                    }
                    assert!(to.write(&buf[0..cnt]).unwrap()==cnt)
                }
            }
        }

        if let Some(x) = on_close_opt {
            x()
        };
    })
}

pub fn Redirect( from: File,
                 to: File,
                 on_close_opt: Option<Box<dyn FnOnce() + Send + Sync>>,
                 readPipe: i32) -> JoinHandle<()> {
    std::thread::spawn(move || {
        // don't know why the read end of the pty slave is not closed correctly.
        // have to use pipe to trigger the close
        // todo: fix this
        defer!({
            if let Some(x) = on_close_opt {
                x()
            };

            drop(&from);
            drop(&to);

            unsafe {
                libc::close(readPipe);
            }
        });

        let mut events = Events::New();
        let epoll = Epoll::New().unwrap();

        epoll.Addfd(from.as_raw_fd(), (EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLET) as u32).unwrap();
        epoll.Addfd(readPipe, (EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLET) as u32).unwrap();

        loop {
            epoll.Poll(&mut events).unwrap();

            for event in events.Slice() {
                let fd = event.u64 as i32;
                if event.events & (EPOLLHUP | EPOLLERR) as u32 != 0 {
                    return
                }

                let cnt = if fd == from.as_raw_fd() {
                    Copy(from.as_raw_fd(), to.as_raw_fd()).unwrap()
                } else { //if fd == readPipe {
                    return
                };

                if cnt == 0 {
                    //eof
                    return
                }
            }
        }
    })

}

pub fn Copy(from: i32, to: i32) -> Result<usize> {
    let mut buf : [u8; 256] = [0; 256];
    let mut cnt = 0;
    loop {
        let ret = unsafe {
            read(from, &mut buf[0] as * mut _ as *mut c_void, buf.len())
        };

        if ret == 0 {
            return Ok(cnt);
        }

        if ret < 0 {
            if errno::errno().0 as i32 == SysErr::EAGAIN {
                return Ok(cnt)
            }

            return Err(Error::SysError(errno::errno().0 as i32))
        }

        let ret = ret as usize;
        cnt += ret;
        let mut offset = 0;
        while offset < ret {
            let writeCnt = unsafe {
                write(to, &buf[offset] as * const _ as *const c_void, ret - offset)
            };

            if writeCnt < 0 {
                return Err(Error::SysError(errno::errno().0 as i32))
            }

            offset += writeCnt as usize;
        }
    }
}

