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

use containerd_shim::util::IntoOption;

use nix::{ioctl_write_ptr_bad};
ioctl_write_ptr_bad!(ioctl_set_winsz, libc::TIOCSWINSZ, libc::winsize);

use super::super::super::console::pty::*;
use super::super::super::qlib::common::*;

#[derive(Clone, Debug)]
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

pub enum ContainerIO {
    PtyIO(PtyIO),
    FifoIO(FifoIO),
    NullIO(NullIO),
    None,
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

    pub fn StdioFds(&self) -> Result<[i32; 3]> {
        match self {
            Self::PtyIO(c) => return c.StdioFds(),
            Self::FifoIO(c) => return c.StdioFds(),
            Self::NullIO(c) => return c.StdioFds(),
            Self::None => panic!("ContainerIO::StdioFds"),
        }
    }
}

pub struct PtyIO {
    pub master: Master,
    pub slave: Slave,
}

impl PtyIO {
    pub fn New() -> Result<Self> {
        let (master, slave) = NewPty()?;
        return Ok(Self {
            master: master,
            slave: slave
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

    pub fn CloseAfterStart(&self) {}

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
            let tty = self.master.dup().unwrap();
            let f = unsafe { File::from_raw_fd(tty) };
            debug!("copy_console: pipe stdin to console");
            let stdin = OpenOptions::new()
                .read(true)
                .write(true)
                .open(stdio.stdin.as_str())
                .map_err(|e| Error::IOError(format!("IOErr {:?}", e)))?;
            spawn_copy(stdin, f, None);
        }

        if !stdio.stdout.is_empty() {
            let tty = self.master.pty;
            let f = unsafe { File::from_raw_fd(tty) };
            debug!("copy_console: pipe stdout from console");
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
            spawn_copy(
                f,
                stdout,
                Some(Box::new(move || {
                    drop(stdout_r);
                })),
            );
        }

        return Ok(())
    }

    pub fn StdioFds(&self) -> Result<[i32; 3]> {
        let tty = self.slave.pty;
        return Ok([tty, tty, tty])
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

    pub fn StdioFds(&self) -> Result<[i32; 3]> {
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

        return Ok([fd0, fd1, fd2])
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

    pub fn StdioFds(&self) -> Result<[i32; 3]> {
        let fd = if let Some(null) = self.dev_null.lock().unwrap().as_ref() {
            null.as_raw_fd()
        } else {
            panic!("NullIO StdioFds fail")
        };
        return Ok([fd, fd, fd])
    }
}

pub fn spawn_copy<R: Read + Send + 'static, W: Write + Send + 'static>(
    mut from: R,
    mut to: W,
    on_close_opt: Option<Box<dyn FnOnce() + Send + Sync>>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = std::io::copy(&mut from, &mut to) {
            debug!("copy io error: {}", e);
        }
        if let Some(x) = on_close_opt {
            x()
        };
    })
}

