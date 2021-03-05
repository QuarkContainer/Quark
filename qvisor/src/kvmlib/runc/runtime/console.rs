// Copyright (c) 2021 QuarkSoft LLC
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

use std::fs::File;
use std::os::unix::io::FromRawFd;
use std::os::unix::io::AsRawFd;
use std::io;
use libc::*;

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::util::*;
use super::super::super::console::pty::*;

#[derive(Debug)]
pub enum Console {
    StdioConsole(StdioConsole),
    PtyConsole(PtyConsole),
    Detach,
}

impl Default for Console {
    fn default() -> Self {
        return Self::Detach;
    }
}

#[derive(Debug)]
pub struct StdioConsole {
    pub stdin: File,
    pub stdout: File,
    pub stderr: File,
}

impl StdioConsole {
    pub fn New(stdin: i32, stdout: i32, stderr: i32) -> Self {
        unsafe {
            return Self {
                stdin: File::from_raw_fd(stdin),
                stdout: File::from_raw_fd(stdout),
                stderr: File::from_raw_fd(stderr),
            }
        }
    }

    pub fn Redirect(&self) -> Result<()> {
        let mut events = Events::New();
        let epoll = Epoll::New()?;

        epoll.Addfd(io::stdin().as_raw_fd(), (EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLET) as u32)?;
        epoll.Addfd(self.stdout.as_raw_fd(), (EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLET) as u32)?;
        epoll.Addfd(self.stderr.as_raw_fd(), (EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLET) as u32)?;

        loop {
            epoll.Poll(&mut events)?;

            for event in events.Slice() {
                if event.events & (EPOLLHUP | EPOLLERR) as u32 != 0{
                    return Ok(())
                }

                let fd = event.u64 as i32;
                let cnt = if fd == self.stdout.as_raw_fd() {
                    Copy(fd, io::stdout().as_raw_fd())?
                } else if fd == self.stderr.as_raw_fd() {
                    Copy(fd, io::stderr().as_raw_fd())?
                } else { //io::stdin()
                    Copy(fd, self.stdin.as_raw_fd())?
                };

                if cnt == 0 {
                    //eof
                    return Ok(())
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PtyConsole {
    pub master: Master,
}

impl PtyConsole {
    pub fn New(master: Master) -> Self {
        return Self {
            master: master
        }
    }

    pub fn Redirect(&self) -> Result<()> {
        let mut events = Events::New();
        let epoll = Epoll::New()?;

        epoll.Addfd(io::stdin().as_raw_fd(), (EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLET) as u32)?;
        epoll.Addfd(self.master.as_raw_fd(), (EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLET) as u32)?;

        loop {
            epoll.Poll(&mut events)?;

            for event in events.Slice() {
                if event.events & (EPOLLHUP | EPOLLERR) as u32 != 0{
                    return Ok(())
                }

                let fd = event.u64 as i32;
                let cnt = if fd == io::stdin().as_raw_fd() {
                    Copy(fd, self.master.as_raw_fd())?
                } else {
                    Copy(fd, io::stdin().as_raw_fd())?
                };

                if cnt == 0 {
                    //eof
                    return Ok(())
                }
            }
        }
    }
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

