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

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;

use super::tty::*;
use super::util::*;

pub fn ioctlGetTermios(fd: i32, termios: &mut Termios) -> Result<()> {
    let ret = Ioctl(fd, IoCtlCmd::TCGETS, termios as *mut Termios as u64);

    if ret < 0 {
        return Err(Error::SysError(-ret));
    }

    return Ok(());
}

pub fn ioctlSetTermios(fd: i32, req: u64, termios: &Termios) -> Result<()> {
    let ret = Ioctl(fd, req, termios as *const Termios as u64);

    if ret < 0 {
        return Err(Error::SysError(-ret));
    }

    return Ok(());
}

pub fn ioctlGetWinsize(fd: i32, w: &mut Winsize) -> Result<()> {
    let ret = Ioctl(fd, IoCtlCmd::TIOCGWINSZ, w as *mut Winsize as u64);

    if ret < 0 {
        return Err(Error::SysError(-ret));
    }

    return Ok(());
}

pub fn ioctlSetWinsize(fd: i32, w: &Winsize) -> Result<()> {
    let ret = Ioctl(fd, IoCtlCmd::TIOCSWINSZ, w as *const Winsize as u64);

    if ret < 0 {
        return Err(Error::SysError(-ret));
    }

    return Ok(());
}
