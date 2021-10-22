// Copyright (c) 2021 Quark Container Authors
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

use alloc::string::String;
use std::fs::File;
use std::io::Write;
use std::fs::OpenOptions;
use spin::Mutex;
use lazy_static::lazy_static;
use chrono::prelude::*;
use std::os::unix::io::AsRawFd;

use super::kvmlib::qlib::qmsg::input::*;

lazy_static! {
    pub static ref LOG : Mutex<Log> = Mutex::new(Log::New());
}

pub struct Log {
    pub file: File,
    pub kernelPrint: bool,
}

pub fn EnableKernelPrint() {
    LOG.lock().EnableKernelPrint();
}

pub const LOG_FILE : &str = "/var/log/quark/quark.log";
pub  const TIME_FORMAT: &str = "%H:%M:%S%.3f";
impl Log {
    pub fn New() -> Self {
        let file = OpenOptions::new().create(true).append(true).open(LOG_FILE).expect("Log Open fail");
        return Self {
            file: file,
            kernelPrint: false,
        }
    }

    pub fn Logfd(&self) -> i32 {
        return self.file.as_raw_fd();
    }

    pub fn EnableKernelPrint(&mut self) {
        self.kernelPrint = false;
    }

    pub fn Write(&mut self, str: &str) {
        if !self.kernelPrint {
            self.WriteBytes(str.as_bytes());
        } else {
            let uringLog = super::kvmlib::VMS.lock().shareSpace.config.read().UringLog;
            let trigger = super::kvmlib::VMS.lock().shareSpace.Log(str.as_bytes());
            if trigger {
                if uringLog {
                    super::kvmlib::VMS.lock().shareSpace.AQHostInputCall(&HostInputMsg::LogFlush);
                }
            }


            if !uringLog {
                super::kvmlib::VMS.lock().shareSpace.LogFlush();
            }
        }
    }

    pub fn WriteBytes(&mut self, buf: &[u8]) {
        self.file.write_all(buf).expect("log write fail");
    }

    pub fn Now() -> String {
        return Local::now().format(TIME_FORMAT).to_string()
    }

    pub fn Print(&mut self, level: &str, str: &str) {
        //self.Write(&format!("{} [{}] {}\n", Self::Now(), level, str));
        self.Write(&format!("[{}] {}\n", level, str));
    }
}

#[macro_export]
macro_rules! raw {
 // macth like arm for macro
    ($a:expr,$b:expr,$c:expr)=>{
        {
           error!("raw:: {:x}/{:x}/{:x}", $a, $b, $c);
        }
    }
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.lock().Write(&format!("{}\n",&s));
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.lock().Print("Print", &s);
    });
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.lock().Print("ERROR", &s);
    });
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.lock().Print("INFO", &s);
    });
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.lock().Print("DEBUG", &s);
    });
}
