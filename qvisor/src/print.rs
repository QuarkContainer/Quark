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

use super::qlib::ShareSpace;
use super::qlib::kernel::IOURING;
use super::qlib::kernel::Timestamp;
use super::ThreadId;

lazy_static! {
    pub static ref LOG : Mutex<Log> = Mutex::new(Log::New());
}

pub struct Log {
    pub file: File,
    pub syncPrint: bool,
    pub shareSpace: &'static ShareSpace,
}

pub fn SetSyncPrint(syncPrint: bool) {
    LOG.lock().SetSyncPrint(syncPrint);
}

pub fn SetSharespace(sharespace: &'static ShareSpace) {
    LOG.lock().shareSpace = sharespace;
}

pub const LOG_FILE_DEFAULT : &str = "/var/log/quark/quark.log";
pub const LOG_FILE_FORMAT : &str = "/var/log/quark/{}.log";
pub  const TIME_FORMAT: &str = "%H:%M:%S%.3f";
impl Log {
    pub fn New() -> Self {
        let file = OpenOptions::new().create(true).append(true).open(LOG_FILE_DEFAULT).expect("Log Open fail");
        return Self {
            file: file,
            syncPrint: true,
            shareSpace: unsafe {
                &mut *(0 as * mut ShareSpace)
            },
        }
    }

    pub fn Reset(&mut self, name: &str) {
        let filename = format!( "/var/log/quark/{}.log", name);
        let file = OpenOptions::new().create(true).append(true).open(filename).expect("Log Open fail");
        self.file = file;
    }

    pub fn SetSharespace(&mut self, sharespace: &'static ShareSpace) {
        self.shareSpace = sharespace;
    }

    pub fn Logfd(&self) -> i32 {
        return self.file.as_raw_fd();
    }

    pub fn SetSyncPrint(&mut self, syncPrint: bool) {
        self.syncPrint = syncPrint;
    }

    pub fn RawWrite(&mut self, str: &str) {
        self.WriteBytes(str.as_bytes());
    }

    pub fn Write(&mut self, str: &str) {
        if self.syncPrint {
            self.RawWrite(str);
        } else {
            let trigger = self.shareSpace.Log(str.as_bytes());
            if trigger && self.shareSpace.config.read().Async() {
                //self.shareSpace.AQHostInputCall(&HostInputMsg::LogFlush);
                IOURING.LogFlush();
            }
        }
    }

    pub fn Flush(&self, partial: bool) {
        self.shareSpace.LogFlush(partial);
    }

    pub fn WriteBytes(&mut self, buf: &[u8]) {
        self.file.write_all(buf).expect("log write fail");
    }

    pub fn Now() -> String {
        return Local::now().format(TIME_FORMAT).to_string()
    }

    pub fn Print(&mut self, level: &str, str: &str) {
        let now = Timestamp();
        self.Write(&format!("[{}] [{}/{}] {}\n", level, ThreadId(), now, str));
    }

    pub fn RawPrint(&mut self, level: &str, str: &str) {
        //self.Write(&format!("{} [{}] {}\n", Self::Now(), level, str));
        self.RawWrite(&format!("[{}] {}\n", level, str));
    }

    pub fn Clear(&mut self) {
        if !self.syncPrint {
            self.shareSpace.LogFlush(false);
            self.syncPrint = true;
        }
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
        crate::print::LOG.lock().RawWrite(&format!("{}\n",&s));
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.lock().RawPrint("Print", &s);
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
