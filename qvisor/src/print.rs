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
use chrono::prelude::*;
use lazy_static::lazy_static;
use std::fs::OpenOptions;
use std::os::unix::io::IntoRawFd;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::Ordering;

use super::qlib::kernel::Timestamp;
use super::qlib::kernel::IOURING;
use super::qlib::kernel::SHARESPACE;
use super::ThreadId;

lazy_static! {
    pub static ref LOG: Log = Log::New();
}

pub struct Log {
    pub fd: AtomicI32,
    pub syncPrint: AtomicBool,
}

pub fn SetSyncPrint(syncPrint: bool) {
    LOG.SetSyncPrint(syncPrint);
}

pub const LOG_FILE_DEFAULT: &str = "/var/log/quark/quark.log";
pub const LOG_FILE_FORMAT: &str = "/var/log/quark/{}.log";
pub const TIME_FORMAT: &str = "%H:%M:%S%.3f";

impl Drop for Log {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.Logfd());
        }
    }
}

impl Log {
    pub fn New() -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(LOG_FILE_DEFAULT)
            .expect("Log Open fail");
        return Self {
            fd: AtomicI32::new(file.into_raw_fd()),
            syncPrint: AtomicBool::new(true),
        };
    }

    pub fn Reset(&self, name: &str) {
        let filename = format!("/var/log/quark/{}.log", name);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .expect("Log Open fail");

        unsafe {
            libc::close(self.Logfd());
        }

        self.fd.store(file.into_raw_fd(), Ordering::SeqCst);
    }

    pub fn Logfd(&self) -> i32 {
        return self.fd.load(Ordering::Relaxed);
    }

    pub fn SyncPrint(&self) -> bool {
        return self.syncPrint.load(Ordering::Relaxed);
    }

    pub fn SetSyncPrint(&self, syncPrint: bool) {
        self.syncPrint.store(syncPrint, Ordering::SeqCst);
    }

    pub fn RawWrite(&self, str: &str) {
        self.WriteAll(str.as_bytes());
    }

    pub fn Write(&self, str: &str) {
        if self.SyncPrint() {
            self.WriteAll(str.as_bytes());
        } else {
            let trigger = SHARESPACE.Log(str.as_bytes());
            if trigger && SHARESPACE.config.read().Async() {
                //self.shareSpace.AQHostInputCall(&HostInputMsg::LogFlush);
                IOURING.LogFlush();
            }
        }
    }

    pub fn Flush(&self, partial: bool) {
        SHARESPACE.LogFlush(partial);
    }

    fn write(&self, buf: &[u8]) -> i32 {
        let ret = unsafe {
            libc::write(self.Logfd(), &buf[0] as * const _ as u64 as * const _, buf.len() as _)
        };

        if ret < 0 {
            panic!("log write fail ...")
        }

        return ret as i32;
    }

    pub fn WriteAll(&self, buf: &[u8]) {
        let mut count = 0;
        while count < buf.len() {
            let n = self.write(&buf[count..]);
            count += n as usize;
        }
    }

    pub fn Now() -> String {
        return Local::now().format(TIME_FORMAT).to_string();
    }

    pub fn Print(&self, level: &str, str: &str) {
        let now = Timestamp();
        //let now = RawTimestamp();
        self.Write(&format!("[{}] [{}/{}] {}\n", level, ThreadId(), now, str));
    }

    pub fn RawPrint(&self, level: &str, str: &str) {
        //self.Write(&format!("{} [{}] {}\n", Self::Now(), level, str));
        self.RawWrite(&format!("[{}] {}\n", level, str));
    }

    pub fn Clear(&self) {
        if !self.SyncPrint() {
            SHARESPACE.LogFlush(false);
            self.SetSyncPrint(true);
        }
    }
}

#[macro_export]
macro_rules! raw {
    // macth like arm for macro
    ($a:expr,$b:expr,$c:expr) => {{
        error!("raw:: {:x}/{:x}/{:x}", $a, $b, $c);
    }};
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.RawWrite(&format!("{}\n",&s));
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.RawPrint("Print", &s);
    });
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.Print("ERROR", &s);
    });
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.Print("INFO", &s);
    });
}

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.Print("WARN", &s);
    });
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        crate::print::LOG.Print("DEBUG", &s);
    });
}
