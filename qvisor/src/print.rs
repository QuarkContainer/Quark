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

lazy_static! {
    pub static ref LOG : Mutex<Log> = Mutex::new(Log::New());
}

pub struct Log {
    pub file: File,
}

pub const LOG_FILE : &str = "/var/log/quark/quark.log";
pub  const TIME_FORMAT: &str = "%H:%M:%S%.3f";
impl Log {
    pub fn New() -> Self {
        let file = OpenOptions::new().create(true).append(true).open(LOG_FILE).expect("Log Open fail");
        return Self {
            file: file
        }
    }

    pub fn Write(&mut self, str: &str) {
        self.file.write_all(str.as_bytes()).expect("log write fail");
    }

    pub fn Now() -> String {
        return Local::now().format(TIME_FORMAT).to_string()
    }

    pub fn Print(&mut self, level: &str, str: &str) {
        self.Write(&format!("{} [{}] {}\n", Self::Now(), level, str));
    }
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
