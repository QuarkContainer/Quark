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

use super::qlib::kernel::Timestamp;
use super::qlib::vcpu_mgr::*;
use super::task::*;
use alloc::string::String;
use log::{Record, Level, Metadata, SetLoggerError, LevelFilter};

pub struct SimpleLogger;

static LOGGER: SimpleLogger = SimpleLogger;

pub fn init() -> core::result::Result<(), SetLoggerError> {
    let res = log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(LevelFilter::Trace));
    res
}

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {

        let res = match super::SHARESPACE.config.read().DebugLevel {
            crate::qlib::config::DebugLevel::Off =>  metadata.level() < Level::Error,
            crate::qlib::config::DebugLevel::Error =>  metadata.level() <= Level::Error,
            crate::qlib::config::DebugLevel::Warn =>  metadata.level() <= Level::Warn,
            crate::qlib::config::DebugLevel::Info =>  metadata.level() <= Level::Info,
            crate::qlib::config::DebugLevel::Debug =>  metadata.level() <= Level::Debug,
            crate::qlib::config::DebugLevel::Trace =>  metadata.level() <= Level::Trace,            
        };

        res
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            if crate::SHARESPACE.config.read().SyncPrint() {
                let str = format!("[{}] {}", record.level(), record.args());
                crate::Kernel::HostSpace::SyncPrint(crate::qlib::config::DebugLevel::Error, &str);
            } else {
                let str = format!("[{}] {}", record.level(), record.args());
                crate::Kernel::HostSpace::Kprint(&str);
            }
        }
    }
    fn flush(&self) {}
}

pub fn PrintPrefix() -> String {
    let now = if super::SHARESPACE.config.read().PerfDebug {
        Timestamp()
    } else {
        0
    };

    return format!(
        "[{}/{:x}|{}]",
        CPULocal::CpuId(),
        Task::TaskAddr(),
        now
    );
}

#[macro_export]
macro_rules! raw {
    // macth like arm for macro
    ($a:expr,$b:expr,$c:expr,$d:expr) => {{
        $crate::Kernel::HostSpace::KernelMsg($a, $b, $c, $d);
    }};
}

#[macro_export]
macro_rules! raw_print {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.read().DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let s = &format!($($arg)*);
            let str = format!("[Print] {}", s);

            $crate::Kernel::HostSpace::SyncPrint($crate::qlib::config::DebugLevel::Error, &str);
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.read().DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);
            let str = format!("[Print] {} {}", prefix, s);

            $crate::Kernel::HostSpace::SyncPrint($crate::qlib::config::DebugLevel::Error, &str);
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.read().DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);

            if $crate::SHARESPACE.config.read().SyncPrint() {
                let str = format!("[ERROR] {} {}", prefix, s);
                $crate::Kernel::HostSpace::SyncPrint($crate::qlib::config::DebugLevel::Error, &str);
            } else {
                let str = format!("[ERROR] {} {}\n", prefix, s);
                $crate::Kernel::HostSpace::Kprint(&str);
            }

            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.read().DebugLevel >= $crate::qlib::config::DebugLevel::Info {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);

            if $crate::SHARESPACE.config.read().SyncPrint() {
                let str = format!("[INFO] {} {}", prefix, s);
                $crate::Kernel::HostSpace::SyncPrint($crate::qlib::config::DebugLevel::Error, &str);
            } else {
                 let str = format!("[INFO] {} {}\n", prefix, s);
                 $crate::Kernel::HostSpace::Kprint(&str);
            }
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.read().DebugLevel >= $crate::qlib::config::DebugLevel::Info {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);

            if $crate::SHARESPACE.config.read().SyncPrint() {
                let str = format!("[WARN] {} {}", prefix, s);
                $crate::Kernel::HostSpace::SyncPrint($crate::qlib::config::DebugLevel::Error, &str);
            } else {
                 let str = format!("[WARN] {} {}\n", prefix, s);
                 $crate::Kernel::HostSpace::Kprint(&str);
            }
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.read().DebugLevel >= $crate::qlib::config::DebugLevel::Debug {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);

            if $crate::SHARESPACE.config.read().SyncPrint() {
                let str = format!("[DEBUG] {} {}", prefix, s);
                $crate::Kernel::HostSpace::SyncPrint($crate::qlib::config::DebugLevel::Error, &str);
            } else {
                let str = format!("[DEBUG] {} {}\n", prefix, s);
                $crate::Kernel::HostSpace::Kprint(&str);
            }
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}
