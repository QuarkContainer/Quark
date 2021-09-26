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

use alloc::string::String;
use lazy_static::lazy_static;

use super::task::*;
use super::qlib::vcpu_mgr::*;
use super::qlib::config::DebugLevel;
use super::asm::*;

pub const SCALE : i64 = 2_000;
pub const ERROR : DebugLevel = DebugLevel::Error;
pub const INFO : DebugLevel = DebugLevel::Info;
pub const DEBUG : DebugLevel = DebugLevel::Debug;

lazy_static! {
    pub static ref DEBUG_LEVEL: DebugLevel = super::SHARESPACE.config.DebugLevel;
}

pub fn PrintPrefix() -> String {
    let now = if super::SHARESPACE.config.PerfDebug {
        Rdtsc()/SCALE
    } else {
        0
    };

    return format!("[{}/{:x}|{}]", CPULocal::CpuId() , Task::TaskId().Addr(), now);
}

pub fn DebugLevel() -> DebugLevel {
    let level = super::SHARESPACE.config.DebugLevel;
    return level;
}

#[macro_export]
macro_rules! raw_print {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let s = &format!($($arg)*);
            let str = format!("[Print] {}", s);

            $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &str);
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        let current = $crate::print::ERROR;
        let level = *$crate::print::DEBUG_LEVEL;
        let cmp = level >= current;

        if cmp {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);
            let str = format!("[Print] {} {}", prefix, s);

            $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &str);
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({

        let repro = true;
        let cmp;

        if !repro {
            let current = $crate::print::ERROR;
            let level = *$crate::print::DEBUG_LEVEL;
            cmp = level >= current;
        } else {
            cmp = $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error;
        }

        if cmp {
        //if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);

            if $crate::SHARESPACE.config.SlowPrint {
                let str = format!("[ERROR] {} {}", prefix, s);
                $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &str);
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
        let current = $crate::print::INFO;
        let level = *$crate::print::DEBUG_LEVEL;
        let cmp = level >= current;

        if cmp  {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);

            if $crate::SHARESPACE.config.SlowPrint {
                let str = format!("[INFO] {} {}", prefix, s);
                $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &str);
            } else {
                 let str = format!("[INFO] {} {}\n", prefix, s);
                 $crate::Kernel::HostSpace::Kprint(&str);
            }
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => ({
        let current = $crate::print::DEBUG;
        let level = *$crate::print::DEBUG_LEVEL;
        let cmp = level >= current;

        if cmp {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);

            if $crate::SHARESPACE.config.SlowPrint {
                let str = format!("[DEBUG] {} {}", prefix, s);
                $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &str);
            } else {
                let str = format!("[DEBUG] {} {}\n", prefix, s);
                $crate::Kernel::HostSpace::Kprint(&str);
            }
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

