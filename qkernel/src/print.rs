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

use super::task::*;
use super::qlib::vcpu_mgr::*;
use super::asm::*;

pub const SCALE : i64 = 2_000;

pub fn PrintPrefix() -> String {
    return format!("[{}/{:x}|{}]", CPULocal::CpuId() , Task::TaskId().Addr(), Rdtsc()/SCALE);
}

#[macro_export]
macro_rules! raw_print {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let s = &format!($($arg)*);
            let str = format!("[ERROR] {}", s);

            $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &str);
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let prefix = $crate::print::PrintPrefix();
            let s = &format!($($arg)*);
            let str = format!("[ERROR] {} {}", prefix, s);

            $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &str);
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error {
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
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Info {
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
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Debug {
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

