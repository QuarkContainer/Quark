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

#![macro_use]

use alloc::string::String;

use super::task::*;
use super::qlib::vcpu_mgr::*;
use super::asm::*;

pub const SCALE : i64 = 2_000;

pub fn PrintPrefix() -> String {
    return format!("[{}/{:x}|{}] ", CPULocal::CpuId() , Task::TaskId().Addr(), Rdtsc()/SCALE);
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let mut s = $crate::print::PrintPrefix();
            s += &format!($($arg)*);
            //s += "\n";
            $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &s);
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({
        if $crate::SHARESPACE.config.DebugLevel >= $crate::qlib::config::DebugLevel::Error {
            //$crate::qlib::perf_tunning::PerfGoto($crate::qlib::perf_tunning::PerfType::Print);
            let mut s = $crate::print::PrintPrefix();
            s += &format!($($arg)*);

            if $crate::SHARESPACE.config.SlowPrint {
                $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &s);
            } else {
                $crate::Kernel::HostSpace::Kprint($crate::qlib::config::DebugLevel::Error, &s);
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
            let mut s = $crate::print::PrintPrefix();
            s += &format!($($arg)*);

            if $crate::SHARESPACE.config.SlowPrint {
                $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &s);
            } else {
                $crate::Kernel::HostSpace::Kprint($crate::qlib::config::DebugLevel::Error, &s);
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
            let mut s = $crate::print::PrintPrefix();
            s += &format!($($arg)*);

            if $crate::SHARESPACE.config.SlowPrint {
                $crate::Kernel::HostSpace::SlowPrint($crate::qlib::config::DebugLevel::Error, &s);
            } else {
                $crate::Kernel::HostSpace::Kprint($crate::qlib::config::DebugLevel::Error, &s);
            }
            //$crate::qlib::perf_tunning::PerfGofrom($crate::qlib::perf_tunning::PerfType::Print);
        }
    });
}

