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

pub enum Arch {
    // AMD64 is the x86-64 architecture.
    AMD64,
    // ARM64 is the aarch64 architecture.
    ARM64,
}

// LinuxSysname is the OS name advertised by Quark.
pub const LINUX_SYSNAME: &'static str = "Linux";

// LinuxRelease is the Linux release version number advertised by Quark.
pub const LINUX_RELEASE: &'static str = "4.4.0";

// LinuxVersion is the version info advertised by gVisor.
pub const LINUX_VERSION: &'static str = "#1 SMP Sun Jan 10 15:06:54 PST 2016";

pub struct Version {
    // Operating system name (e.g. "Linux").
    pub OS: &'static str,

    pub Arch: Arch,

    // Operating system name (e.g. "Linux").
    pub Sysname: &'static str,

    // Operating system release (e.g. "4.4-amd64").
    pub Release: &'static str,

    // Operating system version. On Linux this takes the shape
    // "#VERSION CONFIG_FLAGS TIMESTAMP"
    // where:
    // - VERSION is a sequence counter incremented on every successful build
    // - CONFIG_FLAGS is a space-separated list of major enabled kernel features
    //   (e.g. "SMP" and "PREEMPT")
    // - TIMESTAMP is the build timestamp as returned by `date`
    pub Version: &'static str,
}

pub const VERSION: Version = Version {
    OS: LINUX_SYSNAME,
    #[cfg(target_arch = "x86_64")]
    Arch: Arch::AMD64,
    #[cfg(target_arch = "aarch64")]
    Arch: Arch::ARM64,
    Sysname: LINUX_SYSNAME,
    Release: LINUX_RELEASE,
    Version: LINUX_VERSION,
};
