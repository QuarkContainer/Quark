// Copyright (c) 2021 Quark Container Authors
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

// ioctl(2) requests provided by uapi/linux/sockios.h

pub struct SIOCGIReq {}

impl SIOCGIReq {
    pub const SIOCGIFNAME    : u64 = 0x8910;
	pub const SIOCGIFCONF    : u64 = 0x8912;
	pub const SIOCGIFFLAGS   : u64 = 0x8913;
	pub const SIOCGIFADDR    : u64 = 0x8915;
	pub const SIOCGIFDSTADDR : u64 = 0x8917;
	pub const SIOCGIFBRDADDR : u64 = 0x8919;
	pub const SIOCGIFNETMASK : u64 = 0x891b;
	pub const SIOCGIFMETRIC  : u64 = 0x891d;
	pub const SIOCGIFMTU     : u64 = 0x8921;
	pub const SIOCGIFMEM     : u64 = 0x891f;
	pub const SIOCGIFHWADDR  : u64 = 0x8927;
	pub const SIOCGIFINDEX   : u64 = 0x8933;
	pub const SIOCGIFPFLAGS  : u64 = 0x8935;
	pub const SIOCGIFTXQLEN  : u64 = 0x8942;
	pub const SIOCETHTOOL    : u64 = 0x8946;
	pub const SIOCGMIIPHY    : u64 = 0x8947;
	pub const SIOCGMIIREG    : u64 = 0x8948;
	pub const SIOCGIFMAP     : u64 = 0x8970;
}

// ioctl(2) requests provided by uapi/asm-generic/sockios.h
pub const SIOCGSTAMP : u64 = 0x8906;

// ioctl(2) directions. Used to calculate requests number.
// Constants from asm-generic/ioctl.h.
pub struct IOCDirection {}

impl IOCDirection {
    pub const IOC_NONE  : u32 = 0;
	pub const IOC_WRITE : u32 = 1;
	pub const IOC_READ  : u32 = 2;
}

pub struct IOCBits {}

impl IOCBits {
    pub const IOC_NRBITS   : usize = 8;
	pub const IOC_TYPEBITS : usize = 8;
	pub const IOC_SIZEBITS : usize = 14;
	pub const IOC_DIRBITS  : usize = 2;

	pub const IOC_NRSHIFT   : usize = 0;
	pub const IOC_TYPESHIFT : usize = Self::IOC_NRSHIFT + Self::IOC_NRBITS;
	pub const IOC_SIZESHIFT : usize = Self::IOC_TYPESHIFT + Self::IOC_TYPEBITS;
	pub const IOC_DIRSHIFT  : usize = Self::IOC_SIZESHIFT + Self::IOC_SIZEBITS;
}

impl IOCBits {
    // IOC outputs the result of _IOC macro in include/uapi/asm-generic/ioctl.h.
    pub const fn IOC(dir: u32, typ: u32, nr: u32, size: u32) -> u32 {
        return dir << Self::IOC_DIRSHIFT |
                typ << Self::IOC_TYPESHIFT |
                nr << Self::IOC_NRSHIFT |
                size << Self::IOC_SIZESHIFT
    }

    // IO outputs the result of _IO macro in include/uapi/asm-generic/ioctl.h.
    pub const fn IO(typ: u32, nr: u32) -> u32 {
        return Self::IOC(IOCDirection::IOC_NONE, typ, nr, 0);
    }

    // IOR outputs the result of _IOR macro in include/uapi/asm-generic/ioctl.h.
    pub const fn IOR(typ: u32, nr: u32, size: u32) -> u32 {
        return Self::IOC(IOCDirection::IOC_READ, typ, nr, size);
    }

    // IOW outputs the result of _IOW macro in include/uapi/asm-generic/ioctl.h.
    pub const fn IOW(typ: u32, nr: u32, size: u32) -> u32 {
        return Self::IOC(IOCDirection::IOC_WRITE, typ, nr, size);
    }

    // IOWR outputs the result of _IOWR macro in include/uapi/asm-generic/ioctl.h.
    pub const fn IOWR(typ: u32, nr: u32, size: u32) -> u32 {
        return Self::IOC(IOCDirection::IOC_WRITE | IOCDirection::IOC_READ, typ, nr, size);
    }

    // IOC_NR outputs the result of IOC_NR macro in
    // include/uapi/asm-generic/ioctl.h.
    pub const fn IOC_NR(nr: u32) -> u32 {
        return (nr << Self::IOC_NRSHIFT) & ((1<< Self::IOC_NRBITS) - 1)
    }

    // IOC_SIZE outputs the result of IOC_SIZE macro in
    // include/uapi/asm-generic/ioctl.h.
    pub const fn IOC_SIZE(nr: u32) -> u32 {
        return (nr << Self::IOC_SIZESHIFT) & ((1<< Self::IOC_SIZEBITS) - 1)
    }

    // Kcov ioctls from include/uapi/linux/kcov.h.
    pub const KCOV_INIT_TRACE : u32 = Self::IOR('c' as u32, 1, 8);
	pub const KCOV_ENABLE     : u32 = Self::IO('c' as u32, 100);
	pub const KCOV_DISABLE    : u32 = Self::IO('c' as u32, 101);

    // Kcov trace types from include/uapi/linux/kcov.h.
    pub const KCOV_TRACE_PC  : i32 = 0;
	pub const KCOV_TRACE_CMP : i32 = 1;

    // Kcov state constants from include/uapi/linux/kcov.h.
    pub const KCOV_MODE_DISABLED  : i32 = 0;
	pub const KCOV_MODE_INIT      : i32 = 1;
	pub const KCOV_MODE_TRACE_PC  : i32 = 2;
	pub const KCOV_MODE_TRACE_CMP : i32 = 3;
}