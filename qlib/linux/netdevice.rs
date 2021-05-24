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

pub const IFNAMSIZ : usize = 16;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct IFReq {
    // IFName is an encoded name, normally null-terminated. This should be
    // accessed via the Name and SetName functions.
    pub IFName: [u8; IFNAMSIZ],

    // Data is the union of the following structures:
    //
    //	struct sockaddr ifr_addr;
    //	struct sockaddr ifr_dstaddr;
    //	struct sockaddr ifr_broadaddr;
    //	struct sockaddr ifr_netmask;
    //	struct sockaddr ifr_hwaddr;
    //	short           ifr_flags;
    //	int             ifr_ifindex;
    //	int             ifr_metric;
    //	int             ifr_mtu;
    //	struct ifmap    ifr_map;
    //	char            ifr_slave[IFNAMSIZ];
    //	char            ifr_newname[IFNAMSIZ];
    //	char           *ifr_data;
    pub Data: [u8; 24],
}

impl IFReq {
    pub fn Name(&self) -> String {
        let len = self.IFName.len();
        let mut idx = len;
        for i in 0..len {
            if self.IFName[i] == 0{
                idx = i;
            }
        }

        return String::from_utf8(self.IFName[0..idx].to_vec()).expect("IFReq Name() fail");
    }

    pub fn SetName(&mut self, name: &str) {
        assert!(name.len() <= IFNAMSIZ, "IFReq setname is too large");
        for i in 0..name.len() {
            self.IFName[i] = name.as_bytes()[i];
        }

        for i in name.len()..IFNAMSIZ {
            self.IFName[i] = 0;
        }
    }
}

// SizeOfIFReq is the binary size of an IFReq struct (40 bytes).
pub const SIZE_OF_IFREQ : usize = core::mem::size_of::<IFReq>();

// IFMap contains interface hardware parameters.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct IFMap {
    pub MemStart : u64,
    pub MemEnd   : u64,
    pub BaseAddr : u16,
    pub IRQ      : u8,
    pub DMA      : u8,
    pub Port     : u8,
    pub _pad     : [u8; 3], // Pad to sizeof(struct ifmap).
}

// IFConf is used to return a list of interfaces and their addresses. See
// netdevice(7) and struct ifconf for more detail on its use.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct IFConf {
    pub Len: i32,
    pub _pad: [u8; 4],
    pub Ptr: u64,
}