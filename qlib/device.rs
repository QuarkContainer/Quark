// Copyright (c) 2021 QuarkSoft LLC
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
use alloc::collections::btree_map::BTreeMap;
use core::cmp::Ordering;
use alloc::sync::Arc;
use spin::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref SIMPLE_DEVICES: Mutex<Registry> = Mutex::new(Registry::New());
    pub static ref HOSTFILE_DEVICE: Mutex<MultiDevice> = Mutex::new(NewAnonMultiDevice());
    pub static ref PSEUDO_DEVICE: Arc<Mutex<Device>> = NewAnonDevice();
    pub static ref DEV_DEVICE: Arc<Mutex<Device>> = NewAnonDevice();
    pub static ref PTS_DEVICE: Arc<Mutex<Device>> = NewAnonDevice();
    pub static ref PROC_DEVICE: Arc<Mutex<Device>> = NewAnonDevice();
    pub static ref SHM_DEVICE: Arc<Mutex<Device>> = NewAnonDevice();
    pub static ref SYS_DEVICE: Arc<Mutex<Device>> = NewAnonDevice();
    pub static ref TMPFS_DEVICE: Arc<Mutex<Device>> = NewAnonDevice();
}

// TTYAUX_MAJOR is the major device number for alternate TTY devices.
pub const TTYAUX_MAJOR: u16 = 5;

// UNIX98_PTY_MASTER_MAJOR is the initial major device number for
// Unix98 PTY masters.
pub const UNIX98_PTY_MASTER_MAJOR: u16 = 128;

// UNIX98_PTY_SLAVE_MAJOR is the initial major device number for
// Unix98 PTY slaves.
pub const UNIX98_PTY_SLAVE_MAJOR: u16 = 136;

// PTMX_MINOR is the minor device number for /dev/ptmx.
pub const PTMX_MINOR: u32 = 2;

pub struct Device {
    pub id: ID,
    pub last: u64,
}

impl Device {
    pub fn NextIno(&mut self) -> u64 {
        self.last += 1;
        return self.last;
    }

    pub fn DeviceID(&self) -> u64 {
        return self.id.DeviceID()
    }
}

pub struct Registry {
    pub last: u32,
    pub devices: BTreeMap<ID, Arc<Mutex<Device>>>
}

impl Registry {
    pub fn New() -> Self {
        return Self {
            last: 0,
            devices: BTreeMap::new(),
        }
    }

    fn newAnonID(&mut self) -> ID {
        self.last += 1;
        return ID {
            Major: 0,
            Minor: self.last,
        }
    }

    pub fn NewAnonDevice(&mut self) -> Arc<Mutex<Device>> {
        let d = Arc::new(Mutex::new(Device {
            id: self.newAnonID(),
            last: 0,
        }));

        self.devices.insert(d.lock().id, d.clone());
        return d;
    }
}

#[derive(Debug, Copy, Clone, Eq)]
pub struct ID {
    pub Major: u16,
    pub Minor: u32,
}

impl ID {
    pub fn New(rdev: u32) -> Self {
        return ID {
            Major: ((rdev >> 8) & 0xfff) as u16,
            Minor: (rdev & 0xff) | ((rdev >> 20) << 8),
        }
    }

    // Bits 7:0   - minor bits 7:0
    // Bits 19:8  - major bits 11:0
    // Bits 31:20 - minor bits 19:8
    pub fn MakeDeviceID(&self) -> u32 {
        return (self.Minor & 0xff) | ((self.Major as u32 & 0xfff) << 8) | ((self.Minor >> 8) << 20);
    }

    pub fn DeviceID(&self) -> u64 {
        return self.MakeDeviceID() as u64
    }
}

impl Ord for ID {
    fn cmp(&self, other: &Self) -> Ordering {
        let MajorCmp = self.Major.cmp(&other.Major);
        if MajorCmp != Ordering::Equal {
            return MajorCmp;
        }

        let MinorCmp = self.Minor.cmp(&other.Minor);
        if MinorCmp != Ordering::Equal {
            return MinorCmp;
        }

        return Ordering::Equal
    }
}

impl PartialOrd for ID {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ID {
    fn eq(&self, other: &Self) -> bool {
        self.Major == other.Major && self.Minor == other.Minor
    }
}

#[derive(Debug, Clone, Eq)]
pub struct MultiDeviceKey {
    pub Device: u64,
    pub SecondaryDevice: String,
    pub Inode: u64,
}

impl Ord for MultiDeviceKey {
    fn cmp(&self, other: &Self) -> Ordering {
        let DeviceCmp = self.Device.cmp(&other.Device);
        if DeviceCmp != Ordering::Equal {
            return DeviceCmp;
        }

        let SecondaryDevicCmp = self.SecondaryDevice.cmp(&other.SecondaryDevice);
        if SecondaryDevicCmp != Ordering::Equal {
            return SecondaryDevicCmp;
        }

        let InodeCmp = self.Inode.cmp(&other.Inode);
        if InodeCmp != Ordering::Equal {
            return InodeCmp;
        }

        return Ordering::Equal
    }
}

impl PartialOrd for MultiDeviceKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for MultiDeviceKey {
    fn eq(&self, other: &Self) -> bool {
        self.Device == other.Device &&
            self.SecondaryDevice == other.SecondaryDevice &&
            self.Inode == other.Inode
    }
}

pub struct MultiDevice {
    pub id: ID,
    pub last: u64,
    pub cache: BTreeMap<MultiDeviceKey, u64>,
    pub rcache: BTreeMap<u64, MultiDeviceKey>,
}

impl MultiDevice {
    pub fn New(id: ID) -> Self {
        return Self {
            id: id,
            last: 0,
            cache: BTreeMap::new(),
            rcache: BTreeMap::new(),
        }
    }

    pub fn Map(&mut self, key: MultiDeviceKey) -> u64 {
        match self.cache.get(&key) {
            Some(id) => return *id,
            None => ()
        }

        let mut idx = self.last + 1;
        loop {
            match self.rcache.get(&idx) {
                Some(_) => idx += 1,
                None => break,
            }
        }

        self.last = idx;
        self.cache.insert(key.clone(), idx);
        self.rcache.insert(idx, key);

        return idx;
    }

    pub fn DeviceID(&self) -> u64 {
        return self.id.DeviceID()
    }
}

pub fn NewAnonDevice() -> Arc<Mutex<Device>> {
    return SIMPLE_DEVICES.lock().NewAnonDevice()
}

pub fn NewAnonMultiDevice() -> MultiDevice {
    return MultiDevice::New(SIMPLE_DEVICES.lock().newAnonID())
}

pub fn MakeDeviceID(major: u16, minor: u32) -> u32 {
    return (minor & 0xff) | (((major as u32 & 0xfff) << 8) | ((minor >> 8) << 20))
}

pub fn DecodeDeviceId(rdev: u32) -> (u16, u32) {
    let major = ((rdev >> 8) & 0xfff) as u16;
    let minor = (rdev & 0xff) | ((rdev >> 20) << 8);
    return (major, minor)
}