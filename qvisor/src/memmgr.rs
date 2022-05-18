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

#![allow(non_snake_case)]
#![allow(dead_code)]

use std::collections::HashMap;

use super::addr::Addr;
use super::qlib::common::Error;
use super::qlib::common::Result;

trait AddrRange {
    fn Start() -> u64;
    fn End() -> u64;
}

trait PhyRegionTrait {
    fn HostStartAddr(&self) -> Addr;
    fn PhyStartAddr(&self) -> Addr;
    fn Len(&self) -> u64;
}

/*pub struct Range {
    start : Addr,
    end : Addr,
}*/

pub struct FileInfo {
    offset: Addr,
    hostFileName: String,
}

pub struct PhyRegion {
    fileInfo: Option<FileInfo>,

    hostBaseAddr: Addr,
    mr: Box<MappedRegion>,
}

impl PhyRegion {
    pub fn InitAnan(
        hostBaseAddr: Addr,
        hostAddrLimit: Addr,
        len: u64,
        hugePage: bool,
    ) -> Result<PhyRegion> {
        let mut option = &mut MapOption::New();
        option = option
            .Addr(hostBaseAddr.0)
            .Len(len)
            .MapAnan()
            .MapPrivate()
            .ProtoRead()
            .ProtoWrite()
            .ProtoExec();
        if hugePage {
            option = option.MapHugeTLB();
        }

        let mr = Box::new(option.Map()?);
        //let mr = Box::new(MappedRegion::Init(hostBaseAddr, len, hugePage, libc::PROT_READ |  libc::PROT_WRITE |  libc::PROT_EXEC)?);
        if mr.End()?.0 >= hostAddrLimit.0 {
            return Err(Error::AddressNotInRange);
        }

        return Ok(PhyRegion {
            fileInfo: None,
            hostBaseAddr: hostBaseAddr,
            mr: mr,
        });
    }

    fn HostStartAddr(&self) -> Addr {
        return self.mr.Start();
    }

    fn PhyStartAddr(&self) -> Addr {
        return self.mr.Start().Offset(self.hostBaseAddr).unwrap();
    }

    fn Len(&self) -> u64 {
        return self.mr.Len();
    }

    fn IsAnan(&self) -> bool {
        match self.fileInfo {
            None => true,
            _ => false,
        }
    }
}

pub struct PhyAddrMgr {
    hostBaseAddr: Addr,
    hostAddrLimit: Addr,
    regions: HashMap<u64, Box<PhyRegion>>,
}

impl PhyAddrMgr {
    pub fn Init(hostBaseAddr: Addr, len: u64) -> Result<Self> {
        return Ok(PhyAddrMgr {
            hostBaseAddr: hostBaseAddr,
            hostAddrLimit: hostBaseAddr.AddLen(len)?,
            regions: HashMap::new(),
        });
    }

    pub fn PhyToHostAddr(&mut self, phyStartAddr: Addr) -> Result<Addr> {
        if let Some(region) = self.regions.get(&phyStartAddr.0) {
            return Ok(region.HostStartAddr());
        } else {
            return Err(Error::UnmatchRegion);
        }
    }

    //start is the physical address
    pub fn Free(&mut self, start: u64, len: u64) -> Result<()> {
        if let Some(region) = self.regions.get(&start) {
            if region.Len() != len {
                return Err(Error::UnmatchRegion);
            }
        } else {
            return Err(Error::UnmatchRegion);
        }

        self.regions.remove(&start);
        Ok(())
    }
}

#[derive(Debug)]
pub struct MapOption {
    Addr: u64,
    len: u64,
    flags: libc::c_int,
    proto: libc::c_int,
    fd: libc::c_int,
    fileOffset: libc::off_t,
}

impl MapOption {
    pub fn New() -> Self {
        return MapOption {
            Addr: 0,
            len: 0,
            flags: 0,
            proto: libc::PROT_NONE,
            fd: -1,
            fileOffset: 0,
        };
    }

    pub fn FileId(&mut self, id: i32) -> &mut Self {
        self.fd = id as libc::c_int;
        self
    }

    pub fn Addr(&mut self, addr: u64) -> &mut Self {
        self.Addr = addr;
        self
    }

    pub fn Len(&mut self, len: u64) -> &mut Self {
        self.len = len;
        self
    }

    pub fn FileOffset(&mut self, offset: u64) -> &mut Self {
        self.fileOffset = offset as libc::off_t;
        self
    }

    pub fn Proto(&mut self, proto: i32) -> &mut Self {
        self.proto = proto as libc::c_int;
        self
    }

    pub fn ProtoExec(&mut self) -> &mut Self {
        self.proto |= libc::PROT_EXEC;
        self
    }

    pub fn ProtoRead(&mut self) -> &mut Self {
        self.proto |= libc::PROT_READ;
        self
    }

    pub fn ProtoWrite(&mut self) -> &mut Self {
        self.proto |= libc::PROT_WRITE;
        self
    }

    pub fn MapShare(&mut self) -> &mut Self {
        self.flags |= libc::MAP_SHARED;
        self
    }

    pub fn MapShareValidate(&mut self) -> &mut Self {
        self.flags |= libc::MAP_SHARED_VALIDATE;
        self
    }

    pub fn MapPrivate(&mut self) -> &mut Self {
        self.flags |= libc::MAP_PRIVATE;
        self
    }

    pub fn MapAnan(&mut self) -> &mut Self {
        self.flags |= libc::MAP_ANONYMOUS;
        self
    }

    pub fn MapFixed(&mut self) -> &mut Self {
        self.flags |= libc::MAP_FIXED;
        self
    }

    pub fn MapFixedNoReplace(&mut self) -> &mut Self {
        self.flags |= libc::MAP_FIXED_NOREPLACE;
        self
    }

    pub fn MapHugeTLB(&mut self) -> &mut Self {
        self.flags |= libc::MAP_HUGETLB;
        self
    }

    pub fn MapPrecommit(&mut self) -> &mut Self {
        self.flags |= libc::MAP_POPULATE;
        self
    }

    pub fn MapLocked(&mut self) -> &mut Self {
        self.flags |= libc::MAP_LOCKED;
        self
    }

    pub fn MapNonBlock(&mut self) -> &mut Self {
        self.flags |= libc::MAP_NONBLOCK;
        self
    }

    pub fn Map(&self) -> Result<MappedRegion> {
        //info!("the MapOption prot is {:b} flag is {:b} ", self.proto, self.flags);

        MappedRegion::New(
            self.Addr as *mut libc::c_void,
            self.len as libc::size_t,
            self.proto as libc::c_int,
            self.flags as libc::c_int,
            self.fd as libc::c_int,
            self.fileOffset as libc::off_t,
        )
    }

    pub fn MMap(&self) -> Result<u64> {
        let addr = self.Addr as *mut libc::c_void;
        let len = self.len as libc::size_t;
        let prot = self.proto as libc::c_int;
        let flags = self.flags as libc::c_int;
        let fd = self.fd as libc::c_int;
        let offset = self.fileOffset as libc::off_t;

        //error!("mmap addr is {:x}, len is {:x}", self.Addr, self.len);
        unsafe {
            let ret = libc::mmap(addr, len, prot, flags, fd, offset);

            if (ret as i64) < 0 {
                Err(Error::SysError(errno::errno().0))
            } else {
                Ok(ret as u64)
            }
        }
    }

    pub fn MSync(addr: u64, len: u64) -> Result<()> {
        unsafe {
            if libc::msync(addr as *mut libc::c_void, len as usize, libc::MS_SYNC) != 0 {
                let errno = errno::errno().0;
                return Err(Error::SysError(errno));
            }
        }

        return Ok(());
    }

    pub fn MUnmap(addr: u64, len: u64) -> Result<()> {
        unsafe {
            if libc::munmap(addr as *mut libc::c_void, len as usize) != 0 {
                let errno = errno::errno().0;
                return Err(Error::SysError(errno));
            }
        }

        return Ok(());
    }
}

#[derive(Debug, Clone, Default)]
pub struct MappedRegion {
    pub sz: u64,
    pub ptr: u64,
}

impl MappedRegion {
    pub fn New(
        addr: *mut libc::c_void,
        len: libc::size_t,
        prot: libc::c_int,
        flags: libc::c_int,
        fd: libc::c_int,
        offset: libc::off_t,
    ) -> Result<Self> {
        //info!("addr is {:x}, len is {:x}, prot is {:x}, flags = {:b}, fd = {}, offset = {}", addr as u64, len, prot, flags, fd, offset);

        unsafe {
            let ret = libc::mmap(addr, len, prot, flags, fd, offset);

            if (ret as i64) < 0 {
                Err(Error::SysError(errno::errno().0))
            } else {
                Ok(MappedRegion {
                    ptr: ret as u64,
                    sz: len as u64,
                })
            }
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        return self.ptr as *mut u8;
    }
    pub fn Start(&self) -> Addr {
        return Addr(self.ptr as u64);
    }

    pub fn End(&self) -> Result<Addr> {
        return self.Start().AddLen(self.sz);
    }

    pub fn Len(&self) -> u64 {
        return self.sz;
    }
}

impl Drop for MappedRegion {
    fn drop(&mut self) {
        unsafe {
            info!("unmap ptr is {:x}, len is {:x}", self.ptr as u64, self.sz);

            if libc::munmap(self.ptr as *mut libc::c_void, self.sz as usize) != 0 {
                panic!("munmap: {}", std::io::Error::last_os_error());
            }
        }
    }
}
