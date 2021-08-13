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

use std::slice;
use std::fs::File;
use xmas_elf::program::ProgramHeader::{Ph64};
use xmas_elf::program::Type;
//use xmas_elf::program::{ProgramIter, SegmentData, Type};
//use xmas_elf::sections::SectionData;
use xmas_elf::*;
use memmap::Mmap;

pub use xmas_elf::program::{Flags, ProgramHeader, ProgramHeader64};
pub use xmas_elf::sections::Rela;
pub use xmas_elf::symbol_table::{Entry, Entry64};
pub use xmas_elf::{P32, P64};
pub use xmas_elf::header::HeaderPt2;

use super::addr::Addr;

//use xmas_elf::dynamic::Tag;
//use xmas_elf::header;
use super::qlib::common::Error;
use super::qlib::common::Result;

use super::memmgr::{MappedRegion, MapOption};

pub struct KernelELF {
    pub mmap: Option<Mmap>,
    pub startAddr: Addr,
    pub endAddr: Addr,
    pub entry: u64,
    pub mr: Option<MappedRegion>,

    pub vdsoStart: u64,
    pub vdsoLen: u64,
    pub vdsomr: Option<MappedRegion>,
}

impl KernelELF {
    pub fn Init(fileName: &String) -> Result<Self> {
        let f = File::open(fileName).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let mmap = unsafe { Mmap::map(&f).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))? };
        let elfFile = ElfFile::new(&mmap).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let mut startAddr: Addr = Addr(0xfffff_fffff_fffff);
        let mut endAddr: Addr = Addr(0);

        let entry = match &elfFile.header.pt2 {
            HeaderPt2::Header64(pt2) => pt2.entry_point,
            _ => return Err(Error::WrongELFFormat),
        };

        for p in elfFile.program_iter() {
            //todo : add more check
            if let Ph64(header) = p {
                if header.get_type().map_err(Error::ELFLoadError)? == Type::Load {
                    let startMem = Addr(header.virtual_addr).RoundDown()?;
                    //let endMem = Addr(header.virtual_addr).AddLen(header.file_size)?.RoundUp()?;
                    let endMem = Addr(header.virtual_addr).AddLen(header.mem_size)?.RoundUp()?;

                    if startMem.0 < startAddr.0 {
                        startAddr = startMem;
                    }

                    if endAddr.0 < endMem.0 {
                        endAddr = endMem;
                    }
                }
            }
        }

        return Ok(KernelELF {
            mmap: Some(mmap),
            startAddr,
            endAddr,
            entry,
            mr: None,
            vdsoStart: 0,
            vdsoLen: 0,
            vdsomr: None,
        })
    }

    pub fn StartAddr(&self) -> Addr {
        return self.startAddr;
    }

    pub fn EndAddr(&self) -> Addr {
        return self.endAddr;
    }

    pub fn LoadKernel(&mut self) -> Result<u64> {
        let mut option = &mut MapOption::New();
        option = option.Addr(self.startAddr.0).Len(self.endAddr.0 - self.startAddr.0).MapAnan().MapPrivate().ProtoRead().ProtoWrite().ProtoExec();

        let mr = option.Map()?;
        //let mr = MappedRegion::Init(self.startAddr, self.endAddr.0 - self.startAddr.0, false, libc::PROT_READ |  libc::PROT_WRITE |  libc::PROT_EXEC)?;
        let hostAddr = Addr(mr.ptr as u64);
        if hostAddr.0 != self.startAddr.0 {
            return Err(Error::AddressDoesMatch)
        }

        info!("loadKernel: get address is {:x}, len is {:x}, self.endAddr.0 - self.startAddr.0 is {:x}", mr.ptr as u64, mr.sz, self.endAddr.0 - self.startAddr.0);

        let mmap = self.mmap.take().unwrap();
        let elfFile = ElfFile::new(&mmap).map_err(Error::ELFLoadError)?;
        for p in elfFile.program_iter() {
            //todo : add more check
            if let Ph64(header) = p {
                if header.get_type().map_err(Error::ELFLoadError)? == Type::Load {
                    let startMem = Addr(header.virtual_addr).RoundDown()?;
                    let pageOffset = Addr(header.virtual_addr).0 - Addr(header.virtual_addr).RoundDown()?.0;

                    let target = unsafe { slice::from_raw_parts_mut((startMem.0 + pageOffset) as *mut u8, header.file_size as usize) };
                    let source = &mmap[header.offset as usize..(header.offset + header.file_size) as usize];

                    target.clone_from_slice(source);

                    //VMS.lock().KernelMap(startMem, endMem, startMem, PageOpts::Zero().SetPresent().SetWrite().Val())?;
                }
            }
        }

        self.mr = Some(mr);

        return Ok(self.entry)
    }

    pub fn LoadVDSO(&mut self, fileName: &String) -> Result<()> {
        let f = File::open(fileName).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let mmap = unsafe { Mmap::map(&f).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))? };
        let len = mmap.len();

        assert!(Addr(len as u64).RoundUp()?.0 == 2 * 4096);

        let mut option = &mut MapOption::New();
        option = option.Addr(self.EndAddr().0).Len(3 * 4096).MapAnan().MapPrivate().ProtoRead().ProtoWrite().ProtoExec();
        let mr = option.Map()?;
        //let mr = MappedRegion::Init(self.startAddr, self.endAddr.0 - self.startAddr.0, false, libc::PROT_READ |  libc::PROT_WRITE |  libc::PROT_EXEC)?;
        let hostAddr = mr.ptr as u64;
        if hostAddr != self.EndAddr().0 {
            return Err(Error::AddressDoesMatch)
        }

        let target = unsafe { slice::from_raw_parts_mut((hostAddr + 4096) as *mut u8, len) };
        let source = &mmap[..];
        target.clone_from_slice(source);

        self.vdsoStart = hostAddr;
        self.vdsoLen = 3 * 4096;
        self.vdsomr = Some(mr);

        return Ok(())
    }
}

