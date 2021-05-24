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

use alloc::vec::Vec;
use alloc::sync::Arc;
use alloc::slice;
use core::ops::Deref;
use spin::RwLock;
use xmas_elf::program::Type;
use xmas_elf::program::ProgramHeader::{Ph64};
pub use xmas_elf::header::HeaderPt2;
use xmas_elf::program::ProgramHeader64;
use xmas_elf::*;
use lazy_static::lazy_static;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;

lazy_static! {
    pub static ref VDSO: Vdso = Vdso::default();
}

#[derive(Default)]
pub struct VdsoInternal {
    pub vdsoParamAddr: u64,
    pub vdsoAddr: u64,
    pub vdsoLen: usize,
    pub phdrs: Vec<ProgramHeader64>,
}

impl VdsoInternal {
    pub fn Init(&mut self, vdsoParamAddr: u64) -> Result<()> {
        self.vdsoParamAddr = vdsoParamAddr;
        self.vdsoAddr = vdsoParamAddr + MemoryDef::PAGE_SIZE;

        //todo: align with vdso.so's len
        self.vdsoLen = 2 * MemoryDef::PAGE_SIZE as usize;

        return self.LoadVDSO();
    }

    pub fn LoadVDSO(&mut self) -> Result<()> {
        let slice = unsafe { slice::from_raw_parts(self.vdsoAddr as *const u8, self.vdsoLen) };
        let elfFile = ElfFile::new(&slice).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        match &elfFile.header.pt2 {
            HeaderPt2::Header64(pt2) => {
                match pt2.type_.as_type() {
                    xmas_elf::header::Type::SharedObject => (),
                    xmas_elf::header::Type::Executable => {
                        info!("VDSO::LoadVDSO: invalid vdso format, not sharedobject");
                        return Err(Error::WrongELFFormat);
                    },
                    _ => return Err(Error::WrongELFFormat),
                }
            }
            _ => return Err(Error::WrongELFFormat),
        };

        for p in elfFile.program_iter() {
            if let Ph64(header) = p {
                let headerType = header.get_type().map_err(Error::ELFLoadError)?;
                if headerType == Type::Interp || headerType == Type::Load {
                    self.phdrs.push(*header);
                }
            }
        }

        return Ok(())
    }
}

#[derive(Clone, Default)]
pub struct Vdso(Arc<RwLock<VdsoInternal>>);

impl Deref for Vdso {
    type Target = Arc<RwLock<VdsoInternal>>;

    fn deref(&self) -> &Arc<RwLock<VdsoInternal>> {
        &self.0
    }
}

impl Vdso {
    pub fn Init(&self, vdsoParamPageAddr: u64) {
        self.write().Init(vdsoParamPageAddr).expect("VDSO init fail");
    }

    pub fn VDSOAddr(&self) -> u64 {
        return self.read().vdsoAddr;
    }
}