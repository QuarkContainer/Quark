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

use crate::qlib::mutex::*;
use alloc::slice;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
pub use xmas_elf::header::HeaderPt2;
use xmas_elf::program::ProgramHeader::Ph64;
use xmas_elf::program::ProgramHeader64;
use xmas_elf::program::Type;
use xmas_elf::sections::SectionData::SymbolTable64;
#[cfg(target_arch = "aarch64")]
use xmas_elf::symbol_table::Entry;
use xmas_elf::symbol_table::Entry64;
use xmas_elf::*;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::singleton::*;

pub static VDSO: Singleton<Vdso> = Singleton::<Vdso>::New();

pub unsafe fn InitSingleton() {
    VDSO.Init(Vdso::default());
}

#[derive(Default)]
pub struct VdsoInternal {
    pub vdsoParamAddr: u64,
    pub vdsoAddr: u64,
    pub vdsoLen: usize,
    pub phdrs: Vec<ProgramHeader64>,
    pub vdso_symbols: Option<VdsoArchSymbols>,
}

struct VdsoSymbol {
    name: &'static str,
    page_offset: Option<u64>,
}

/// Bookkeep the exported symbols for x86_64.
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub struct VdsoArchSymbols {}

#[cfg(target_arch = "x86_64")]
impl VdsoArchSymbols {
    pub fn set_symbols_page_offset(&mut self, _elf_file: &ElfFile, _symbol_table: &[Entry64]) {
        todo!();
    }

    pub fn get_symbol_page_offset(&self, _name: &str) -> Option<u64> {
        todo!();
    }
}

/// Bookkeep exported symbols for aarch64.
#[cfg(target_arch = "aarch64")]
pub struct VdsoArchSymbols {
    symbols: [VdsoSymbol; Self::TOTAL_SYMBOLS],
}

#[cfg(target_arch = "aarch64")]
impl Default for VdsoArchSymbols {
    fn default() -> Self {
        let _symbols: [VdsoSymbol; Self::TOTAL_SYMBOLS] = [
            VdsoSymbol {
                name: "__kernel_gettimeofday",
                page_offset: None,
            },
            VdsoSymbol {
                name: "__kernel_clock_getres",
                page_offset: None,
            },
            VdsoSymbol {
                name: "__kernel_rt_sigreturn",
                page_offset: None,
            },
            VdsoSymbol {
                name: "__kernel_clock_gettime",
                page_offset: None,
            },
        ];
        Self { symbols: _symbols }
    }
}

#[cfg(target_arch = "aarch64")]
impl VdsoArchSymbols {
    const TOTAL_SYMBOLS: usize = 4;

    pub fn set_symbols_page_offset(&mut self, elf_file: &ElfFile, symbol_table: &[Entry64]) {
        let mut found_symbols = 0;
        for sym in 0..Self::TOTAL_SYMBOLS {
            for entry in 0..symbol_table.len() {
                if self.symbols[sym].name
                    == symbol_table[entry].get_name(&elf_file).unwrap_or("None")
                {
                    self.symbols[sym].page_offset = Some(symbol_table[entry].value() & 0xFFFF);
                    found_symbols += 1;
                    continue;
                }
            }
        }
        if found_symbols < Self::TOTAL_SYMBOLS {
            panic!("aarch64-VDSO - failed to find symbols.");
        }
        for sym in 0..Self::TOTAL_SYMBOLS {
            info!(
                "aarch64-VDSO - name:{}, offset:{:#x}",
                self.symbols[sym].name,
                self.symbols[sym].page_offset.unwrap()
            );
        }
    }

    pub fn get_symbol_page_offset(&self, name: &str) -> Option<u64> {
        for entry in 0..Self::TOTAL_SYMBOLS {
            if name == self.symbols[entry].name {
                return self.symbols[entry].page_offset;
            }
        }
        None
    }
}

impl VdsoInternal {
    pub fn Initialization(&mut self, vdsoParamAddr: u64) -> Result<()> {
        self.vdsoParamAddr = vdsoParamAddr;
        self.vdsoAddr = vdsoParamAddr + MemoryDef::PAGE_SIZE;

        //todo: align with vdso.so's len
        self.vdsoLen = 2 * MemoryDef::PAGE_SIZE as usize;

        return self.LoadVDSO();
    }

    pub fn LoadVDSO(&mut self) -> Result<()> {
        let slice = unsafe { slice::from_raw_parts(self.vdsoAddr as *const u8, self.vdsoLen) };
        let elfFile =
            ElfFile::new(&slice).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        match &elfFile.header.pt2 {
            HeaderPt2::Header64(pt2) => match pt2.type_.as_type() {
                xmas_elf::header::Type::SharedObject => (),
                xmas_elf::header::Type::Executable => {
                    info!("VDSO::LoadVDSO: invalid vdso format, not sharedobject");
                    return Err(Error::WrongELFFormat);
                }
                _ => return Err(Error::WrongELFFormat),
            },
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

        #[cfg(target_arch = "aarch64")]
        self.load_symbols(&elfFile);

        return Ok(());
    }

    /// Load symbol table section
    fn load_symbols(&mut self, elf_file: &ElfFile) {
        if let Some(symtab_section) = elf_file.find_section_by_name(".symtab") {
            if let Ok(SymbolTable64(symbols)) = symtab_section.get_data(&elf_file) {
                let mut _vdso_symbols: VdsoArchSymbols = Default::default();
                _vdso_symbols.set_symbols_page_offset(&elf_file, symbols);
                self.vdso_symbols = Some(_vdso_symbols);
            } else if cfg!(target_arch = "aarch64") {
                panic!("aarch64-VDSO .symtab has not valid data.");
            }
        } else if cfg!(target_arch = "aarch64") {
            panic!("aarch64-VDSO .symtab section not present.");
        }
    }

    pub fn get_symbol_page_offset(&self, name: &str) -> Option<u64> {
        if self.vdso_symbols.is_some() {
            return self
                .vdso_symbols
                .as_ref()
                .unwrap()
                .get_symbol_page_offset(&name);
        }
        None
    }
}

#[derive(Clone, Default)]
pub struct Vdso(Arc<QRwLock<VdsoInternal>>);

impl Deref for Vdso {
    type Target = Arc<QRwLock<VdsoInternal>>;

    fn deref(&self) -> &Arc<QRwLock<VdsoInternal>> {
        &self.0
    }
}

impl Vdso {
    pub fn Initialization(&self, vdsoParamPageAddr: u64) {
        self.write()
            .Initialization(vdsoParamPageAddr)
            .expect("VDSO init fail");
    }

    pub fn VDSOAddr(&self) -> u64 {
        return self.read().vdsoAddr;
    }

    pub fn get_symbol_page_offset(&self, name: &str) -> Option<u64> {
        self.read().get_symbol_page_offset(&name)
    }
}
