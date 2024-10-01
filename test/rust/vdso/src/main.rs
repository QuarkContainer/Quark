// SPDX-License-Identifier: Apache-2.0

//! This module provides functions for reading symbols from the Linux vDSO.

#![deny(missing_docs)]
#![deny(clippy::all)]

use crt0stack::{Entry, Reader};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice::from_raw_parts;

#[cfg(target_pointer_width = "64")]
#[allow(unused_imports)]
mod elf {
    pub use goblin::elf64::dynamic::*;
    pub use goblin::elf64::header::*;
    pub use goblin::elf64::program_header::*;
    pub use goblin::elf64::section_header::*;
    pub use goblin::elf64::sym::Sym;

    pub const CLASS: u8 = ELFCLASS64;
    pub type Word = u64;
}

#[cfg(target_pointer_width = "32")]
#[allow(unused_imports)]
mod elf {
    pub use goblin::elf32::dynamic::*;
    pub use goblin::elf32::header::*;
    pub use goblin::elf32::program_header::*;
    pub use goblin::elf32::section_header::*;
    pub use goblin::elf32::sym::Sym;

    pub const CLASS: u8 = ELFCLASS32;
    pub type Word = u32;
}

#[repr(transparent)]
#[derive(Debug)]
struct Header(elf::Header);

impl Header {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub unsafe fn from_ptr(ptr: &()) -> Option<&Self> {
        let hdr = &*(ptr as *const _ as *const Self);

        if hdr.0.e_ident[..elf::ELFMAG.len()] != elf::ELFMAG[..] {
            return None;
        }

        if hdr.0.e_ident[elf::EI_CLASS] != elf::CLASS {
            return None;
        }

        Some(hdr)
    }

    unsafe fn ptr<T>(&self, off: impl Into<elf::Word>) -> *const T {
        let addr = self as *const _ as *const u8;
        addr.add(off.into() as usize) as *const T
    }

    unsafe fn slice<T>(&self, off: impl Into<elf::Word>, len: impl Into<elf::Word>) -> &[T] {
        from_raw_parts::<u8>(self.ptr(off), len.into() as usize)
            .align_to()
            .1
    }

    unsafe fn shtab(&self) -> &[elf::SectionHeader] {
        self.slice(self.0.e_shoff, self.0.e_shentsize * self.0.e_shnum)
    }

    unsafe fn section<T>(&self, kind: u32) -> Option<&[T]> {
        for sh in self.shtab() {
            if sh.sh_type == kind {
                return Some(self.slice(sh.sh_offset, sh.sh_size));
            }
        }

        None
    }

    unsafe fn symbol(&self, name: &str) -> Option<&Symbol> {
        println!("symbol 1");
        let symstrtab: &[c_char] = self.section(elf::SHT_STRTAB)?;
        println!("symbol 2");
        let symtab: &[elf::Sym] = self.section(elf::SHT_DYNSYM)?;

        // Yes, we could speed up the lookup by checking against the hash
        // table. But the reality is that there is less than a dozen symbols
        // in the vDSO, so the gains are trivial.

        println!("symbol 3");
        for sym in symtab {
            let cstr = CStr::from_ptr(&symstrtab[sym.st_name as usize]);
            let s = String::from_utf8_lossy(cstr.to_bytes()).to_string();
            println!("string buffer size without nul terminator: {}", s);
            if let Ok(s) = cstr.to_str() {
                let addr = self.ptr(sym.st_value);
                println!("symbol4 addr is {:x}", addr as *const _ as u64);
                if s == name {
                    return Some(&*addr);
                }
            }
        }

        None
    }
}

/// A resolved symbol
///
/// Since vDSO symbols have no type information, this type is opaque.
/// Generally, you will cast a `&Symbol` to the appropriate reference type.
pub enum Symbol {}

/// This structure represents the Linux vDSO
pub struct Vdso<'a>(&'a Header);

impl Vdso<'static> {
    /// Locates the vDSO by parsing the auxiliary vectors
    pub fn locate() -> Option<Self> {
        for aux in Reader::from_environ().done() {
            if let Entry::SysInfoEHdr(addr) = aux {
                println!("vdso adddr is {:x}", addr);
                let hdr = unsafe { Header::from_ptr(&*(addr as *const _))? };
                //println!("vdso adddr hdr is {:#x?}", hdr);
                return Some(Self(hdr));
            }
        }

        None
    }

    /// tewst
    pub fn new(addr: u64) -> Option<Self> {
        println!("vdso adddr is {:x}", addr);
        let hdr = unsafe { Header::from_ptr(&*(addr as *const _))? };
        //println!("vdso adddr hdr is {:#x?}", hdr);
        return Some(Self(hdr));
    }
}

impl<'a> Vdso<'a> {
    /// Find a vDSO symbol by its name
    ///
    /// The return type is essentially a void pointer. You will need to cast
    /// it for the type of the symbol you are looking up.
    pub fn lookup(&self, name: &str) -> Option<&'a Symbol> {
        unsafe { self.0.symbol(name) }
    }
}

fn get_file_as_byte_vec(filename: &str) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;

    let mut f = File::open(&filename).expect("no file found");
    let metadata = std::fs::metadata(&filename).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");

    buffer
}

fn main() {
    use libc::time_t;
    use std::mem::transmute;
    use std::ptr::null_mut;

    let v = get_file_as_byte_vec("/test/vdso.so");
    println!("data is {:x?}", &v[0..100]);

    let addr: u64 = 0xa000001000;

    let slice = unsafe { std::slice::from_raw_parts(addr as *const u8, v.len()) };
    println!("slice is {:x?}", &slice[0..100]);

    for i in 0..v.len() {
        assert!(v[i] == slice[i]);
    }

    println!("compare done ...");

    let vdso = Vdso::locate().unwrap();

    //let vdso = Vdso::new(&v[0] as *const _ as u64).unwrap();

    let func = vdso.lookup("time").unwrap();
    println!("Hello xxx, world! {:x}", func as *const _ as u64);

    let func: extern "C" fn(*mut time_t) -> time_t = unsafe { transmute(func) };

    let libc = unsafe { libc::time(null_mut()) };
    println!("Hello xxx1");
    let vdso = func(null_mut());
    println!("Hello xxx2");
    assert!(vdso - libc <= 1);
    println!("Hello xxx3");
}
