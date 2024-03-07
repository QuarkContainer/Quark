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

use std::collections::BTreeMap;
use spin::Mutex;
use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::qlib::common::*;
use crate::qlib::proxy::*;
use crate::qlib::kernel::util::cstring::CString;

use libelf::raw::*;

lazy_static! {
    pub static ref XPU_LIBRARY_HANDLERS:Mutex<BTreeMap<XpuLibrary, HandlerAddr>> = Mutex::new(BTreeMap::new());
    pub static ref KERNEL_INFOS:Mutex<BTreeMap<String, Arc<KernelInfo>>> = Mutex::new(BTreeMap::new());
    pub static ref MODULES:Mutex<BTreeMap<CUmoduleKey, CUmoduleAddr>> = Mutex::new(BTreeMap::new());
    pub static ref FUNCTIONS:Mutex<BTreeMap<CUhostFunction, CUfunctionAddr>> = Mutex::new(BTreeMap::new());
    pub static ref GLOBALS:Mutex<BTreeMap<u64, CUdeviceAddr>> = Mutex::new(BTreeMap::new());
}

pub type HandlerAddr = u64;
pub type CUmoduleKey = u64;
pub type CUmoduleAddr = u64;
pub type CUhostFunction = u64;
pub type CUfunctionAddr = u64;
pub type CUdeviceAddr = u64;

#[repr(C)]
#[derive(Default, Debug)]
pub struct KernelInfo {
    pub name: String,
    pub paramSize: usize,
    pub paramNum: usize,
    pub paramOffsets: Vec<u16>,
    pub paramSizes: Vec<u16>,
    pub hostFun: u64
}

pub fn GetFatbinInfo(addr:u64, fatElfHeader:&FatElfHeader) -> Result<i64> {
    error!("fatElfHeader magic is :{:x}, version is :{:x}, header size is :{:x}, size is :{:x}", fatElfHeader.magic, fatElfHeader.version, fatElfHeader.header_size, fatElfHeader.size);
    let mut inputPosition = addr + fatElfHeader.header_size as u64;
    let endPosition = inputPosition + fatElfHeader.size as u64;
    let mut textDatAddr:u64;
    let mut textDataSize:u64 = 0; 
    error!("inputPosition:{:x} endPosition:{:x}", inputPosition, endPosition);
    while inputPosition < endPosition {
        let fatTextHeader = unsafe { &*(inputPosition as *const u8 as *const FatTextHeader) };
        let decompressedByte: Vec<u8> = vec![0; (fatTextHeader.compressed_size + 7) as usize];

        error!("fatTextHeader:{:x?}", *fatTextHeader);
        error!("FATBIN_FLAG_COMPRESS:{:x}, fatTextHeader.flags:{:x}, result of &:{:x}", FATBIN_FLAG_COMPRESS, fatTextHeader.flags, fatTextHeader.flags & FATBIN_FLAG_COMPRESS);
        
        inputPosition += fatTextHeader.header_size as u64;
        if fatTextHeader.kind != 2 { // section does not cotain device code (but e.g. PTX)
            inputPosition += fatTextHeader.size;
            continue;
        }
        if fatTextHeader.flags & FATBIN_FLAG_DEBUG > 0 {
            error!("fatbin contains debug information.");
        }
        if (fatTextHeader.flags & FATBIN_FLAG_COMPRESS) > 0 {
            error!("fatbin contains compressed device code. Decompressing...");
            let input_read:i64;
            
            textDatAddr = &decompressedByte[0] as *const _ as u64;   //decompressedByte.as_ptr() as u64 ;
            error!("yiwang textDatAddr is {:x}", textDatAddr );

            match DecompressSingleSection(inputPosition, textDatAddr, &mut textDataSize, fatTextHeader) {
                Ok(inputRead) => {input_read = inputRead;  },
                Err(error) => {
                    return Err(error)
                },
            }

            error!("yiwang decompressedByte is: {:?}", decompressedByte);
            error!("yiwang input_read is: {:x}", input_read);

            if input_read < 0 {
                error!("Something went wrong while decompressing text section.");
                return Err(Error::DecompressFatbinError(String::from("Something went wrong while decompressing text section.")));
            } 
            inputPosition += input_read as u64;                           
        }else {
            textDatAddr = inputPosition;
            textDataSize = fatTextHeader.size;

            inputPosition += fatTextHeader.size;   
        }

        match GetParameterInfo(fatTextHeader, textDatAddr, textDataSize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

       
    }
    error!("Complete handling FatTextHeader");
    Ok(0)
}
                                                                                                                           //Result<(i64, Vec<u8>)>
fn DecompressSingleSection(inputPosition:u64, outputPosition:u64, outputSize:*mut u64,fatTextHeader:&FatTextHeader) -> Result<i64> {

    let mut padding:u64;
    let mut inputRead:u64 = 0;
    let mut outputWritten:u64 = 0;
    let decompressResult;
    let zeros: [u8; 8] = [0; 8];

    // let mut decompressedByte: Vec<u8> = vec![0; (fatTextHeader.compressed_size + 7) as usize];

    // // unsafe{ *outputPosition = &decompressedByte[0] as *const _ as u64 };
    // unsafe{ *outputPosition = decompressedByte.as_ptr() as u64 };

    error!("fatTextHeader: fatbin_kind:{:x}, header_size:{:x}, size:{:x}, compressed_size:{:x}, minor:{:x}, major:{:x}, arch:{:x}, decompressed_size:{:x}, flags:{:#x?}",
    fatTextHeader.kind,
    fatTextHeader.header_size, 
    fatTextHeader.size, 
    fatTextHeader.compressed_size, 
    fatTextHeader.minor, 
    fatTextHeader.major, 
    fatTextHeader.arch, 
    fatTextHeader.decompressed_size, 
    fatTextHeader.flags
    );
    error!("fatTextHeader unknown fields: unknown1: {:x}, unknown2: {:x}, zeros: {:x}", fatTextHeader.unknown1, fatTextHeader.unknown2, fatTextHeader.zero);

                                                                                                                          // &mut decompressedByte 
    decompressResult = decompress(inputPosition, fatTextHeader.compressed_size as u64, outputPosition, fatTextHeader.decompressed_size);
    if decompressResult != fatTextHeader.decompressed_size {
        error!("Decompression failed: decompressed size is {:x}, but header says {:x}.", decompressResult, fatTextHeader.decompressed_size);
        hexdump(inputPosition, 0x160);
        if decompressResult >= 0x60 {
                hexdump(outputPosition + decompressResult - 0x60, 0x60); 
        }
        return Err(Error::DecompressFatbinError(String::from("decompression failed")));
    }

    error!("decompressResult should equal to fatTextHeader.decompressed_size, decompressResult: {:x}, fatTextHeader.decompressed_size: {:x}", decompressResult, fatTextHeader.decompressed_size);
    inputRead += fatTextHeader.compressed_size as u64;
    outputWritten += fatTextHeader.decompressed_size;

    // error!("yiwang inputPosition is: {:x}, inputRead is: {:x}", inputPosition, inputRead);
    // error!("yiwang inputPosition + inputRead = {:x}", inputPosition + inputRead);
    // good 
    padding = 8u64.wrapping_sub(inputPosition +inputRead);
    // error!("yiwang padding after subtraction is: {:x}", padding);
    padding = padding % 8;
    // error!("yiwang padding after %8 is {}", padding );
       
    let slice1 = unsafe{std::slice::from_raw_parts((inputPosition + inputRead) as *const u8, padding as usize) };
    let slice2 = unsafe {std::slice::from_raw_parts(&zeros[0] as *const u8, padding as usize)};

    if slice1 != slice2 {
        error!("yiwang expected {:x} zero bytes", padding);
        hexdump(inputPosition + inputRead, 0x60);
        return Err(Error::DecompressFatbinError(String::from("padding length is wrong")));
    }
    error!("slice1 is: {:?}", slice1);
    error!("slice2 is: {:?}", slice2);

    inputRead += padding;
    
    padding = (8u64.wrapping_sub(fatTextHeader.decompressed_size)) % 8;
    error!("yiwang padding is {}", padding );
    
    error!("yiwang the address gonna be write over is: {:x}", outputPosition);

    unsafe{
        std::ptr::write_bytes((outputPosition) as *mut u8, 0, padding as usize)
    };
   
    outputWritten += padding;
    unsafe {
    *outputSize = outputWritten;
    }
    return Ok(inputRead as i64);
}

/* Decompressed a fatbin file 
* @param inputPosition: inputPosition for compressed input data 
* @param inputSize: Size of compressed data
* @param output: Preallocated memory where decompressed output should be stored
* @param outputSize: Size of output buffer. Should be equal to the size of the decompressed data
*/                                                            //&mut [u8]
fn decompress(inputPosition:u64, inputSize:u64, outputPosition: u64, outputSize: u64) -> u64 {
    let mut ipos:u64 = 0;
    let mut opos:u64 = 0;
    let mut nextNonCompressed_length:u64;
    let mut nextCompressed_length:u64;
    let mut backOffset:u64;

     // may be i8, in rust, char is i8
    while ipos < inputSize {      
        nextNonCompressed_length = unsafe { ((*((inputPosition + ipos) as *const u8) & 0xf0) >> 4) as u64};
        nextCompressed_length = unsafe{ 4 + (*((inputPosition + ipos) as *const u8) & 0xf) as u64  };
        // error!("nextNonCompressed_length is {}", nextNonCompressed_length);
        // error!("nextCompressed_length is {}", nextCompressed_length);
        
        if nextNonCompressed_length == 0xf{
            loop {
                ipos += 1;
                unsafe{
                nextNonCompressed_length += *((inputPosition + ipos) as *const u8) as u64
                };
                if unsafe{ *((inputPosition + ipos) as *const u8)} != 0xff  {
                    break; 
                }
            }
        }

        unsafe{
            std::ptr::copy_nonoverlapping((inputPosition + ipos + 1) as *const u8, 
            (outputPosition + opos) as *mut u8,
            nextNonCompressed_length as usize)
        };
        // let slice1 = unsafe{std::slice::from_raw_parts((outputPosition + opos) as *const u8, nextNonCompressed_length as usize) };
        // error!("yiwang decompress first copy: {:?}",slice1);

        ipos += 1 + nextNonCompressed_length;
        opos += nextNonCompressed_length;

        if ipos >= inputSize || opos >= outputSize {
            break;
        }
        // backOffset = unsafe { *((inputPosition as *mut u8).offset(ipos as isize)) as u64 + (*((inputPosition as *mut u8).offset(ipos as isize + 1)) << 8) as u64};
        // backOffset = unsafe{ (*((inputPosition + ipos) as *const u8)) as u64 + (*((inputPosition + ipos +1) as *const u8) as u64 ) << 8 };
       backOffset = unsafe{ (*((inputPosition + ipos) as *const u8)) as u64 };

       let mut inputvalue:u64 =unsafe{ *((inputPosition + ipos +1) as *const u8) as u64 };
       

       let shiftvalue = inputvalue << 8;
    //    error!("backOffset before shift is: {}", backOffset);
    //    error!("input[ipos+1] is {}",inputvalue );
    //    error!("input[ipos+1] << 8 is {}",shiftvalue);
       backOffset =  backOffset + shiftvalue;
    //    error!("backOffset after shift is {}",backOffset);
        ipos += 2;

        if nextCompressed_length == 0xf + 4{
            loop {
                nextCompressed_length += unsafe{ *((inputPosition + ipos) as *const u8) as u64 };
                ipos += 1;
                if unsafe {*((inputPosition + ipos - 1) as *const u8)} != 0xff {
                    break;
                }
            }        
        }

        if nextCompressed_length <= backOffset {
            unsafe{
                std::ptr::copy_nonoverlapping((outputPosition + opos - backOffset) as *const u8,
                 (outputPosition + opos) as *mut u8,
                 nextCompressed_length as usize)
            };

            // let slice2 = unsafe{std::slice::from_raw_parts((outputPosition + opos) as *const u8, nextCompressed_length as usize) };
            // error!("yiwang decompress if copy: {:?}",slice2);
        }else{
            unsafe{
                std::ptr::copy_nonoverlapping((outputPosition + opos -backOffset) as *const u8 ,(outputPosition + opos) as *mut u8 , backOffset as usize)
            };
            // let slice3 = unsafe{std::slice::from_raw_parts((outputPosition + opos) as *const u8, backOffset as usize) };
            // error!("yiwang decompress else copy: {:?}",slice3);


            for i in backOffset..nextCompressed_length {
               unsafe {*((outputPosition + opos + i) as *mut u8) =  *((outputPosition + opos + i - backOffset) as *mut u8);
            //    error!("yiwang decompress for loop: {}",*((outputPosition + opos + i) as *mut u8));
               };
            }
        }

        opos += nextCompressed_length;
        // error!("next_nclen is {}",nextNonCompressed_length);
        // error!("next_clen is {}",nextCompressed_length);
        // error!("back_offset is {}",backOffset);

    }
    error!("yiwang ipos: {:x}, opos: {:x}, input size: {:x}, output size: {:x}", ipos, opos, inputSize, outputSize);
    return opos ; 
}

fn hexdump(dataAddress: u64, size: u64){
    let mut pos: u64 = 0;
    while pos < size {
        error!("yiwang hexdump debug {:#05x}", pos);
        for i in 0..16 {
            if pos + i < size {
                unsafe {
                error!("yiwang gexdump debug {:02x}", *((dataAddress + i) as *const u8))
                };
            } else {
                error!(" ");
            }
            if i % 4 == 3 {
                error!(" ");
            }
        }
        for i in 0..16 {
            unsafe{
                if pos + i < size {
                
                if  *((dataAddress + i) as *const u8) >= 0x20 &&  *((dataAddress + i) as *const u8) <= 0x7e {
                    error!("{}", *((dataAddress + i) as *const char));    
                } else {
                    error!(".");
                }
            } else {
                error!("");
            }
        };
        }
        pos += 16;
    }
}

fn GetParameterInfo(fatTextHeader:&FatTextHeader, inputPosition:u64, memSize:u64) -> Result<i64> {
    let mut section:MaybeUninit<Elf_Scn>  = MaybeUninit::uninit();
    let mut ptr_section = section.as_mut_ptr();
    let mut shdr:MaybeUninit<GElf_Shdr> = MaybeUninit::uninit();
    let mut symtab_shdr = shdr.as_mut_ptr();
    let mut sym:MaybeUninit<GElf_Sym> = MaybeUninit::uninit();
    let mut ptr_sym = sym.as_mut_ptr();

    // let memsize = fatTextHeader.size as usize;
    let elf = unsafe { elf_memory(inputPosition as *mut i8, memSize as usize) };
    match CheckElf(elf) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    match GetSectionByName(elf, String::from(".symtab"), &mut ptr_section) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    symtab_shdr = unsafe { gelf_getshdr(ptr_section, symtab_shdr) };
    let shdr = unsafe { *symtab_shdr };

    if shdr.sh_type != SHT_SYMTAB {
        return Err(Error::ELFLoadError("not a symbol table"));
    }

    let symbol_table_data_p = unsafe { elf_getdata(ptr_section, 0 as _) };
    let symbol_table_data = unsafe { &*symbol_table_data_p };
    error!("symbol_table_data: {:?}", symbol_table_data);
    let symbol_table_size = shdr.sh_size / shdr.sh_entsize;

    match GetSectionByName(elf, String::from(".nv.info"), &mut ptr_section) {
        Ok(v) => v,
        Err(_e) => {
            return Err(Error::ELFLoadError("could not find .nv.info section. This means this binary does not contain any kernels."));
        },
    };

    let data_p = unsafe { elf_getdata(ptr_section, 0 as _) };
    let data = unsafe { &*data_p };
    error!("data: {:?}", data);

    let mut secpos:usize = 0;
    let infoSize = std::mem::size_of::<NvInfoEntry>();
    while secpos < data.d_size {
        let position = data.d_buf as u64 + secpos as u64;
        let entry_p = position as *const u8 as *const NvInfoEntry;
        let entry = unsafe { &*entry_p };
        error!("entry: {:x?}", entry);
        if entry.values_size != 8 {
            error!("unexpected values_size: {:x}", entry.values_size);
            return Err(Error::ELFLoadError("unexpected values_size")); 
        }

        if entry.attribute as u64 != EIATTR_FRAME_SIZE {
            secpos += infoSize;
            continue;
        }

        if entry.kernel_id as u64 >= symbol_table_size {
            error!("kernel_id {:x} out of bounds: {:x}", entry.kernel_id, symbol_table_size);
            secpos += infoSize;
            continue;
        }

        ptr_sym = unsafe { gelf_getsym(symbol_table_data_p, entry.kernel_id as libc::c_int, ptr_sym) };
        
        let kernel_str = unsafe { CString::FromAddr(elf_strptr(elf, (*symtab_shdr).sh_link as usize, (*ptr_sym).st_name as usize) as u64).Str().unwrap().to_string() };
        error!("kernel_str: {}", kernel_str);

        if KERNEL_INFOS.lock().contains_key(&kernel_str) {
            secpos += infoSize;
            continue;
        }
        
        error!("found new kernel: {} (symbol table id: {:x})", kernel_str, entry.kernel_id);

        let mut ki = KernelInfo::default();
        ki.name = kernel_str.clone();
        
        if kernel_str.chars().next().unwrap() != '$' {
            match GetParamForKernel(elf, &mut ki){
                Ok(_) => {},
                Err(e) => {
                    return Err(e); 
                },
            }
        }
        error!("ki: {:x?}", ki);

        KERNEL_INFOS.lock().insert(kernel_str.clone(), Arc::new(ki));

        secpos += infoSize;
    }

    Ok(0)
}

pub fn GetParamForKernel(elf: *mut Elf, kernel: *mut KernelInfo) -> Result<i64> {
    let sectionName = GetKernelSectionFromKernelName(unsafe { (*kernel).name.clone() });

    let mut section = &mut(0 as u64 as *mut Elf_Scn);
    match GetSectionByName(elf, sectionName.clone(), &mut section) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    error!("GetSectionByName({}) got section: {:?}", sectionName, section);
    let data = unsafe { &*(elf_getdata(*section, 0 as _)) };
    error!("data: {:x?}", data);

    let mut secpos:usize = 0;
    while secpos < data.d_size {
        let position = data.d_buf as u64 + secpos as u64;
        let entry = unsafe { &*(position as *const u8 as *const NvInfoKernelEntry) };
        error!("entry: {:x?}", entry);
        if entry.format as u64 == EIFMT_SVAL && entry.attribute as u64 == EIATTR_KPARAM_INFO {
            if entry.values_size != 0xc {
                return Err(Error::ELFLoadError("EIATTR_KPARAM_INFO values size has not the expected value of 0xc"));
            }
            let kparam = unsafe { &*(&entry.values as *const _ as *const u8 as *const NvInfoKParamInfo) };
            error!("kparam: {:x?}", *kparam);

            unsafe {
                if kparam.ordinal as usize >= (*kernel).paramNum {
                    (*kernel).paramNum = kparam.ordinal as usize + 1;
                    while (*kernel).paramOffsets.len() < (*kernel).paramNum {
                        (*kernel).paramOffsets.push(0);
                        (*kernel).paramSizes.push(0);
                        error!("in while kernel: {:x?}", *kernel);
                    }
                    error!("end while kernel: {:x?}", *kernel);                    
                }
                (*kernel).paramOffsets[kparam.ordinal as usize] = kparam.offset;
                (*kernel).paramSizes[kparam.ordinal as usize] = kparam.GetSize();
                error!("changed value kernel: {:x?}, kparam: {:x?}", *kernel, *kparam);
            }

            secpos += std::mem::size_of::<NvInfoKernelEntry>() - 4 + entry.values_size as usize;
        } else if entry.format as u64 == EIFMT_HVAL && entry.attribute as u64 == EIATTR_CBANK_PARAM_SIZE {
            unsafe { 
                (*kernel).paramSize = entry.values_size as usize; 
            }
            error!("cbank_param_size: {:x}", entry.values_size);
            secpos += std::mem::size_of::<NvInfoKernelEntry>() - 4;
        } else if entry.format as u64 == EIFMT_HVAL {
            secpos += std::mem::size_of::<NvInfoKernelEntry>() - 4;
        } else if entry.format as u64 == EIFMT_SVAL {
            secpos += std::mem::size_of::<NvInfoKernelEntry>() + entry.values_size as usize - 4;
        } else if entry.format as u64 == EIFMT_NVAL {
            secpos += std::mem::size_of::<NvInfoKernelEntry>() - 4;
        } else {
            error!("unknown format: {:x}", entry.format);
            secpos += std::mem::size_of::<NvInfoKernelEntry>() - 4;
        }            
    }

    Ok(0)
}

fn GetKernelSectionFromKernelName(kernelName:String) -> String {
    if kernelName.chars().next().unwrap() == '$' {
        todo!();
    }

    format!(".nv.info.{}", kernelName)
}
                                                          // a mutable reference to a mutable raw pointer
pub fn GetSectionByName(elf: *mut Elf, name: String,  section: &mut *mut Elf_Scn) -> Result<i64> {
    let mut scn = 0 as u64 as *mut Elf_Scn;
    let mut size:usize = 0;
    let str_section_index = &mut size as *mut _ as u64 as *mut usize;
    let ret = unsafe { elf_getshdrstrndx(elf, str_section_index) };
    if ret !=0 {
        return Err(Error::ELFLoadError("elf_getshstrndx failed"));
    }

    let mut found = false;
    loop {
        let scnNew = unsafe { elf_nextscn(elf, scn) };
        if scnNew == std::ptr::null_mut() {
            break;
        }
        
        let mut shdr : MaybeUninit<GElf_Shdr> = MaybeUninit::uninit();
        let mut symtab_shdr = shdr.as_mut_ptr();
        symtab_shdr = unsafe { gelf_getshdr(scnNew, symtab_shdr) };
        let section_name = CString::FromAddr(unsafe { elf_strptr(elf, *str_section_index, (*symtab_shdr).sh_name as usize) as u64 }).Str().unwrap().to_string();
        error!("section_name {}", section_name);
        if name.eq(&section_name) {
            error!("Found section {}", section_name);
            *section = scnNew;
            found = true;
            break;
        }
        scn = scnNew;
    }

    if found {
        Ok(0)
    } else {
        Err(Error::ELFLoadError("Named section not found!"))
    }    
}

pub fn CheckElf(elf: *mut Elf) -> Result<i64> {    
    let ek = unsafe { elf_kind(elf) };
    if ek != libelf::raw::Elf_Kind::ELF_K_ELF {
        error!("elf_kind is not ELF_K_ELF, but {}", ek);
        return Err(Error::ELFLoadError("elf_kind is not ELF_K_ELF"));            
    }

    let mut ehdr : MaybeUninit<GElf_Ehdr> = MaybeUninit::uninit();
    let ptr_ehdr = ehdr.as_mut_ptr();
    unsafe { gelf_getehdr(elf, ptr_ehdr); }

    let elfclass = unsafe { gelf_getclass(elf) };
    if elfclass == libelf::raw::ELFCLASSNONE as i32 {
        return Err(Error::ELFLoadError("gelf_getclass failed"));            
    }

    let nbytes = 0 as *mut usize;
    let id = unsafe { elf_getident(elf, nbytes) };
    let idStr = CString::FromAddr(id as u64).Str().unwrap().to_string();
    error!("id: {:?}, nbytes: {:?}, idStr: {}", id, nbytes, idStr);

    let mut size:usize = 0;
    let sections_num = &mut size as *mut _ as u64 as *mut usize;
    let ret = unsafe { elf_getshdrnum(elf, sections_num) };
    error!("yiwang result of elf_getshdrnum: {:?}",ret);
    if ret != 0 {
        return Err(Error::ELFLoadError("elf_getshdrnum failed"));
    }
    
    let mut size:usize = 0;
    let program_header_num = &mut size as *mut _ as u64 as *mut usize;
    let ret = unsafe { elf_getphdrnum(elf, program_header_num) };
    if ret != 0 {
        return Err(Error::ELFLoadError("elf_getphdrnum failed"));    
    }
    
    let mut size:usize = 0;
    let section_str_num = &mut size as *mut _ as u64 as *mut usize;
    let ret = unsafe { elf_getshdrstrndx(elf, section_str_num) };
    if ret != 0 {
        return Err(Error::ELFLoadError("elf_getshdrstrndx failed"));    
    }
    return Ok(ret as i64);
}