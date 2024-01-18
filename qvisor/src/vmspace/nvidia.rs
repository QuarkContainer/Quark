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
use core::ops::Deref;
use std::ffi::CString;
use std::os::raw::*;
use std::mem::MaybeUninit;

use crate::qlib::common::*;
use crate::qlib::linux_def::SysErr;
use crate::qlib::proxy::*;
use crate::qlib::range::Range;
use crate::qlib::kernel::util::cstring::CString as QString;
use std::sync::Arc;

use cuda_driver_sys::*;
use libelf::raw::*;

lazy_static! {
    pub static ref NVIDIA_HANDLERS: NvidiaHandlers = NvidiaHandlers::New();
    pub static ref FUNC_MAP: BTreeMap<ProxyCommand,(XpuLibrary, &'static str)> = BTreeMap::from(
        [
            (ProxyCommand::CudaSetDevice, (XpuLibrary::CudaRuntime, "cudaSetDevice")),
            (ProxyCommand::CudaDeviceSynchronize, (XpuLibrary::CudaRuntime, "cudaDeviceSynchronize")),
            (ProxyCommand::CudaMalloc, (XpuLibrary::CudaRuntime, "cudaMalloc")),
            (ProxyCommand::CudaMemcpy, (XpuLibrary::CudaRuntime, "cudaMemcpy")),
            (ProxyCommand::CudaRegisterFatBinary, (XpuLibrary::CudaDriver, "cuModuleLoadData")),
            (ProxyCommand::CudaRegisterFunction, (XpuLibrary::CudaDriver, "cuModuleGetFunction")),
            (ProxyCommand::CudaLaunchKernel, (XpuLibrary::CudaDriver, "cuLaunchKernel")),
        ]
    );
    pub static ref XPU_LIBRARY_HANDLERS:Mutex<BTreeMap<XpuLibrary, u64>> = Mutex::new(BTreeMap::new());
    pub static ref KERNEL_INFOS:Mutex<BTreeMap<String, Arc<KernelInfo>>> = Mutex::new(BTreeMap::new());
    pub static ref MODULES:Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());
    pub static ref FUNCTIONS:Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());
}

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

pub fn  NvidiaProxy(cmd: ProxyCommand, parameters: &ProxyParameters) -> Result<i64> {
    error!("NvidiaProxy 0 cmd {:?}", cmd);
    let handler = NVIDIA_HANDLERS.GetFuncHandler(cmd)?;
    error!("NvidiaProxy 0 cmd {:?} After getting handler", cmd);
    match cmd {
        ProxyCommand::None => {
            panic!("get impossible proxy command");
        }
        ProxyCommand::CudaSetDevice => {
            error!("hochan CudaSetDevice 1");
            let func: extern "C" fn(libc::c_int) -> i32 = unsafe {
                std::mem::transmute(handler)
            }; 

            error!("hochan CudaSetDevice 2");
            let ret = func(parameters.para1 as i32);
            error!("hochan CudaSetDevice 3");
            return Ok(ret as i64);
        }
        ProxyCommand::CudaDeviceSynchronize => {
            let func: extern "C" fn() -> i32 = unsafe {
                std::mem::transmute(handler)
            }; 

            let ret = func();
            return Ok(ret as i64);
        }
        ProxyCommand::CudaMalloc => {
            let func: extern "C" fn(*mut *mut ::std::os::raw::c_void, usize) -> i32 = unsafe {
                std::mem::transmute(handler)
            }; 

            error!("hochan CudaMalloc before parameters:{:x?}", parameters);
            let mut para1 = parameters.para1 as *mut ::std::os::raw::c_void;
            let addr = &mut para1;

            error!("hochan before cuda_runtime_sys::cudaMalloc addr {:x}", *addr as u64);
            let ret = func(addr, parameters.para2 as usize);
            error!("hochan cuda_runtime_sys::cudaMalloc ret {:x?} addr {:x}", ret, *addr as u64);

            let mut paramInfo = parameters.para3 as *const u8 as *mut ParamInfo;
            unsafe { (*paramInfo).addr = *addr as u64; }

            error!("hochan CudaMalloc after parameters:{:x?} ret {:x?}", parameters, ret);
            return Ok(ret as i64);
        }
        ProxyCommand::CudaMemcpy => {
            return CudaMemcpy(handler, parameters);
        }
        ProxyCommand::CudaRegisterFatBinary => {
            let fatElfHeader = unsafe { &*(parameters.para2 as *const u8 as *const FatElfHeader) };
            let moduleKey = parameters.para3;
            error!("hochan moduleKey:{:x}", moduleKey);

            let mut inputPosition = parameters.para2 + fatElfHeader.header_size as u64;
            let endPosition = inputPosition + fatElfHeader.size as u64;
            error!("hochan inputPosition:{:x} endPosition:{:x}", inputPosition, endPosition);
            while inputPosition < endPosition {
                let fatTextHeader = unsafe { &*(inputPosition as *const u8 as *const FatTextHeader) };
                error!("hochan fatTextHeader:{:x?}", *fatTextHeader);
                error!("hochan FATBIN_FLAG_COMPRESS:{:x}, fatTextHeader.flags:{:x}, result of &:{:x}", FATBIN_FLAG_COMPRESS, fatTextHeader.flags, fatTextHeader.flags & FATBIN_FLAG_COMPRESS);
                
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
                    todo!()
                }
                
                let mut section:MaybeUninit<Elf_Scn>  = MaybeUninit::uninit();
                let mut ptr_section = section.as_mut_ptr();
                let mut shdr:MaybeUninit<GElf_Shdr> = MaybeUninit::uninit();
                let mut symtab_shdr = shdr.as_mut_ptr();
                let mut sym:MaybeUninit<GElf_Sym> = MaybeUninit::uninit();
                let mut ptr_sym = sym.as_mut_ptr();

                let memsize = fatTextHeader.size as usize;
                let elf = unsafe { elf_memory(inputPosition as *mut i8, memsize) };
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
                error!("hochan symbol_table_data: {:?}", symbol_table_data);
                let symbol_table_size = shdr.sh_size / shdr.sh_entsize;

                match GetSectionByName(elf, String::from(".nv.info"), &mut ptr_section) {
                    Ok(v) => v,
                    Err(_e) => {
                        error!("could not find .nv.info section. This means this binary does not contain any kernels.");
                        break;
                    },
                };

                let data_p = unsafe { elf_getdata(ptr_section, 0 as _) };
                let data = unsafe { &*data_p };
                error!("hochan data: {:?}", data);

                let mut secpos:usize = 0;
                let infoSize = std::mem::size_of::<NvInfoEntry>();
                while secpos < data.d_size {
                    let position = data.d_buf as u64 + secpos as u64;
                    let entry_p = position as *const u8 as *const NvInfoEntry;
                    let entry = unsafe { *entry_p };
                    error!("hochan entry: {:x?}", entry);
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

                    ptr_sym = unsafe { gelf_getsym(symbol_table_data_p, entry.kernel_id as c_int, ptr_sym) };
                    
                    let kernel_str = unsafe { QString::FromAddr(elf_strptr(elf, (*symtab_shdr).sh_link as usize, (*ptr_sym).st_name as usize) as u64).Str().unwrap().to_string() };
                    error!("hochan kernel_str: {}", kernel_str);

                    if KERNEL_INFOS.lock().contains_key(&kernel_str) {
                        continue;
                    }

                    error!("found new kernel: {} (symbol table id: {:x})", kernel_str, entry.kernel_id);

                    let mut ki = KernelInfo::default();
                    ki.name = kernel_str.clone();
                    
                    if kernel_str.chars().next().unwrap() != '$' {
                        match GetParmForKernel(elf, &mut ki){
                            Ok(_) => {},
                            Err(e) => {
                                return Err(e); 
                            },
                        }
                    }
                    error!("hochan ki: {:x?}", ki);

                    KERNEL_INFOS.lock().insert(kernel_str.clone(), Arc::new(ki));
                    
                    secpos += infoSize;
                }

                inputPosition += fatTextHeader.size;
            }
            error!("hochan Complete handling FatTextHeader");
            
            let mut module:u64 = 0;
            let ret = unsafe{ cuda_driver_sys::cuModuleLoadData(&mut module as *mut _ as u64 as *mut CUmodule, parameters.para2 as *const c_void)};
            MODULES.lock().insert(moduleKey, module);            
            error!("hochan called func ret {:?} module ptr {:x?} MODULES {:x?}", ret,  module, MODULES.lock());            
            return Ok(ret as i64);
        }
        ProxyCommand::CudaRegisterFunction => {
                let info = unsafe { &*(parameters.para1 as *const u8 as *const RegisterFunctionInfo) };
                
                let bytes = unsafe { std::slice::from_raw_parts(info.deviceName as *const u8, parameters.para2 as usize) };
                let deviceName = std::str::from_utf8(bytes).unwrap();       
                let mut module = *MODULES.lock().get(&info.fatCubinHandle).unwrap();
                error!("hochan deviceName {}, parameters {:x?} module {:x}", deviceName, parameters, module);

                let mut hfunc:u64 = 0;
                let ret = unsafe { cuda_driver_sys::cuModuleGetFunction(
                    &mut hfunc as *mut _ as u64 as *mut CUfunction, 
                    *(&mut module as *mut _ as u64 as *mut CUmodule), 
                    CString::new(deviceName).unwrap().clone().as_ptr()) };
                FUNCTIONS.lock().insert(info.hostFun, hfunc);
                error!("hochan cuModuleGetFunction ret {:x?}, hfunc {:x}, &hfunc {:x}, FUNCTIONS  {:x?}", ret, hfunc, &hfunc, FUNCTIONS.lock());

                let kernelInfo = KERNEL_INFOS.lock().get(&deviceName.to_string()).unwrap().clone();
                let paramInfo = parameters.para3 as *const u8 as *mut ParamInfo;
                unsafe {
                    (*paramInfo).paramNum = kernelInfo.paramNum;
                    for i in 0..(*paramInfo).paramNum {
                        (*paramInfo).paramSizes[i] = kernelInfo.paramSizes[i];
                    }
                    error!("hochan paramInfo in nvidia {:x?}", (*paramInfo));
                }
                return Ok(ret as i64);
        }
        ProxyCommand::CudaLaunchKernel => {
            error!("hochan CudaLaunchKernel in host parameters {:x?}", parameters);
            let info = unsafe { &*(parameters.para1 as *const u8 as *const LaunchKernelInfo) };
            let func = FUNCTIONS.lock().get(&info.func).unwrap().clone();
            error!("hochan CudaLaunchKernel in host info {:x?}, func {:x}", info, func);
                        
            let ret = unsafe { cuda_driver_sys::cuLaunchKernel(
                func as CUfunction,
                info.gridDim.x, info.gridDim.y, info.gridDim.z,
                info.blockDim.x, info.blockDim.y, info.blockDim.z,
                info.sharedMem as u32,
                info.stream as *mut CUstream_st,
                info.args as *mut *mut ::std::os::raw::c_void,
                0 as *mut *mut ::std::os::raw::c_void) };
            error!("hochan cuLaunchKernel ret {:x?}", ret);

            return Ok(0 as i64);
        }
        _ => todo!()
    }
}

fn GetParmForKernel(elf: *mut Elf, kernel: *mut KernelInfo) -> Result<i64> {
    let sectionName = GetKernelSectionFromKernelName(unsafe { (*kernel).name.clone() });

    let mut section = &mut(0 as u64 as *mut Elf_Scn);
    match GetSectionByName(elf, sectionName.clone(), &mut section) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    error!("hochan GetSectionByName({}) got section: {:?}", sectionName, section);
    let data = unsafe { &*(elf_getdata(*section, 0 as _)) };
    error!("hochan data: {:x?}", data);

    let mut secpos:usize = 0;
    while secpos < data.d_size {
        let position = data.d_buf as u64 + secpos as u64;
        let entry = unsafe { &*(position as *const u8 as *const NvInfoKernelEntry) };
        error!("hochan entry: {:x?}", entry);
        if entry.format as u64 == EIFMT_SVAL && entry.attribute as u64 == EIATTR_KPARAM_INFO {
            if entry.values_size != 0xc {
                return Err(Error::ELFLoadError("EIATTR_KPARAM_INFO values size has not the expected value of 0xc"));
            }
            let kparam = unsafe { &*(&entry.values as *const _ as *const u8 as *const NvInfoKParamInfo) };
            error!("hochan kparam: {:x?}", *kparam);

            unsafe {
                if kparam.ordinal as usize >= (*kernel).paramNum {
                    (*kernel).paramNum = kparam.ordinal as usize + 1;
                    while (*kernel).paramOffsets.len() < (*kernel).paramNum {
                        (*kernel).paramOffsets.push(0);
                        (*kernel).paramSizes.push(0);
                        error!("hochan in while kernel: {:x?}", *kernel);
                    }
                    error!("hochan end while kernel: {:x?}", *kernel);                    
                }
                (*kernel).paramOffsets[kparam.ordinal as usize] = kparam.offset;
                (*kernel).paramSizes[kparam.ordinal as usize] = kparam.GetSize();
                error!("hochan changed value kernel: {:x?}, kparam: {:x?}", *kernel, *kparam);
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

fn GetSectionByName(elf: *mut Elf, name: String,  section: &mut *mut Elf_Scn) -> Result<i64> {
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
        let section_name = QString::FromAddr(unsafe { elf_strptr(elf, *str_section_index, (*symtab_shdr).sh_name as usize) as u64 }).Str().unwrap().to_string();
        error!("hochan section_name {}", section_name);
        if name.eq(&section_name) {
            error!("hochan Found section {}", section_name);
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

fn CheckElf(elf: *mut Elf) -> Result<i64> {
    unsafe {
        let ek = elf_kind(elf);
        if ek != libelf::raw::Elf_Kind::ELF_K_ELF {
            error!("elf_kind is not ELF_K_ELF, but {}", ek);
            return Err(Error::ELFLoadError("elf_kind is not ELF_K_ELF"));            
        }

        let mut ehdr : MaybeUninit<GElf_Ehdr> = MaybeUninit::uninit();
        let ptr_ehdr = ehdr.as_mut_ptr();
        gelf_getehdr(elf, ptr_ehdr);
        error!("hochan ehdr {:?}", *ptr_ehdr);

        let elfclass = gelf_getclass(elf);
        if elfclass == libelf::raw::ELFCLASSNONE as i32 {
            return Err(Error::ELFLoadError("gelf_getclass failed"));            
        }

        let nbytes = 0 as *mut usize;
        let id = elf_getident(elf, nbytes);
        let idStr = QString::FromAddr(id as u64).Str().unwrap().to_string();
        error!("hochan id: {:?}, nbytes: {:?}, idStr: {}", id, nbytes, idStr);

        let mut size:usize = 0;
        let sections_num = &mut size as *mut _ as u64 as *mut usize;
        let ret = elf_getshdrnum(elf, sections_num);
        if ret != 0 {
            return Err(Error::ELFLoadError("elf_getshdrnum failed"));
        }
        error!("hochan sections_num: {}", *sections_num);
        
        let mut size:usize = 0;
        let program_header_num = &mut size as *mut _ as u64 as *mut usize;
        let ret = elf_getphdrnum(elf, program_header_num);
        if ret != 0 {
            return Err(Error::ELFLoadError("elf_getphdrnum failed"));    
        }
        error!("hochan program_header_num: {}", *program_header_num);
        
        let mut size:usize = 0;
        let section_str_num = &mut size as *mut _ as u64 as *mut usize;
        let ret = elf_getshdrstrndx(elf, section_str_num);
        if ret != 0 {
            return Err(Error::ELFLoadError("elf_getshdrstrndx failed"));    
        }
        error!("hochan section_str_num: {}", *section_str_num);
        
        error!("elf contains {} sections, {} program_headers, string table section: {}", *sections_num, *program_header_num, *section_str_num);
    }

    return Ok(0);
}

pub fn CudaMemcpy(handle: u64, parameters: &ProxyParameters) -> Result<i64> {
    let func: extern "C" fn(u64, u64, u64, u64) -> u32 = unsafe {
        std::mem::transmute(handle)
    };

    let kind = parameters.para5;
    
    match kind {
        CUDA_MEMCPY_HOST_TO_HOST => todo!(),
        CUDA_MEMCPY_HOST_TO_DEVICE => {
            let dst = parameters.para1;
            let ptr = parameters.para2 as *const Range;
            let len = parameters.para3 as usize;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = func(dst + offset, r.start, r.len, kind);
                if ret != 0 {
                    error!(
                        "CUDA_MEMCPY_HOST_TO_DEVICE ret is {:x}/{:x}/{:x}/{} {}", 
                        dst+offset, r.start, r.len, kind, ret);
                    return Ok(ret as i64);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0)   
        }
        CUDA_MEMCPY_DEVICE_TO_HOST => {
            let ptr = parameters.para1 as *const Range;
            let len = parameters.para2 as usize;
            let src = parameters.para3;
            let count = parameters.para4;

            let ranges = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut offset = 0;
            for r in ranges {
                let ret = func(r.start, src + offset, r.len, kind);
                if ret != 0 {
                    return Ok(ret as i64);
                }
                offset += r.len;
            }

            assert!(offset == count);

            return Ok(0)   
        }
        CUDA_MEMCPY_DEVICE_TO_DEVICE => {
            let dst = parameters.para1;
            let src = parameters.para3;
            let count = parameters.para4;
            let ret = func(dst, src, count, kind);
            return Ok(ret as i64);
        }
        _ => todo!()
    }
}

pub struct NvidiaHandlersInner {
    pub cudaHandler: u64,
    pub cudaRuntimeHandler: u64, //*mut libc::c_void,
    pub handlers: BTreeMap<ProxyCommand, u64>,
}

impl NvidiaHandlersInner {
    pub fn GetFuncHandler(&mut self, cmd: ProxyCommand) -> Result<u64> {
        match self.handlers.get(&cmd) {
            None => {
                let func = self.DLSym(cmd)?;
                self.handlers.insert(cmd, func);
                return Ok(func);
            }
            Some(func) => {
                return Ok(*func);
            }
        }
    }

    pub fn DLSym(&self, cmd: ProxyCommand) -> Result<u64> {
        match FUNC_MAP.get(&cmd) {
            Some(&pair) => {
                let func_name = CString::new(pair.1).unwrap();
                let handler = XPU_LIBRARY_HANDLERS.lock().get(&pair.0).unwrap().clone();
                let handler: u64 = unsafe {
                    std::mem::transmute(libc::dlsym(handler as *mut libc::c_void, func_name.as_ptr()))
                };
                
                if handler != 0 {
                    error!("hochan got handler {:x}", handler);
                    return Ok(handler as u64)
                }
            }
            None => (),
        }

        return Err(Error::SysError(SysErr::ENOTSUP))
    }
}

pub struct NvidiaHandlers(Mutex<NvidiaHandlersInner>);

impl Deref for NvidiaHandlers {
    type Target = Mutex<NvidiaHandlersInner>;

    fn deref(&self) -> &Mutex<NvidiaHandlersInner> {
        &self.0
    }
}

impl NvidiaHandlers {
    pub fn New() -> Self {
        // This code piece is necessary. Otherwise cuModuleLoadData will return CUDA_ERROR_JIT_COMPILER_NOT_FOUND
        let lib = CString::new("libnvidia-ptxjitcompiler.so").unwrap();
        let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };  
        error!("hochan libnvidia-ptxjitcompiler.so handle {:?}", handle);

        let initResult = unsafe { cuda_driver_sys::cuInit(0) };
        error!("hochan initResult {:?}", initResult);

        let mut ctx : MaybeUninit<CUcontext> = MaybeUninit::uninit();
        let ptr_ctx = ctx.as_mut_ptr();
        let ret = unsafe { cuda_driver_sys::cuCtxCreate_v2(ptr_ctx,0,0) };
        error!("hochan cuCtxCreate ret {:?}", ret);

        let ret = unsafe { cuCtxPushCurrent_v2(*ptr_ctx) };
        error!("hochan cuCtxPushCurrent ret {:?}", ret);

        let cuda = format!("/usr/lib/x86_64-linux-gnu/libcuda.so");
        let cudalib = CString::new(&*cuda).unwrap();
        let cudaHandler = unsafe {
            libc::dlopen(
                cudalib.as_ptr(), 
            libc::RTLD_LAZY
            )
        } as u64;
        assert!(cudaHandler != 0, "can't open libcuda.so");
        XPU_LIBRARY_HANDLERS.lock().insert(XpuLibrary::CudaDriver, cudaHandler);
        
        let handlers = BTreeMap::new();
        
        let cudart = format!("/usr/lib/x86_64-linux-gnu/libcudart.so");
        let cudartlib = CString::new(&*cudart).unwrap();
        let cudaRuntimeHandler = unsafe {
            libc::dlopen(
            cudartlib.as_ptr(), 
            libc::RTLD_LAZY
            )
        } as u64;

        assert!(cudaRuntimeHandler != 0, "/usr/lib/x86_64-linux-gnu/libcudart.so");
        XPU_LIBRARY_HANDLERS.lock().insert(XpuLibrary::CudaRuntime, cudaRuntimeHandler);

        let inner = NvidiaHandlersInner {
            cudaHandler: cudaHandler,
            cudaRuntimeHandler: cudaRuntimeHandler,
            handlers: handlers,
        };

        return Self(Mutex::new(inner));  
    }

    // trigger the NvidiaHandlers initialization
    pub fn Trigger(&self) {}

    pub fn GetFuncHandler(&self, cmd: ProxyCommand) -> Result<u64> {
        let mut inner = self.lock();
        return inner.GetFuncHandler(cmd);
    } 
}