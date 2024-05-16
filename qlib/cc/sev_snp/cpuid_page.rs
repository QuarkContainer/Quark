use super::super::super::common::*;
use alloc::string::ToString;
use core::arch::x86_64::CpuidResult;
use core::mem::size_of;

pub const SNP_CPUID_FUNCTION_MAXCOUNT: usize = 64;
#[repr(packed)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct SnpCpuidFunc {
    pub eax_in: u32,
    pub ecx_in: u32,
    pub xcr0_in: u64,
    pub xss_in: u64,
    pub eax: u32,
    pub ebx: u32,
    pub ecx: u32,
    pub edx: u32,
    pub reserved: u64,
}

#[repr(packed)]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SnpCpuidInfo {
    pub count: u32,
    pub reserved1: u32,
    pub reserved2: u64,
    pub entries: [SnpCpuidFunc; SNP_CPUID_FUNCTION_MAXCOUNT],
}

impl Default for SnpCpuidInfo {
    fn default() -> Self {
        return SnpCpuidInfo {
            count: 0,
            reserved1: 0,
            reserved2: 0,
            entries: [SnpCpuidFunc::default(); SNP_CPUID_FUNCTION_MAXCOUNT],
        };
    }
}

#[repr(C)]
#[repr(packed)]
#[derive(Copy, Clone, Debug)]
pub struct CpuidPage {
    entry: SnpCpuidInfo,
    reserved: [u8; 4096 - size_of::<SnpCpuidInfo>()],
}

impl Default for CpuidPage {
    fn default() -> Self {
        return CpuidPage {
            entry: SnpCpuidInfo::default(),
            reserved: [0; 4096 - size_of::<SnpCpuidInfo>()],
        };
    }
}

impl CpuidPage {
    pub fn AddEntry(&mut self, entry: &SnpCpuidFunc) -> Result<()> {
        if self.entry.count as usize >= SNP_CPUID_FUNCTION_MAXCOUNT {
            return Err(Error::Common("Cpuid Page is full!".to_string()));
        }
        self.entry.entries[self.entry.count as usize] = *entry;
        self.entry.count += 1;
        Ok(())
    }

    /// Get all entries
    #[inline]
    pub fn get_functions(&self) -> &[SnpCpuidFunc] {
        &self.entry.entries[..self.entry.count as usize]
    }
    //&CpuidPage::from(MemoryDef::CPUID_PAGE);
    pub fn check_cpuid(&self, leaf: u32, sub_leaf: u32) -> CpuidResult {
        let res = self
            .get_functions()
            .iter()
            .find_map(|e| {
                if e.eax_in == 0xd {
                    if e.eax_in == leaf && e.ecx_in == sub_leaf {
                        Some(CpuidResult {
                            eax: e.eax,
                            ebx: e.ebx,
                            ecx: e.ecx,
                            edx: e.edx,
                        })
                    } else {
                        None
                    }
                } else {
                    if e.eax_in == leaf {
                        Some(CpuidResult {
                            eax: e.eax,
                            ebx: e.ebx,
                            ecx: e.ecx,
                            edx: e.edx,
                        })
                    } else {
                        None
                    }
                }
            })
            .unwrap_or(CpuidResult {
                eax: 0,
                ebx: 0,
                ecx: 0,
                edx: 0,
            });
        return res;
    }

    pub fn dump_cpuid(&self) {
        for i in 0..self.entry.count {
            let entry = self.entry.entries[i as usize];
            info!("Cpuid entry:{:#x?}", entry);
        }
    }
}

impl From<u64> for CpuidPage {
    fn from(addr: u64) -> Self {
        unsafe {
            return *(addr as *mut CpuidPage);
        }
    }
}

impl CpuidPage {
    pub fn get_ref(addr: u64) -> &'static mut Self {
        unsafe {
            return &mut *(addr as *mut CpuidPage);
        }
    }
}
