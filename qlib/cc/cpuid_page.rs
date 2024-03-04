use core::mem::size_of;
use super::super::common::*;
use alloc::string::ToString;


pub const SNP_CPUID_FUNCTION_MAXCOUNT: usize = 64;
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SnpCpuidInfo {
    pub count: u32,
    pub reserved1: u32,
    pub reserved2: u64,
    pub entries: [SnpCpuidFunc; SNP_CPUID_FUNCTION_MAXCOUNT],
}

impl Default for SnpCpuidInfo {
    fn default()-> Self {
        return SnpCpuidInfo {
            count: 0,
            reserved1: 0,
            reserved2: 0,
            entries: [SnpCpuidFunc::default();SNP_CPUID_FUNCTION_MAXCOUNT],
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CpuidPage {
    entry: SnpCpuidInfo,
    reserved: [u8; 4096-size_of::<SnpCpuidInfo>()],
}

impl Default for CpuidPage {
    fn default() -> Self {
        return CpuidPage {
            entry: SnpCpuidInfo::default(),
            reserved:[0; 4096-size_of::<SnpCpuidInfo>()],
        }
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
}
