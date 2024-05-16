use core::sync::atomic::{AtomicU64, Ordering};

/// The C-Bit mask indicating encrypted physical addresses
pub static C_BIT_MASK: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
pub fn get_cbit() -> u64 {
    let ebx;
    unsafe {
        let ret = core::arch::x86_64::__cpuid(0x8000001f);
        ebx = ret.ebx;
    }
    let c_bit = (ebx & 0x3f) as u64;
    return c_bit;
}

#[inline(always)]
pub fn set_cbit_mask() {
    C_BIT_MASK.store(1 << get_cbit(), Ordering::Release);
}

///Test, if is AMD cpu
pub fn check_amd() -> bool {
    let ebx;
    let edx;
    let ecx;
    unsafe {
        let ret = core::arch::x86_64::__cpuid(0x0);
        ebx = ret.ebx;
        edx = ret.edx;
        ecx = ret.ecx;
    }
    let ret = (ebx == 0x68747541) && (ecx == 0x444D4163) && (edx == 0x69746E65);
    return ret;
}

/// Test, if SEV-SNP is supported, should only used in host,
#[inline(always)]
pub fn check_snp_support() -> bool {
    let eax;
    unsafe {
        let ret = core::arch::x86_64::__cpuid(0x8000001f);
        eax = ret.eax;
    }
    return (eax & (1 << 4)) != 0;
}
