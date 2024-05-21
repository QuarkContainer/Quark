use self::Error::{FailInput, FailSizeMismatch, Unknown};
use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::VirtAddr;

pub mod cpuid_page;
pub mod ghcb;
pub mod vc;

/// The C-Bit mask indicating encrypted physical addresses
pub static C_BIT_MASK: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Reasons:
    /// - Page size is 2MB and page is not 2MB aligned
    FailInput,
    /// Reasons:
    /// - 2MB validation backed by 4KB pages
    FailSizeMismatch,
    /// Unknown error
    Unknown(u32),
}

#[derive(Debug, PartialEq)]
pub enum VCResult {
    ///All good
    EsOk,
    ///Requested operation not supported
    EsUnsupported,
    ///Unexpected state from the VMM
    EsVmmError,
    ///Instruction decoding failed
    EsDecodeFailed,
    ///Instruction caused exception
    EsException(u64),
    ///Retry instruction emulation
    EsRetry,
}

/// The size of the page to `pvalidate`
#[repr(u64)]
pub enum PvalidateSize {
    /// A 4k page
    Size4K = 0,
    /// A 2M page
    Size2M = 1,
}

/// Get the SEV C-Bit mask
#[inline(always)]
pub fn get_cbit_mask() -> u64 {
    C_BIT_MASK.load(Ordering::Relaxed)
}

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

/// Test, if SEV-SNP is enabled
#[inline(always)]
pub fn snp_active() -> bool {
    get_cbit_mask() > 0
}
/// AMD pvalidate
///
/// returns `Ok(rmp_changed)` on success with `rmp_changed` indicating if the contents
/// of the RMP entry was changed or not.
///
/// - If `addr` is not a readable mapped page, `pvalidate` will result in a Page Fault, #PF exception.
/// - This is a privileged instruction. Attempted execution at a privilege level other than CPL0 will result in
///   a #GP(0) exception.
/// - VMPL or CPL not zero will result in a #GP(0) exception.
#[inline(always)]
pub fn pvalidate(addr: VirtAddr, size: PvalidateSize, validated: bool) -> Result<bool, Error> {
    let rmp_changed: u32;
    let ret: u64;
    let flag: u32 = validated.into();

    // pvalidate and output the carry bit in edx
    // return value in rax
    unsafe {
        asm!(
        "pvalidate",
        "setc    dl",
        inout("rax") addr.as_u64() & (!0xFFF) => ret,
        in("rcx") size as u64,
        inout("edx") flag => rmp_changed,
        options(nostack, nomem)
        );
    }

    match ret as u32 {
        0 => Ok(rmp_changed as u8 == 0),
        1 => Err(FailInput),
        6 => Err(FailSizeMismatch),
        ret => Err(Unknown(ret)),
    }
}

pub struct SVMExitDef {}
impl SVMExitDef {
    pub const SVM_EXIT_READ_CR0: u64 = 0x000;
    pub const SVM_EXIT_READ_CR2: u64 = 0x002;
    pub const SVM_EXIT_READ_CR3: u64 = 0x003;
    pub const SVM_EXIT_READ_CR4: u64 = 0x004;
    pub const SVM_EXIT_READ_CR8: u64 = 0x008;
    pub const SVM_EXIT_WRITE_CR0: u64 = 0x010;
    pub const SVM_EXIT_WRITE_CR2: u64 = 0x012;
    pub const SVM_EXIT_WRITE_CR3: u64 = 0x013;
    pub const SVM_EXIT_WRITE_CR4: u64 = 0x014;
    pub const SVM_EXIT_WRITE_CR8: u64 = 0x018;
    pub const SVM_EXIT_READ_DR0: u64 = 0x020;
    pub const SVM_EXIT_READ_DR1: u64 = 0x021;
    pub const SVM_EXIT_READ_DR2: u64 = 0x022;
    pub const SVM_EXIT_READ_DR3: u64 = 0x023;
    pub const SVM_EXIT_READ_DR4: u64 = 0x024;
    pub const SVM_EXIT_READ_DR5: u64 = 0x025;
    pub const SVM_EXIT_READ_DR6: u64 = 0x026;
    pub const SVM_EXIT_READ_DR7: u64 = 0x027;
    pub const SVM_EXIT_WRITE_DR0: u64 = 0x030;
    pub const SVM_EXIT_WRITE_DR1: u64 = 0x031;
    pub const SVM_EXIT_WRITE_DR2: u64 = 0x032;
    pub const SVM_EXIT_WRITE_DR3: u64 = 0x033;
    pub const SVM_EXIT_WRITE_DR4: u64 = 0x034;
    pub const SVM_EXIT_WRITE_DR5: u64 = 0x035;
    pub const SVM_EXIT_WRITE_DR6: u64 = 0x036;
    pub const SVM_EXIT_WRITE_DR7: u64 = 0x037;
    pub const SVM_EXIT_EXCP_BASE: u64 = 0x040;
    pub const SVM_EXIT_LAST_EXCP: u64 = 0x05f;
    pub const SVM_EXIT_INTR: u64 = 0x060;
    pub const SVM_EXIT_NMI: u64 = 0x061;
    pub const SVM_EXIT_SMI: u64 = 0x062;
    pub const SVM_EXIT_INIT: u64 = 0x063;
    pub const SVM_EXIT_VINTR: u64 = 0x064;
    pub const SVM_EXIT_CR0_SEL_WRITE: u64 = 0x065;
    pub const SVM_EXIT_IDTR_READ: u64 = 0x066;
    pub const SVM_EXIT_GDTR_READ: u64 = 0x067;
    pub const SVM_EXIT_LDTR_READ: u64 = 0x068;
    pub const SVM_EXIT_TR_READ: u64 = 0x069;
    pub const SVM_EXIT_IDTR_WRITE: u64 = 0x06a;
    pub const SVM_EXIT_GDTR_WRITE: u64 = 0x06b;
    pub const SVM_EXIT_LDTR_WRITE: u64 = 0x06c;
    pub const SVM_EXIT_TR_WRITE: u64 = 0x06d;
    pub const SVM_EXIT_RDTSC: u64 = 0x06e;
    pub const SVM_EXIT_RDPMC: u64 = 0x06f;
    pub const SVM_EXIT_PUSHF: u64 = 0x070;
    pub const SVM_EXIT_POPF: u64 = 0x071;
    pub const SVM_EXIT_CPUID: u64 = 0x072;
    pub const SVM_EXIT_RSM: u64 = 0x073;
    pub const SVM_EXIT_IRET: u64 = 0x074;
    pub const SVM_EXIT_SWINT: u64 = 0x075;
    pub const SVM_EXIT_INVD: u64 = 0x076;
    pub const SVM_EXIT_PAUSE: u64 = 0x077;
    pub const SVM_EXIT_HLT: u64 = 0x078;
    pub const SVM_EXIT_INVLPG: u64 = 0x079;
    pub const SVM_EXIT_INVLPGA: u64 = 0x07a;
    pub const SVM_EXIT_IOIO: u64 = 0x07b;
    pub const SVM_EXIT_MSR: u64 = 0x07c;
    pub const SVM_EXIT_TASK_SWITCH: u64 = 0x07d;
    pub const SVM_EXIT_FERR_FREEZE: u64 = 0x07e;
    pub const SVM_EXIT_SHUTDOWN: u64 = 0x07f;
    pub const SVM_EXIT_VMRUN: u64 = 0x080;
    pub const SVM_EXIT_VMMCALL: u64 = 0x081;
    pub const SVM_EXIT_VMLOAD: u64 = 0x082;
    pub const SVM_EXIT_VMSAVE: u64 = 0x083;
    pub const SVM_EXIT_STGI: u64 = 0x084;
    pub const SVM_EXIT_CLGI: u64 = 0x085;
    pub const SVM_EXIT_SKINIT: u64 = 0x086;
    pub const SVM_EXIT_RDTSCP: u64 = 0x087;
    pub const SVM_EXIT_ICEBP: u64 = 0x088;
    pub const SVM_EXIT_WBINVD: u64 = 0x089;
    pub const SVM_EXIT_MONITOR: u64 = 0x08a;
    pub const SVM_EXIT_MWAIT: u64 = 0x08b;
    pub const SVM_EXIT_MWAIT_COND: u64 = 0x08c;
    pub const SVM_EXIT_XSETBV: u64 = 0x08d;
    pub const SVM_EXIT_RDPRU: u64 = 0x08e;
    pub const SVM_EXIT_EFER_WRITE_TRAP: u64 = 0x08f;
    pub const SVM_EXIT_CR0_WRITE_TRAP: u64 = 0x090;
    pub const SVM_EXIT_CR1_WRITE_TRAP: u64 = 0x091;
    pub const SVM_EXIT_CR2_WRITE_TRAP: u64 = 0x092;
    pub const SVM_EXIT_CR3_WRITE_TRAP: u64 = 0x093;
    pub const SVM_EXIT_CR4_WRITE_TRAP: u64 = 0x094;
    pub const SVM_EXIT_CR5_WRITE_TRAP: u64 = 0x095;
    pub const SVM_EXIT_CR6_WRITE_TRAP: u64 = 0x096;
    pub const SVM_EXIT_CR7_WRITE_TRAP: u64 = 0x097;
    pub const SVM_EXIT_CR8_WRITE_TRAP: u64 = 0x098;
    pub const SVM_EXIT_CR9_WRITE_TRAP: u64 = 0x099;
    pub const SVM_EXIT_CR10_WRITE_TRAP: u64 = 0x09a;
    pub const SVM_EXIT_CR11_WRITE_TRAP: u64 = 0x09b;
    pub const SVM_EXIT_CR12_WRITE_TRAP: u64 = 0x09c;
    pub const SVM_EXIT_CR13_WRITE_TRAP: u64 = 0x09d;
    pub const SVM_EXIT_CR14_WRITE_TRAP: u64 = 0x09e;
    pub const SVM_EXIT_CR15_WRITE_TRAP: u64 = 0x09f;
    pub const SVM_EXIT_INVPCID: u64 = 0x0a2;
    pub const SVM_EXIT_NPF: u64 = 0x400;
    pub const SVM_EXIT_AVIC_INCOMPLETE_IPI: u64 = 0x401;
    pub const SVM_EXIT_AVIC_UNACCELERATED_ACCESS: u64 = 0x402;
    pub const SVM_EXIT_VMGEXIT: u64 = 0x403;

    // SEV-ES software-defined VMGEXIT events
    pub const SVM_VMGEXIT_MMIO_READ: u64 = 0x80000001;
    pub const SVM_VMGEXIT_MMIO_WRITE: u64 = 0x80000002;
    pub const SVM_VMGEXIT_NMI_COMPLETE: u64 = 0x80000003;
    pub const SVM_VMGEXIT_AP_HLT_LOOP: u64 = 0x80000004;
    pub const SVM_VMGEXIT_AP_JUMP_TABLE: u64 = 0x80000005;
    pub const SVM_VMGEXIT_SET_AP_JUMP_TABLE: u64 = 0;
    pub const SVM_VMGEXIT_GET_AP_JUMP_TABLE: u64 = 1;
    pub const SVM_VMGEXIT_PSC: u64 = 0x80000010;
    pub const SVM_VMGEXIT_GUEST_REQUEST: u64 = 0x80000011;
    pub const SVM_VMGEXIT_EXT_GUEST_REQUEST: u64 = 0x80000012;
    pub const SVM_VMGEXIT_AP_CREATION: u64 = 0x80000013;
    pub const SVM_VMGEXIT_AP_CREATE_ON_INIT: u64 = 0;
    pub const SVM_VMGEXIT_AP_CREATE: u64 = 1;
    pub const SVM_VMGEXIT_AP_DESTROY: u64 = 2;
    pub const SVM_VMGEXIT_HV_FEATURES: u64 = 0x8000fffd;
    pub const SVM_VMGEXIT_TERM_REQUEST: u64 = 0x8000fffe;
    pub fn svm_vmgexit_term_reason(reason_set: u64, reason_code: u64) -> u64 {
        // SW_EXITINFO1[3:0]
        (reason_set & 0xf) |
    // SW_EXITINFO1[11:4]
    ((reason_code & 0xff) << 4)
    }
    pub const SVM_VMGEXIT_UNSUPPORTED_EVENT: u64 = 0x8000ffff;

    // Exit code reserved for hypervisor/software use
    pub const SVM_EXIT_SW: u64 = 0xf0000000;

    // Error exit code
    pub const SVM_EXIT_ERR: i64 = -1;

    pub fn X86_MODRM_REG(modrm: u8) -> u8 {
        return ((modrm) & 0x38) >> 3;
    }
}
