use super::super::super::qlib::cc::sev_snp::cpuid_page::*;
use super::super::super::qlib::common::*;
use super::super::super::runc::runtime::loader::*;

use super::vm::*;
use kvm_bindings::*;

impl CpuidPage {
    pub fn FillCpuidPage(&mut self, kvm_cpuid_entries: &CpuId) -> Result<()> {
        let mut has_entries = false;

        for kvm_entry in kvm_cpuid_entries.as_slice() {
            if kvm_entry.function == 0 && kvm_entry.index == 0 && has_entries {
                break;
            }

            if kvm_entry.function == 0xFFFFFFFF {
                break;
            }

            // range check, see:
            // SEV Secure Nested Paging Firmware ABI Specification
            // 8.17.2.6 PAGE_TYPE_CPUID
            if !((0x0000_0000..=0x0000_FFFF).contains(&kvm_entry.function)
                || (0x8000_0000..=0x8000_FFFF).contains(&kvm_entry.function))
            {
                continue;
            }
            has_entries = true;

            let mut snp_cpuid_entry = SnpCpuidFunc {
                eax_in: kvm_entry.function,
                ecx_in: {
                    if (kvm_entry.flags & KVM_CPUID_FLAG_SIGNIFCANT_INDEX) != 0 {
                        kvm_entry.index
                    } else {
                        0
                    }
                },
                xcr0_in: 0,
                xss_in: 0,
                eax: kvm_entry.eax,
                ebx: kvm_entry.ebx,
                ecx: kvm_entry.ecx,
                edx: kvm_entry.edx,
                ..Default::default()
            };
            if snp_cpuid_entry.eax_in == 0xD
                && (snp_cpuid_entry.ecx_in == 0x0 || snp_cpuid_entry.ecx_in == 0x1)
            {
                /*
                 * Guest kernels will calculate EBX themselves using the 0xD
                 * subfunctions corresponding to the individual XSAVE areas, so only
                 * encode the base XSAVE size in the initial leaves, corresponding
                 * to the initial XCR0=1 state.
                 */
                snp_cpuid_entry.ebx = 0x240;
                snp_cpuid_entry.xcr0_in = 1;
                snp_cpuid_entry.xss_in = 0;
            }

            self.AddEntry(&snp_cpuid_entry)
                .expect("Failed to add CPUID entry to the CPUID page");
        }
        Ok(())
    }
}

impl VirtualMachine {
    pub fn InitSevSnp(args: Args /*args: &Args, kvmfd: i32*/) -> Result<Self> {
        todo!();
    }
}
