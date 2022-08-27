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

use super::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::collections::btree_set::BTreeSet;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::common::*;
use super::singleton::*;
use super::sort_arr::*;
use crate::qlib::kernel::asm::AsmHostID;

pub static X86_FEATURES_FROM_STRING: Singleton<QMutex<BTreeMap<String, i32>>> =
    Singleton::<QMutex<BTreeMap<String, i32>>>::New();
pub static X86_FEATURE_STRINGS: Singleton<SortArr<i32, &'static str>> =
    Singleton::<SortArr<i32, &'static str>>::New();
pub static X86_FEATURE_PARSE_ONLY_STRINGS: Singleton<SortArr<i32, &'static str>> =
    Singleton::<SortArr<i32, &'static str>>::New();
pub static CPU_FREQ_MHZ: Singleton<QMutex<f64>> = Singleton::<QMutex<f64>>::New();

pub unsafe fn InitSingleton() {
    X86_FEATURES_FROM_STRING.Init(QMutex::new(BTreeMap::new()));
    X86_FEATURE_STRINGS.Init(SortArr::New(&[
        // Block 0.
        (X86Feature::X86FeatureSSE3 as i32, "pni"),
        (X86Feature::X86FeaturePCLMULDQ as i32, "pclmulqdq"),
        (X86Feature::X86FeatureDTES64 as i32, "dtes64"),
        (X86Feature::X86FeatureMONITOR as i32, "monitor"),
        (X86Feature::X86FeatureDSCPL as i32, "ds_cpl"),
        (X86Feature::X86FeatureVMX as i32, "vmx"),
        (X86Feature::X86FeatureSMX as i32, "smx"),
        (X86Feature::X86FeatureEST as i32, "est"),
        (X86Feature::X86FeatureTM2 as i32, "tm2"),
        (X86Feature::X86FeatureSSSE3 as i32, "ssse3"),
        (X86Feature::X86FeatureCNXTID as i32, "cid"),
        (X86Feature::X86FeatureSDBG as i32, "sdbg"),
        (X86Feature::X86FeatureFMA as i32, "fma"),
        (X86Feature::X86FeatureCX16 as i32, "cx16"),
        (X86Feature::X86FeatureXTPR as i32, "xtpr"),
        (X86Feature::X86FeaturePDCM as i32, "pdcm"),
        (X86Feature::X86FeaturePCID as i32, "pcid"),
        (X86Feature::X86FeatureDCA as i32, "dca"),
        (X86Feature::X86FeatureSSE4_1 as i32, "sse4_1"),
        (X86Feature::X86FeatureSSE4_2 as i32, "sse4_2"),
        (X86Feature::X86FeatureX2APIC as i32, "x2apic"),
        (X86Feature::X86FeatureMOVBE as i32, "movbe"),
        (X86Feature::X86FeaturePOPCNT as i32, "popcnt"),
        (X86Feature::X86FeatureTSCD as i32, "tsc_deadline_timer"),
        (X86Feature::X86FeatureAES as i32, "aes"),
        (X86Feature::X86FeatureXSAVE as i32, "xsave"),
        (X86Feature::X86FeatureAVX as i32, "avx"),
        (X86Feature::X86FeatureF16C as i32, "f16c"),
        (X86Feature::X86FeatureRDRAND as i32, "rdrand"),
        // Block 1.
        (X86Feature::X86FeatureFPU as i32, "fpu"),
        (X86Feature::X86FeatureVME as i32, "vme"),
        (X86Feature::X86FeatureDE as i32, "de"),
        (X86Feature::X86FeaturePSE as i32, "pse"),
        (X86Feature::X86FeatureTSC as i32, "tsc"),
        (X86Feature::X86FeatureMSR as i32, "msr"),
        (X86Feature::X86FeaturePAE as i32, "pae"),
        (X86Feature::X86FeatureMCE as i32, "mce"),
        (X86Feature::X86FeatureCX8 as i32, "cx8"),
        (X86Feature::X86FeatureAPIC as i32, "apic"),
        (X86Feature::X86FeatureSEP as i32, "sep"),
        (X86Feature::X86FeatureMTRR as i32, "mtrr"),
        (X86Feature::X86FeaturePGE as i32, "pge"),
        (X86Feature::X86FeatureMCA as i32, "mca"),
        (X86Feature::X86FeatureCMOV as i32, "cmov"),
        (X86Feature::X86FeaturePAT as i32, "pat"),
        (X86Feature::X86FeaturePSE36 as i32, "pse36"),
        (X86Feature::X86FeaturePSN as i32, "pn"),
        (X86Feature::X86FeatureCLFSH as i32, "clflush"),
        (X86Feature::X86FeatureDS as i32, "dts"),
        (X86Feature::X86FeatureACPI as i32, "acpi"),
        (X86Feature::X86FeatureMMX as i32, "mmx"),
        (X86Feature::X86FeatureFXSR as i32, "fxsr"),
        (X86Feature::X86FeatureSSE as i32, "sse"),
        (X86Feature::X86FeatureSSE2 as i32, "sse2"),
        (X86Feature::X86FeatureSS as i32, "ss"),
        (X86Feature::X86FeatureHTT as i32, "ht"),
        (X86Feature::X86FeatureTM as i32, "tm"),
        (X86Feature::X86FeatureIA64 as i32, "ia64"),
        (X86Feature::X86FeaturePBE as i32, "pbe"),
        // Block 2.
        (X86Feature::X86FeatureFSGSBase as i32, "fsgsbase"),
        (X86Feature::X86FeatureTSC_ADJUST as i32, "tsc_adjust"),
        (X86Feature::X86FeatureBMI1 as i32, "bmi1"),
        (X86Feature::X86FeatureHLE as i32, "hle"),
        (X86Feature::X86FeatureAVX2 as i32, "avx2"),
        (X86Feature::X86FeatureSMEP as i32, "smep"),
        (X86Feature::X86FeatureBMI2 as i32, "bmi2"),
        (X86Feature::X86FeatureERMS as i32, "erms"),
        (X86Feature::X86FeatureINVPCID as i32, "invpcid"),
        (X86Feature::X86FeatureRTM as i32, "rtm"),
        (X86Feature::X86FeatureCQM as i32, "cqm"),
        (X86Feature::X86FeatureMPX as i32, "mpx"),
        (X86Feature::X86FeatureRDT as i32, "rdt_a"),
        (X86Feature::X86FeatureAVX512F as i32, "avx512f"),
        (X86Feature::X86FeatureAVX512DQ as i32, "avx512dq"),
        (X86Feature::X86FeatureRDSEED as i32, "rdseed"),
        (X86Feature::X86FeatureADX as i32, "adx"),
        (X86Feature::X86FeatureSMAP as i32, "smap"),
        (X86Feature::X86FeatureCLWB as i32, "clwb"),
        (X86Feature::X86FeatureAVX512PF as i32, "avx512pf"),
        (X86Feature::X86FeatureAVX512ER as i32, "avx512er"),
        (X86Feature::X86FeatureAVX512CD as i32, "avx512cd"),
        (X86Feature::X86FeatureSHA as i32, "sha_ni"),
        (X86Feature::X86FeatureAVX512BW as i32, "avx512bw"),
        (X86Feature::X86FeatureAVX512VL as i32, "avx512vl"),
        // Block 3.
        (X86Feature::X86FeatureAVX512VBMI as i32, "avx512vbmi"),
        (X86Feature::X86FeatureUMIP as i32, "umip"),
        (X86Feature::X86FeaturePKU as i32, "pku"),
        // Block 4.
        (X86Feature::X86FeatureXSAVEOPT as i32, "xsaveopt"),
        (X86Feature::X86FeatureXSAVEC as i32, "xsavec"),
        (X86Feature::X86FeatureXGETBV1 as i32, "xgetbv1"),
        (X86Feature::X86FeatureXSAVES as i32, "xsaves"),
        // Block 5.
        (X86Feature::X86FeatureLAHF64 as i32, "lahf_lm"), // LAHF/SAHF in long mode
        (X86Feature::X86FeatureCMP_LEGACY as i32, "cmp_legacy"),
        (X86Feature::X86FeatureSVM as i32, "svm"),
        (X86Feature::X86FeatureEXTAPIC as i32, "extapic"),
        (X86Feature::X86FeatureCR8_LEGACY as i32, "cr8_legacy"),
        (X86Feature::X86FeatureLZCNT as i32, "abm"), // Advanced bit manipulation
        (X86Feature::X86FeatureSSE4A as i32, "sse4a"),
        (X86Feature::X86FeatureMISALIGNSSE as i32, "misalignsse"),
        (X86Feature::X86FeaturePREFETCHW as i32, "3dnowprefetch"),
        (X86Feature::X86FeatureOSVW as i32, "osvw"),
        (X86Feature::X86FeatureIBS as i32, "ibs"),
        (X86Feature::X86FeatureXOP as i32, "xop"),
        (X86Feature::X86FeatureSKINIT as i32, "skinit"),
        (X86Feature::X86FeatureWDT as i32, "wdt"),
        (X86Feature::X86FeatureLWP as i32, "lwp"),
        (X86Feature::X86FeatureFMA4 as i32, "fma4"),
        (X86Feature::X86FeatureTCE as i32, "tce"),
        (X86Feature::X86FeatureTBM as i32, "tbm"),
        (X86Feature::X86FeatureTOPOLOGY as i32, "topoext"),
        (X86Feature::X86FeaturePERFCTR_CORE as i32, "perfctr_core"),
        (X86Feature::X86FeaturePERFCTR_NB as i32, "perfctr_nb"),
        (X86Feature::X86FeatureBPEXT as i32, "bpext"),
        (X86Feature::X86FeaturePERFCTR_TSC as i32, "ptsc"),
        (X86Feature::X86FeaturePERFCTR_LLC as i32, "perfctr_llc"),
        (X86Feature::X86FeatureMWAITX as i32, "mwaitx"),
        // Block 6.
        (X86Feature::X86FeatureSYSCALL as i32, "syscall"),
        (X86Feature::X86FeatureNX as i32, "nx"),
        (X86Feature::X86FeatureMMXEXT as i32, "mmxext"),
        (X86Feature::X86FeatureFXSR_OPT as i32, "fxsr_opt"),
        (X86Feature::X86FeatureGBPAGES as i32, "pdpe1gb"),
        (X86Feature::X86FeatureRDTSCP as i32, "rdtscp"),
        (X86Feature::X86FeatureLM as i32, "lm"),
        (X86Feature::X86Feature3DNOWEXT as i32, "3dnowext"),
        (X86Feature::X86Feature3DNOW as i32, "3dnow"),
    ]));

    X86_FEATURE_PARSE_ONLY_STRINGS.Init(SortArr::New(&[
        // Block 0.
        (X86Feature::X86FeatureOSXSAVE as i32, "osxsave"),
        // Block 2.
        (
            X86Feature::X86FeatureFDP_EXCPTN_ONLY as i32,
            "fdp_excptn_only",
        ),
        (X86Feature::X86FeatureFPCSDS as i32, "fpcsds"),
        (X86Feature::X86FeatureIPT as i32, "pt"),
        (X86Feature::X86FeatureCLFLUSHOPT as i32, "clfushopt"),
        // Block 3.
        (X86Feature::X86FeaturePREFETCHWT1 as i32, "prefetchwt1"),
    ]));

    CPU_FREQ_MHZ.Init(QMutex::new(0.0));
}

pub type Block = i32;

const BLOCK_SIZE: i32 = 32;

#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
pub enum X86Feature {
    // Block 0 constants are all of the "basic" feature bits returned by a cpuid in
    // ecx with eax=1.
    X86FeatureSSE3 = 0,
    X86FeaturePCLMULDQ,
    X86FeatureDTES64,
    X86FeatureMONITOR,
    X86FeatureDSCPL,
    X86FeatureVMX,
    X86FeatureSMX,
    X86FeatureEST,
    X86FeatureTM2,
    X86FeatureSSSE3,
    // Not a typo, "supplemental" SSE3.
    X86FeatureCNXTID,
    X86FeatureSDBG,
    X86FeatureFMA,
    X86FeatureCX16,
    X86FeatureXTPR,
    X86FeaturePDCM,
    _Dummy1,
    // ecx bit 16 is reserved.
    X86FeaturePCID,
    X86FeatureDCA,
    X86FeatureSSE4_1,
    X86FeatureSSE4_2,
    X86FeatureX2APIC,
    X86FeatureMOVBE,
    X86FeaturePOPCNT,
    X86FeatureTSCD,
    X86FeatureAES,
    X86FeatureXSAVE,
    X86FeatureOSXSAVE,
    X86FeatureAVX,
    X86FeatureF16C,
    X86FeatureRDRAND,
    _Dummy2,
    // ecx bit 31 is reserved.

    // Block 1 constants are all of the "basic" feature bits returned by a cpuid in
    // edx with eax=1.
    X86FeatureFPU = 32,
    X86FeatureVME,
    X86FeatureDE,
    X86FeaturePSE,
    X86FeatureTSC,
    X86FeatureMSR,
    X86FeaturePAE,
    X86FeatureMCE,
    X86FeatureCX8,
    X86FeatureAPIC,
    _Dummy3,
    // edx bit 10 is reserved.
    X86FeatureSEP,
    X86FeatureMTRR,
    X86FeaturePGE,
    X86FeatureMCA,
    X86FeatureCMOV,
    X86FeaturePAT,
    X86FeaturePSE36,
    X86FeaturePSN,
    X86FeatureCLFSH,
    _Dummy4,
    // edx bit 20 is reserved.
    X86FeatureDS,
    X86FeatureACPI,
    X86FeatureMMX,
    X86FeatureFXSR,
    X86FeatureSSE,
    X86FeatureSSE2,
    X86FeatureSS,
    X86FeatureHTT,
    X86FeatureTM,
    X86FeatureIA64,
    X86FeaturePBE,

    // Block 2 bits are the "structured extended" features returned in ebx for
    // eax=7, ecx=0.
    X86FeatureFSGSBase = 2 * 32,
    X86FeatureTSC_ADJUST,
    _Dummy5,
    // ebx bit 2 is reserved.
    X86FeatureBMI1,
    X86FeatureHLE,
    X86FeatureAVX2,
    X86FeatureFDP_EXCPTN_ONLY,
    X86FeatureSMEP,
    X86FeatureBMI2,
    X86FeatureERMS,
    X86FeatureINVPCID,
    X86FeatureRTM,
    X86FeatureCQM,
    X86FeatureFPCSDS,
    X86FeatureMPX,
    X86FeatureRDT,
    X86FeatureAVX512F,
    X86FeatureAVX512DQ,
    X86FeatureRDSEED,
    X86FeatureADX,
    X86FeatureSMAP,
    X86FeatureAVX512IFMA,
    X86FeaturePCOMMIT,
    X86FeatureCLFLUSHOPT,
    X86FeatureCLWB,
    X86FeatureIPT,
    // Intel processor trace.
    X86FeatureAVX512PF,
    X86FeatureAVX512ER,
    X86FeatureAVX512CD,
    X86FeatureSHA,
    X86FeatureAVX512BW,
    X86FeatureAVX512VL,

    // Block 3 bits are the "extended" features returned in ecx for eax=7, ecx=0.
    X86FeaturePREFETCHWT1 = 3 * 32,
    X86FeatureAVX512VBMI,
    X86FeatureUMIP,
    X86FeaturePKU,

    // Block 4 constants are for xsave capabilities in CPUID.(EAX=0DH,ECX=01H):EAX.
    // The CPUID leaf is available only if 'X86FeatureXSAVE' is present.
    X86FeatureXSAVEOPT = 4 * 32,
    X86FeatureXSAVEC,
    X86FeatureXGETBV1,
    X86FeatureXSAVES,
    // EAX[31:4] are reserved.

    // Block 5 constants are the extended feature bits in
    // CPUID.(EAX=0x80000001):ECX.
    X86FeatureLAHF64 = 5 * 32,
    X86FeatureCMP_LEGACY,
    X86FeatureSVM,
    X86FeatureEXTAPIC,
    X86FeatureCR8_LEGACY,
    X86FeatureLZCNT,
    X86FeatureSSE4A,
    X86FeatureMISALIGNSSE,
    X86FeaturePREFETCHW,
    X86FeatureOSVW,
    X86FeatureIBS,
    X86FeatureXOP,
    X86FeatureSKINIT,
    X86FeatureWDT,
    _Dummy6,
    // ecx bit 14 is reserved.
    X86FeatureLWP,
    X86FeatureFMA4,
    X86FeatureTCE,
    _Dummy7,
    // ecx bit 18 is reserved.
    _Dummy8,
    // ecx bit 19 is reserved.
    _Dummy9,
    // ecx bit 20 is reserved.
    X86FeatureTBM,
    X86FeatureTOPOLOGY,
    X86FeaturePERFCTR_CORE,
    X86FeaturePERFCTR_NB,
    _Dummy10,
    // ecx bit 25 is reserved.
    X86FeatureBPEXT,
    X86FeaturePERFCTR_TSC,
    X86FeaturePERFCTR_LLC,
    X86FeatureMWAITX,
    // ECX[31:30] are reserved.

    // Block 6 constants are the extended feature bits in
    // CPUID.(EAX=0x80000001):EDX.
    //
    // These are sparse, and so the bit positions are assigned manually.
    Block6DuplicateMask = 0x183f3ff,

    X86FeatureSYSCALL = 6 * 32 + 11,
    X86FeatureNX = 6 * 32 + 20,
    X86FeatureMMXEXT = 6 * 32 + 22,
    X86FeatureFXSR_OPT = 6 * 32 + 25,
    X86FeatureGBPAGES = 6 * 32 + 26,
    X86FeatureRDTSCP = 6 * 32 + 27,
    X86FeatureLM = 6 * 32 + 29,
    X86Feature3DNOWEXT = 6 * 32 + 30,
    X86Feature3DNOW = 6 * 32 + 31,
}

// linuxBlockOrder defines the order in which linux organizes the feature
// blocks. Linux also tracks feature bits in 32-bit blocks, but in an order
// which doesn't match well here, so for the /proc/cpuinfo generation we simply
// re-map the blocks to Linux's ordering and then go through the bits in each
// block.
const LINUX_BLOCK_ORDER: [Block; 7] = [1, 6, 0, 5, 2, 4, 3];

// The constants below are the lower or "standard" cpuid functions, ordered as
// defined by the hardware.
#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
pub enum CpuidFunction {
    vendorID = 0,
    // Returns vendor ID and largest standard function.
    featureInfo,
    // Returns basic feature bits and processor signature.
    intelCacheDescriptors,
    // Returns list of cache descriptors. Intel only.
    intelSerialNumber,
    // Returns processor serial number (obsolete on new hardware). Intel only.
    intelDeterministicCacheParams,
    // Returns deterministic cache information. Intel only.
    monitorMwaitParams,
    // Returns information about monitor/mwait instructions.
    powerParams,
    // Returns information about power management and thermal sensors.
    extendedFeatureInfo,
    // Returns extended feature bits.
    _dummy1,
    // Function 0x8 is reserved.
    intelDCAParams,
    // Returns direct cache access information. Intel only.
    intelPMCInfo,
    // Returns information about performance monitoring features. Intel only.
    intelX2APICInfo,
    // Returns core/logical processor topology. Intel only.
    _dummy12,
    // Function 0xc is reserved.(/
    xSaveInfo,
    // Returns information about extended state management.
}

// The "extended" functions start at 0x80000000.
#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
pub enum ExtendedFunction {
    extendedFunctionInfo = 0x80000000,
    // Returns highest available extended function in eax.
    extendedFeatures,
    // Returns some extended feature bits in edx and ecx.
}

// These are the extended floating point state features. They are used to
// enumerate floating point features in XCR0, XSTATE_BV, etc.
#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
pub enum XSAVEFeature {
    XSAVEFeatureX87 = 1 << 0,
    XSAVEFeatureSSE = 1 << 1,
    XSAVEFeatureAVX = 1 << 2,
    XSAVEFeatureBNDREGS = 1 << 3,
    XSAVEFeatureBNDCSR = 1 << 4,
    XSAVEFeatureAVX512op = 1 << 5,
    XSAVEFeatureAVX512zmm0 = 1 << 6,
    XSAVEFeatureAVX512zmm16 = 1 << 7,
    XSAVEFeaturePKRU = 1 << 9,
}

#[derive(Debug)]
pub struct Feature(pub i32);

impl Feature {
    pub fn String(&self) -> String {
        let s = self.flagString(false).to_string();
        if s.as_str() != "" {
            return s;
        }

        let block = self.0 / 32;
        let bit = self.0 % 32;

        return format!("<cpuflag {}; block {} bit {}>", self.0, block, bit);
    }

    fn flagString(&self, cpuinfoOnly: bool) -> &'static str {
        match X86_FEATURE_STRINGS.Get(self.0) {
            Some(s) => return s,
            None => {
                if cpuinfoOnly {
                    match X86_FEATURE_PARSE_ONLY_STRINGS.Get(self.0) {
                        Some(s) => return s,
                        None => return "",
                    }
                }
                return "";
            }
        }
    }
}

fn featureID(b: Block, bit: i32) -> Feature {
    return Feature(32 * b + bit);
}

#[derive(Debug, Default)]
pub struct FeatureSet {
    // Set is the set of features that are enabled in this FeatureSet.
    pub Set: BTreeSet<i32>,

    // VendorID is the 12-char string returned in ebx:edx:ecx for eax=0.
    pub VendorID: String,

    // ExtendedFamily is part of the processor signature.
    pub ExtendedFamily: u8,

    // ExtendedModel is part of the processor signature.
    pub ExtendedModel: u8,

    // ProcessorType is part of the processor signature.
    pub ProcessorType: u8,

    // Family is part of the processor signature.
    pub Family: u8,

    // Model is part of the processor signature.
    pub Model: u8,

    // SteppingID is part of the processor signature.
    pub SteppingID: u8,
}

impl FeatureSet {
    pub fn FlagsString(&self, cpuinfoOnly: bool) -> String {
        let mut s = Vec::new();

        for b in &LINUX_BLOCK_ORDER {
            for i in 0..BLOCK_SIZE as usize {
                let f = featureID(*b, i as i32);
                if self.Set.contains(&f.0) {
                    let fstr = f.flagString(cpuinfoOnly);
                    if fstr != "" {
                        s.push(fstr)
                    }
                }
            }
        }

        return s.join(" ");
    }

    pub fn CPUInfo(&self, cpu: u32) -> String {
        let mut res = "".to_string();

        let fs = self;

        //error!("self.HasFeature(Feature(X86Feature::X86FeatureXSAVE as i32) is {}", self.HasFeature(Feature(X86Feature::X86FeatureXSAVE as i32)));
        //error!("self.HasFeature(Feature(X86Feature::X86FeatureOSXSAVE as i32)) is {}", self.HasFeature(Feature(X86Feature::X86FeatureOSXSAVE as i32)));
        //error!("self.HasFeature(Feature(X86Feature::X86FeatureXSAVEOPT as i32)) is {}", self.HasFeature(Feature(X86Feature::X86FeatureXSAVEOPT as i32)));

        res += &format!("processor\t: {}\n", cpu);
        res += &format!("vendor_id\t: {}\n", fs.VendorID);
        res += &format!(
            "cpu family\t: {}\n",
            ((fs.ExtendedFamily << 4) & 0xff) | fs.Family
        );
        res += &format!(
            "model\t\t: {}\n",
            ((fs.ExtendedModel << 4) & 0xff) | fs.Model
        );
        res += &format!("model name\t: {}\n", "unknown");
        res += &format!("stepping\t: {}\n", "unknown");
        res += &format!("cpu MHz\t\t: {}\n", *CPU_FREQ_MHZ.lock());
        res += &format!("fpu\t\t: yes\n");
        res += &format!("fpu_exception\t: yes\n");
        res += &format!("cpuid level\t: {}\n", CpuidFunction::xSaveInfo as u32);
        res += &format!("wp\t\t: yes\n");
        res += &format!("flags\t\t: {}\n", fs.FlagsString(true));
        res += &format!("bogomips\t: {}\n", *CPU_FREQ_MHZ.lock());
        res += &format!("clflush size\t: {}\n", 64);
        res += &format!("cache_alignment\t: {}\n", 64);
        res += &format!(
            "address sizes\t: {} bits physical, {} bits virtual\n",
            46, 48
        );
        res += &format!("power management:\n");
        res += &format!("");
        return res;
    }

    pub fn AMD(&self) -> bool {
        return self.VendorID.as_str() == "AuthenticAMD";
    }

    pub fn Intel(&self) -> bool {
        return self.VendorID.as_str() == "GenuineIntel";
    }

    // blockMask returns the 32-bit mask associated with a block of features.
    fn blockMask(&self, b: Block) -> u32 {
        let mut mask: u32 = 0;
        for i in 0..BLOCK_SIZE as usize {
            if self.Set.contains(&featureID(b, i as i32).0) {
                mask |= 1 << i;
            }
        }

        return mask;
    }

    // Remove removes a Feature from a FeatureSet. It ignores features
    // that are not in the FeatureSet.
    pub fn Remove(&mut self, feature: Feature) {
        self.Set.remove(&feature.0);
    }

    // Add adds a Feature to a FeatureSet. It ignores duplicate features.
    pub fn Add(&mut self, feature: Feature) {
        self.Set.insert(feature.0);
    }

    // HasFeature tests whether or not a feature is in the given feature set.
    pub fn HasFeature(&self, feature: Feature) -> bool {
        return self.Set.contains(&feature.0);
    }

    // IsSubset returns true if the FeatureSet is a subset of the FeatureSet passed in.
    // This is useful if you want to see if a FeatureSet is compatible with another
    // FeatureSet, since you can only run with a given FeatureSet if it's a subset of
    // the host's.
    pub fn IsSubet(&self, other: &Self) -> bool {
        return self.Subtract(other).len() == 0;
    }

    // Subtract returns the features present in fs that are not present in other.
    // If all features in fs are present in other, Subtract returns nil.
    pub fn Subtract(&self, other: &Self) -> Vec<Feature> {
        let mut diff = Vec::new();
        for f in &self.Set {
            if !other.Set.contains(f) {
                diff.push(Feature(*f))
            }
        }

        return diff;
    }

    pub fn TakeFeatureIntersection(&mut self, other: &Self) {
        let mut removes = Vec::with_capacity(self.Set.len());

        for f in &self.Set {
            if !other.Set.contains(f) {
                removes.push(*f)
            }
        }

        for r in removes {
            self.Set.remove(&r);
        }
    }

    pub fn EmulateID(&self, origAx: u32, origCx: u32) -> (u32, u32, u32, u32) {
        //(ax, bx, cx, dx)
        let mut ax: u32 = 0;
        let mut bx: u32 = 0;
        let mut cx: u32 = 0;
        let mut dx: u32 = 0;

        if origAx == CpuidFunction::vendorID as u32 {
            let ax = CpuidFunction::xSaveInfo as u32;
            let (bx, dx, cx) = self.vendorIDRegs();
            return (ax, bx, cx, dx);
        } else if origAx == CpuidFunction::featureInfo as u32 {
            let bx = 8 << 8;
            let cx = self.blockMask(0);
            let dx = self.blockMask(1);
            let ax = self.signature();
            return (ax, bx, cx, dx);
        } else if origAx == CpuidFunction::intelCacheDescriptors as u32 {
            if !self.Intel() {
                return (0, 0, 0, 0);
            }

            ax = 1;
        } else if origAx == CpuidFunction::xSaveInfo as u32 {
            if !self.UseXsave() {
                return (0, 0, 0, 0);
            }

            return HostID(CpuidFunction::xSaveInfo as u32, origCx);
        } else if origAx == CpuidFunction::extendedFeatureInfo as u32 {
            if origCx == 0 {
                bx = self.blockMask(2);
                cx = self.blockMask(3);
            }
        } else if origAx == ExtendedFunction::extendedFunctionInfo as u32 {
            ax = ExtendedFunction::extendedFeatures as u32;
            cx = 0;
        } else if origAx == ExtendedFunction::extendedFeatures as u32 {
            cx = self.blockMask(5);
            dx = self.blockMask(6);
            if self.AMD() {
                dx |= self.blockMask(1) & X86Feature::Block6DuplicateMask as u32
            }
        } else {
            ax = 0;
            bx = 0;
            cx = 0;
            dx = 0;
        }

        return (ax, bx, cx, dx);
    }

    pub fn UseXsave(&self) -> bool {
        return self.HasFeature(Feature(X86Feature::X86FeatureXSAVE as i32))
            && self.HasFeature(Feature(X86Feature::X86FeatureOSXSAVE as i32));
    }

    pub fn UseXsaveopt(&self) -> bool {
        return self.UseXsave() && self.HasFeature(Feature(X86Feature::X86FeatureXSAVEOPT as i32));
    }

    // CheckHostCompatible returns nil if fs is a subset of the host feature set.
    pub fn CheckHostCompatible(&self) -> Result<()> {
        let hfs = HostFeatureSet();
        let diff = self.Subtract(&hfs);
        if diff.len() != 0 {
            return Err(Error::Common(format!(
                "CPU feature set {:?} incompatible with host feature set {:?} (missing: {:?})",
                self.FlagsString(false),
                self.FlagsString(false),
                diff
            )));
        }

        return Ok(());
    }

    // ExtendedStateSize returns the number of bytes needed to save the "extended
    // state" for this processor and the boundary it must be aligned to. Extended
    // state includes floating point registers, and other cpu state that's not
    // associated with the normal task context.
    //
    // Note: We can save some space here with an optimiazation where we use a
    // smaller chunk of memory depending on features that are actually enabled.
    // Currently we just use the largest possible size for simplicity (which is
    // about 2.5K worst case, with avx512).
    pub fn ExtendedStateSize(&self) -> (u32, u32) {
        if self.UseXsave() {
            // Leaf 0 of xsaveinfo function returns the size for currently
            // enabled xsave features in ebx, the maximum size if all valid
            // features are saved with xsave in ecx, and valid XCR0 bits in
            // edx:eax.
            let (_, _, maxSize, _) = HostID(CpuidFunction::xSaveInfo as u32, 0);
            return (maxSize as u32, 64);
        }

        return (512, 16);
    }

    // ValidXCR0Mask returns the bits that may be set to 1 in control register
    // XCR0.
    pub fn ValidXCR0Mask(&self) -> u64 {
        if !self.UseXsave() {
            return 0;
        }

        let (eax, _, _, edx) = HostID(CpuidFunction::xSaveInfo as u32, 0);
        return (edx as u64) << 32 | eax as u64;
    }

    // vendorIDRegs returns the 3 register values used to construct the 12-byte
    // vendor ID string for eax=0.
    pub fn vendorIDRegs(&self) -> (u32, u32, u32) {
        //(bx, dx, cx)
        let mut bx: u32 = 0;
        let mut cx: u32 = 0;
        let mut dx: u32 = 0;

        for i in 0..4 {
            bx |= (self.VendorID.as_bytes()[i] as u32) << (i * 8);
        }

        for i in 0..4 {
            dx |= (self.VendorID.as_bytes()[i + 4] as u32) << (i * 8);
        }

        for i in 0..4 {
            cx |= (self.VendorID.as_bytes()[i + 8] as u32) << (i * 8);
        }

        return (bx, dx, cx);
    }

    pub fn signature(&self) -> u32 {
        let mut s: u32 = 0;
        s |= (self.SteppingID & 0xf) as u32;
        s |= ((self.Model & 0xf) as u32) << 4;
        s |= ((self.Family & 0xf) as u32) << 8;
        s |= ((self.ProcessorType & 0x3) as u32) << 12;
        s |= ((self.ExtendedModel & 0xf) as u32) << 16;
        s |= ((self.ExtendedFamily & 0xf) as u32) << 20;
        return s;
    }
}

fn signatureSplit(v: u32) -> (u8, u8, u8, u8, u8, u8) {
    //ef, em, pt, f, m, sid
    let sid = (v & 0xf) as u8;
    let m = (v >> 4) as u8 & 0xf;
    let f = (v >> 8) as u8 & 0xf;
    let pt = (v >> 12) as u8 & 0x3;
    let em = (v >> 16) as u8 & 0xf;
    let ef = (v >> 20) as u8;

    return (ef, em, pt, f, m, sid);
}

fn setFromBlockMasks(blocks: &[u32]) -> BTreeSet<i32> {
    let mut s = BTreeSet::new();

    for b in 0..blocks.len() {
        let mut blockMask = blocks[b];
        for i in 0..BLOCK_SIZE as usize {
            if blockMask & 1 != 0 {
                s.insert(featureID(b as i32, i as i32).0);
            }

            blockMask >>= 1;
        }
    }

    return s;
}

pub fn PrintHostId(axArg: u32, cxArg: u32) {
    let (ax, bx, cx, dx) = HostID(axArg, cxArg);
    info!(
        "Host({}, {}) => ax = {} bx = {}, cx = {}, dx= {}",
        axArg, cxArg, ax, bx, cx, dx
    );
}

pub fn HostID(axArg: u32, cxArg: u32) -> (u32, u32, u32, u32) {
    let (ax, bx, cx, dx) = AsmHostID(axArg, cxArg);
    return (ax, bx, cx, dx);
}

// HostFeatureSet uses cpuid to get host values and construct a feature set
// that matches that of the host machine. Note that there are several places
// where there appear to be some unnecessary assignments between register names
// (ax, bx, cx, or dx) and featureBlockN variables. This is to explicitly show
// where the different feature blocks come from, to make the code easier to
// inspect and read.
pub fn HostFeatureSet() -> FeatureSet {
    let (_, bx, cx, dx) = HostID(0, 0);
    let vendorID = vendorIDFromRegs(bx, cx, dx);

    let (ax, _, cx, dx) = HostID(1, 0);
    let featureBlock0 = cx;
    let featureBlock1 = dx;
    let (ef, em, pt, f, m, sid) = signatureSplit(ax);

    let (_, bx, cx, _) = HostID(7, 0);
    let featureBlock2 = bx;
    let featureBlock3 = cx;

    let mut featureBlock4 = 0;
    if (featureBlock0 & (1 << 26)) != 0 {
        let (tmp, _, _, _) = HostID(CpuidFunction::xSaveInfo as u32, 1);
        featureBlock4 = tmp;
    }

    let mut featureBlock5 = 0;
    let mut featureBlock6 = 0;
    let (ax, _, _, _) = HostID(ExtendedFunction::extendedFunctionInfo as u32, 0);
    if ax > ExtendedFunction::extendedFunctionInfo as u32 {
        let (_, _, cx, dx) = HostID(ExtendedFunction::extendedFeatures as u32, 0);
        featureBlock5 = cx;
        featureBlock6 = dx & !(X86Feature::Block6DuplicateMask as u32);
    }

    let set = setFromBlockMasks(&[
        featureBlock0,
        featureBlock1,
        featureBlock2,
        featureBlock3,
        featureBlock4,
        featureBlock5,
        featureBlock6,
    ]);
    return FeatureSet {
        Set: set,
        VendorID: vendorID,
        ExtendedFamily: ef,
        ExtendedModel: em,
        ProcessorType: pt,
        Family: f,
        Model: m,
        SteppingID: sid,
    };
}

// Helper to convert 3 regs into 12-byte vendor ID.
fn vendorIDFromRegs(bx: u32, cx: u32, dx: u32) -> String {
    let mut bytes: Vec<u8> = Vec::with_capacity(12);

    for i in 0..4 {
        let b = (bx >> (i * 8)) as u8;
        bytes.push(b);
    }

    for i in 0..4 {
        let b = (dx >> (i * 8)) as u8;
        bytes.push(b);
    }

    for i in 0..4 {
        let b = (cx >> (i * 8)) as u8;
        bytes.push(b);
    }

    return String::from_utf8(bytes).unwrap();
}

fn initFeaturesFromString() {
    let mut map = X86_FEATURES_FROM_STRING.lock();
    for (f, s) in &X86_FEATURE_STRINGS.0 {
        map.insert(s.to_string(), *f);
    }

    for (f, s) in &X86_FEATURE_PARSE_ONLY_STRINGS.0 {
        map.insert(s.to_string(), *f);
    }
}
